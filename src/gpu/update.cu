#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <curand_kernel.h>
#include <cmath>
#include "gpu_api.h"
#include "particle.hpp"

__global__ void updateKernel(
    Particle* ps, int n, float dt,
    float xmin, float xmax,
    float ymin, float ymax,
    int   nCellsX, int nCellsY,
    float cellW, float cellH,
    int* cellIndices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto& p = ps[idx];

    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.y < ymin) {
        p.y = ymin;
        p.vx = 0;
        p.vy = 0;
    }

    int tmpx = int((p.x - xmin) / cellW);
    if (tmpx < 0) tmpx = 0;
    else if (tmpx > nCellsX - 1) tmpx = nCellsX - 1;
    int tmpy = int((p.y - ymin) / cellH);
    if (tmpy < 0) tmpy = 0;
    else if (tmpy > nCellsY - 1) tmpy = nCellsY - 1;

    cellIndices[idx] = tmpy * nCellsX + tmpx;
}

__global__ void clampBottom(
    Particle* ps, int n, float ymin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    auto& p = ps[idx];
    if (p.y <= ymin) {
        p.vx = 0;
        p.vy = 0;
    }
}

__global__ void findCellBounds(
    int* sortedCell, int n,
    int  nCells,
    int* cellStart,
    int* cellEnd
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int c = sortedCell[tid];
    if (tid == 0 || sortedCell[tid - 1] != c) cellStart[c] = tid;
    if (tid == n - 1 || sortedCell[tid + 1] != c) cellEnd[c] = tid + 1;
}

__global__ void initRNG(curandState* states, unsigned long seed) {
    int cell = blockIdx.x;
    curand_init(seed, cell, 0, &states[cell]);
}

// DSMC collision kernel
__global__ void collideDSMC(
    float2* vel,
    int* sortedIdx,
    int* cellStart,
    int* cellEnd,
    curandState* states,
    float   dt,
    float   sigma,
    float   Vcell,
    float   maxRelSpeed
) {
    int cell = blockIdx.x;
    int start = cellStart[cell];
    int end = cellEnd[cell];
    int Nc = end - start;
    if (Nc < 2) return;

    int nPairs = int(0.5f * Nc * (Nc - 1) * sigma * dt / Vcell + 0.5f);
    curandState& st = states[cell];

    for (int i = 0; i < nPairs; ++i) {
        int a = start + (curand(&st) % Nc);
        int b = start + (curand(&st) % Nc);
        if (a == b) continue;
        int ia = sortedIdx[a];
        int ib = sortedIdx[b];
        float2 v1 = vel[ia];
        float2 v2 = vel[ib];
        float2 dv = { v1.x - v2.x, v1.y - v2.y };
        float  rel = sqrtf(dv.x * dv.x + dv.y * dv.y);
        if (curand_uniform(&st) * maxRelSpeed < rel) {
            float phi = 2.f * 3.1415926535f * curand_uniform(&st);
            float2 gnew = { rel * cosf(phi), rel * sinf(phi) };
            float2 vcm = { 0.5f * (v1.x + v2.x), 0.5f * (v1.y + v2.y) };
            vel[ia] = { vcm.x + 0.5f * gnew.x, vcm.y + 0.5f * gnew.y };
            vel[ib] = { vcm.x - 0.5f * gnew.x, vcm.y - 0.5f * gnew.y };
        }
    }
}

void launchUpdate(
    Particle* devPs, int n, float dt,
    float xmin, float xmax, float ymin, float ymax,
    int nCellsX, int nCellsY,
    float cellW, float cellH,
    int* cellIndices,
    int* particleIndices,
    int* cellStart,
    int* cellEnd,
    curandState* randStates,
    float sigma, float Vcell, float maxRelSpeed
) {
    int blocks = (n + 255) / 256;

    updateKernel << <blocks, 256 >> > (
        devPs, n, dt,
        xmin, xmax,
        ymin, ymax,
        nCellsX, nCellsY,
        cellW, cellH,
        cellIndices
        );
    cudaDeviceSynchronize();

    thrust::device_ptr<int> dCell(cellIndices);
    thrust::device_ptr<int> dIdx(particleIndices);
    thrust::sequence(dIdx, dIdx + n);
    thrust::sort_by_key(dCell, dCell + n, dIdx);

    findCellBounds << <blocks, 256 >> > (
        cellIndices, n,
        nCellsX * nCellsY,
        cellStart, cellEnd
        );
    cudaDeviceSynchronize();

    // collisions
    collideDSMC << <nCellsX * nCellsY, 1 >> > (
        reinterpret_cast<float2*>(devPs),
        dIdx.get(),
        cellStart, cellEnd,
        randStates,
        dt, sigma, Vcell, maxRelSpeed
        );
    cudaDeviceSynchronize();


    clampBottom << <blocks, 256 >> > (devPs, n, ymin);
    cudaDeviceSynchronize();
}

void initRNGStates(curandState* states, int nCells, unsigned long seed) {
    initRNG << <nCells, 1 >> > (states, seed);
    cudaDeviceSynchronize();
}
