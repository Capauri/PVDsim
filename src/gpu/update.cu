#include "particle.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>

__global__
void rngInitKernel(curandState* states, int nCells, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCells) return;
    curand_init(seed, idx, 0, &states[idx]);
}

void initRNGStates(curandState* states, int nCells, unsigned long seed) {
    int threads = 256;
    int blocks = (nCells + threads - 1) / threads;
    rngInitKernel << <blocks, threads >> > (states, nCells, seed);
    cudaDeviceSynchronize();
}

__global__
void collideDSMC(
    Particle* ps,
    int* particleIndices,
    int* cellStart,
    int* cellEnd,
    curandState* states,
    float        dt,
    float        sigma,
    float        Vcell,
    float        maxRelSpeed
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

        int ia = particleIndices[a];
        int ib = particleIndices[b];

        float vx1 = (ps[ia].x - ps[ia].x_prev) / dt;
        float vy1 = (ps[ia].y - ps[ia].y_prev) / dt;
        float vx2 = (ps[ib].x - ps[ib].x_prev) / dt;
        float vy2 = (ps[ib].y - ps[ib].y_prev) / dt;

        float2 dv = { vx1 - vx2, vy1 - vy2 };
        float  rel = sqrtf(dv.x * dv.x + dv.y * dv.y);

        if (curand_uniform(&st) * maxRelSpeed < rel) {
            float phi = 2.f * 3.141592653589793f * curand_uniform(&st);
            float2 gnew = { rel * cosf(phi), rel * sinf(phi) };
            float2 vcm = { 0.5f * (vx1 + vx2), 0.5f * (vy1 + vy2) };


            float newVx1 = vcm.x + 0.5f * gnew.x;
            float newVy1 = vcm.y + 0.5f * gnew.y;
            float newVx2 = vcm.x - 0.5f * gnew.x;
            float newVy2 = vcm.y - 0.5f * gnew.y;

            // encode them back into x_prev
            ps[ia].x_prev = ps[ia].x - newVx1 * dt;
            ps[ia].y_prev = ps[ia].y - newVy1 * dt;
            ps[ib].x_prev = ps[ib].x - newVx2 * dt;
            ps[ib].y_prev = ps[ib].y - newVy2 * dt;
        }
    }
}

__global__
void verletKernel(
    Particle* ps,
    int       N,
    float     dt,
    float     xmin, float xmax,
    float     ymin, float ymax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = ps[idx].x;
    float xp = ps[idx].x_prev;
    float y = ps[idx].y;
    float yp = ps[idx].y_prev;

    if (y == ymin && yp == ymin) {
        return;
    }

    float x_new = 2 * x - xp;
    float y_new = 2 * y - yp;

    if (y_new <= ymin) {
        float freezeX = x;
        ps[idx].x = freezeX;
        ps[idx].x_prev = freezeX;
        ps[idx].y = ymin;
        ps[idx].y_prev = ymin;
        return;
    }

    ps[idx].x_prev = x;
    ps[idx].y_prev = y;
    ps[idx].x = x_new;
    ps[idx].y = y_new;
}

void launchUpdate(
    Particle* devPs,
    int          n,
    float        dt,
    float        xmin, float xmax,
    float        ymin, float ymax,
    int          nCellsX, int nCellsY,
    float        cellW, float cellH,
    int* particleIndices,
    int* cellStart,
    int* cellEnd,
    curandState* randStates,
    float        sigma,
    float        Vcell,
    float        maxRelSpeed
) {
    int numCells = nCellsX * nCellsY;
    collideDSMC << <numCells, 1 >> > (
        devPs,
        particleIndices,
        cellStart,
        cellEnd,
        randStates,
        dt,
        sigma,
        Vcell,
        maxRelSpeed
        );

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    verletKernel << <blocks, threads >> > (
        devPs, n, dt,
        xmin, xmax,
        ymin, ymax
        );

    cudaDeviceSynchronize();
}
