#include "gpu_api.h"
#include "particle.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <curand_kernel.h>


// simulation update kernel
__global__
void updateKernel(Particle* ps,
    int n,
    float dt,
    float xmin, float xmax,
    float ymin, float ymax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Particle& p = ps[idx];

    // collision and position update per dt
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.x < xmin || p.x > xmax)  p.vx = -p.vx;
    if (p.y < ymin || p.y > ymax)  p.vy = 0;
}

extern "C" void launchUpdate(
    Particle* devPs,
    int       n,
    float     dt,
    float     xmin,
    float     xmax,
    float     ymin,
    float     ymax
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    updateKernel << <blocks, threads >> > (devPs, n, dt, xmin, xmax, ymin, ymax);
    cudaDeviceSynchronize();
}
