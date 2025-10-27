#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "particle.hpp"

__global__
void updateKernel(
    Particle* ps,
    int n,
    float dt,
    float gravity,
    float ymin
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle& p = ps[i];

    float vx = p.x - p.x_prev;
    float vy = p.y - p.y_prev;
    float vz = p.z - p.z_prev;

    float ax = 0.f;
    float ay = (p.type == SPECIES_ARGON) ? 0.f : -gravity;
    float az = 0.f;

    float x_new = 2.f * p.x - p.x_prev + ax * dt * dt;
    float y_new = 2.f * p.y - p.y_prev + ay * dt * dt;
    float z_new = 2.f * p.z - p.z_prev + az * dt * dt;

    if (y_new <= ymin) {
        y_new = ymin;
        x_new = p.x;
        z_new = p.z;
    }

    p.x_prev = p.x;
    p.y_prev = p.y;
    p.z_prev = p.z;

    p.x = x_new;
    p.y = y_new;
    p.z = z_new;
}


void launchUpdate(
    Particle* devPs,
    int n,
    float dt,
    float gravity,
    float ymin
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    updateKernel << <blocks, threads >> > (devPs, n, dt, gravity, ymin);
    cudaDeviceSynchronize();
}
__global__
void injectArKernel(
    Particle* pOut,
    int       N_emit,
    float     xmin,
    float     xmax,
    float     zmin,
    float     zmax,
    float     y_fixed,
    float     vy_down
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N_emit) return;

    curandState rng;
    curand_init(clock64(), i, 0, &rng);

    float x = xmin + (xmax - xmin) * curand_uniform(&rng);
    float z = zmin + (zmax - zmin) * curand_uniform(&rng);
    float vy = vy_down;

    Particle p;
    p.x = x;
    p.y = y_fixed;
    p.z = z;
    p.x_prev = x;
    p.y_prev = y_fixed - vy * 0.01f;
    p.z_prev = z;
    p.type = SPECIES_ARGON;

    pOut[i] = p;
}

void launchInjectArIons(
    Particle* devOut,
    int       N_emit,
    float     xmin, float xmax,
    float     zmin, float zmax,
    float     y_fixed,
    float     vy_down
) {
    int threads = 128;
    int blocks = (N_emit + threads - 1) / threads;

    injectArKernel << <blocks, threads >> > (
        devOut, N_emit,
        xmin, xmax,
        zmin, zmax,
        y_fixed, vy_down
        );

    cudaDeviceSynchronize();
}

__global__
void countArgon(Particle* ps, int n, int* outCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (ps[i].type == SPECIES_ARGON) {
        atomicAdd(outCount, 1);
    }
}

void debugCountArgon(Particle* devPs, int n) {
    int* devCount;
    int hostCount = 0;
    cudaMalloc(&devCount, sizeof(int));
    cudaMemcpy(devCount, &hostCount, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    countArgon << <blocks, threads >> > (devPs, n, devCount);

    cudaMemcpy(&hostCount, devCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devCount);

}
