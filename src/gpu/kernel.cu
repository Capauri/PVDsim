#include <cuda_runtime.h>
#include <cstdio>

__global__ void dummyKernel() {
    printf("Hello from GPU block %d, thread %d\n",
        blockIdx.x, threadIdx.x);
}

extern "C" void launchDummyKernel() {
    dummyKernel <<<2, 4 >>> ();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(err));
    }
}
