#include <cuda_runtime.h>
#include <cstdio>

// A simple device kernel — __global__ **must** be on the function, not as a declspec
__global__ void dummyKernel() {
    printf("Hello from GPU block %d, thread %d\n",
        blockIdx.x, threadIdx.x);
}

// Exposed with C linkage for the host to call
extern "C" void launchDummyKernel() {
    // launch 2 blocks of 4 threads each
    dummyKernel <<<2, 4 >>> ();

    // wait and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(err));
    }
}
