#include <iostream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <curand_kernel.h>
#include "chamber.hpp"
#include "particle.hpp"
#include "gpu_api.h"
#include "renderer.hpp"

static Particle* allocDev(size_t n, const Particle* host) {
    Particle* devPtr = nullptr;
    if (cudaMalloc(&devPtr, n * sizeof(Particle)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        return nullptr;
    }
    if (cudaMemcpy(devPtr, host, n * sizeof(Particle), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "cudaMemcpy to device failed\n";
        cudaFree(devPtr);
        return nullptr;
    }
    return devPtr;
}

int main() {
    const int   N = 10000;
    const float dt = 0.001f;
    const float dt_init = dt;
    const float cellSize = 0.01f;
    const float sigma = 1e-4f;
    const float initSpeedMin = 0.5f;
    const float initSpeedMax = 2.0f;
    const float maxRelSpeed = 10.0f;

    Chamber chamber(-1.0f, 1.0f,
        -1.0f, 1.0f,
        cellSize);

    int nCellsX = static_cast<int>((chamber.xmax() - chamber.xmin()) / cellSize);
    int nCellsY = static_cast<int>((chamber.ymax() - chamber.ymin()) / cellSize);
    int nCells = nCellsX * nCellsY;
    float Vcell = cellSize * cellSize;

    std::vector<Particle> hostP(N);
    initParticles(hostP, chamber, initSpeedMin, initSpeedMax, dt_init);

    Particle* devP = allocDev(N, hostP.data());
    if (!devP) return -1;

    curandState* d_randStates = nullptr;
    cudaMalloc(&d_randStates, nCells * sizeof(curandState));
    initRNGStates(d_randStates, nCells, static_cast<unsigned long>(time(nullptr)));

    int* d_cellIdx = nullptr;
    int* d_partIdx = nullptr;
    int* d_cellStart = nullptr;
    int* d_cellEnd = nullptr;
    cudaMalloc(&d_cellIdx, N * sizeof(int));
    cudaMalloc(&d_partIdx, N * sizeof(int));
    cudaMalloc(&d_cellStart, nCells * sizeof(int));
    cudaMalloc(&d_cellEnd, nCells * sizeof(int));

    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "PVDsim", nullptr, nullptr);
    if (!window) {
        std::cerr << "glfwCreateWindow failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "gladLoadGLLoader failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    initRenderer(N);

    while (!glfwWindowShouldClose(window)) {
        launchUpdate(
            devP, N, dt,
            chamber.xmin(), chamber.xmax(),
            chamber.ymin(), chamber.ymax(),
            nCellsX, nCellsY,
            cellSize, cellSize, 
            d_partIdx,
            d_cellStart, d_cellEnd,
            d_randStates,
            sigma, Vcell, maxRelSpeed
        );

        glClear(GL_COLOR_BUFFER_BIT);
        updateAndDraw(dt, devP, static_cast<size_t>(N));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanupRenderer();
    cudaFree(devP);
    cudaFree(d_randStates);
    cudaFree(d_cellIdx);
    cudaFree(d_partIdx);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}