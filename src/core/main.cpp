#include <iostream>
#include <vector>

#include <cuda_runtime.h>      // cudaMalloc, cudaMemcpy
#include <glad/glad.h>         // openGL functions & types
#include <GLFW/glfw3.h>        // window/context management

#include "chamber.hpp"
#include "particle.hpp"
#include "gpu_api.h"           // launchUpdate
#include "renderer.hpp"        // initRenderer, updateAndDraw, cleanupRenderer

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
    // set program params here
    const int   N = 10000;
    const float dt = 0.001f;
    Chamber chamber(
        -1.0f, 1.0f,
        -1.0f, 1.0f,
        0.01f
    );

    // init
    std::vector<Particle> hostP(N);
    initParticles(hostP, chamber, 1.0f, 5.0f);
    Particle* devP = allocDev(N, hostP.data());
    if (!devP) return -1;

    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        cudaFree(devP);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "PVDsim", nullptr, nullptr);
    if (!window) {
        std::cerr << "glfwCreateWindow failed\n";
        glfwTerminate();
        cudaFree(devP);
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "gladLoadGLLoader failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        cudaFree(devP);
        return -1;
    }

    initRenderer(N);

    while (!glfwWindowShouldClose(window)) {
        // a) advance simulation
        launchUpdate(devP, N, dt,
            chamber.xmin(), chamber.xmax(),
            chamber.ymin(), chamber.ymax());

        // b) sample & render
        glClear(GL_COLOR_BUFFER_BIT);
        updateAndDraw(dt, devP, N);

        // c) swap buffers & poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanupRenderer();
    cudaFree(devP);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
