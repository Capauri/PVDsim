#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include <cuda_runtime.h>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "particle.hpp"
#include "chamber.hpp"
#include "chamber_device.cuh"
#include "update.cuh"
#include "renderer.hpp"

// Allocate device buffer and copy host particles to GPU
static Particle* allocDev(size_t n, const Particle* host) {
    Particle* devPtr = nullptr;
    cudaMalloc(&devPtr, n * sizeof(Particle));
    cudaMemcpy(devPtr, host, n * sizeof(Particle), cudaMemcpyHostToDevice);
    return devPtr;
}

int main() {
    const int N = 1000;
    const float B = 1.0f;        // Cube bounds: [-B, B]^3
    const float dt = 0.01f;
    const float gravity = -0.5f;
    const float ymin = -B;

    // Injection parameters
    const int N_max = 10000;
    const float sourceXmin = -0.5f, sourceXmax = 0.5f;
    const float sourceZmin = -0.5f, sourceZmax = 0.5f;
    const float sourceY = 1.0f;    // top of chamber
    const float flux = 1e19f;      // atoms/m^2/s (real units)
    const float ionEnergy_eV = 500.0f;
    const float m_Ar = 6.63e-26f;
    const float e_J = ionEnergy_eV * 1.602e-19f;
    const float ionSpeed = std::sqrt(2.0f * e_J / m_Ar);
    const float vy_down = -0.5f;

    // Macro-particle weight: 1 sim-particle = 1e15 real atoms
    const float macroWeight = 1e15f;

    // 1) Initialize particles in cube with downward velocity
    std::vector<Particle> hostP(N);
    std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(-B, B);
    for (auto& p : hostP) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
        float vy = -2.0f;
        p.x_prev = p.x;
        p.y_prev = p.y - vy * dt;
        p.z_prev = p.z;
    }

    // 2) Copy to GPU (with padding)
    std::vector<Particle> paddedHostP(N_max);
    std::copy(hostP.begin(), hostP.end(), paddedHostP.begin());
    Particle* devP = allocDev(N_max, paddedHostP.data());
    int currentCount = N;

    // 3) Initialize GLFW / OpenGL
    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "PVDsim 3D", nullptr, nullptr);
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

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, int w, int h) {
        glViewport(0, 0, w, h);
        });

    // 5) Enable depth test and point size
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // 6) Initialize renderer with max capacity
    initRenderer(N_max);

    // 7) Main loop
    std::cout << "Entering render loop…\n";
    while (!glfwWindowShouldClose(window)) {
        float sourceArea = (sourceXmax - sourceXmin) * (sourceZmax - sourceZmin);
        int N_emit = static_cast<int>(flux * sourceArea * dt / macroWeight);
        N_emit = std::min(N_emit, N_max - currentCount);

        if (N_emit > 0) {
            launchInjectArIons(
                devP + currentCount,
                N_emit,
                sourceXmin, sourceXmax,
                sourceZmin, sourceZmax,
                sourceY,
                vy_down
            );
            currentCount += N_emit;
        }

        // Update particles on GPU
        launchUpdate(devP, currentCount, dt, gravity, ymin);

        // Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        updateAndDraw(0.0f, devP, static_cast<size_t>(currentCount));

        glfwSwapBuffers(window);
        glfwPollEvents();

        debugCountArgon(devP, currentCount);
    }

    // Cleanup
    cleanupRenderer();
    cudaFree(devP);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}