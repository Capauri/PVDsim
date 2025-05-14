#include "renderer.hpp"
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstddef>
#include "particle.hpp"

static GLuint                particleVBO = 0;
static GLuint                particleVAO = 0;
static cudaGraphicsResource* cudaVboResource = nullptr;
static size_t                numSamplePoints = 0;

__global__
void sampleParticlesKernel(Particle* vboData,
    const Particle* fullData,
    int             totalParticles,
    int             numSamplePoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSamplePoints) return;
    int step = totalParticles / numSamplePoints;
    vboData[idx] = fullData[idx * step];
}

void initRenderer(size_t totalParticles)
{
    glClearColor(137.0f, 190.0f, 196.0f, 1.0f);
    glPointSize(3.0f);

    numSamplePoints = totalParticles / 50;

    glGenVertexArrays(1, &particleVAO);
    glBindVertexArray(particleVAO);
    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER,
        numSamplePoints * sizeof(Particle),
        nullptr,
        GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Particle),
        (void*)offsetof(Particle, x));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    cudaGraphicsGLRegisterBuffer(
        &cudaVboResource,
        particleVBO,
        cudaGraphicsMapFlagsWriteDiscard
    );
}

void updateAndDraw(float         dt,
    Particle* devParticleArray,
    size_t        totalParticles)
{
    cudaGraphicsMapResources(1, &cudaVboResource, 0);
    void* devPtr = nullptr;
    size_t mappedSize = 0;
    cudaGraphicsResourceGetMappedPointer(
        &devPtr,
        &mappedSize,
        cudaVboResource
    );

    // launch sampling kernel
    Particle* vboPtr = reinterpret_cast<Particle*>(devPtr);
    int threads = 256;
    int blocks = int((numSamplePoints + threads - 1) / threads);
    sampleParticlesKernel << <blocks, threads >> > (
        vboPtr,
        devParticleArray,
        int(totalParticles),
        int(numSamplePoints)
        );
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cudaVboResource, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(particleVAO);
    glDrawArrays(GL_POINTS, 0, (GLsizei)numSamplePoints);
    glBindVertexArray(0);
}

void cleanupRenderer()
{
    if (cudaVboResource) {
        cudaGraphicsUnregisterResource(cudaVboResource);
        cudaVboResource = nullptr;
    }
    if (particleVBO) {
        glDeleteBuffers(1, &particleVBO);
        particleVBO = 0;
    }
    if (particleVAO) {
        glDeleteVertexArrays(1, &particleVAO);
        particleVAO = 0;
    }
}
