// src/render/renderer.cu

#include "renderer.hpp"
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstddef>              // for offsetof
#include "particle.hpp"

// ------------ globals ------------
static GLuint                particleVBO = 0;
static GLuint                particleVAO = 0;   // Vertex Array Object
static cudaGraphicsResource* cudaVboResource = nullptr;
static size_t                numSamplePoints = 0;

// ------------ sampling kernel ------------
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

// ------------ initRenderer ------------
void initRenderer(size_t totalParticles)
{
    // set clear color
    glClearColor(137.0f, 190.0f, 196.0f, 1.0f);
    glPointSize(3.0f);

    // determine points to draw (1% of total)
    numSamplePoints = totalParticles / 50;

    // create and bind VAO (required in core profile)
    glGenVertexArrays(1, &particleVAO);
    glBindVertexArray(particleVAO);

    // create and allocate VBO
    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER,
        numSamplePoints * sizeof(Particle),
        nullptr,
        GL_DYNAMIC_DRAW);

    // configure vertex attribute (position = first two floats of Particle)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Particle),
        (void*)offsetof(Particle, x));

    // unbind for cleanliness
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(
        &cudaVboResource,
        particleVBO,
        cudaGraphicsMapFlagsWriteDiscard
    );
}

// ------------ updateAndDraw ------------
void updateAndDraw(float         dt,
    Particle* devParticleArray,
    size_t        totalParticles)
{
    // map OpenGL buffer for CUDA write
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

    // unmap buffer
    cudaGraphicsUnmapResources(1, &cudaVboResource, 0);

    // clear and draw points
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(particleVAO);
    glDrawArrays(GL_POINTS, 0, (GLsizei)numSamplePoints);
    glBindVertexArray(0);
}

// ------------ cleanupRenderer ------------
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
