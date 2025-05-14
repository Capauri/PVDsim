#pragma once

typedef struct cudaGraphicsResource cudaGraphicsResource;
#include "particle.hpp"
#include "chamber.hpp"

/**
 * init rendering resources for particle visualization.
 * creates a VBO registered with CUDA to display a subset of particles.
 *
 * @param totalParticles  total number of particles in the simulation.
 */
void initRenderer(size_t totalParticles);

/**
 * map CUDA-registered VBO, launch kernel to sample and copy particle data into it,
 * unmap the VBO, and render the sampled particles.
 *
 * @param dt               time step for simulation update.
 * @param devParticleArray pointer to device memory for full particle array.
 * @param totalParticles   total number of particles in devParticleArray.
 */
void updateAndDraw(float dt, Particle* devParticleArray, size_t totalParticles);

/**
 * clean up rendering resources: unregister CUDA resource and delete VBO.
 */
void cleanupRenderer();
