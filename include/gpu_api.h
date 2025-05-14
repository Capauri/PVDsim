#pragma once
#include "particle.hpp"
#include <curand_kernel.h>

void launchUpdate(
    Particle* devPs, int n, float dt,
    float xmin, float xmax, float ymin, float ymax,
    int nCellsX, int nCellsY,
    float cellW, float cellH,
    int* particleIndices,
    int* cellStart,
    int* cellEnd,
    curandState* randStates,
    float sigma, float Vcell, float maxRelSpeed
);

void initRNGStates(
    curandState* states, 
    int nCells, 
    unsigned long seed
);

