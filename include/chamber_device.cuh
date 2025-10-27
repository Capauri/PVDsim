#pragma once
#include <cuda_runtime.h> 
#include "particle.hpp"

__device__
inline void enforceBoundsDevice(Particle& p,
    float xmin, float xmax,
    float ymin, float ymax,
    float zmin, float zmax) {

    p.x = fminf(fmaxf(p.x, xmin), xmax);
    p.y = fminf(fmaxf(p.y, ymin), ymax);
    p.z = fminf(fmaxf(p.z, zmin), zmax);
}



void launchInjectArIons(
    Particle* devOut,
    int       N_emit,
    float     xmin, float xmax,
    float     zmin, float zmax,
    float     y_fixed,
    float     vy_down
);
