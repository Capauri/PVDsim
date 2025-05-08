#pragma once
#include <cuda_runtime.h> 
#include "particle.hpp"

__device__
inline void enforceBoundsDevice(Particle& p,
    float xmin, float xmax,
    float ymin, float ymax) {
    if (p.x < xmin || p.x > xmax) p.vx = -p.vx;

    if (p.y <= ymin)           p.vy = 0.0f;
    else if (p.y > ymax)       p.vy = -p.vy;

    p.x = fminf(fmaxf(p.x, xmin), xmax);
    p.y = fminf(fmaxf(p.y, ymin), ymax);
}
