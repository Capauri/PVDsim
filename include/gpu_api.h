#pragma once
#include "particle.hpp"

extern "C" void launchUpdate(
    Particle* devPs,
    int       n,
    float     dt,
    float     xmin,
    float     xmax,
    float     ymin,
    float     ymax
);
