#pragma once

#include "particle.hpp"

void debugCountArgon(Particle* devPs, int n);

void launchUpdate(
    Particle* devPs,
    int       n,
    float     dt,
    float     gravity,
    float     ymin
);
