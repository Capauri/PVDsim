#pragma once

#include <vector>

class Chamber;

struct Particle {
    float x, y;
    float vx, vy;
};

void initParticles(
    std::vector<Particle>& particles,
    const Chamber& chamber,
    float initSpeedMin,
    float initSpeedMax
);
