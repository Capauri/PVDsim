#pragma once

#include <cstdint>
#include <vector>

enum Species : uint8_t {
    SPECIES_ARGON = 0,
    SPECIES_GOLD = 1,
};

struct Particle {
    float x, y, z;
    float x_prev, y_prev, z_prev;
    Species type;
};

class Chamber;

void initParticles(
    std::vector<Particle>& particles,
    const Chamber& chamber,
    float initSpeedMin,
    float initSpeedMax,
    float dt_init
);
