#pragma once
#include <vector>

#pragma once
#include <vector>
class Chamber;
struct Particle {
    float x, y;
    float x_prev, y_prev;
};
void initParticles(
    std::vector<Particle>& particles,
    const Chamber& chamber,
    float initSpeedMin,
    float initSpeedMax,
    float dt_init
);