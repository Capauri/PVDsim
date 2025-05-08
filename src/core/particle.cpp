#include "particle.hpp"
#include "chamber.hpp"
#include <random>

void initParticles(std::vector<Particle>& hostP,
    const Chamber& chamber,
    float initSpeedMin,
    float initSpeedMax) {
    std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> posX(chamber.xmin(), chamber.xmax());
    std::uniform_real_distribution<float> distV(initSpeedMin, initSpeedMax);
    std::uniform_real_distribution<float> distX(-4.0, 4.0);

    for (auto& p : hostP) {
        p.x = posX(rng);
        p.y = chamber.ymax();
        p.vx = distX(rng);
        p.vy = -distV(rng);
    }
}
