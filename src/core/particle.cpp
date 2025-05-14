#include "particle.hpp"
#include "chamber.hpp"
#include <random>
#include <cmath>

void initParticles(std::vector<Particle>& hostP,
    const Chamber& chamber,
    float initSpeedMin,
    float initSpeedMax,
    float dt_init)
{
    std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> posX(chamber.xmin(), chamber.xmax());
    std::uniform_real_distribution<float> speedDist(initSpeedMin, initSpeedMax);
    const float deg = 15.f * 3.141592653589793f / 180.f;
    std::uniform_real_distribution<float> angDist(
        -0.5f * 3.141592653589793f - deg,
        -0.5f * 3.141592653589793f + deg
    );

    for (auto& p : hostP) {
        p.x = posX(rng);
        p.y = chamber.ymax();

        float speed = speedDist(rng);
        float phi = angDist(rng);
        float vx = speed * std::cos(phi);
        float vy = speed * std::sin(phi);

        p.x_prev = p.x - vx * dt_init;
        p.y_prev = p.y - vy * dt_init;
    }
}
