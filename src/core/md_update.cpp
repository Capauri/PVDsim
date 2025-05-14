#include "particle.hpp"
#include <vector>
#include <iostream>

void cpuUpdate(std::vector<Particle>& particles, float dt) {
    for (auto& p : particles) {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
    }
}

void testCPU() {
    std::vector<Particle> v(4);
    for (int i = 0; i < 4; ++i) {
        v[i] = { float(i), float(i), 1.0f, 0.5f };
    }
    cpuUpdate(v, 0.1f);
    for (auto& p : v)
        std::cout << "CPU-> x=" << p.x << " y=" << p.y << "\n";
}
