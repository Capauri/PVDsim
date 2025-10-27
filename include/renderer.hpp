#pragma once
#include <cstddef>
#include "particle.hpp"

void initRenderer(size_t N);
void updateAndDraw(float t, const Particle* devP, size_t N);
void cleanupRenderer();
