#pragma once
#include <utility>
#include "particle.hpp"

class Chamber {
public:
    Chamber(float xmin, float xmax,
        float ymin, float ymax,
        float cellSize = 1.0f);

    float xmin() const;
    float xmax() const;
    float ymin() const;
    float ymax() const;

    std::pair<int, int> cellIndex(float x, float y) const;
    std::pair<int, int> cellIndex(const Particle& p) const;

private:
    float xmin_, xmax_, ymin_, ymax_;
    float cellSize_;
    int   nCellsX_, nCellsY_;
};
