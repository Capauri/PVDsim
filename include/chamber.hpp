#pragma once
#include <tuple>
#include "particle.hpp"

class Chamber {
public:
    Chamber(float xmin, float xmax,
        float ymin, float ymax,
        float zmin, float zmax,
        float cellSize = 1.0f);

    float xmin() const;
    float xmax() const;
    float ymin() const;
    float ymax() const;
    float zmin() const;
    float zmax() const;

    std::tuple<int, int, int> cellIndex(float x, float y, float z) const;
    std::tuple<int, int, int> cellIndex(const Particle& p) const;

private:
    float xmin_, xmax_;
    float ymin_, ymax_;
    float zmin_, zmax_;
    float cellSize_;
    int   nCellsX_, nCellsY_, nCellsZ_;
};
