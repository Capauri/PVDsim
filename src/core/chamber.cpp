// src/core/chamber.cpp

#include "chamber.hpp"
#include "particle.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Chamber::Chamber(float xmin, float xmax,
    float ymin, float ymax,
    float cellSize)
    : xmin_(xmin),
    xmax_(xmax),
    ymin_(ymin),
    ymax_(ymax),
    cellSize_(cellSize)
{
    float domainX = xmax_ - xmin_;
    float domainY = ymax_ - ymin_;

    // compute floating-point cell counts
    float cellsX = domainX / cellSize_;
    float cellsY = domainY / cellSize_;

    int nX = static_cast<int>(std::round(cellsX));
    int nY = static_cast<int>(std::round(cellsY));

    // error tolerance
    constexpr float eps = 1e-6f;
    if (std::fabs(cellsX - nX) > eps) {
        throw std::invalid_argument(
            "chamber: (xmax - xmin)/cellSize must be an integer (within tolerance)");
    }
    if (std::fabs(cellsY - nY) > eps) {
        throw std::invalid_argument(
            "chamber: (ymax - ymin)/cellSize must be an integer (within tolerance)");
    }

    nCellsX_ = nX;
    nCellsY_ = nY;
}

std::pair<int, int> Chamber::cellIndex(float x, float y) const {
    // local chamber coords
    float fx = x - xmin_;
    float fy = y - ymin_;

    int i = static_cast<int>(fx / cellSize_);
    int j = static_cast<int>(fy / cellSize_);

    i = std::clamp(i, 0, nCellsX_ - 1);
    j = std::clamp(j, 0, nCellsY_ - 1);

    return { i, j };
}

std::pair<int, int> Chamber::cellIndex(const Particle& p) const {
    return cellIndex(p.x, p.y);
}

float Chamber::xmin() const { return xmin_; }
float Chamber::xmax() const { return xmax_; }
float Chamber::ymin() const { return ymin_; }
float Chamber::ymax() const { return ymax_; }
