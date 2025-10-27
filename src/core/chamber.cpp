#include "chamber.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Chamber::Chamber(float xmin, float xmax,
    float ymin, float ymax,
    float zmin, float zmax,
    float cellSize)
    : xmin_(xmin), xmax_(xmax),
    ymin_(ymin), ymax_(ymax),
    zmin_(zmin), zmax_(zmax),
    cellSize_(cellSize)
{
    float domainX = xmax_ - xmin_;
    float domainY = ymax_ - ymin_;
    float domainZ = zmax_ - zmin_;

    float cellsX = domainX / cellSize_;
    float cellsY = domainY / cellSize_;
    float cellsZ = domainZ / cellSize_;

    int nX = static_cast<int>(std::round(cellsX));
    int nY = static_cast<int>(std::round(cellsY));
    int nZ = static_cast<int>(std::round(cellsZ));

    constexpr float eps = 1e-6f;
    if (std::fabs(cellsX - nX) > eps ||
        std::fabs(cellsY - nY) > eps ||
        std::fabs(cellsZ - nZ) > eps) {
        throw std::invalid_argument(
            "chamber: each (max-min)/cellSize must be an integer (within tolerance)");
    }

    nCellsX_ = nX;
    nCellsY_ = nY;
    nCellsZ_ = nZ;
}

std::tuple<int, int, int> Chamber::cellIndex(float x, float y, float z) const {
    float fx = x - xmin_;
    float fy = y - ymin_;
    float fz = z - zmin_;

    int i = static_cast<int>(fx / cellSize_);
    int j = static_cast<int>(fy / cellSize_);
    int k = static_cast<int>(fz / cellSize_);

    i = std::clamp(i, 0, nCellsX_ - 1);
    j = std::clamp(j, 0, nCellsY_ - 1);
    k = std::clamp(k, 0, nCellsZ_ - 1);

    return { i, j, k };
}

std::tuple<int, int, int> Chamber::cellIndex(const Particle& p) const {
    return cellIndex(p.x, p.y, p.z);
}

float Chamber::xmin() const { return xmin_; }
float Chamber::xmax() const { return xmax_; }
float Chamber::ymin() const { return ymin_; }
float Chamber::ymax() const { return ymax_; }
float Chamber::zmin() const { return zmin_; }
float Chamber::zmax() const { return zmax_; }