#ifndef WAFERLENS_CPU_DENSITY_GRADIENT_H
#define WAFERLENS_CPU_DENSITY_GRADIENT_H

#include <cstdint>
#include <vector>

namespace waferlens {
namespace cpu {

/**
 * @brief Compute the 2D gradient magnitude of local defect density on CPU.
 *
 * The wafer map is partitioned into a grid_size x grid_size grid.  Each cell
 * accumulates the count of defective pixels, then a 3x3 Sobel operator
 * computes the gradient magnitude at each grid cell.  The output is
 * normalized by the maximum gradient value (or all zeros if no gradient
 * exists).
 *
 * @param data       Row-major uint8_t wafer map.
 * @param rows       Number of rows.
 * @param cols       Number of columns.
 * @param grid_size  Side length of the density grid.
 * @return           Flattened row-major gradient magnitude, length grid_size^2.
 */
std::vector<float> density_gradient(const uint8_t* data, int rows, int cols,
                                    int grid_size);

}  // namespace cpu
}  // namespace waferlens

#endif  // WAFERLENS_CPU_DENSITY_GRADIENT_H
