#ifndef WAFERLENS_CPU_RADIAL_DENSITY_H
#define WAFERLENS_CPU_RADIAL_DENSITY_H

#include <cstdint>
#include <vector>

namespace waferlens {
namespace cpu {

/**
 * @brief Compute the radial defect density profile on CPU.
 *
 * Bins every defective cell by its normalized distance from the wafer center
 * into @p num_bins concentric rings.  Each bin value is the fraction of total
 * defects falling in that ring.  If the wafer contains no defects, the
 * returned vector is all zeros.
 *
 * @param data      Row-major uint8_t wafer map (CellState encoding).
 * @param rows      Number of rows.
 * @param cols      Number of columns.
 * @param num_bins  Number of radial bins.
 * @return          Vector of length num_bins with normalized densities.
 */
std::vector<float> radial_density(const uint8_t* data, int rows, int cols,
                                  int num_bins);

}  // namespace cpu
}  // namespace waferlens

#endif  // WAFERLENS_CPU_RADIAL_DENSITY_H
