#ifndef WAFERLENS_CPU_EDGE_CONCENTRATION_H
#define WAFERLENS_CPU_EDGE_CONCENTRATION_H

#include <cstdint>

namespace waferlens {
namespace cpu {

/**
 * @brief Compute the edge concentration ratio on CPU.
 *
 * Returns the fraction of defective cells whose distance from the wafer
 * center exceeds (1 - edge_fraction) * max_radius.
 *
 * @param data           Row-major uint8_t wafer map.
 * @param rows           Number of rows.
 * @param cols           Number of columns.
 * @param edge_fraction  Fractional width of the edge annulus (0, 1).
 * @return               Edge concentration ratio in [0, 1].
 */
float edge_concentration(const uint8_t* data, int rows, int cols,
                         float edge_fraction);

}  // namespace cpu
}  // namespace waferlens

#endif  // WAFERLENS_CPU_EDGE_CONCENTRATION_H
