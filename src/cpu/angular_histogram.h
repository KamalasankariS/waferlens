#ifndef WAFERLENS_CPU_ANGULAR_HISTOGRAM_H
#define WAFERLENS_CPU_ANGULAR_HISTOGRAM_H

#include <cstdint>
#include <vector>

namespace waferlens {
namespace cpu {

/**
 * @brief Compute the angular defect histogram on CPU.
 *
 * Bins defective cells by their angle from the wafer center, measured
 * counter-clockwise from the notch position.  Each bin is normalized
 * by the total defect count.
 *
 * @param data          Row-major uint8_t wafer map.
 * @param rows          Number of rows.
 * @param cols          Number of columns.
 * @param num_sectors   Number of angular sectors.
 * @param notch_angle   Notch position in radians.
 * @return              Vector of length num_sectors with normalized counts.
 */
std::vector<float> angular_histogram(const uint8_t* data, int rows, int cols,
                                     int num_sectors, float notch_angle);

}  // namespace cpu
}  // namespace waferlens

#endif  // WAFERLENS_CPU_ANGULAR_HISTOGRAM_H
