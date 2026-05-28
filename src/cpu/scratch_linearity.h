#ifndef WAFERLENS_CPU_SCRATCH_LINEARITY_H
#define WAFERLENS_CPU_SCRATCH_LINEARITY_H

#include <cstdint>

namespace waferlens {
namespace cpu {

/**
 * @brief Result of the scratch linearity analysis.
 */
struct ScratchResult {
    float linearity;  ///< Normalized peak strength in [0, 1].
    float angle;      ///< Peak angle in degrees.
};

/**
 * @brief Detect linear (scratch-like) defect patterns via Radon transform.
 *
 * Projects the binary defect map onto a set of angles and identifies the
 * strongest linear accumulation.  The linearity score is the peak sinogram
 * value normalized by the theoretical maximum for the given defect count.
 *
 * @param data        Row-major uint8_t wafer map.
 * @param rows        Number of rows.
 * @param cols        Number of columns.
 * @param num_angles  Number of projection angles in [0, 180).
 * @return            ScratchResult with linearity score and peak angle.
 */
ScratchResult scratch_linearity(const uint8_t* data, int rows, int cols,
                                int num_angles);

}  // namespace cpu
}  // namespace waferlens

#endif  // WAFERLENS_CPU_SCRATCH_LINEARITY_H
