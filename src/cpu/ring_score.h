#ifndef WAFERLENS_CPU_RING_SCORE_H
#define WAFERLENS_CPU_RING_SCORE_H

#include <cstdint>
#include <vector>

namespace waferlens {
namespace cpu {

/**
 * @brief Result of the ring-score computation.
 */
struct RingScoreResult {
    float score;          ///< Peak-to-uniform concentration ratio.
    int   peak_bin_index; ///< Index of the radial bin with highest density.
};

/**
 * @brief Compute the ring score from a precomputed radial density profile.
 *
 * The ring score measures how strongly defects cluster in a single radial
 * band relative to a uniform baseline.  Defined as:
 *
 *     score = max(radial_density) * num_bins
 *
 * A perfectly uniform distribution yields a score of 1.0.  Higher values
 * indicate stronger ring-like concentration.
 *
 * @param radial_density  Normalized radial density vector (sums to ~1).
 * @return                RingScoreResult with score and peak index.
 */
RingScoreResult ring_score(const std::vector<float>& radial_density);

}  // namespace cpu
}  // namespace waferlens

#endif  // WAFERLENS_CPU_RING_SCORE_H
