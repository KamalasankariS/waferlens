#include "ring_score.h"

#include <algorithm>

namespace waferlens {
namespace cpu {

RingScoreResult ring_score(const std::vector<float>& radial_density) {
    RingScoreResult result{1.0f, 0};

    if (radial_density.empty()) {
        return result;
    }

    auto it = std::max_element(radial_density.begin(), radial_density.end());
    result.peak_bin_index = static_cast<int>(it - radial_density.begin());

    float peak = *it;
    float num_bins = static_cast<float>(radial_density.size());

    result.score = peak * num_bins;

    return result;
}

}  // namespace cpu
}  // namespace waferlens
