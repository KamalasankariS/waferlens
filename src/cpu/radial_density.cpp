#include "radial_density.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace waferlens {
namespace cpu {

std::vector<float> radial_density(const uint8_t* data, int rows, int cols,
                                  int num_bins) {
    std::vector<float> bins(num_bins, 0.0f);

    const float center_r = static_cast<float>(rows - 1) / 2.0f;
    const float center_c = static_cast<float>(cols - 1) / 2.0f;
    const float max_radius = std::sqrt(center_r * center_r + center_c * center_c);

    if (max_radius < 1e-6f) {
        return bins;
    }

    int total_defects = 0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (data[r * cols + c] != 2) {
                continue;
            }
            float dr = static_cast<float>(r) - center_r;
            float dc = static_cast<float>(c) - center_c;
            float dist = std::sqrt(dr * dr + dc * dc) / max_radius;

            int bin = static_cast<int>(dist * num_bins);
            bin = std::min(bin, num_bins - 1);

            bins[bin] += 1.0f;
            ++total_defects;
        }
    }

    if (total_defects > 0) {
        float inv_total = 1.0f / static_cast<float>(total_defects);
        for (float& v : bins) {
            v *= inv_total;
        }
    }

    return bins;
}

}  // namespace cpu
}  // namespace waferlens
