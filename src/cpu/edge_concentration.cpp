#include "edge_concentration.h"

#include <cmath>

namespace waferlens {
namespace cpu {

float edge_concentration(const uint8_t* data, int rows, int cols,
                         float edge_fraction) {
    const float center_r = static_cast<float>(rows - 1) / 2.0f;
    const float center_c = static_cast<float>(cols - 1) / 2.0f;
    const float max_radius = std::sqrt(center_r * center_r + center_c * center_c);
    const float threshold = (1.0f - edge_fraction) * max_radius;

    int edge_defects = 0;
    int total_defects = 0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (data[r * cols + c] != 2) {
                continue;
            }
            ++total_defects;

            float dr = static_cast<float>(r) - center_r;
            float dc = static_cast<float>(c) - center_c;
            float dist = std::sqrt(dr * dr + dc * dc);

            if (dist >= threshold) {
                ++edge_defects;
            }
        }
    }

    if (total_defects == 0) {
        return 0.0f;
    }

    return static_cast<float>(edge_defects) / static_cast<float>(total_defects);
}

}  // namespace cpu
}  // namespace waferlens
