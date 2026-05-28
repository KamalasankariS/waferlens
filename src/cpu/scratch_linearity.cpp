#include "scratch_linearity.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace waferlens {
namespace cpu {

ScratchResult scratch_linearity(const uint8_t* data, int rows, int cols,
                                int num_angles) {
    ScratchResult result{0.0f, 0.0f};

    const float center_r = static_cast<float>(rows - 1) / 2.0f;
    const float center_c = static_cast<float>(cols - 1) / 2.0f;
    const float max_radius = std::sqrt(center_r * center_r + center_c * center_c);
    const int   num_offsets = static_cast<int>(2.0f * max_radius) + 1;

    std::vector<std::pair<float, float>> defect_coords;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (data[r * cols + c] == 2) {
                defect_coords.push_back({
                    static_cast<float>(r) - center_r,
                    static_cast<float>(c) - center_c
                });
            }
        }
    }

    int total_defects = static_cast<int>(defect_coords.size());
    if (total_defects == 0) {
        return result;
    }

    float global_max = 0.0f;
    int   best_angle_idx = 0;

    for (int ai = 0; ai < num_angles; ++ai) {
        float theta = static_cast<float>(ai) * static_cast<float>(M_PI)
                      / static_cast<float>(num_angles);
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        std::vector<int> sinogram(num_offsets, 0);
        float offset_center = static_cast<float>(num_offsets - 1) / 2.0f;

        for (auto& [dr, dc] : defect_coords) {
            float proj = dr * cos_t + dc * sin_t;
            int bin = static_cast<int>(proj + offset_center + 0.5f);
            bin = std::clamp(bin, 0, num_offsets - 1);
            sinogram[bin] += 1;
        }

        int local_max = *std::max_element(sinogram.begin(), sinogram.end());
        if (static_cast<float>(local_max) > global_max) {
            global_max = static_cast<float>(local_max);
            best_angle_idx = ai;
        }
    }

    result.linearity = global_max / static_cast<float>(total_defects);
    result.angle = static_cast<float>(best_angle_idx) * 180.0f
                   / static_cast<float>(num_angles);

    return result;
}

}  // namespace cpu
}  // namespace waferlens
