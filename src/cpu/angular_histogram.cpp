#include "angular_histogram.h"

#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace waferlens {
namespace cpu {

std::vector<float> angular_histogram(const uint8_t* data, int rows, int cols,
                                     int num_sectors, float notch_angle) {
    std::vector<float> sectors(num_sectors, 0.0f);

    const float center_r = static_cast<float>(rows - 1) / 2.0f;
    const float center_c = static_cast<float>(cols - 1) / 2.0f;
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    const float sector_width = two_pi / static_cast<float>(num_sectors);

    int total_defects = 0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (data[r * cols + c] != 2) {
                continue;
            }
            float dr = static_cast<float>(r) - center_r;
            float dc = static_cast<float>(c) - center_c;

            float angle = std::atan2(dr, dc) - notch_angle;
            if (angle < 0.0f) {
                angle += two_pi;
            }

            int sector = static_cast<int>(angle / sector_width);
            sector = std::min(sector, num_sectors - 1);

            sectors[sector] += 1.0f;
            ++total_defects;
        }
    }

    if (total_defects > 0) {
        float inv_total = 1.0f / static_cast<float>(total_defects);
        for (float& v : sectors) {
            v *= inv_total;
        }
    }

    return sectors;
}

}  // namespace cpu
}  // namespace waferlens
