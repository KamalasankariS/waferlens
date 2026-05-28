#include "density_gradient.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace waferlens {
namespace cpu {

std::vector<float> density_gradient(const uint8_t* data, int rows, int cols,
                                    int grid_size) {
    std::vector<float> density(grid_size * grid_size, 0.0f);
    std::vector<float> counts(grid_size * grid_size, 0.0f);

    float row_scale = static_cast<float>(grid_size) / static_cast<float>(rows);
    float col_scale = static_cast<float>(grid_size) / static_cast<float>(cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            uint8_t val = data[r * cols + c];
            if (val == 0) {
                continue;
            }

            int gr = std::min(static_cast<int>(r * row_scale), grid_size - 1);
            int gc = std::min(static_cast<int>(c * col_scale), grid_size - 1);
            int gidx = gr * grid_size + gc;

            counts[gidx] += 1.0f;
            if (val == 2) {
                density[gidx] += 1.0f;
            }
        }
    }

    for (int i = 0; i < grid_size * grid_size; ++i) {
        if (counts[i] > 0.0f) {
            density[i] /= counts[i];
        }
    }

    std::vector<float> gradient(grid_size * grid_size, 0.0f);

    for (int gr = 1; gr < grid_size - 1; ++gr) {
        for (int gc = 1; gc < grid_size - 1; ++gc) {
            float gx = -1.0f * density[(gr - 1) * grid_size + (gc - 1)]
                        -2.0f * density[gr * grid_size + (gc - 1)]
                        -1.0f * density[(gr + 1) * grid_size + (gc - 1)]
                        +1.0f * density[(gr - 1) * grid_size + (gc + 1)]
                        +2.0f * density[gr * grid_size + (gc + 1)]
                        +1.0f * density[(gr + 1) * grid_size + (gc + 1)];

            float gy = -1.0f * density[(gr - 1) * grid_size + (gc - 1)]
                        -2.0f * density[(gr - 1) * grid_size + gc]
                        -1.0f * density[(gr - 1) * grid_size + (gc + 1)]
                        +1.0f * density[(gr + 1) * grid_size + (gc - 1)]
                        +2.0f * density[(gr + 1) * grid_size + gc]
                        +1.0f * density[(gr + 1) * grid_size + (gc + 1)];

            gradient[gr * grid_size + gc] = std::sqrt(gx * gx + gy * gy);
        }
    }

    float max_grad = *std::max_element(gradient.begin(), gradient.end());
    if (max_grad > 1e-8f) {
        for (float& v : gradient) {
            v /= max_grad;
        }
    }

    return gradient;
}

}  // namespace cpu
}  // namespace waferlens
