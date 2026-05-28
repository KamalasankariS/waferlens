#include "connected_components.h"

#include <cmath>
#include <queue>
#include <vector>

namespace waferlens {
namespace cpu {

std::vector<ClusterDescriptor> connected_components(const uint8_t* data,
                                                    int rows, int cols) {
    std::vector<int> labels(rows * cols, 0);
    std::vector<ClusterDescriptor> clusters;
    int current_label = 0;

    static constexpr int dr[] = {-1, 1, 0, 0};
    static constexpr int dc[] = {0, 0, -1, 1};

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            if (data[idx] != 2 || labels[idx] != 0) {
                continue;
            }

            ++current_label;

            std::queue<std::pair<int, int>> frontier;
            frontier.push({r, c});
            labels[idx] = current_label;

            double sum_r = 0.0;
            double sum_c = 0.0;
            int area = 0;

            std::vector<std::pair<int, int>> members;

            while (!frontier.empty()) {
                auto [cr, cc] = frontier.front();
                frontier.pop();

                sum_r += cr;
                sum_c += cc;
                ++area;
                members.push_back({cr, cc});

                for (int d = 0; d < 4; ++d) {
                    int nr = cr + dr[d];
                    int nc = cc + dc[d];
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                        continue;
                    }
                    int nidx = nr * cols + nc;
                    if (data[nidx] == 2 && labels[nidx] == 0) {
                        labels[nidx] = current_label;
                        frontier.push({nr, nc});
                    }
                }
            }

            double centroid_r = sum_r / area;
            double centroid_c = sum_c / area;

            double mu20 = 0.0;
            double mu02 = 0.0;
            double mu11 = 0.0;

            for (auto [mr, mc] : members) {
                double dr_val = mr - centroid_r;
                double dc_val = mc - centroid_c;
                mu20 += dr_val * dr_val;
                mu02 += dc_val * dc_val;
                mu11 += dr_val * dc_val;
            }

            mu20 /= area;
            mu02 /= area;
            mu11 /= area;

            float eccentricity = 0.0f;
            double trace = mu20 + mu02;
            if (trace > 1e-12) {
                double discriminant = std::sqrt((mu20 - mu02) * (mu20 - mu02)
                                                + 4.0 * mu11 * mu11);
                double lambda1 = (trace + discriminant) / 2.0;
                double lambda2 = (trace - discriminant) / 2.0;

                if (lambda1 > 1e-12) {
                    eccentricity = static_cast<float>(
                        std::sqrt(1.0 - lambda2 / lambda1));
                }
            }

            ClusterDescriptor desc;
            desc.label       = current_label;
            desc.area        = area;
            desc.centroid_row = static_cast<float>(centroid_r);
            desc.centroid_col = static_cast<float>(centroid_c);
            desc.eccentricity = eccentricity;

            clusters.push_back(desc);
        }
    }

    return clusters;
}

}  // namespace cpu
}  // namespace waferlens
