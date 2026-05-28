#include <gtest/gtest.h>

#include "waferlens/waferlens.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

std::vector<uint8_t> make_empty_wafer(int rows, int cols) {
    return std::vector<uint8_t>(rows * cols, 1);
}

std::vector<uint8_t> make_center_defect(int rows, int cols, int radius) {
    std::vector<uint8_t> wafer(rows * cols, 1);
    float cr = static_cast<float>(rows - 1) / 2.0f;
    float cc = static_cast<float>(cols - 1) / 2.0f;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float dr = static_cast<float>(r) - cr;
            float dc = static_cast<float>(c) - cc;
            if (std::sqrt(dr * dr + dc * dc) <= static_cast<float>(radius)) {
                wafer[r * cols + c] = 2;
            }
        }
    }
    return wafer;
}

std::vector<uint8_t> make_edge_ring(int rows, int cols, float inner_frac) {
    std::vector<uint8_t> wafer(rows * cols, 1);
    float cr = static_cast<float>(rows - 1) / 2.0f;
    float cc = static_cast<float>(cols - 1) / 2.0f;
    float max_r = std::sqrt(cr * cr + cc * cc);
    float threshold = inner_frac * max_r;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float dr = static_cast<float>(r) - cr;
            float dc = static_cast<float>(c) - cc;
            if (std::sqrt(dr * dr + dc * dc) >= threshold) {
                wafer[r * cols + c] = 2;
            }
        }
    }
    return wafer;
}

std::vector<uint8_t> make_scratch(int rows, int cols) {
    std::vector<uint8_t> wafer(rows * cols, 1);
    int mid_row = rows / 2;
    for (int c = 0; c < cols; ++c) {
        wafer[mid_row * cols + c] = 2;
        if (mid_row + 1 < rows) {
            wafer[(mid_row + 1) * cols + c] = 2;
        }
    }
    return wafer;
}

}  // namespace

TEST(WaferAnalyzer, NoDefectsProducesZeroFeatures) {
    auto wafer = make_empty_wafer(64, 64);
    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    EXPECT_EQ(feat.total_defects, 0);
    EXPECT_GT(feat.total_normal, 0);
    EXPECT_FLOAT_EQ(feat.edge_concentration, 0.0f);
    EXPECT_TRUE(feat.clusters.empty());

    for (float v : feat.radial_density) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(WaferAnalyzer, CenterDefectHasLowEdgeConcentration) {
    auto wafer = make_center_defect(64, 64, 5);
    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    EXPECT_GT(feat.total_defects, 0);
    EXPECT_LT(feat.edge_concentration, 0.05f);
    EXPECT_EQ(feat.clusters.size(), 1u);
}

TEST(WaferAnalyzer, EdgeRingHasHighEdgeConcentration) {
    auto wafer = make_edge_ring(64, 64, 0.85f);
    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    EXPECT_GT(feat.edge_concentration, 0.8f);
    EXPECT_GT(feat.ring_score, 2.0f);
}

TEST(WaferAnalyzer, RadialDensitySumsToOne) {
    auto wafer = make_center_defect(64, 64, 10);
    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    float sum = std::accumulate(feat.radial_density.begin(),
                                feat.radial_density.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 0.01f);
}

TEST(WaferAnalyzer, AngularHistogramSumsToOne) {
    auto wafer = make_center_defect(64, 64, 10);
    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    float sum = std::accumulate(feat.angular_histogram.begin(),
                                feat.angular_histogram.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 0.01f);
}

TEST(WaferAnalyzer, ScratchHasHighLinearity) {
    auto wafer = make_scratch(64, 64);
    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    EXPECT_GT(feat.scratch_linearity, 0.4f);
}

TEST(WaferAnalyzer, ConnectedComponentsCountsClusters) {
    std::vector<uint8_t> wafer(64 * 64, 1);
    wafer[10 * 64 + 10] = 2;
    wafer[10 * 64 + 11] = 2;
    wafer[11 * 64 + 10] = 2;
    wafer[50 * 64 + 50] = 2;
    wafer[50 * 64 + 51] = 2;

    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    EXPECT_EQ(feat.clusters.size(), 2u);
}

TEST(WaferAnalyzer, DensityGradientHasCorrectSize) {
    auto wafer = make_center_defect(64, 64, 10);
    waferlens::AnalyzerConfig cfg;
    cfg.gradient_grid = 8;
    waferlens::WaferAnalyzer analyzer(cfg);
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    EXPECT_EQ(feat.gradient_grid_size, 8);
    EXPECT_EQ(static_cast<int>(feat.density_gradient.size()), 64);
}

TEST(WaferAnalyzer, BatchProducesCorrectCount) {
    auto wafer = make_center_defect(32, 32, 5);
    std::vector<uint8_t> batch_data;
    for (int i = 0; i < 4; ++i) {
        batch_data.insert(batch_data.end(), wafer.begin(), wafer.end());
    }

    waferlens::WaferAnalyzer analyzer;
    auto results = analyzer.compute_batch(batch_data.data(), 4, 32, 32);
    EXPECT_EQ(results.size(), 4u);

    for (const auto& feat : results) {
        EXPECT_GT(feat.total_defects, 0);
    }
}

TEST(WaferAnalyzer, InvalidInputThrows) {
    waferlens::WaferAnalyzer analyzer;
    EXPECT_THROW(analyzer.compute(nullptr, 64, 64), std::invalid_argument);
    EXPECT_THROW(analyzer.compute(nullptr, 0, 64), std::invalid_argument);
}

TEST(WaferAnalyzer, ClusterEccentricityRange) {
    auto wafer = make_scratch(64, 64);
    waferlens::WaferAnalyzer analyzer;
    auto feat = analyzer.compute(wafer.data(), 64, 64);

    for (const auto& cluster : feat.clusters) {
        EXPECT_GE(cluster.eccentricity, 0.0f);
        EXPECT_LE(cluster.eccentricity, 1.0f);
    }
}
