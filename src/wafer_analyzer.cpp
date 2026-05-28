#include "waferlens/waferlens.h"

#include "cpu/radial_density.h"
#include "cpu/angular_histogram.h"
#include "cpu/edge_concentration.h"
#include "cpu/ring_score.h"
#include "cpu/connected_components.h"
#include "cpu/scratch_linearity.h"
#include "cpu/density_gradient.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef WAFERLENS_CUDA_ENABLED
#include "kernels/cuda_dispatch.h"
#endif

namespace waferlens {

struct WaferAnalyzer::Impl {
    AnalyzerConfig config;
    bool           gpu_available;

    explicit Impl(const AnalyzerConfig& cfg)
        : config(cfg), gpu_available(false) {
#ifdef WAFERLENS_CUDA_ENABLED
        if (cfg.prefer_gpu) {
            gpu_available = cuda::is_available();
        }
#endif
    }
};

WaferAnalyzer::WaferAnalyzer(const AnalyzerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

WaferAnalyzer::~WaferAnalyzer() = default;

WaferAnalyzer::WaferAnalyzer(WaferAnalyzer&&) noexcept = default;
WaferAnalyzer& WaferAnalyzer::operator=(WaferAnalyzer&&) noexcept = default;

bool WaferAnalyzer::using_gpu() const {
    return impl_->gpu_available;
}

const AnalyzerConfig& WaferAnalyzer::config() const {
    return impl_->config;
}

WaferFeatures WaferAnalyzer::compute(const uint8_t* data, int rows, int cols) const {
    if (!data || rows <= 0 || cols <= 0) {
        throw std::invalid_argument("waferlens::compute: invalid input dimensions");
    }

    const auto& cfg = impl_->config;
    WaferFeatures feat;

    int total_defects = 0;
    int total_normal  = 0;
    for (int i = 0; i < rows * cols; ++i) {
        if (data[i] == 2) ++total_defects;
        else if (data[i] == 1) ++total_normal;
    }
    feat.total_defects = total_defects;
    feat.total_normal  = total_normal;

#ifdef WAFERLENS_CUDA_ENABLED
    if (impl_->gpu_available) {
        return cuda::compute_all(data, rows, cols, cfg);
    }
#endif

    feat.radial_density = cpu::radial_density(data, rows, cols, cfg.radial_bins);

    float notch_rad = cfg.notch_angle * static_cast<float>(M_PI) / 180.0f;
    feat.angular_histogram = cpu::angular_histogram(data, rows, cols,
                                                    cfg.angular_sectors, notch_rad);

    feat.edge_concentration = cpu::edge_concentration(data, rows, cols,
                                                      cfg.edge_fraction);

    auto rs = cpu::ring_score(feat.radial_density);
    feat.ring_score      = rs.score;
    feat.peak_ring_index = rs.peak_bin_index;

    feat.clusters = cpu::connected_components(data, rows, cols);

    auto scratch = cpu::scratch_linearity(data, rows, cols, cfg.radon_angles);
    feat.scratch_linearity = scratch.linearity;
    feat.scratch_angle     = scratch.angle;

    feat.density_gradient  = cpu::density_gradient(data, rows, cols, cfg.gradient_grid);
    feat.gradient_grid_size = cfg.gradient_grid;

    return feat;
}

std::vector<WaferFeatures> WaferAnalyzer::compute_batch(const uint8_t* data,
                                                        int n, int rows,
                                                        int cols) const {
    if (!data || n <= 0 || rows <= 0 || cols <= 0) {
        throw std::invalid_argument("waferlens::compute_batch: invalid input dimensions");
    }

    std::vector<WaferFeatures> results;
    results.reserve(n);

    const int wafer_size = rows * cols;
    for (int i = 0; i < n; ++i) {
        results.push_back(compute(data + i * wafer_size, rows, cols));
    }

    return results;
}

WaferFeatures compute(const uint8_t* data, int rows, int cols,
                       const AnalyzerConfig& config) {
    WaferAnalyzer analyzer(config);
    return analyzer.compute(data, rows, cols);
}

}  // namespace waferlens
