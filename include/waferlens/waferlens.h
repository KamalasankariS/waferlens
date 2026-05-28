#ifndef WAFERLENS_WAFERLENS_H
#define WAFERLENS_WAFERLENS_H

#include "waferlens/wafer_features.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace waferlens {

/**
 * @brief Configuration for the wafer geometry analyzer.
 *
 * All fields carry sensible defaults.  Override only what you need.
 */
struct AnalyzerConfig {
    int   radial_bins       = 32;     ///< Number of concentric rings for radial profiling.
    int   angular_sectors   = 36;     ///< Number of angular sectors (degrees per sector = 360 / sectors).
    float edge_fraction     = 0.15f;  ///< Fractional radius defining the "edge" annulus.
    int   gradient_grid     = 16;     ///< Side length of the grid used for density gradient.
    int   radon_angles      = 180;    ///< Number of projection angles for the Radon transform.
    float notch_angle       = 0.0f;   ///< Notch position in degrees (counter-clockwise from +x).
    bool  prefer_gpu        = true;   ///< Use CUDA kernels when available.
};

/**
 * @brief Primary interface for wafer-map geometric feature extraction.
 *
 * Accepts a wafer map encoded as a row-major 2D array of uint8_t values
 * (see CellState) and produces a complete WaferFeatures descriptor.
 *
 * Usage:
 * @code
 *   waferlens::AnalyzerConfig cfg;
 *   cfg.radial_bins = 64;
 *
 *   waferlens::WaferAnalyzer analyzer(cfg);
 *
 *   // Single wafer -- data is row-major uint8_t, rows x cols.
 *   waferlens::WaferFeatures feats = analyzer.compute(data, rows, cols);
 *
 *   // Batch -- contiguous array of N wafers, each rows x cols.
 *   auto batch = analyzer.compute_batch(data, n, rows, cols);
 * @endcode
 */
class WaferAnalyzer {
public:
    explicit WaferAnalyzer(const AnalyzerConfig& config = AnalyzerConfig{});
    ~WaferAnalyzer();

    WaferAnalyzer(const WaferAnalyzer&)            = delete;
    WaferAnalyzer& operator=(const WaferAnalyzer&) = delete;
    WaferAnalyzer(WaferAnalyzer&&) noexcept;
    WaferAnalyzer& operator=(WaferAnalyzer&&) noexcept;

    /**
     * @brief Compute geometric features for a single wafer map.
     *
     * @param data  Row-major uint8_t buffer of size rows * cols.
     *              Values must conform to CellState encoding.
     * @param rows  Number of rows in the wafer map.
     * @param cols  Number of columns in the wafer map.
     * @return      Fully populated WaferFeatures struct.
     */
    WaferFeatures compute(const uint8_t* data, int rows, int cols) const;

    /**
     * @brief Compute features for a batch of identically sized wafer maps.
     *
     * @param data  Contiguous row-major buffer of size n * rows * cols.
     * @param n     Number of wafer maps in the batch.
     * @param rows  Number of rows per wafer map.
     * @param cols  Number of columns per wafer map.
     * @return      Vector of WaferFeatures, one per wafer.
     */
    std::vector<WaferFeatures> compute_batch(const uint8_t* data, int n,
                                             int rows, int cols) const;

    /** @brief Returns true if CUDA acceleration is available and enabled. */
    bool using_gpu() const;

    /** @brief Returns the active configuration. */
    const AnalyzerConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Free-function convenience wrapper for single-wafer computation.
 *
 * Constructs a temporary WaferAnalyzer with the given config.  For repeated
 * calls, prefer constructing a WaferAnalyzer once and reusing it.
 */
WaferFeatures compute(const uint8_t* data, int rows, int cols,
                       const AnalyzerConfig& config = AnalyzerConfig{});

}  // namespace waferlens

#endif  // WAFERLENS_WAFERLENS_H
