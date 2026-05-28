#ifndef WAFERLENS_WAFER_FEATURES_H
#define WAFERLENS_WAFER_FEATURES_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace waferlens {

/**
 * @brief Cell state encoding for wafer map arrays.
 *
 * Follows the WM-811K convention:
 *   0 -- off-wafer (background / padding)
 *   1 -- normal die (no defect)
 *   2 -- defective die
 */
enum class CellState : uint8_t {
    kOffWafer = 0,
    kNormal   = 1,
    kDefect   = 2
};

/**
 * @brief Per-cluster geometric descriptors produced by connected-component
 *        analysis of defective regions.
 */
struct ClusterDescriptor {
    int       label;            ///< Cluster label (1-indexed).
    int       area;             ///< Number of defective cells in the cluster.
    float     centroid_row;     ///< Row coordinate of the cluster centroid.
    float     centroid_col;     ///< Column coordinate of the cluster centroid.
    float     eccentricity;     ///< Eccentricity of the best-fit ellipse [0, 1).
};

/**
 * @brief Complete geometric feature vector for a single wafer map.
 *
 * All spatial metrics are computed relative to the wafer center and,
 * where applicable, the notch position (assumed at angle 0 unless
 * overridden).  Density values are normalized to [0, 1].
 */
struct WaferFeatures {
    /// Radial density profile -- fraction of defective cells in each
    /// concentric ring, from center outward.  Length equals the bin
    /// count passed to the analyzer (default 32).
    std::vector<float> radial_density;

    /// Angular histogram -- fraction of defective cells in each sector,
    /// measured counter-clockwise from the notch.  Length equals the
    /// sector count passed to the analyzer (default 36, i.e. 10-degree bins).
    std::vector<float> angular_histogram;

    /// Fraction of defects located in the outermost annular region.
    /// The boundary is controlled by the edge_fraction parameter
    /// (default 0.15, meaning the outer 15 % of the wafer radius).
    float edge_concentration;

    /// Ring score quantifying how strongly defects concentrate in a
    /// single radial band.  Defined as the peak-to-uniform ratio of
    /// the radial density profile.  A value of 1.0 indicates perfectly
    /// uniform radial distribution.
    float ring_score;

    /// Peak ring index -- the radial bin with the highest defect density.
    int peak_ring_index;

    /// Connected-component descriptors for each defect cluster.
    std::vector<ClusterDescriptor> clusters;

    /// Scratch linearity score from the Radon-transform peak detector.
    /// Higher values indicate stronger linear (scratch-like) structure
    /// in the defect pattern.  Range [0, 1].
    float scratch_linearity;

    /// Radon peak angle in degrees -- orientation of the strongest
    /// linear feature, measured counter-clockwise from horizontal.
    float scratch_angle;

    /// Flattened 2D gradient magnitude of the local defect density.
    /// Dimensions match the grid used for density binning (default 16x16).
    std::vector<float> density_gradient;

    /// Grid side length for the density gradient (rows == cols).
    int gradient_grid_size;

    /// Total number of defective cells on the wafer.
    int total_defects;

    /// Total number of normal (non-defective, on-wafer) cells.
    int total_normal;
};

}  // namespace waferlens

#endif  // WAFERLENS_WAFER_FEATURES_H
