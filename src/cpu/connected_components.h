#ifndef WAFERLENS_CPU_CONNECTED_COMPONENTS_H
#define WAFERLENS_CPU_CONNECTED_COMPONENTS_H

#include "waferlens/wafer_features.h"

#include <cstdint>
#include <vector>

namespace waferlens {
namespace cpu {

/**
 * @brief Find connected components of defective cells and compute
 *        per-cluster geometric descriptors on CPU.
 *
 * Uses 4-connectivity flood fill over cells with value 2 (defect).
 * For each connected component the function computes the area, centroid,
 * and eccentricity derived from the second-order central moments of
 * the cluster's spatial distribution.
 *
 * @param data  Row-major uint8_t wafer map.
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 * @return      Vector of ClusterDescriptor, one per connected component.
 */
std::vector<ClusterDescriptor> connected_components(const uint8_t* data,
                                                    int rows, int cols);

}  // namespace cpu
}  // namespace waferlens

#endif  // WAFERLENS_CPU_CONNECTED_COMPONENTS_H
