#ifndef WAFERLENS_CUDA_DISPATCH_H
#define WAFERLENS_CUDA_DISPATCH_H

/**
 * @file cuda_dispatch.h
 * @brief CUDA kernel dispatch interface.
 *
 * This header is only included when WAFERLENS_CUDA_ENABLED is defined.
 * Each kernel has a host-callable entry point that manages device memory
 * allocation, kernel launch, and result transfer.
 */

#include "waferlens/waferlens.h"

namespace waferlens {
namespace cuda {

/**
 * @brief Check whether a CUDA-capable device is available.
 */
bool is_available();

/**
 * @brief Run the full feature extraction pipeline on GPU.
 *
 * Transfers the wafer map to device memory once, launches all kernels,
 * and returns the assembled WaferFeatures on the host.
 */
WaferFeatures compute_all(const uint8_t* data, int rows, int cols,
                           const AnalyzerConfig& config);

}  // namespace cuda
}  // namespace waferlens

#endif  // WAFERLENS_CUDA_DISPATCH_H
