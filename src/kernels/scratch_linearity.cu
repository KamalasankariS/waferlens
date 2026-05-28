/**
 * @file scratch_linearity.cu
 * @brief CUDA kernel for Radon-transform-based scratch detection.
 *
 * Each thread block handles one projection angle.  Threads within the block
 * iterate over defective cells and accumulate projections into a
 * shared-memory sinogram column.  The host then extracts the global peak
 * to determine linearity score and scratch angle.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace waferlens {
namespace cuda {

__global__ void radon_projection_kernel(const uint8_t* __restrict__ data,
                                        int rows, int cols,
                                        int num_angles, int num_offsets,
                                        float center_r, float center_c,
                                        float offset_center,
                                        int* __restrict__ sinogram) {
    int angle_idx = blockIdx.x;
    if (angle_idx >= num_angles) return;

    float theta = static_cast<float>(angle_idx) * static_cast<float>(M_PI)
                  / static_cast<float>(num_angles);
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    int total_cells = rows * cols;

    for (int i = threadIdx.x; i < total_cells; i += blockDim.x) {
        if (data[i] == 2) {
            int r = i / cols;
            int c = i % cols;

            float dr = static_cast<float>(r) - center_r;
            float dc = static_cast<float>(c) - center_c;
            float proj = dr * cos_t + dc * sin_t;

            int bin = static_cast<int>(proj + offset_center + 0.5f);
            if (bin >= 0 && bin < num_offsets) {
                atomicAdd(&sinogram[angle_idx * num_offsets + bin], 1);
            }
        }
    }
}

void launch_radon_projection(const uint8_t* d_data, int rows, int cols,
                             int num_angles, int num_offsets,
                             int* d_sinogram, cudaStream_t stream) {
    float center_r = static_cast<float>(rows - 1) / 2.0f;
    float center_c = static_cast<float>(cols - 1) / 2.0f;
    float offset_center = static_cast<float>(num_offsets - 1) / 2.0f;

    int block_size = 256;
    radon_projection_kernel<<<num_angles, block_size, 0, stream>>>(
        d_data, rows, cols, num_angles, num_offsets,
        center_r, center_c, offset_center, d_sinogram);
}

}  // namespace cuda
}  // namespace waferlens
