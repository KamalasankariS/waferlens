/**
 * @file radial_density.cu
 * @brief CUDA kernel for radial defect density profiling.
 *
 * Each thread processes one cell of the wafer map.  Defective cells
 * contribute to a shared-memory histogram via atomicAdd, which is then
 * reduced to global memory.  A final normalization pass divides by the
 * total defect count.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace waferlens {
namespace cuda {

__global__ void radial_density_kernel(const uint8_t* __restrict__ data,
                                      int rows, int cols, int num_bins,
                                      float center_r, float center_c,
                                      float inv_max_radius,
                                      float* __restrict__ bins,
                                      int* __restrict__ defect_count) {
    extern __shared__ float shared_bins[];

    int tid = threadIdx.x;
    for (int b = tid; b < num_bins; b += blockDim.x) {
        shared_bins[b] = 0.0f;
    }
    __syncthreads();

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = rows * cols;

    if (global_idx < total_cells) {
        uint8_t val = data[global_idx];
        if (val == 2) {
            int r = global_idx / cols;
            int c = global_idx % cols;

            float dr = static_cast<float>(r) - center_r;
            float dc = static_cast<float>(c) - center_c;
            float dist = sqrtf(dr * dr + dc * dc) * inv_max_radius;

            int bin = min(static_cast<int>(dist * num_bins), num_bins - 1);
            atomicAdd(&shared_bins[bin], 1.0f);
            atomicAdd(defect_count, 1);
        }
    }

    __syncthreads();

    for (int b = tid; b < num_bins; b += blockDim.x) {
        if (shared_bins[b] > 0.0f) {
            atomicAdd(&bins[b], shared_bins[b]);
        }
    }
}

__global__ void normalize_kernel(float* __restrict__ bins, int num_bins,
                                 const int* __restrict__ defect_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins && *defect_count > 0) {
        bins[idx] /= static_cast<float>(*defect_count);
    }
}

void launch_radial_density(const uint8_t* d_data, int rows, int cols,
                           int num_bins, float* d_bins, int* d_defect_count,
                           cudaStream_t stream) {
    float center_r = static_cast<float>(rows - 1) / 2.0f;
    float center_c = static_cast<float>(cols - 1) / 2.0f;
    float max_radius = sqrtf(center_r * center_r + center_c * center_c);
    float inv_max_radius = (max_radius > 1e-6f) ? 1.0f / max_radius : 0.0f;

    int total_cells = rows * cols;
    int block_size = 256;
    int grid_size = (total_cells + block_size - 1) / block_size;
    size_t shared_mem = num_bins * sizeof(float);

    radial_density_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        d_data, rows, cols, num_bins, center_r, center_c, inv_max_radius,
        d_bins, d_defect_count);

    int norm_grid = (num_bins + block_size - 1) / block_size;
    normalize_kernel<<<norm_grid, block_size, 0, stream>>>(
        d_bins, num_bins, d_defect_count);
}

}  // namespace cuda
}  // namespace waferlens
