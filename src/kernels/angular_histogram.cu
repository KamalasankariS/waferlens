/**
 * @file angular_histogram.cu
 * @brief CUDA kernel for angular defect histogram computation.
 *
 * Bins defective cells by angle from the wafer center relative to the
 * notch position.  Uses shared-memory atomic histogram accumulation
 * with a final global reduction and normalization pass.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace waferlens {
namespace cuda {

__global__ void angular_histogram_kernel(const uint8_t* __restrict__ data,
                                         int rows, int cols, int num_sectors,
                                         float center_r, float center_c,
                                         float notch_rad, float sector_width,
                                         float* __restrict__ sectors,
                                         int* __restrict__ defect_count) {
    extern __shared__ float shared_sectors[];

    int tid = threadIdx.x;
    for (int s = tid; s < num_sectors; s += blockDim.x) {
        shared_sectors[s] = 0.0f;
    }
    __syncthreads();

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = rows * cols;

    if (global_idx < total_cells && data[global_idx] == 2) {
        int r = global_idx / cols;
        int c = global_idx % cols;

        float dr = static_cast<float>(r) - center_r;
        float dc = static_cast<float>(c) - center_c;

        float angle = atan2f(dr, dc) - notch_rad;
        if (angle < 0.0f) {
            angle += 2.0f * static_cast<float>(M_PI);
        }

        int sector = min(static_cast<int>(angle / sector_width), num_sectors - 1);
        atomicAdd(&shared_sectors[sector], 1.0f);
        atomicAdd(defect_count, 1);
    }

    __syncthreads();

    for (int s = tid; s < num_sectors; s += blockDim.x) {
        if (shared_sectors[s] > 0.0f) {
            atomicAdd(&sectors[s], shared_sectors[s]);
        }
    }
}

__global__ void normalize_sectors_kernel(float* __restrict__ sectors,
                                         int num_sectors,
                                         const int* __restrict__ defect_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sectors && *defect_count > 0) {
        sectors[idx] /= static_cast<float>(*defect_count);
    }
}

void launch_angular_histogram(const uint8_t* d_data, int rows, int cols,
                              int num_sectors, float notch_rad,
                              float* d_sectors, int* d_defect_count,
                              cudaStream_t stream) {
    float center_r = static_cast<float>(rows - 1) / 2.0f;
    float center_c = static_cast<float>(cols - 1) / 2.0f;
    float two_pi = 2.0f * static_cast<float>(M_PI);
    float sector_width = two_pi / static_cast<float>(num_sectors);

    int total_cells = rows * cols;
    int block_size = 256;
    int grid_size = (total_cells + block_size - 1) / block_size;
    size_t shared_mem = num_sectors * sizeof(float);

    angular_histogram_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        d_data, rows, cols, num_sectors, center_r, center_c,
        notch_rad, sector_width, d_sectors, d_defect_count);

    int norm_grid = (num_sectors + block_size - 1) / block_size;
    normalize_sectors_kernel<<<norm_grid, block_size, 0, stream>>>(
        d_sectors, num_sectors, d_defect_count);
}

}  // namespace cuda
}  // namespace waferlens
