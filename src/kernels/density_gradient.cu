/**
 * @file density_gradient.cu
 * @brief CUDA kernel for 2D defect density gradient computation.
 *
 * Two-phase approach:
 *   1. A binning kernel maps each wafer cell to a grid cell and
 *      accumulates defect and total counts via atomicAdd.
 *   2. A Sobel kernel computes the gradient magnitude at each grid point.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace waferlens {
namespace cuda {

__global__ void density_bin_kernel(const uint8_t* __restrict__ data,
                                   int rows, int cols, int grid_size,
                                   float row_scale, float col_scale,
                                   float* __restrict__ defect_counts,
                                   float* __restrict__ total_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    uint8_t val = data[idx];
    if (val == 0) return;

    int r = idx / cols;
    int c = idx % cols;
    int gr = min(static_cast<int>(r * row_scale), grid_size - 1);
    int gc = min(static_cast<int>(c * col_scale), grid_size - 1);
    int gidx = gr * grid_size + gc;

    atomicAdd(&total_counts[gidx], 1.0f);
    if (val == 2) {
        atomicAdd(&defect_counts[gidx], 1.0f);
    }
}

__global__ void sobel_gradient_kernel(const float* __restrict__ defect_counts,
                                      const float* __restrict__ total_counts,
                                      int grid_size,
                                      float* __restrict__ gradient) {
    int gr = blockIdx.y * blockDim.y + threadIdx.y;
    int gc = blockIdx.x * blockDim.x + threadIdx.x;

    if (gr < 1 || gr >= grid_size - 1 || gc < 1 || gc >= grid_size - 1) {
        if (gr < grid_size && gc < grid_size) {
            gradient[gr * grid_size + gc] = 0.0f;
        }
        return;
    }

    auto density_at = [&](int row, int col) -> float {
        int idx = row * grid_size + col;
        return (total_counts[idx] > 0.0f)
                   ? defect_counts[idx] / total_counts[idx]
                   : 0.0f;
    };

    float gx = -1.0f * density_at(gr - 1, gc - 1)
               - 2.0f * density_at(gr, gc - 1)
               - 1.0f * density_at(gr + 1, gc - 1)
               + 1.0f * density_at(gr - 1, gc + 1)
               + 2.0f * density_at(gr, gc + 1)
               + 1.0f * density_at(gr + 1, gc + 1);

    float gy = -1.0f * density_at(gr - 1, gc - 1)
               - 2.0f * density_at(gr - 1, gc)
               - 1.0f * density_at(gr - 1, gc + 1)
               + 1.0f * density_at(gr + 1, gc - 1)
               + 2.0f * density_at(gr + 1, gc)
               + 1.0f * density_at(gr + 1, gc + 1);

    gradient[gr * grid_size + gc] = sqrtf(gx * gx + gy * gy);
}

void launch_density_gradient(const uint8_t* d_data, int rows, int cols,
                             int grid_size, float* d_defect_counts,
                             float* d_total_counts, float* d_gradient,
                             cudaStream_t stream) {
    float row_scale = static_cast<float>(grid_size) / static_cast<float>(rows);
    float col_scale = static_cast<float>(grid_size) / static_cast<float>(cols);

    int total = rows * cols;
    int block1d = 256;
    int grid1d = (total + block1d - 1) / block1d;

    density_bin_kernel<<<grid1d, block1d, 0, stream>>>(
        d_data, rows, cols, grid_size, row_scale, col_scale,
        d_defect_counts, d_total_counts);

    dim3 block2d(16, 16);
    dim3 grid2d((grid_size + 15) / 16, (grid_size + 15) / 16);

    sobel_gradient_kernel<<<grid2d, block2d, 0, stream>>>(
        d_defect_counts, d_total_counts, grid_size, d_gradient);
}

}  // namespace cuda
}  // namespace waferlens
