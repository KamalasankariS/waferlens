/**
 * @file connected_components.cu
 * @brief GPU-accelerated connected-component labeling for defect clusters.
 *
 * Implements a parallel union-find algorithm based on the approach described
 * by Komura (2015).  The algorithm proceeds in three phases:
 *
 *   1. Initialization: each defective cell is its own root.
 *   2. Merge: neighboring defective cells are unioned using atomic
 *      compare-and-swap on a parent array.
 *   3. Path compression: all labels are flattened to their root.
 *
 * After labeling, a separate kernel computes per-cluster moments
 * (area, centroid, second-order central moments) for eccentricity
 * calculation on the host.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace waferlens {
namespace cuda {

__device__ int find_root(int* labels, int idx) {
    while (labels[idx] != idx) {
        labels[idx] = labels[labels[idx]];
        idx = labels[idx];
    }
    return idx;
}

__device__ void union_cells(int* labels, int a, int b) {
    int root_a = find_root(labels, a);
    int root_b = find_root(labels, b);
    while (root_a != root_b) {
        int high = max(root_a, root_b);
        int low  = min(root_a, root_b);
        int old = atomicCAS(&labels[high], high, low);
        if (old == high) {
            break;
        }
        root_a = find_root(labels, high);
        root_b = find_root(labels, low);
    }
}

__global__ void init_labels_kernel(const uint8_t* __restrict__ data,
                                   int* __restrict__ labels,
                                   int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        labels[idx] = (data[idx] == 2) ? idx : -1;
    }
}

__global__ void merge_kernel(const uint8_t* __restrict__ data,
                             int* __restrict__ labels,
                             int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total || data[idx] != 2) {
        return;
    }

    int r = idx / cols;
    int c = idx % cols;

    if (r + 1 < rows && data[(r + 1) * cols + c] == 2) {
        union_cells(labels, idx, (r + 1) * cols + c);
    }
    if (c + 1 < cols && data[r * cols + c + 1] == 2) {
        union_cells(labels, idx, r * cols + c + 1);
    }
}

__global__ void flatten_kernel(int* __restrict__ labels, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && labels[idx] >= 0) {
        labels[idx] = find_root(labels, idx);
    }
}

void launch_connected_components(const uint8_t* d_data, int* d_labels,
                                 int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    init_labels_kernel<<<grid_size, block_size, 0, stream>>>(
        d_data, d_labels, rows, cols);
    merge_kernel<<<grid_size, block_size, 0, stream>>>(
        d_data, d_labels, rows, cols);
    flatten_kernel<<<grid_size, block_size, 0, stream>>>(
        d_labels, total);
}

}  // namespace cuda
}  // namespace waferlens
