/**
 * @file cuda_dispatch.cu
 * @brief Orchestrates the full GPU feature extraction pipeline.
 *
 * Manages device memory allocation, data transfer, kernel launches
 * across a single CUDA stream, and result assembly.
 */

#include "cuda_dispatch.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace waferlens {
namespace cuda {

void launch_radial_density(const uint8_t* d_data, int rows, int cols,
                           int num_bins, float* d_bins, int* d_defect_count,
                           cudaStream_t stream);

void launch_angular_histogram(const uint8_t* d_data, int rows, int cols,
                              int num_sectors, float notch_rad,
                              float* d_sectors, int* d_defect_count,
                              cudaStream_t stream);

void launch_connected_components(const uint8_t* d_data, int* d_labels,
                                 int rows, int cols, cudaStream_t stream);

void launch_radon_projection(const uint8_t* d_data, int rows, int cols,
                             int num_angles, int num_offsets,
                             int* d_sinogram, cudaStream_t stream);

void launch_density_gradient(const uint8_t* d_data, int rows, int cols,
                             int grid_size, float* d_defect_counts,
                             float* d_total_counts, float* d_gradient,
                             cudaStream_t stream);

bool is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

WaferFeatures compute_all(const uint8_t* data, int rows, int cols,
                           const AnalyzerConfig& config) {
    WaferFeatures feat;
    int total_cells = rows * cols;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint8_t* d_data;
    cudaMalloc(&d_data, total_cells);
    cudaMemcpyAsync(d_data, data, total_cells, cudaMemcpyHostToDevice, stream);

    int total_defects = 0;
    int total_normal = 0;
    for (int i = 0; i < total_cells; ++i) {
        if (data[i] == 2) ++total_defects;
        else if (data[i] == 1) ++total_normal;
    }
    feat.total_defects = total_defects;
    feat.total_normal = total_normal;

    float* d_radial_bins;
    int*   d_radial_count;
    cudaMalloc(&d_radial_bins, config.radial_bins * sizeof(float));
    cudaMalloc(&d_radial_count, sizeof(int));
    cudaMemsetAsync(d_radial_bins, 0, config.radial_bins * sizeof(float), stream);
    cudaMemsetAsync(d_radial_count, 0, sizeof(int), stream);

    launch_radial_density(d_data, rows, cols, config.radial_bins,
                          d_radial_bins, d_radial_count, stream);

    float* d_angular_sectors;
    int*   d_angular_count;
    cudaMalloc(&d_angular_sectors, config.angular_sectors * sizeof(float));
    cudaMalloc(&d_angular_count, sizeof(int));
    cudaMemsetAsync(d_angular_sectors, 0,
                    config.angular_sectors * sizeof(float), stream);
    cudaMemsetAsync(d_angular_count, 0, sizeof(int), stream);

    float notch_rad = config.notch_angle * static_cast<float>(M_PI) / 180.0f;
    launch_angular_histogram(d_data, rows, cols, config.angular_sectors,
                             notch_rad, d_angular_sectors, d_angular_count,
                             stream);

    int* d_labels;
    cudaMalloc(&d_labels, total_cells * sizeof(int));
    launch_connected_components(d_data, d_labels, rows, cols, stream);

    float max_radius = std::sqrt(static_cast<float>((rows - 1) * (rows - 1)
                                 + (cols - 1) * (cols - 1))) / 2.0f;
    int num_offsets = static_cast<int>(2.0f * max_radius) + 1;

    int* d_sinogram;
    cudaMalloc(&d_sinogram, config.radon_angles * num_offsets * sizeof(int));
    cudaMemsetAsync(d_sinogram, 0,
                    config.radon_angles * num_offsets * sizeof(int), stream);
    launch_radon_projection(d_data, rows, cols, config.radon_angles,
                            num_offsets, d_sinogram, stream);

    int gs = config.gradient_grid;
    float* d_defect_counts;
    float* d_total_counts;
    float* d_gradient;
    cudaMalloc(&d_defect_counts, gs * gs * sizeof(float));
    cudaMalloc(&d_total_counts, gs * gs * sizeof(float));
    cudaMalloc(&d_gradient, gs * gs * sizeof(float));
    cudaMemsetAsync(d_defect_counts, 0, gs * gs * sizeof(float), stream);
    cudaMemsetAsync(d_total_counts, 0, gs * gs * sizeof(float), stream);
    cudaMemsetAsync(d_gradient, 0, gs * gs * sizeof(float), stream);
    launch_density_gradient(d_data, rows, cols, gs,
                            d_defect_counts, d_total_counts, d_gradient,
                            stream);

    cudaStreamSynchronize(stream);

    feat.radial_density.resize(config.radial_bins);
    cudaMemcpy(feat.radial_density.data(), d_radial_bins,
               config.radial_bins * sizeof(float), cudaMemcpyDeviceToHost);

    auto rs_result = [&]() {
        float max_val = 0.0f;
        int peak_idx = 0;
        for (int i = 0; i < config.radial_bins; ++i) {
            if (feat.radial_density[i] > max_val) {
                max_val = feat.radial_density[i];
                peak_idx = i;
            }
        }
        feat.ring_score = max_val * static_cast<float>(config.radial_bins);
        feat.peak_ring_index = peak_idx;
    };
    rs_result();

    feat.angular_histogram.resize(config.angular_sectors);
    cudaMemcpy(feat.angular_histogram.data(), d_angular_sectors,
               config.angular_sectors * sizeof(float), cudaMemcpyDeviceToHost);

    float threshold_radius = (1.0f - config.edge_fraction) * max_radius;
    float center_r = static_cast<float>(rows - 1) / 2.0f;
    float center_c = static_cast<float>(cols - 1) / 2.0f;
    int edge_defects = 0;
    for (int i = 0; i < total_cells; ++i) {
        if (data[i] == 2) {
            int r = i / cols;
            int c = i % cols;
            float dr = static_cast<float>(r) - center_r;
            float dc = static_cast<float>(c) - center_c;
            if (std::sqrt(dr * dr + dc * dc) >= threshold_radius) {
                ++edge_defects;
            }
        }
    }
    feat.edge_concentration = (total_defects > 0)
        ? static_cast<float>(edge_defects) / static_cast<float>(total_defects)
        : 0.0f;

    std::vector<int> h_labels(total_cells);
    cudaMemcpy(h_labels.data(), d_labels, total_cells * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::unordered_map<int, std::vector<std::pair<int, int>>> label_map;
    for (int i = 0; i < total_cells; ++i) {
        if (h_labels[i] >= 0) {
            label_map[h_labels[i]].push_back({i / cols, i % cols});
        }
    }

    int cluster_id = 0;
    for (auto& [root, members] : label_map) {
        ++cluster_id;
        ClusterDescriptor desc;
        desc.label = cluster_id;
        desc.area = static_cast<int>(members.size());

        double sum_r = 0, sum_c = 0;
        for (auto& [r, c] : members) {
            sum_r += r;
            sum_c += c;
        }
        desc.centroid_row = static_cast<float>(sum_r / desc.area);
        desc.centroid_col = static_cast<float>(sum_c / desc.area);

        double mu20 = 0, mu02 = 0, mu11 = 0;
        for (auto& [r, c] : members) {
            double dr = r - desc.centroid_row;
            double dc = c - desc.centroid_col;
            mu20 += dr * dr;
            mu02 += dc * dc;
            mu11 += dr * dc;
        }
        mu20 /= desc.area;
        mu02 /= desc.area;
        mu11 /= desc.area;

        double trace = mu20 + mu02;
        desc.eccentricity = 0.0f;
        if (trace > 1e-12) {
            double disc = std::sqrt((mu20 - mu02) * (mu20 - mu02)
                                    + 4.0 * mu11 * mu11);
            double l1 = (trace + disc) / 2.0;
            double l2 = (trace - disc) / 2.0;
            if (l1 > 1e-12) {
                desc.eccentricity = static_cast<float>(std::sqrt(1.0 - l2 / l1));
            }
        }

        feat.clusters.push_back(desc);
    }

    std::vector<int> h_sinogram(config.radon_angles * num_offsets);
    cudaMemcpy(h_sinogram.data(), d_sinogram,
               config.radon_angles * num_offsets * sizeof(int),
               cudaMemcpyDeviceToHost);

    int global_max = 0;
    int best_angle = 0;
    for (int a = 0; a < config.radon_angles; ++a) {
        for (int o = 0; o < num_offsets; ++o) {
            int val = h_sinogram[a * num_offsets + o];
            if (val > global_max) {
                global_max = val;
                best_angle = a;
            }
        }
    }
    feat.scratch_linearity = (total_defects > 0)
        ? static_cast<float>(global_max) / static_cast<float>(total_defects)
        : 0.0f;
    feat.scratch_angle = static_cast<float>(best_angle) * 180.0f
                         / static_cast<float>(config.radon_angles);

    feat.density_gradient.resize(gs * gs);
    cudaMemcpy(feat.density_gradient.data(), d_gradient,
               gs * gs * sizeof(float), cudaMemcpyDeviceToHost);

    float max_grad = *std::max_element(feat.density_gradient.begin(),
                                       feat.density_gradient.end());
    if (max_grad > 1e-8f) {
        for (float& v : feat.density_gradient) {
            v /= max_grad;
        }
    }
    feat.gradient_grid_size = gs;

    cudaFree(d_data);
    cudaFree(d_radial_bins);
    cudaFree(d_radial_count);
    cudaFree(d_angular_sectors);
    cudaFree(d_angular_count);
    cudaFree(d_labels);
    cudaFree(d_sinogram);
    cudaFree(d_defect_counts);
    cudaFree(d_total_counts);
    cudaFree(d_gradient);
    cudaStreamDestroy(stream);

    return feat;
}

}  // namespace cuda
}  // namespace waferlens
