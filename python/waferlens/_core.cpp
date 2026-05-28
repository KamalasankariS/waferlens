/**
 * @file _core.cpp
 * @brief pybind11 bindings for the waferlens C++ library.
 *
 * Exposes WaferAnalyzer, WaferFeatures, and the free-function compute()
 * to Python via the _waferlens_core extension module.  NumPy arrays are
 * accepted directly through pybind11's numpy support.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "waferlens/waferlens.h"

namespace py = pybind11;

namespace {

waferlens::WaferFeatures compute_from_numpy(
        waferlens::WaferAnalyzer& analyzer,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> wafer_map) {
    auto buf = wafer_map.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument(
            "wafer_map must be a 2D array with shape (rows, cols)");
    }
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    const auto* data = static_cast<const uint8_t*>(buf.ptr);
    return analyzer.compute(data, rows, cols);
}

std::vector<waferlens::WaferFeatures> compute_batch_from_numpy(
        waferlens::WaferAnalyzer& analyzer,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> wafer_maps) {
    auto buf = wafer_maps.request();
    if (buf.ndim != 3) {
        throw std::invalid_argument(
            "wafer_maps must be a 3D array with shape (n, rows, cols)");
    }
    int n    = static_cast<int>(buf.shape[0]);
    int rows = static_cast<int>(buf.shape[1]);
    int cols = static_cast<int>(buf.shape[2]);
    const auto* data = static_cast<const uint8_t*>(buf.ptr);
    return analyzer.compute_batch(data, n, rows, cols);
}

waferlens::WaferFeatures free_compute(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> wafer_map,
        const waferlens::AnalyzerConfig& config) {
    auto buf = wafer_map.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument(
            "wafer_map must be a 2D array with shape (rows, cols)");
    }
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    const auto* data = static_cast<const uint8_t*>(buf.ptr);
    return waferlens::compute(data, rows, cols, config);
}

}  // namespace

PYBIND11_MODULE(_waferlens_core, m) {
    m.doc() = "waferlens: CUDA-accelerated wafer-map defect geometry";

    py::class_<waferlens::ClusterDescriptor>(m, "ClusterDescriptor")
        .def_readonly("label",        &waferlens::ClusterDescriptor::label)
        .def_readonly("area",         &waferlens::ClusterDescriptor::area)
        .def_readonly("centroid_row", &waferlens::ClusterDescriptor::centroid_row)
        .def_readonly("centroid_col", &waferlens::ClusterDescriptor::centroid_col)
        .def_readonly("eccentricity", &waferlens::ClusterDescriptor::eccentricity)
        .def("__repr__", [](const waferlens::ClusterDescriptor& d) {
            return "<ClusterDescriptor label=" + std::to_string(d.label)
                   + " area=" + std::to_string(d.area)
                   + " eccentricity=" + std::to_string(d.eccentricity) + ">";
        });

    py::class_<waferlens::WaferFeatures>(m, "WaferFeatures")
        .def_readonly("radial_density",    &waferlens::WaferFeatures::radial_density)
        .def_readonly("angular_histogram", &waferlens::WaferFeatures::angular_histogram)
        .def_readonly("edge_concentration",&waferlens::WaferFeatures::edge_concentration)
        .def_readonly("ring_score",        &waferlens::WaferFeatures::ring_score)
        .def_readonly("peak_ring_index",   &waferlens::WaferFeatures::peak_ring_index)
        .def_readonly("clusters",          &waferlens::WaferFeatures::clusters)
        .def_readonly("scratch_linearity", &waferlens::WaferFeatures::scratch_linearity)
        .def_readonly("scratch_angle",     &waferlens::WaferFeatures::scratch_angle)
        .def_readonly("density_gradient",  &waferlens::WaferFeatures::density_gradient)
        .def_readonly("gradient_grid_size",&waferlens::WaferFeatures::gradient_grid_size)
        .def_readonly("total_defects",     &waferlens::WaferFeatures::total_defects)
        .def_readonly("total_normal",      &waferlens::WaferFeatures::total_normal);

    py::class_<waferlens::AnalyzerConfig>(m, "AnalyzerConfig")
        .def(py::init<>())
        .def_readwrite("radial_bins",     &waferlens::AnalyzerConfig::radial_bins)
        .def_readwrite("angular_sectors", &waferlens::AnalyzerConfig::angular_sectors)
        .def_readwrite("edge_fraction",   &waferlens::AnalyzerConfig::edge_fraction)
        .def_readwrite("gradient_grid",   &waferlens::AnalyzerConfig::gradient_grid)
        .def_readwrite("radon_angles",    &waferlens::AnalyzerConfig::radon_angles)
        .def_readwrite("notch_angle",     &waferlens::AnalyzerConfig::notch_angle)
        .def_readwrite("prefer_gpu",      &waferlens::AnalyzerConfig::prefer_gpu);

    py::class_<waferlens::WaferAnalyzer>(m, "WaferAnalyzer")
        .def(py::init<const waferlens::AnalyzerConfig&>(),
             py::arg("config") = waferlens::AnalyzerConfig{})
        .def("compute", &compute_from_numpy,
             py::arg("wafer_map"),
             "Compute geometric features for a single wafer map (2D uint8 array).")
        .def("compute_batch", &compute_batch_from_numpy,
             py::arg("wafer_maps"),
             "Compute features for a batch of wafer maps (3D uint8 array).")
        .def("using_gpu", &waferlens::WaferAnalyzer::using_gpu)
        .def_property_readonly("config", &waferlens::WaferAnalyzer::config);

    m.def("compute", &free_compute,
          py::arg("wafer_map"),
          py::arg("config") = waferlens::AnalyzerConfig{},
          "Compute geometric features for a single wafer map.");
}
