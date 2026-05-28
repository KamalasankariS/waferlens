# waferlens

CUDA-accelerated geometric feature extraction for semiconductor wafer maps.

`waferlens` computes a comprehensive set of spatial defect descriptors from wafer bin maps (WM-811K format), with optional GPU acceleration via CUDA and transparent CPU fallback.

## Features

Seven geometric kernels, each available in both C++/CUDA and pure-CPU implementations:

| Kernel | Description | CUDA Pattern |
|--------|-------------|--------------|
| Radial density profile | Defect distribution across concentric rings | Parallel reduction with shared-memory histogram |
| Angular histogram | Notch-relative angular defect distribution | Shared-memory atomic accumulation |
| Edge concentration | Fraction of defects in the outer annulus | Derived from radial profile |
| Ring score | Peak-to-uniform radial concentration ratio | Composition over radial density |
| Connected components | Cluster labeling with per-cluster eccentricity | Parallel union-find (Komura) |
| Scratch linearity | Radon-transform peak for linear feature detection | Block-per-angle sinogram projection |
| Density gradient | 2D Sobel gradient of local defect density | Binning kernel + stencil computation |

## Quick start

```python
import numpy as np
import waferlens

wafer_map = np.load("wafer.npy").astype(np.uint8)

features = waferlens.compute(wafer_map)

print(f"Ring score:          {features.ring_score:.3f}")
print(f"Edge concentration:  {features.edge_concentration:.3f}")
print(f"Scratch linearity:   {features.scratch_linearity:.3f}")
print(f"Number of clusters:  {len(features.clusters)}")
```

## Installation

### From source (recommended)

```bash
git clone https://github.com/kamalasankaris/waferlens.git
cd waferlens
pip install -e ".[dev]"
```

Requires CMake 3.18+, a C++17 compiler, and Python 3.9+. CUDA is optional -- if `nvcc` is not found, the library builds with CPU-only support.

### C++ only

```bash
mkdir build && cd build
cmake .. -DWAFERLENS_BUILD_PYTHON=OFF
cmake --build . --config Release
ctest --output-on-failure
```

## API

### Python

```python
import waferlens

config = waferlens.AnalyzerConfig()
config.radial_bins = 64
config.edge_fraction = 0.10

analyzer = waferlens.WaferAnalyzer(config)

features = analyzer.compute(wafer_map)

batch_results = analyzer.compute_batch(wafer_maps_3d)
```

### C++

```cpp
#include <waferlens/waferlens.h>

waferlens::AnalyzerConfig config;
config.radial_bins = 64;

waferlens::WaferAnalyzer analyzer(config);
waferlens::WaferFeatures features = analyzer.compute(data, rows, cols);
```

## WaferFeatures fields

| Field | Type | Description |
|-------|------|-------------|
| `radial_density` | `vector<float>` | Normalized radial defect profile |
| `angular_histogram` | `vector<float>` | Normalized angular defect distribution |
| `edge_concentration` | `float` | Fraction of defects in the edge annulus |
| `ring_score` | `float` | Peak-to-uniform radial concentration |
| `peak_ring_index` | `int` | Radial bin with highest density |
| `clusters` | `vector<ClusterDescriptor>` | Per-cluster geometry (area, centroid, eccentricity) |
| `scratch_linearity` | `float` | Radon-based linear feature strength [0, 1] |
| `scratch_angle` | `float` | Orientation of strongest linear feature (degrees) |
| `density_gradient` | `vector<float>` | 2D gradient magnitude of local density |
| `total_defects` | `int` | Total defective cell count |
| `total_normal` | `int` | Total normal cell count |

## Benchmarks

Run the throughput benchmark against the NumPy reference implementation:

```bash
python benchmarks/benchmark_throughput.py --num-wafers 10000 --size 64
```

## Dataset

`waferlens` is designed for the [WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) wafer map dataset. Download it with:

```bash
python scripts/download_wm811k.py
```

## Project structure

```
waferlens/
  include/waferlens/     C++ public headers
  src/cpu/               CPU kernel implementations
  src/kernels/           CUDA kernel implementations
  python/waferlens/      Python package and pybind11 bindings
  python/tests/          Python test suite
  tests/cpp/             C++ test suite (Google Test)
  benchmarks/            Throughput benchmarks and NumPy baseline
  probe/                 VLM faithfulness probe (Layer 2)
  scripts/               Dataset download utilities
```

## License

MIT
