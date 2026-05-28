"""
waferlens -- CUDA-accelerated wafer-map defect geometry.

Provides fast geometric feature extraction for semiconductor wafer maps,
with automatic GPU acceleration when CUDA is available and transparent
CPU fallback otherwise.

Quick start::

    import numpy as np
    import waferlens

    wafer_map = np.load("wafer.npy").astype(np.uint8)
    features = waferlens.compute(wafer_map)

    print(features.ring_score)
    print(features.edge_concentration)
    print(features.clusters)
"""

from waferlens._waferlens_core import (  # type: ignore[import-not-found]
    AnalyzerConfig,
    ClusterDescriptor,
    WaferAnalyzer,
    WaferFeatures,
    compute,
)

__version__ = "0.1.0"

__all__ = [
    "AnalyzerConfig",
    "ClusterDescriptor",
    "WaferAnalyzer",
    "WaferFeatures",
    "compute",
]
