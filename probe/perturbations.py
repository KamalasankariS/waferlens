"""
Wafer map perturbation generators for faithfulness probing.

Each function takes a wafer map and produces a modified version that
invalidates a specific geometric property, enabling counterfactual
evaluation of VLM reasoning.
"""

import numpy as np


def remove_edge_defects(wafer: np.ndarray, edge_fraction: float = 0.15) -> np.ndarray:
    """Remove all defects from the outer annulus, invalidating edge concentration."""
    result = wafer.copy()
    rows, cols = result.shape
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_r = np.sqrt(cr ** 2 + cc ** 2)
    threshold = (1.0 - edge_fraction) * max_r

    for r in range(rows):
        for c in range(cols):
            if result[r, c] == 2:
                dist = np.sqrt((r - cr) ** 2 + (c - cc) ** 2)
                if dist >= threshold:
                    result[r, c] = 1
    return result


def remove_center_defects(wafer: np.ndarray, center_fraction: float = 0.3) -> np.ndarray:
    """Remove defects from the inner region, invalidating center concentration."""
    result = wafer.copy()
    rows, cols = result.shape
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_r = np.sqrt(cr ** 2 + cc ** 2)
    threshold = center_fraction * max_r

    for r in range(rows):
        for c in range(cols):
            if result[r, c] == 2:
                dist = np.sqrt((r - cr) ** 2 + (c - cc) ** 2)
                if dist <= threshold:
                    result[r, c] = 1
    return result


def break_linear_pattern(wafer: np.ndarray) -> np.ndarray:
    """Scatter linear defects randomly to destroy scratch-like structure."""
    result = wafer.copy()
    defect_coords = np.argwhere(result == 2)
    if len(defect_coords) == 0:
        return result

    rows, cols = result.shape
    on_wafer = np.argwhere(result >= 1)
    if len(on_wafer) == 0:
        return result

    rng = np.random.default_rng(42)

    for r, c in defect_coords:
        result[r, c] = 1

    indices = rng.choice(len(on_wafer), size=len(defect_coords), replace=False)
    for idx in indices:
        r, c = on_wafer[idx]
        result[r, c] = 2

    return result


def scatter_clusters(wafer: np.ndarray) -> np.ndarray:
    """Redistribute clustered defects uniformly to destroy spatial clustering."""
    result = wafer.copy()
    defect_coords = np.argwhere(result == 2)
    n_defects = len(defect_coords)
    if n_defects == 0:
        return result

    on_wafer = np.argwhere(result >= 1)
    if len(on_wafer) < n_defects:
        return result

    for r, c in defect_coords:
        result[r, c] = 1

    rng = np.random.default_rng(42)
    indices = rng.choice(len(on_wafer), size=n_defects, replace=False)
    for idx in indices:
        r, c = on_wafer[idx]
        result[r, c] = 2

    return result


def flatten_radial_profile(wafer: np.ndarray) -> np.ndarray:
    """Redistribute defects to create a uniform radial distribution."""
    result = wafer.copy()
    defect_coords = np.argwhere(result == 2)
    n_defects = len(defect_coords)
    if n_defects == 0:
        return result

    rows, cols = result.shape
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0

    on_wafer = np.argwhere(result >= 1)
    distances = np.sqrt((on_wafer[:, 0] - cr) ** 2 + (on_wafer[:, 1] - cc) ** 2)
    max_d = distances.max()
    if max_d < 1e-6:
        return result

    for r, c in defect_coords:
        result[r, c] = 1

    rng = np.random.default_rng(42)
    num_bins = 10
    bin_edges = np.linspace(0, max_d, num_bins + 1)
    per_bin = n_defects // num_bins
    remainder = n_defects % num_bins

    placed = 0
    for b in range(num_bins):
        mask = (distances >= bin_edges[b]) & (distances < bin_edges[b + 1])
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue

        count = per_bin + (1 if b < remainder else 0)
        count = min(count, len(candidates))
        chosen = rng.choice(candidates, size=count, replace=False)
        for idx in chosen:
            r, c = on_wafer[idx]
            result[r, c] = 2
            placed += 1

    return result


PERTURBATION_MAP = {
    "edge_concentration": remove_edge_defects,
    "radial_density": flatten_radial_profile,
    "ring_score": flatten_radial_profile,
    "clusters": scatter_clusters,
    "scratch_linearity": break_linear_pattern,
    "angular_histogram": scatter_clusters,
    "density_gradient": scatter_clusters,
}


def get_counterfactual(wafer: np.ndarray, cited_feature: str) -> np.ndarray:
    """Return a perturbed wafer that invalidates the given cited feature."""
    perturb_fn = PERTURBATION_MAP.get(cited_feature)
    if perturb_fn is None:
        return wafer.copy()
    return perturb_fn(wafer)
