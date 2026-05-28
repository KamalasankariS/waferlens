"""
NumPy reference implementations of all waferlens geometric features.

These serve two purposes:
  1. Correctness oracle for testing the C++/CUDA implementations.
  2. Performance baseline for throughput benchmarks.
"""

import numpy as np
from scipy import ndimage
from typing import NamedTuple


class ClusterInfo(NamedTuple):
    label: int
    area: int
    centroid_row: float
    centroid_col: float
    eccentricity: float


class WaferFeaturesNP(NamedTuple):
    radial_density: np.ndarray
    angular_histogram: np.ndarray
    edge_concentration: float
    ring_score: float
    peak_ring_index: int
    clusters: list
    scratch_linearity: float
    scratch_angle: float
    density_gradient: np.ndarray
    total_defects: int
    total_normal: int


def radial_density(wafer: np.ndarray, num_bins: int = 32) -> np.ndarray:
    rows, cols = wafer.shape
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_r = np.sqrt(cr ** 2 + cc ** 2)

    defect_mask = wafer == 2
    if not defect_mask.any() or max_r < 1e-6:
        return np.zeros(num_bins, dtype=np.float32)

    rr, cc_arr = np.where(defect_mask)
    dist = np.sqrt((rr - cr) ** 2 + (cc_arr - cc) ** 2) / max_r
    bins = np.clip((dist * num_bins).astype(int), 0, num_bins - 1)

    profile = np.bincount(bins, minlength=num_bins).astype(np.float32)
    return profile / profile.sum()


def angular_histogram(wafer: np.ndarray, num_sectors: int = 36,
                      notch_angle: float = 0.0) -> np.ndarray:
    rows, cols = wafer.shape
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0

    defect_mask = wafer == 2
    if not defect_mask.any():
        return np.zeros(num_sectors, dtype=np.float32)

    rr, cc_arr = np.where(defect_mask)
    angles = np.arctan2(rr - cr, cc_arr - cc) - notch_angle
    angles = angles % (2 * np.pi)

    sector_width = 2 * np.pi / num_sectors
    sector_idx = np.clip((angles / sector_width).astype(int), 0, num_sectors - 1)

    hist = np.bincount(sector_idx, minlength=num_sectors).astype(np.float32)
    return hist / hist.sum()


def edge_concentration(wafer: np.ndarray, edge_fraction: float = 0.15) -> float:
    rows, cols = wafer.shape
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_r = np.sqrt(cr ** 2 + cc ** 2)
    threshold = (1.0 - edge_fraction) * max_r

    defect_mask = wafer == 2
    total = defect_mask.sum()
    if total == 0:
        return 0.0

    rr, cc_arr = np.where(defect_mask)
    dist = np.sqrt((rr - cr) ** 2 + (cc_arr - cc) ** 2)

    return float((dist >= threshold).sum()) / float(total)


def ring_score(radial_profile: np.ndarray) -> tuple:
    if len(radial_profile) == 0 or radial_profile.sum() == 0:
        return 1.0, 0
    peak_idx = int(np.argmax(radial_profile))
    score = float(radial_profile[peak_idx] * len(radial_profile))
    return score, peak_idx


def connected_components(wafer: np.ndarray) -> list:
    defect_mask = (wafer == 2).astype(np.int32)
    labeled, num_features = ndimage.label(defect_mask)

    clusters = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        area = len(coords)
        centroid = coords.mean(axis=0)

        centered = coords - centroid
        cov = np.cov(centered.T) if area > 1 else np.zeros((2, 2))
        if cov.ndim == 0:
            cov = np.array([[cov, 0], [0, 0]])

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]

        ecc = 0.0
        if eigvals[0] > 1e-12:
            ecc = float(np.sqrt(1.0 - eigvals[1] / eigvals[0]))

        clusters.append(ClusterInfo(
            label=i,
            area=area,
            centroid_row=float(centroid[0]),
            centroid_col=float(centroid[1]),
            eccentricity=ecc
        ))

    return clusters


def scratch_linearity(wafer: np.ndarray, num_angles: int = 180) -> tuple:
    rows, cols = wafer.shape
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_r = np.sqrt(cr ** 2 + cc ** 2)
    num_offsets = int(2 * max_r) + 1

    defect_mask = wafer == 2
    total = defect_mask.sum()
    if total == 0:
        return 0.0, 0.0

    rr, cc_arr = np.where(defect_mask)
    dr = rr - cr
    dc = cc_arr - cc

    global_max = 0
    best_angle = 0

    offset_center = (num_offsets - 1) / 2.0

    for ai in range(num_angles):
        theta = ai * np.pi / num_angles
        proj = dr * np.cos(theta) + dc * np.sin(theta)
        bins = np.clip((proj + offset_center + 0.5).astype(int), 0, num_offsets - 1)
        sinogram_col = np.bincount(bins, minlength=num_offsets)
        local_max = sinogram_col.max()
        if local_max > global_max:
            global_max = local_max
            best_angle = ai

    linearity = float(global_max) / float(total)
    angle = best_angle * 180.0 / num_angles

    return linearity, angle


def density_gradient(wafer: np.ndarray, grid_size: int = 16) -> np.ndarray:
    rows, cols = wafer.shape
    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            if wafer[r, c] == 0:
                continue
            gr = min(int(r * grid_size / rows), grid_size - 1)
            gc = min(int(c * grid_size / cols), grid_size - 1)
            counts[gr, gc] += 1
            if wafer[r, c] == 2:
                density[gr, gc] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(counts > 0, density / counts, 0)

    gx = ndimage.sobel(density, axis=1)
    gy = ndimage.sobel(density, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    max_g = grad.max()
    if max_g > 1e-8:
        grad /= max_g

    return grad.astype(np.float32)


def compute_all(wafer: np.ndarray, num_radial_bins: int = 32,
                num_sectors: int = 36, edge_frac: float = 0.15,
                grid_size: int = 16, num_radon_angles: int = 180,
                notch_angle_deg: float = 0.0) -> WaferFeaturesNP:
    notch_rad = np.radians(notch_angle_deg)

    rd = radial_density(wafer, num_radial_bins)
    ah = angular_histogram(wafer, num_sectors, notch_rad)
    ec = edge_concentration(wafer, edge_frac)
    rs, pri = ring_score(rd)
    cc = connected_components(wafer)
    sl, sa = scratch_linearity(wafer, num_radon_angles)
    dg = density_gradient(wafer, grid_size)

    defect_mask = wafer == 2
    normal_mask = wafer == 1

    return WaferFeaturesNP(
        radial_density=rd,
        angular_histogram=ah,
        edge_concentration=ec,
        ring_score=rs,
        peak_ring_index=pri,
        clusters=cc,
        scratch_linearity=sl,
        scratch_angle=sa,
        density_gradient=dg.ravel(),
        total_defects=int(defect_mask.sum()),
        total_normal=int(normal_mask.sum())
    )
