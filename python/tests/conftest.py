import numpy as np
import pytest


@pytest.fixture
def empty_wafer():
    """64x64 wafer with no defects (all normal cells)."""
    return np.ones((64, 64), dtype=np.uint8)


@pytest.fixture
def center_defect():
    """64x64 wafer with a circular defect cluster at the center."""
    wafer = np.ones((64, 64), dtype=np.uint8)
    cr, cc = 31.5, 31.5
    for r in range(64):
        for c in range(64):
            if np.sqrt((r - cr) ** 2 + (c - cc) ** 2) <= 5:
                wafer[r, c] = 2
    return wafer


@pytest.fixture
def edge_ring():
    """64x64 wafer with defects concentrated in the outer 15% radius."""
    wafer = np.ones((64, 64), dtype=np.uint8)
    cr, cc = 31.5, 31.5
    max_r = np.sqrt(cr ** 2 + cc ** 2)
    threshold = 0.85 * max_r
    for r in range(64):
        for c in range(64):
            if np.sqrt((r - cr) ** 2 + (c - cc) ** 2) >= threshold:
                wafer[r, c] = 2
    return wafer


@pytest.fixture
def scratch_wafer():
    """64x64 wafer with a horizontal scratch through the center."""
    wafer = np.ones((64, 64), dtype=np.uint8)
    wafer[32, :] = 2
    wafer[33, :] = 2
    return wafer


@pytest.fixture
def two_cluster_wafer():
    """64x64 wafer with two separated defect clusters."""
    wafer = np.ones((64, 64), dtype=np.uint8)
    wafer[10, 10] = 2
    wafer[10, 11] = 2
    wafer[11, 10] = 2
    wafer[50, 50] = 2
    wafer[50, 51] = 2
    return wafer
