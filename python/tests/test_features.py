"""
Tests for waferlens Python bindings.

These tests validate the Python API against expected geometric properties
of synthetic wafer maps.  They serve as both correctness tests and as
documentation of the expected behavior of each feature.
"""

import numpy as np
import pytest


def _import_waferlens():
    try:
        import waferlens
        return waferlens
    except ImportError:
        pytest.skip("waferlens C++ extension not built")


class TestRadialDensity:

    def test_sums_to_one(self, center_defect):
        wl = _import_waferlens()
        feat = wl.compute(center_defect)
        assert abs(sum(feat.radial_density) - 1.0) < 0.01

    def test_center_defect_peaks_early(self, center_defect):
        wl = _import_waferlens()
        feat = wl.compute(center_defect)
        peak = np.argmax(feat.radial_density)
        assert peak < len(feat.radial_density) // 2

    def test_no_defects_all_zero(self, empty_wafer):
        wl = _import_waferlens()
        feat = wl.compute(empty_wafer)
        assert all(v == 0.0 for v in feat.radial_density)


class TestAngularHistogram:

    def test_sums_to_one(self, center_defect):
        wl = _import_waferlens()
        feat = wl.compute(center_defect)
        assert abs(sum(feat.angular_histogram) - 1.0) < 0.01

    def test_symmetric_defect_is_roughly_uniform(self, center_defect):
        wl = _import_waferlens()
        feat = wl.compute(center_defect)
        values = np.array(feat.angular_histogram)
        assert np.std(values) < 0.05


class TestEdgeConcentration:

    def test_center_defect_low(self, center_defect):
        wl = _import_waferlens()
        feat = wl.compute(center_defect)
        assert feat.edge_concentration < 0.05

    def test_edge_ring_high(self, edge_ring):
        wl = _import_waferlens()
        feat = wl.compute(edge_ring)
        assert feat.edge_concentration > 0.8

    def test_no_defects_zero(self, empty_wafer):
        wl = _import_waferlens()
        feat = wl.compute(empty_wafer)
        assert feat.edge_concentration == 0.0


class TestRingScore:

    def test_edge_ring_scores_high(self, edge_ring):
        wl = _import_waferlens()
        feat = wl.compute(edge_ring)
        assert feat.ring_score > 2.0


class TestConnectedComponents:

    def test_single_cluster(self, center_defect):
        wl = _import_waferlens()
        feat = wl.compute(center_defect)
        assert len(feat.clusters) == 1

    def test_two_clusters(self, two_cluster_wafer):
        wl = _import_waferlens()
        feat = wl.compute(two_cluster_wafer)
        assert len(feat.clusters) == 2

    def test_no_clusters(self, empty_wafer):
        wl = _import_waferlens()
        feat = wl.compute(empty_wafer)
        assert len(feat.clusters) == 0

    def test_eccentricity_range(self, scratch_wafer):
        wl = _import_waferlens()
        feat = wl.compute(scratch_wafer)
        for cluster in feat.clusters:
            assert 0.0 <= cluster.eccentricity <= 1.0


class TestScratchLinearity:

    def test_scratch_scores_high(self, scratch_wafer):
        wl = _import_waferlens()
        feat = wl.compute(scratch_wafer)
        assert feat.scratch_linearity > 0.4

    def test_center_defect_scores_low(self, center_defect):
        wl = _import_waferlens()
        feat = wl.compute(center_defect)
        assert feat.scratch_linearity < 0.3


class TestDensityGradient:

    def test_correct_size(self, center_defect):
        wl = _import_waferlens()
        cfg = wl.AnalyzerConfig()
        cfg.gradient_grid = 8
        analyzer = wl.WaferAnalyzer(cfg)
        feat = analyzer.compute(center_defect)
        assert len(feat.density_gradient) == 64
        assert feat.gradient_grid_size == 8


class TestBatch:

    def test_batch_matches_individual(self, center_defect):
        wl = _import_waferlens()
        batch = np.stack([center_defect, center_defect, center_defect])
        analyzer = wl.WaferAnalyzer()
        results = analyzer.compute_batch(batch)
        assert len(results) == 3

        single = analyzer.compute(center_defect)
        for feat in results:
            assert feat.total_defects == single.total_defects
            assert abs(feat.ring_score - single.ring_score) < 1e-6
