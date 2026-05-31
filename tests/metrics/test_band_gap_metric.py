"""Tests for the band-gap metric and the pluggable backend registry.

These run WITHOUT the heavy ML backends: the metric reads a ``band_gap`` value
from ``structure.properties`` (set directly here), and the on-the-fly path is
exercised with a tiny in-test fake backend.
"""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from lemat_genbench.metrics.band_gap_metric import BandGapMetric
from lemat_genbench.properties.band_gap_backends import (
    BandGapBackend,
    available_backends,
    get_band_gap_backend,
    register_backend,
)


def _struct(gap=None, species=("Na", "Cl")):
    s = Structure(
        Lattice.cubic(4.0), list(species), [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    if gap is not None:
        s.properties["band_gap"] = gap
    return s


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_aggregate_classification_and_stats():
    m = BandGapMetric(metal_threshold=0.1, insulator_threshold=3.0)
    res = m.compute([_struct(g) for g in (0.0, 1.5, 4.0)])
    met = res.metrics
    assert met["n_valid"] == 3
    assert met["metallic_ratio"] == pytest.approx(1 / 3)
    assert met["semiconductor_ratio"] == pytest.approx(1 / 3)
    assert met["insulator_ratio"] == pytest.approx(1 / 3)
    assert met["mean_band_gap"] == pytest.approx(5.5 / 3)
    assert met["min_band_gap"] == pytest.approx(0.0)
    assert met["max_band_gap"] == pytest.approx(4.0)
    assert res.primary_metric == "mean_band_gap"
    assert res.failed_indices == []


def test_missing_band_gap_is_failure():
    m = BandGapMetric()
    res = m.compute([_struct(gap=None, species=("Si", "Si"))])
    assert 0 in res.failed_indices
    assert np.isnan(res.individual_values[0])


def test_target_window_becomes_primary_metric():
    m = BandGapMetric(target_min=1.0, target_max=2.0)
    res = m.compute([_struct(g) for g in (0.5, 1.5, 1.8, 4.0)])
    assert res.primary_metric == "fraction_in_target_window"
    assert res.metrics["fraction_in_target_window"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Backend registry + on-the-fly path
# --------------------------------------------------------------------------


def test_registry_contains_builtin_backends():
    backends = available_backends()
    assert "hamgnn" in backends
    assert "alignn" in backends


def test_compute_if_missing_uses_backend():
    class _FakeBackend(BandGapBackend):
        name = "fake"

        def predict(self, structure):
            return 1.234

    register_backend("fake_bandgap", _FakeBackend)
    assert isinstance(get_band_gap_backend("fake_bandgap"), _FakeBackend)

    m = BandGapMetric(compute_if_missing=True, backend="fake_bandgap")
    res = m.compute([_struct(gap=None, species=("Si", "Si"))])
    assert res.individual_values[0] == pytest.approx(1.234)
    assert res.failed_indices == []


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_band_gap_backend("does_not_exist")


# --------------------------------------------------------------------------
# HamGNN backend: clear, actionable error until configured
# --------------------------------------------------------------------------


def test_hamgnn_reports_missing_configuration(monkeypatch):
    for key in (
        "HAMGNN_ENV_BIN",
        "HAMGNN_OPENMX_POSTPROCESS",
        "HAMGNN_READ_OPENMX",
        "HAMGNN_MODEL_PKL",
        "HAMGNN_PREDICTOR_SCRIPT",
        "HAMGNN_DFT_DATA",
    ):
        monkeypatch.delenv(key, raising=False)

    from lemat_genbench.properties.band_gap_backends import (
        HamGNNBandGapBackend,
        HamGNNConfig,
        HamGNNNotConfigured,
    )

    backend = HamGNNBandGapBackend(config=HamGNNConfig())
    assert backend.config.missing()  # a concrete checklist of what's unset
    assert backend.is_available() is False
    with pytest.raises(HamGNNNotConfigured):
        backend.predict(_struct(gap=None, species=("Si", "Si")))
