"""Tests for the BVSE ion migration-barrier metric.

The aggregation logic and the "no mobile ion -> NaN" path run without bvlain.
The real BVSE computations (LiF / LiCoO2) are guarded by skipif.
"""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from lemat_genbench.metrics.migration_barrier_metric import MigrationBarrierMetric
from lemat_genbench.properties.migration_barrier import (
    NO_PERCOLATION_SENTINEL,
    is_available,
    is_no_percolation,
)

skip_no_bvlain = pytest.mark.skipif(
    not is_available(), reason="bvlain not installed"
)


# --------------------------------------------------------------------------
# Reference structures (ported from matter_evolve)
# --------------------------------------------------------------------------


def _lif_rocksalt() -> Structure:
    """Rocksalt LiF, a = 4.026 A — 3D Li percolation in all directions."""
    lat = Lattice.cubic(4.026)
    species = ["Li"] * 4 + ["F"] * 4
    coords = [
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
    ]
    return Structure(lat, species, coords)


def _licoo2_layered() -> Structure:
    """Layered LiCoO2 (R-3m): Li percolates in-plane but NOT out-of-plane, so
    the 3D barrier should hit the no-percolation sentinel."""
    lat = Lattice.from_parameters(2.85, 2.85, 14.05, 90, 90, 120)
    return Structure(
        lat,
        ["Li", "Co", "O", "O"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.260], [0.0, 0.0, 0.740]],
    )


def _nacl_no_li() -> Structure:
    lat = Lattice.cubic(5.64)
    return Structure(lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


# --------------------------------------------------------------------------
# Aggregation logic (no bvlain required)
# --------------------------------------------------------------------------


def test_aggregate_classifies_real_sentinel_and_nan():
    m = MigrationBarrierMetric(
        fast_threshold=0.6, no_percolation_sentinel=NO_PERCOLATION_SENTINEL
    )
    # 0.3 -> fast conductor, 0.8 -> slow conductor, sentinel -> non-percolating,
    # nan -> no mobile ion (excluded).
    out = m.aggregate_results([0.3, NO_PERCOLATION_SENTINEL, float("nan"), 0.8])
    met = out["metrics"]
    assert met["n_evaluated"] == 4
    assert met["n_with_mobile_ion"] == 3
    assert met["n_percolating"] == 2
    assert met["n_non_percolating"] == 1
    assert met["fraction_with_mobile_ion"] == pytest.approx(0.75)
    assert met["fraction_percolating"] == pytest.approx(2 / 3)
    assert met["fraction_fast_ion_conductors"] == pytest.approx(1 / 3)
    assert met["mean_barrier"] == pytest.approx(0.55)
    assert met["min_barrier"] == pytest.approx(0.3)
    assert met["max_barrier"] == pytest.approx(0.8)
    assert out["primary_metric"] == "fraction_fast_ion_conductors"


def test_aggregate_all_nan_gives_defined_metrics():
    m = MigrationBarrierMetric()
    out = m.aggregate_results([float("nan"), float("nan")])
    met = out["metrics"]
    assert met["n_with_mobile_ion"] == 0
    assert np.isnan(met["fraction_fast_ion_conductors"])
    assert np.isnan(met["mean_barrier"])
    # primary metric key must exist for MetricResult validation
    assert out["primary_metric"] in met


def test_invalid_dimensionality_raises():
    with pytest.raises(ValueError):
        MigrationBarrierMetric(dimensionality="4d")


def test_no_mobile_ion_returns_nan_not_failure():
    """A structure lacking the mobile ion is excluded (NaN), not a failure."""
    m = MigrationBarrierMetric(dimensionality="3d")
    res = m.compute([_nacl_no_li()])
    assert res.failed_indices == []
    assert np.isnan(res.individual_values[0])
    assert res.metrics["n_with_mobile_ion"] == 0


# --------------------------------------------------------------------------
# Real BVSE computations (need bvlain)
# --------------------------------------------------------------------------


@skip_no_bvlain
def test_lif_rocksalt_3d_is_finite_conductor():
    m = MigrationBarrierMetric(dimensionality="3d")
    res = m.compute([_lif_rocksalt()])
    val = res.individual_values[0]
    assert not np.isnan(val)
    assert not is_no_percolation(val)
    assert 0.0 < val < 5.0
    assert res.metrics["n_percolating"] == 1


@skip_no_bvlain
def test_licoo2_layered_3d_hits_sentinel():
    m = MigrationBarrierMetric(dimensionality="3d")
    res = m.compute([_licoo2_layered()])
    val = res.individual_values[0]
    assert is_no_percolation(val)
    assert res.metrics["n_non_percolating"] == 1
    assert res.metrics["n_percolating"] == 0
