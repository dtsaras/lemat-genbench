"""Tests for the band-gap preprocessor (with a fake backend; no ML deps)."""

import pytest
from pymatgen.core import Lattice, Structure

from lemat_genbench.preprocess.band_gap_preprocess import BandGapPreprocessor
from lemat_genbench.properties.band_gap_backends import (
    BandGapBackend,
    register_backend,
)


class _FakeGapBackend(BandGapBackend):
    name = "fakepre"

    def predict(self, structure):
        # Return None for structures with no Si (to exercise the None path).
        return 2.0 if "Si" in [str(s.specie) for s in structure.sites] else None


register_backend("fake_preproc", _FakeGapBackend)


def _struct(species):
    return Structure(
        Lattice.cubic(4.0), list(species), [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def test_preprocessor_attaches_band_gap():
    pre = BandGapPreprocessor(backend="fake_preproc")
    res = pre.run([_struct(("Si", "Si"))])
    s = res.processed_structures[0]
    assert s.properties["band_gap"] == pytest.approx(2.0)
    assert s.properties["band_gap_backend"] == "fakepre"
    assert res.failed_indices == []


def test_preprocessor_attaches_none_when_backend_cannot_score():
    pre = BandGapPreprocessor(backend="fake_preproc")
    res = pre.run([_struct(("Na", "Cl"))])
    s = res.processed_structures[0]
    # Structure is retained with band_gap=None; the metric later treats it as NaN.
    assert s.properties["band_gap"] is None
    assert res.failed_indices == []
