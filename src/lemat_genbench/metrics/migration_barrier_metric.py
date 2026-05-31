"""Ionic migration-barrier metric (BVSE percolation barriers via bvlain).

Per structure: the minimum saddle-point energy (eV) a mobile ion (Li+ by
default) must clear to percolate through the structure at the requested
dimensionality. Aggregate: the fraction of generated structures that are fast
ion conductors, plus barrier statistics over the conducting subset.

See :mod:`lemat_genbench.properties.migration_barrier` for the underlying BVSE
computation and its NaN / sentinel / raise semantics.
"""

from typing import Any

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.metrics.base import BaseMetric
from lemat_genbench.properties.migration_barrier import (
    DIM_KEYS,
    NO_PERCOLATION_SENTINEL,
    MobileIonAbsent,
    compute_barriers,
)


class MigrationBarrierMetric(BaseMetric):
    """BVSE ion migration-barrier metric.

    Parameters
    ----------
    mobile_ion : str, default="Li1+"
        Pymatgen-style ion string (e.g. "Li1+", "Na1+", "Mg2+").
    dimensionality : str, default="3d"
        One of "1d", "2d", "3d" (require percolation in that many directions),
        or "min" (the easiest/best direction).
    r_cut, resolution, k, encut
        BVSE parameters forwarded to bvlain.
    fast_threshold : float, default=0.6
        A structure counts as a "fast ion conductor" if its barrier is below
        this value (eV). ~0.6 eV is a common usability cutoff; <0.4 eV is fast.
    no_percolation_sentinel : float, default=10.0
        Barrier returned when bvlain runs but finds no percolating path at this
        dimensionality (counted as "not a conductor", not as missing data).
    lower_is_better : bool, default=True
        Lower barrier => better conductor.
    n_jobs : int, default=1
        Parallel workers (each runs bvlain independently; pure CPU).
    timeout : int | None, default=30
        Per-structure wall-clock cap (enforced by BaseMetric.timeout_context).
    """

    def __init__(
        self,
        mobile_ion: str = "Li1+",
        dimensionality: str = "3d",
        r_cut: float = 10.0,
        resolution: float = 0.2,
        k: int = 100,
        encut: float = 5.0,
        fast_threshold: float = 0.6,
        no_percolation_sentinel: float = NO_PERCOLATION_SENTINEL,
        name: str = None,
        description: str = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
        timeout: int | None = 30,
    ):
        dim = dimensionality.lower()
        if dim not in DIM_KEYS and dim != "min":
            raise ValueError(
                f"dimensionality must be one of {sorted(DIM_KEYS)} or 'min', "
                f"got {dimensionality!r}"
            )
        super().__init__(
            name=name or f"MigrationBarrier_{dim}",
            description=description
            or f"BVSE {dim} {mobile_ion} ion migration barrier (eV)",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
            timeout=timeout,
        )
        self.mobile_ion = mobile_ion
        self.dimensionality = dim
        self.r_cut = r_cut
        self.resolution = resolution
        self.k = k
        self.encut = encut
        self.fast_threshold = fast_threshold
        self.no_percolation_sentinel = no_percolation_sentinel

    def _get_compute_attributes(self) -> dict[str, Any]:
        attrs = super()._get_compute_attributes()  # picks up timeout / verbose
        attrs.update(
            {
                "mobile_ion": self.mobile_ion,
                "dimensionality": self.dimensionality,
                "r_cut": self.r_cut,
                "resolution": self.resolution,
                "k": self.k,
                "encut": self.encut,
                "no_percolation_sentinel": self.no_percolation_sentinel,
            }
        )
        return attrs

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Return the migration barrier (eV) for one structure.

        * mobile ion absent          -> NaN (not a failure; excluded from stats)
        * no percolating path at dim  -> sentinel (counted as "not a conductor")
        * real barrier                -> finite float
        * bvlain hard failure         -> raises (BaseMetric -> failed_indices)
        """
        dim = compute_args.get("dimensionality", "3d")
        sentinel = compute_args.get("no_percolation_sentinel", NO_PERCOLATION_SENTINEL)
        try:
            barriers = compute_barriers(
                structure,
                mobile_ion=compute_args.get("mobile_ion", "Li1+"),
                r_cut=compute_args.get("r_cut", 10.0),
                resolution=compute_args.get("resolution", 0.2),
                k=compute_args.get("k", 100),
                encut=compute_args.get("encut", 5.0),
                no_percolation_sentinel=sentinel,
            )
        except MobileIonAbsent:
            return float("nan")

        if dim == "min":
            finite = [v for v in barriers.values() if v < sentinel - 1e-9]
            # If every dimension is non-percolating, the min is the sentinel.
            return float(min(finite)) if finite else float(min(barriers.values()))
        return float(barriers[DIM_KEYS[dim]])

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        arr = np.array(
            [v if v is not None else np.nan for v in values], dtype=float
        )
        total = int(arr.size)
        with_ion_mask = ~np.isnan(arr)  # ran (real barrier or sentinel)
        ran = arr[with_ion_mask]
        n_with_ion = int(ran.size)

        sentinel = self.no_percolation_sentinel
        real = ran[ran < sentinel - 1e-9]  # finite, actually-percolating barriers
        n_percolating = int(real.size)

        metrics: dict[str, Any] = {
            "n_evaluated": total,
            "n_with_mobile_ion": n_with_ion,
            "n_percolating": n_percolating,
            "n_non_percolating": int(n_with_ion - n_percolating),
            "fraction_with_mobile_ion": (n_with_ion / total) if total else np.nan,
        }
        uncertainties: dict[str, dict[str, float]] = {}

        if n_with_ion > 0:
            n_fast = int(np.sum(real < self.fast_threshold))
            fraction_fast = n_fast / n_with_ion
            metrics["fraction_fast_ion_conductors"] = fraction_fast
            metrics["fraction_percolating"] = n_percolating / n_with_ion
            metrics["n_fast_ion_conductors"] = n_fast
            uncertainties["fraction_fast_ion_conductors"] = {
                "std": float(
                    np.sqrt(fraction_fast * (1 - fraction_fast) / n_with_ion)
                ),
                "sample_size": n_with_ion,
            }
        else:
            metrics["fraction_fast_ion_conductors"] = np.nan
            metrics["fraction_percolating"] = np.nan
            metrics["n_fast_ion_conductors"] = 0

        if n_percolating > 0:
            metrics["mean_barrier"] = float(np.mean(real))
            metrics["median_barrier"] = float(np.median(real))
            metrics["min_barrier"] = float(np.min(real))
            metrics["max_barrier"] = float(np.max(real))
            uncertainties["mean_barrier"] = {
                "std": float(np.std(real)),
                "std_error": float(np.std(real) / np.sqrt(n_percolating)),
            }
        else:
            metrics["mean_barrier"] = np.nan
            metrics["median_barrier"] = np.nan
            metrics["min_barrier"] = np.nan
            metrics["max_barrier"] = np.nan

        return {
            "metrics": metrics,
            "primary_metric": "fraction_fast_ion_conductors",
            "uncertainties": uncertainties,
        }
