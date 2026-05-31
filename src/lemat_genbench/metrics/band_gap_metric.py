"""Band-gap metric.

Per structure: the electronic band gap (eV). By default the metric *reads* a
``band_gap`` value attached to ``structure.properties`` by
:class:`~lemat_genbench.preprocess.band_gap_preprocess.BandGapPreprocessor`
(the idiomatic "preprocess -> metric reads properties" pattern used by the
stability metric). For convenience it can also compute on the fly via a backend
when ``compute_if_missing=True``.

Aggregate: distribution statistics plus metallic / semiconductor / insulator
fractions, and an optional "fraction in a target gap window" — useful when a
generative campaign targets, say, photovoltaic-range gaps.
"""

from typing import Any

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.metrics.base import BaseMetric

# Process-local backend cache for the optional on-the-fly path.
_PROCESS_BACKEND_CACHE: dict = {}


def _get_or_create_backend(backend_name: str, backend_kwargs: dict):
    from lemat_genbench.properties.band_gap_backends import get_band_gap_backend

    key = f"{backend_name}_{hash(tuple(sorted(backend_kwargs.items())))}"
    if key not in _PROCESS_BACKEND_CACHE:
        _PROCESS_BACKEND_CACHE[key] = get_band_gap_backend(backend_name, **backend_kwargs)
    return _PROCESS_BACKEND_CACHE[key]


class BandGapMetric(BaseMetric):
    """Electronic band-gap metric (eV).

    Parameters
    ----------
    metal_threshold : float, default=0.1
        Gap <= this (eV) counts as metallic.
    insulator_threshold : float, default=3.0
        Gap > this (eV) counts as an insulator; in between is a semiconductor.
    target_min, target_max : float, optional
        If both set, report ``fraction_in_target_window`` (gaps within
        [target_min, target_max] eV) and make it the primary metric.
    compute_if_missing : bool, default=False
        If ``structure.properties["band_gap"]`` is absent, compute it on the fly
        via ``backend`` instead of failing. Default is to require preprocessing.
    backend, backend_kwargs
        Backend used only for the on-the-fly path.
    """

    def __init__(
        self,
        metal_threshold: float = 0.1,
        insulator_threshold: float = 3.0,
        target_min: float | None = None,
        target_max: float | None = None,
        compute_if_missing: bool = False,
        backend: str = "hamgnn",
        backend_kwargs: dict | None = None,
        name: str = None,
        description: str = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
        timeout: int | None = None,
    ):
        super().__init__(
            name=name or "BandGap",
            description=description or "Electronic band gap (eV)",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
            timeout=timeout,
        )
        self.metal_threshold = metal_threshold
        self.insulator_threshold = insulator_threshold
        self.target_min = target_min
        self.target_max = target_max
        self.compute_if_missing = compute_if_missing
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}

    def _get_compute_attributes(self) -> dict[str, Any]:
        attrs = super()._get_compute_attributes()
        attrs.update(
            {
                "compute_if_missing": self.compute_if_missing,
                "backend": self.backend,
                "backend_kwargs": self.backend_kwargs,
            }
        )
        return attrs

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Return the band gap (eV) for one structure.

        Reads ``structure.properties["band_gap"]``; if absent and
        ``compute_if_missing`` is set, computes it via the backend. Raises if no
        value can be obtained (-> BaseMetric records it in ``failed_indices``).
        """
        value = structure.properties.get("band_gap")
        if value is None and compute_args.get("compute_if_missing", False):
            backend = _get_or_create_backend(
                compute_args.get("backend", "hamgnn"),
                compute_args.get("backend_kwargs", {}),
            )
            value = backend.predict(structure)
        if value is None:
            raise ValueError(
                "No 'band_gap' in structure.properties. Run BandGapPreprocessor "
                "first, or set compute_if_missing=True."
            )
        return float(value)

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        arr = np.array(
            [v if v is not None else np.nan for v in values], dtype=float
        )
        valid = arr[~np.isnan(arr)]
        total = int(arr.size)

        metrics: dict[str, Any] = {
            "n_evaluated": total,
            "n_valid": int(valid.size),
        }
        uncertainties: dict[str, dict[str, float]] = {}

        if valid.size == 0:
            metrics["mean_band_gap"] = np.nan
            return {
                "metrics": metrics,
                "primary_metric": "mean_band_gap",
                "uncertainties": uncertainties,
            }

        n = valid.size
        metrics.update(
            {
                "mean_band_gap": float(np.mean(valid)),
                "std_band_gap": float(np.std(valid)),
                "min_band_gap": float(np.min(valid)),
                "max_band_gap": float(np.max(valid)),
                "metallic_ratio": float(np.mean(valid <= self.metal_threshold)),
                "semiconductor_ratio": float(
                    np.mean(
                        (valid > self.metal_threshold)
                        & (valid <= self.insulator_threshold)
                    )
                ),
                "insulator_ratio": float(np.mean(valid > self.insulator_threshold)),
            }
        )
        uncertainties["mean_band_gap"] = {
            "std": float(np.std(valid)),
            "std_error": float(np.std(valid) / np.sqrt(n)),
        }

        primary = "mean_band_gap"
        if self.target_min is not None and self.target_max is not None:
            in_window = (valid >= self.target_min) & (valid <= self.target_max)
            frac = float(np.mean(in_window))
            metrics["fraction_in_target_window"] = frac
            uncertainties["fraction_in_target_window"] = {
                "std": float(np.sqrt(frac * (1 - frac) / n)),
                "sample_size": n,
            }
            primary = "fraction_in_target_window"

        return {
            "metrics": metrics,
            "primary_metric": primary,
            "uncertainties": uncertainties,
        }
