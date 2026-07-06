"""Li-exchange-only electrochemical stability window metric."""

from typing import Any

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.metrics.base import BaseMetric
from lemat_genbench.properties.esw import compute_esw


class ESWMetric(BaseMetric):
    """Electrochemical stability window metric (eV)."""

    def __init__(
        self,
        target_min: float | None = None,
        target_max: float | None = None,
        compute_if_missing: bool = False,
        esw_kwargs: dict[str, Any] | None = None,
        name: str = None,
        description: str = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
        timeout: int | None = None,
    ):
        super().__init__(
            name=name or "ESW",
            description=description or "Li-exchange-only ESW (eV)",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
            timeout=timeout,
        )
        self.target_min = target_min
        self.target_max = target_max
        self.compute_if_missing = compute_if_missing
        self.esw_kwargs = esw_kwargs or {}

    def _get_compute_attributes(self) -> dict[str, Any]:
        attrs = super()._get_compute_attributes()
        attrs.update(
            {
                "target_min": self.target_min,
                "target_max": self.target_max,
                "compute_if_missing": self.compute_if_missing,
                "esw_kwargs": self.esw_kwargs,
            }
        )
        return attrs

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        value = structure.properties.get("esw")
        if value is None and compute_args.get("compute_if_missing", False):
            result = compute_esw(structure, **compute_args.get("esw_kwargs", {}))
            value = result.esw
        if value is None:
            return float("nan")
        return float(value)

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        arr = np.array(
            [v if v is not None else np.nan for v in values], dtype=float
        )
        valid = arr[~np.isnan(arr)]
        total = int(arr.size)
        n_valid = int(valid.size)
        metrics: dict[str, Any] = {
            "n_evaluated": total,
            "n_valid": n_valid,
            "fraction_esw_valid": (n_valid / total) if total else np.nan,
        }
        uncertainties: dict[str, dict[str, float]] = {}

        if n_valid == 0:
            metrics.update(
                {
                    "mean_esw": np.nan,
                    "median_esw": np.nan,
                    "min_esw": np.nan,
                    "max_esw": np.nan,
                }
            )
            return {
                "metrics": metrics,
                "primary_metric": "mean_esw",
                "uncertainties": uncertainties,
            }

        metrics.update(
            {
                "mean_esw": float(np.mean(valid)),
                "median_esw": float(np.median(valid)),
                "min_esw": float(np.min(valid)),
                "max_esw": float(np.max(valid)),
            }
        )
        uncertainties["mean_esw"] = {
            "std": float(np.std(valid)),
            "std_error": float(np.std(valid) / np.sqrt(n_valid)),
        }

        primary = "mean_esw"
        if self.target_min is not None and self.target_max is not None:
            in_window = (valid >= self.target_min) & (valid <= self.target_max)
            frac = float(np.mean(in_window))
            metrics["fraction_in_target_window"] = frac
            uncertainties["fraction_in_target_window"] = {
                "std": float(np.sqrt(frac * (1 - frac) / n_valid)),
                "sample_size": n_valid,
            }
            primary = "fraction_in_target_window"

        return {
            "metrics": metrics,
            "primary_metric": primary,
            "uncertainties": uncertainties,
        }
