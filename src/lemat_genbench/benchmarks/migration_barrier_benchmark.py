"""Ion migration-barrier benchmark (BVSE percolation barriers via bvlain).

Wraps :class:`~lemat_genbench.metrics.migration_barrier_metric.MigrationBarrierMetric`.
The headline score is the fraction of generated structures that are fast ion
conductors at the requested dimensionality; barrier statistics over the
conducting subset are reported alongside.
"""

from typing import Any, Dict

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluatorConfig
from lemat_genbench.metrics.migration_barrier_metric import MigrationBarrierMetric


def _safe_num(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class MigrationBarrierBenchmark(BaseBenchmark):
    """Evaluate ionic migration barriers of generated structures."""

    def __init__(
        self,
        mobile_ion: str = "Li1+",
        dimensionality: str = "3d",
        r_cut: float = 10.0,
        resolution: float = 0.2,
        k: int = 100,
        encut: float = 5.0,
        fast_threshold: float = 0.6,
        n_jobs: int = 1,
        timeout: int = 30,
        name: str = "MigrationBarrierBenchmark",
        description: str = None,
        metadata: Dict[str, Any] = None,
    ):
        metric = MigrationBarrierMetric(
            mobile_ion=mobile_ion,
            dimensionality=dimensionality,
            r_cut=r_cut,
            resolution=resolution,
            k=k,
            encut=encut,
            fast_threshold=fast_threshold,
            n_jobs=n_jobs,
            timeout=timeout,
        )
        evaluator_configs = {
            "migration_barrier": EvaluatorConfig(
                name="Ion Migration Barrier",
                description=f"BVSE {dimensionality} {mobile_ion} migration barrier",
                metrics={"migration_barrier": metric},
                weights={"migration_barrier": 1.0},
                aggregation_method="weighted_mean",
            )
        }
        super().__init__(
            name=name,
            description=description
            or f"Evaluates {dimensionality} {mobile_ion} migration barriers (BVSE/bvlain).",
            evaluator_configs=evaluator_configs,
            metadata={
                "version": "0.1.0",
                "category": "property",
                "mobile_ion": mobile_ion,
                "dimensionality": dimensionality,
                "fast_threshold": fast_threshold,
                **(metadata or {}),
            },
        )

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        final_scores: Dict[str, float] = {}
        res = evaluator_results.get("migration_barrier")
        if not res:
            return final_scores
        final_scores["fraction_fast_ion_conductors"] = _safe_num(
            res.get("combined_value")
        )
        metric_result = res.get("metric_results", {}).get("migration_barrier")
        if metric_result:
            for key, val in metric_result.metrics.items():
                final_scores[key] = _safe_num(val)
        return final_scores
