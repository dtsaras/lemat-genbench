"""Combined functional-property benchmark.

Runs the band-gap and ion-migration-barrier metrics together in a single
evaluation, so a generated set can be scored on multiple functional properties
at once. Band gaps are attached via the (pluggable) band-gap preprocessor first;
migration barriers compute in-metric (BVlain). Either property can be disabled.

This is a thin convenience layer over the individual
:class:`~lemat_genbench.benchmarks.band_gap_benchmark.BandGapBenchmark` and
:class:`~lemat_genbench.benchmarks.migration_barrier_benchmark.MigrationBarrierBenchmark`.
"""

from typing import Any, Dict, List

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.benchmarks.base import BaseBenchmark, BenchmarkResult
from lemat_genbench.evaluator import EvaluatorConfig
from lemat_genbench.metrics.band_gap_metric import BandGapMetric
from lemat_genbench.metrics.migration_barrier_metric import MigrationBarrierMetric
from lemat_genbench.preprocess.band_gap_preprocess import BandGapPreprocessor
from lemat_genbench.utils.logging import logger


def _safe_num(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class PropertyBenchmark(BaseBenchmark):
    """Evaluate multiple functional properties (band gap + migration barrier)."""

    def __init__(
        self,
        include_band_gap: bool = True,
        include_migration_barrier: bool = True,
        # band-gap options
        band_gap_backend: str = "hamgnn",
        band_gap_backend_kwargs: Dict[str, Any] = None,
        band_gap_preprocess: bool = True,
        metal_threshold: float = 0.1,
        insulator_threshold: float = 3.0,
        target_min: float | None = None,
        target_max: float | None = None,
        band_gap_timeout: int | None = None,
        # migration-barrier options
        mobile_ion: str = "Li1+",
        dimensionality: str = "3d",
        fast_threshold: float = 0.6,
        migration_timeout: int = 30,
        n_jobs: int = 1,
        name: str = "PropertyBenchmark",
        description: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not (include_band_gap or include_migration_barrier):
            raise ValueError("Enable at least one property.")

        self.include_band_gap = include_band_gap
        self.band_gap_backend = band_gap_backend
        self.band_gap_backend_kwargs = band_gap_backend_kwargs or {}
        self.band_gap_preprocess = band_gap_preprocess
        self.n_jobs = n_jobs

        evaluator_configs: Dict[str, EvaluatorConfig] = {}
        if include_band_gap:
            evaluator_configs["band_gap"] = EvaluatorConfig(
                name="Band Gap",
                description=f"Electronic band gap via the {band_gap_backend} backend",
                metrics={
                    "band_gap": BandGapMetric(
                        metal_threshold=metal_threshold,
                        insulator_threshold=insulator_threshold,
                        target_min=target_min,
                        target_max=target_max,
                        compute_if_missing=not band_gap_preprocess,
                        backend=band_gap_backend,
                        backend_kwargs=self.band_gap_backend_kwargs,
                        n_jobs=n_jobs,
                        timeout=band_gap_timeout,
                    )
                },
                weights={"band_gap": 1.0},
                aggregation_method="weighted_mean",
            )
        if include_migration_barrier:
            evaluator_configs["migration_barrier"] = EvaluatorConfig(
                name="Ion Migration Barrier",
                description=f"BVSE {dimensionality} {mobile_ion} migration barrier",
                metrics={
                    "migration_barrier": MigrationBarrierMetric(
                        mobile_ion=mobile_ion,
                        dimensionality=dimensionality,
                        fast_threshold=fast_threshold,
                        n_jobs=n_jobs,
                        timeout=migration_timeout,
                    )
                },
                weights={"migration_barrier": 1.0},
                aggregation_method="weighted_mean",
            )

        super().__init__(
            name=name,
            description=description or "Combined functional-property benchmark.",
            evaluator_configs=evaluator_configs,
            metadata={
                "version": "0.1.0",
                "category": "property",
                "properties": [
                    *(["band_gap"] if include_band_gap else []),
                    *(["migration_barrier"] if include_migration_barrier else []),
                ],
                **(metadata or {}),
            },
        )

    def evaluate(self, structures: List[Structure]) -> BenchmarkResult:
        if self.include_band_gap and self.band_gap_preprocess:
            logger.info(
                "PropertyBenchmark: predicting band gaps with %r backend...",
                self.band_gap_backend,
            )
            structures = BandGapPreprocessor(
                backend=self.band_gap_backend,
                backend_kwargs=self.band_gap_backend_kwargs,
                n_jobs=self.n_jobs,
            ).run(structures).processed_structures
        return super().evaluate(structures)

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        final_scores: Dict[str, float] = {}
        for prop, result in evaluator_results.items():
            if not result:
                continue
            final_scores[f"{prop}_primary"] = _safe_num(result.get("combined_value"))
            metric_result = result.get("metric_results", {}).get(prop)
            if metric_result:
                for key, val in metric_result.metrics.items():
                    final_scores[f"{prop}_{key}"] = _safe_num(val)
        return final_scores
