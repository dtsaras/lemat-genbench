"""Band-gap benchmark.

Wraps :class:`~lemat_genbench.metrics.band_gap_metric.BandGapMetric`. By default
it runs the :class:`~lemat_genbench.preprocess.band_gap_preprocess.BandGapPreprocessor`
first (attaching ``band_gap`` to each structure) so a single CLI call does
prediction + aggregation. Set ``preprocess=False`` if structures already carry a
``band_gap`` property.
"""

from typing import Any, Dict, List

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.benchmarks.base import BaseBenchmark, BenchmarkResult
from lemat_genbench.evaluator import EvaluatorConfig
from lemat_genbench.metrics.band_gap_metric import BandGapMetric
from lemat_genbench.preprocess.band_gap_preprocess import BandGapPreprocessor
from lemat_genbench.utils.logging import logger


def _safe_num(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class BandGapBenchmark(BaseBenchmark):
    """Evaluate electronic band gaps of generated structures."""

    def __init__(
        self,
        backend: str = "hamgnn",
        backend_kwargs: Dict[str, Any] = None,
        preprocess: bool = True,
        metal_threshold: float = 0.1,
        insulator_threshold: float = 3.0,
        target_min: float | None = None,
        target_max: float | None = None,
        n_jobs: int = 1,
        timeout: int | None = None,
        name: str = "BandGapBenchmark",
        description: str = None,
        metadata: Dict[str, Any] = None,
    ):
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.preprocess = preprocess
        self.n_jobs = n_jobs

        metric = BandGapMetric(
            metal_threshold=metal_threshold,
            insulator_threshold=insulator_threshold,
            target_min=target_min,
            target_max=target_max,
            # When we preprocess, the metric reads properties; otherwise let it
            # compute on the fly so the benchmark still works on raw structures.
            compute_if_missing=not preprocess,
            backend=backend,
            backend_kwargs=self.backend_kwargs,
            n_jobs=n_jobs,
            timeout=timeout,
        )
        evaluator_configs = {
            "band_gap": EvaluatorConfig(
                name="Band Gap",
                description=f"Electronic band gap via the {backend} backend",
                metrics={"band_gap": metric},
                weights={"band_gap": 1.0},
                aggregation_method="weighted_mean",
            )
        }
        super().__init__(
            name=name,
            description=description
            or f"Evaluates electronic band gaps (backend: {backend}).",
            evaluator_configs=evaluator_configs,
            metadata={
                "version": "0.1.0",
                "category": "property",
                "backend": backend,
                "target_window": [target_min, target_max],
                **(metadata or {}),
            },
        )

    def evaluate(self, structures: List[Structure]) -> BenchmarkResult:
        """Optionally attach band gaps via the preprocessor, then aggregate."""
        if self.preprocess:
            logger.info(
                "BandGapBenchmark: predicting band gaps with %r backend...",
                self.backend,
            )
            result = BandGapPreprocessor(
                backend=self.backend,
                backend_kwargs=self.backend_kwargs,
                n_jobs=self.n_jobs,
            ).run(structures)
            structures = result.processed_structures
        return super().evaluate(structures)

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        final_scores: Dict[str, float] = {}
        res = evaluator_results.get("band_gap")
        if not res:
            return final_scores
        final_scores["band_gap_primary"] = _safe_num(res.get("combined_value"))
        metric_result = res.get("metric_results", {}).get("band_gap")
        if metric_result:
            for key, val in metric_result.metrics.items():
                final_scores[key] = _safe_num(val)
        return final_scores
