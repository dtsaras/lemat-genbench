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
from lemat_genbench.metrics.esw_metric import ESWMetric
from lemat_genbench.metrics.migration_barrier_metric import MigrationBarrierMetric
from lemat_genbench.preprocess.band_gap_preprocess import BandGapPreprocessor
from lemat_genbench.preprocess.esw_preprocess import ESWPreprocessor
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
        include_esw: bool = False,
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
        # ESW options
        esw_preprocess: bool = True,
        esw_cache_dir: str = "data/esw_cache",
        esw_api_key: str | None = None,
        esw_mace_model: str | None = None,
        esw_device: str | None = "auto",
        esw_default_dtype: str = "float64",
        esw_fmax: float = 0.05,
        esw_max_steps: int = 500,
        esw_mu_min: float = -5.0,
        esw_mu_max: float = 0.0,
        esw_mu_step: float = 0.01,
        esw_gpd_stability_tol: float = 1e-4,
        esw_refresh_relax: bool = False,
        esw_refresh_mp: bool = False,
        esw_target_min: float | None = None,
        esw_target_max: float | None = None,
        esw_timeout: int | None = None,
        n_jobs: int = 1,
        name: str = "PropertyBenchmark",
        description: str = None,
        metadata: Dict[str, Any] = None,
    ):
        if not (include_band_gap or include_migration_barrier or include_esw):
            raise ValueError("Enable at least one property.")

        self.include_band_gap = include_band_gap
        self.include_esw = include_esw
        self.band_gap_backend = band_gap_backend
        self.band_gap_backend_kwargs = band_gap_backend_kwargs or {}
        self.band_gap_preprocess = band_gap_preprocess
        self.esw_preprocess = esw_preprocess
        self.esw_kwargs = {
            "cache_dir": esw_cache_dir,
            "api_key": esw_api_key,
            "mace_model": esw_mace_model,
            "device": esw_device,
            "default_dtype": esw_default_dtype,
            "fmax": esw_fmax,
            "max_steps": esw_max_steps,
            "mu_min": esw_mu_min,
            "mu_max": esw_mu_max,
            "mu_step": esw_mu_step,
            "gpd_stability_tol": esw_gpd_stability_tol,
            "refresh_relax": esw_refresh_relax,
            "refresh_mp": esw_refresh_mp,
        }
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
        if include_esw:
            evaluator_configs["esw"] = EvaluatorConfig(
                name="Electrochemical Stability Window",
                description="Li-exchange-only ESW",
                metrics={
                    "esw": ESWMetric(
                        target_min=esw_target_min,
                        target_max=esw_target_max,
                        compute_if_missing=not esw_preprocess,
                        esw_kwargs=self.esw_kwargs,
                        n_jobs=n_jobs,
                        timeout=esw_timeout,
                    )
                },
                weights={"esw": 1.0},
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
                    *(["esw"] if include_esw else []),
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
        if self.include_esw and self.esw_preprocess:
            logger.info("PropertyBenchmark: computing Li-exchange-only ESW...")
            structures = ESWPreprocessor(
                n_jobs=self.n_jobs,
                **self.esw_kwargs,
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
