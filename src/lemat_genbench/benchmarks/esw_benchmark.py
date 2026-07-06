"""Li-exchange-only electrochemical stability window benchmark."""

from typing import Any, Dict, List

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.benchmarks.base import BaseBenchmark, BenchmarkResult
from lemat_genbench.evaluator import EvaluatorConfig
from lemat_genbench.metrics.esw_metric import ESWMetric
from lemat_genbench.preprocess.esw_preprocess import ESWPreprocessor
from lemat_genbench.utils.logging import logger


def _safe_num(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class ESWBenchmark(BaseBenchmark):
    """Evaluate Li-exchange-only ESW of generated structures."""

    def __init__(
        self,
        preprocess: bool = True,
        cache_dir: str = "data/esw_cache",
        api_key: str | None = None,
        mace_model: str | None = None,
        device: str | None = "auto",
        default_dtype: str = "float64",
        fmax: float = 0.05,
        max_steps: int = 500,
        mu_min: float = -5.0,
        mu_max: float = 0.0,
        mu_step: float = 0.01,
        gpd_stability_tol: float = 1e-4,
        refresh_relax: bool = False,
        refresh_mp: bool = False,
        target_min: float | None = None,
        target_max: float | None = None,
        n_jobs: int = 1,
        timeout: int | None = None,
        name: str = "ESWBenchmark",
        description: str = None,
        metadata: Dict[str, Any] = None,
    ):
        self.preprocess = preprocess
        self.n_jobs = n_jobs
        self.esw_kwargs = {
            "cache_dir": cache_dir,
            "api_key": api_key,
            "mace_model": mace_model,
            "device": device,
            "default_dtype": default_dtype,
            "fmax": fmax,
            "max_steps": max_steps,
            "mu_min": mu_min,
            "mu_max": mu_max,
            "mu_step": mu_step,
            "gpd_stability_tol": gpd_stability_tol,
            "refresh_relax": refresh_relax,
            "refresh_mp": refresh_mp,
        }

        metric = ESWMetric(
            target_min=target_min,
            target_max=target_max,
            compute_if_missing=not preprocess,
            esw_kwargs=self.esw_kwargs,
            n_jobs=n_jobs,
            timeout=timeout,
        )
        evaluator_configs = {
            "esw": EvaluatorConfig(
                name="Electrochemical Stability Window",
                description="Li-exchange-only ESW",
                metrics={"esw": metric},
                weights={"esw": 1.0},
                aggregation_method="weighted_mean",
            )
        }
        super().__init__(
            name=name,
            description=description or "Evaluates Li-exchange-only ESW.",
            evaluator_configs=evaluator_configs,
            metadata={
                "version": "0.1.0",
                "category": "property",
                "target_window": [target_min, target_max],
                **(metadata or {}),
            },
        )

    def evaluate(self, structures: List[Structure]) -> BenchmarkResult:
        if self.preprocess:
            logger.info("ESWBenchmark: computing Li-exchange-only ESW...")
            structures = ESWPreprocessor(
                n_jobs=self.n_jobs,
                **self.esw_kwargs,
            ).run(structures).processed_structures
        return super().evaluate(structures)

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        final_scores: Dict[str, float] = {}
        res = evaluator_results.get("esw")
        if not res:
            return final_scores
        final_scores["esw_primary"] = _safe_num(res.get("combined_value"))
        metric_result = res.get("metric_results", {}).get("esw")
        if metric_result:
            for key, val in metric_result.metrics.items():
                final_scores[key] = _safe_num(val)
        return final_scores
