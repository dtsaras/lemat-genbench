"""ESW preprocessor: attaches Li-exchange-only ESW properties to structures."""

from dataclasses import dataclass
from typing import Any

from pymatgen.core import Structure

from lemat_genbench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lemat_genbench.properties.esw import compute_esw


@dataclass
class ESWPreprocessorConfig(PreprocessorConfig):
    """Configuration for :class:`ESWPreprocessor`."""

    cache_dir: str = "data/esw_cache"
    api_key: str | None = None
    mace_model: str | None = None
    device: str | None = "auto"
    default_dtype: str = "float64"
    fmax: float = 0.05
    max_steps: int = 500
    mu_min: float = -5.0
    mu_max: float = 0.0
    mu_step: float = 0.01
    gpd_stability_tol: float = 1e-4
    refresh_relax: bool = False
    refresh_mp: bool = False


class ESWPreprocessor(BasePreprocessor):
    """Attach Li-exchange-only ESW results to each structure's properties."""

    def __init__(
        self,
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
        name: str = None,
        description: str = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "ESWPreprocessor",
            description=description or "Computes Li-exchange-only ESW",
            n_jobs=n_jobs,
        )
        self.config = ESWPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            cache_dir=cache_dir,
            api_key=api_key,
            mace_model=mace_model,
            device=device,
            default_dtype=default_dtype,
            fmax=fmax,
            max_steps=max_steps,
            mu_min=mu_min,
            mu_max=mu_max,
            mu_step=mu_step,
            gpd_stability_tol=gpd_stability_tol,
            refresh_relax=refresh_relax,
            refresh_mp=refresh_mp,
        )

    def _get_process_attributes(self) -> dict[str, Any]:
        return {
            "cache_dir": self.config.cache_dir,
            "api_key": self.config.api_key,
            "mace_model": self.config.mace_model,
            "device": self.config.device,
            "default_dtype": self.config.default_dtype,
            "fmax": self.config.fmax,
            "max_steps": self.config.max_steps,
            "mu_min": self.config.mu_min,
            "mu_max": self.config.mu_max,
            "mu_step": self.config.mu_step,
            "gpd_stability_tol": self.config.gpd_stability_tol,
            "refresh_relax": self.config.refresh_relax,
            "refresh_mp": self.config.refresh_mp,
        }

    @staticmethod
    def process_structure(
        structure: Structure,
        **process_args: Any,
    ) -> Structure:
        result = compute_esw(structure, **process_args)
        structure.properties["esw"] = result.esw
        structure.properties["reduction_potential"] = result.reduction_potential
        structure.properties["oxidation_potential"] = result.oxidation_potential
        structure.properties["ehull"] = result.ehull
        structure.properties["esw_error"] = result.error
        structure.properties["esw_used_relaxed_structure"] = (
            result.used_relaxed_structure
        )
        structure.properties["esw_mace_energy"] = result.mace_energy
        structure.properties["esw_relaxed_structure_path"] = result.optimized_cif_path
        structure.properties["esw_stable_mu_min"] = result.stable_mu_min
        structure.properties["esw_stable_mu_max"] = result.stable_mu_max
        return structure
