"""Band-gap preprocessor: attaches a predicted band gap to ``structure.properties``.

Mirrors the multi-MLIP stability preprocessor: the (potentially expensive)
band-gap backend is loaded **once per worker process** via a module-level cache,
and the predicted value is written to ``structure.properties["band_gap"]`` so the
cheap :class:`~lemat_genbench.metrics.band_gap_metric.BandGapMetric` can just read
it. The backend is pluggable (``hamgnn`` / ``alignn`` / ...); see
:mod:`lemat_genbench.properties.band_gap_backends`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from pymatgen.core import Structure

from lemat_genbench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lemat_genbench.properties.band_gap_backends import get_band_gap_backend
from lemat_genbench.utils.logging import logger

# Process-local backend cache (each worker process loads its own backend once).
_PROCESS_BACKEND_CACHE: dict = {}


def _get_or_create_backend(backend_name: str, backend_kwargs: Dict[str, Any]):
    """Get the band-gap backend from the process cache, creating it on first use."""
    cache_key = f"{backend_name}_{hash(tuple(sorted(backend_kwargs.items())))}"
    if cache_key not in _PROCESS_BACKEND_CACHE:
        logger.info("Loading band-gap backend %r in worker process...", backend_name)
        _PROCESS_BACKEND_CACHE[cache_key] = get_band_gap_backend(
            backend_name, **backend_kwargs
        )
    return _PROCESS_BACKEND_CACHE[cache_key]


@dataclass
class BandGapPreprocessorConfig(PreprocessorConfig):
    """Configuration for :class:`BandGapPreprocessor`."""

    backend: str = "hamgnn"
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)


class BandGapPreprocessor(BasePreprocessor):
    """Attach a predicted band gap (eV) to each structure's ``properties``.

    Parameters
    ----------
    backend : str, default="hamgnn"
        Band-gap backend name (see
        :func:`lemat_genbench.properties.band_gap_backends.available_backends`).
        ``"hamgnn"`` is the accuracy-first default but requires its environment
        to be configured; ``"alignn"`` works immediately given its checkpoint.
    backend_kwargs : dict, optional
        Constructor kwargs forwarded to the backend.
    n_jobs : int, default=1
        Parallel worker processes.
    """

    def __init__(
        self,
        backend: str = "hamgnn",
        backend_kwargs: Dict[str, Any] = None,
        name: str = None,
        description: str = None,
        n_jobs: int = 1,
    ):
        backend_kwargs = backend_kwargs or {}
        super().__init__(
            name=name or f"BandGapPreprocessor_{backend}",
            description=description or f"Predicts band gap via the {backend} backend",
            n_jobs=n_jobs,
        )
        self.config = BandGapPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )

    def _get_process_attributes(self) -> Dict[str, Any]:
        return {
            "backend_name": self.config.backend,
            "backend_kwargs": self.config.backend_kwargs,
        }

    @staticmethod
    def process_structure(
        structure: Structure,
        backend_name: str = "hamgnn",
        backend_kwargs: Dict[str, Any] = None,
    ) -> Structure:
        """Predict the band gap and attach it to ``structure.properties``.

        ``band_gap`` is set to the predicted value (eV) or ``None`` if the
        backend could not score this structure (the metric then treats it as
        NaN). A backend *configuration* error (e.g. HamGNN not set up) is allowed
        to propagate so the whole run fails loudly rather than silently marking
        every structure as failed.
        """
        backend = _get_or_create_backend(backend_name, backend_kwargs or {})
        value = backend.predict(structure)
        structure.properties["band_gap"] = (
            float(value) if value is not None else None
        )
        structure.properties["band_gap_backend"] = backend.name
        return structure
