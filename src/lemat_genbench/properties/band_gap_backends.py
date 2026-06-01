"""Pluggable band-gap backends.

A *backend* maps a pymatgen :class:`~pymatgen.core.Structure` to an electronic
band gap in eV. The band-gap metric / preprocessor depend only on the
:class:`BandGapBackend` interface, so new predictors drop in via the
``get_band_gap_backend`` registry without touching the metric.

Backends shipped here
---------------------
``hamgnn`` — :class:`HamGNNBandGapBackend`
    HamGNN (https://github.com/QuantumLab-ZY/HamGNN) predicts the *ab-initio*
    (PBE-level) tight-binding Hamiltonian; a band gap is then obtained by
    diagonalising ``H(k)``/``S(k)``. This is the physically-grounded option
    (it yields the full electronic structure, not just a scalar), but it is
    **not** a structure->gap model: every structure must first pass through an
    OpenMX *non-self-consistent* preprocessing step to produce the overlap
    matrix, and HamGNN's pinned dependencies (pytorch-lightning 1.5.x /
    torch 2.5) cannot share lemat-genbench's env. The backend therefore
    orchestrates the documented HamGNN CLI pipeline via **subprocess** into a
    separate, pinned HamGNN environment — the metric stays in-process while the
    conflicting deps stay quarantined. See :class:`HamGNNBandGapBackend` and
    ``scripts/hamgnn/`` for environment bring-up.

``alignn`` — :class:`AlignnBandGapBackend`
    JARVIS ALIGNN ``jv_mbj_bandgap`` checkpoint: a fast structure->property GNN
    (~ms/structure, no DFT, broad chemistry). Targets MBJ gaps (closer to
    experiment than PBE). Useful as an immediately-usable metric and as a
    baseline to compare HamGNN against. Ported from the author's
    ``matter_evolve`` ALIGNN predictor.

``hamgnn`` is the default to honour the project's "accuracy-first" choice, but
it raises an actionable :class:`HamGNNNotConfigured` until the environment is
stood up; ``alignn`` works as soon as its checkpoint + deps are present.
"""

from __future__ import annotations

import abc
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface + registry
# ---------------------------------------------------------------------------


class BandGapBackend(abc.ABC):
    """Abstract band-gap predictor: ``Structure -> band gap (eV) | None``."""

    #: Short registry name (e.g. ``"hamgnn"``, ``"alignn"``).
    name: str = "base"

    @abc.abstractmethod
    def predict(self, structure: Any) -> float | None:
        """Return the band gap (eV) for ``structure``, or ``None`` on failure."""

    def predict_many(self, structures: list[Any]) -> list[float | None]:
        """Predict for a list of structures (override for batched backends)."""
        return [self.predict(s) for s in structures]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's optional dependencies / binaries are present."""
        return True


_BACKENDS: dict[str, type[BandGapBackend]] = {}


def register_backend(name: str, cls: type[BandGapBackend]) -> None:
    """Register a backend class under ``name`` (lower-cased)."""
    _BACKENDS[name.lower()] = cls


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_band_gap_backend(name: str = "hamgnn", **kwargs: Any) -> BandGapBackend:
    """Instantiate the backend registered under ``name``.

    Parameters
    ----------
    name : str
        One of :func:`available_backends` (default ``"hamgnn"``).
    **kwargs
        Forwarded to the backend constructor.
    """
    key = name.lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"Unknown band-gap backend {name!r}. Available: {available_backends()}"
        )
    return _BACKENDS[key](**kwargs)


# ---------------------------------------------------------------------------
# ALIGNN backend (fast, works today) — ported from matter_evolve
# ---------------------------------------------------------------------------


_ALIGNN_LOAD_LOCK = threading.Lock()

# Candidate locations for the ALIGNN ``*_alignn.zip`` checkpoints, in priority
# order. Override with $ALIGNN_MODELS_DIR.
_ALIGNN_DEFAULT_DIRS = (
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "third_party",
        "alignn_models",
        "alignn",
    ),
    # The author's matter_evolve mirror (the user pointed here for weights).
    os.path.expanduser(
        "~/projects/crystal_gen/matter_evolve/third_party/alignn_models/alignn"
    ),
)


def _resolve_alignn_models_dir(models_dir: str | None) -> str | None:
    if models_dir:
        return models_dir
    env = os.environ.get("ALIGNN_MODELS_DIR")
    if env:
        return env
    for cand in _ALIGNN_DEFAULT_DIRS:
        if os.path.isdir(cand):
            return cand
    return None


class AlignnBandGapBackend(BandGapBackend):
    """In-process ALIGNN band-gap predictor (JARVIS ``jv_mbj_bandgap``).

    Wraps ``alignn.pretrained.get_prediction`` with a jarvis ``Atoms``
    conversion. The first call for a model copies the local ``<model>.zip``
    checkpoint into the alignn package dir (where ``get_prediction`` looks).
    Requires the optional ``alignn`` + ``dgl`` + ``jarvis-tools`` extra.
    """

    name = "alignn"

    def __init__(
        self,
        model_name: str = "jv_mbj_bandgap_alignn",
        models_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.models_dir = _resolve_alignn_models_dir(models_dir)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import alignn  # noqa: F401
            import dgl  # noqa: F401
            import jarvis  # noqa: F401
        except Exception:
            return False
        return True

    def _ensure_pkg_zip(self) -> None:
        import alignn

        if not self.models_dir:
            raise FileNotFoundError(
                "No ALIGNN models dir found. Set $ALIGNN_MODELS_DIR to the dir "
                "containing the *_alignn.zip checkpoints (e.g. "
                "jv_mbj_bandgap_alignn.zip)."
            )
        src = os.path.join(self.models_dir, f"{self.model_name}.zip")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"ALIGNN checkpoint not found: {src!r}")
        dst = os.path.join(os.path.dirname(alignn.__file__), f"{self.model_name}.zip")
        if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copyfile(src, dst)

    @staticmethod
    def _patch_torch_load_once() -> None:
        """ALIGNN checkpoints predate torch 2.6's safe loader; trust local zips."""
        import torch

        if getattr(torch.load, "__alignn_patched__", False):
            return
        _orig = torch.load

        def _wrapped(*args: Any, **kwargs: Any):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)

        _wrapped.__alignn_patched__ = True  # type: ignore[attr-defined]
        torch.load = _wrapped  # type: ignore[assignment]

    def predict(self, structure: Any) -> float | None:
        with _ALIGNN_LOAD_LOCK:
            try:
                self._ensure_pkg_zip()
                self._patch_torch_load_once()
                from alignn.pretrained import get_prediction
                from jarvis.core.atoms import Atoms

                atoms = Atoms(
                    lattice_mat=structure.lattice.matrix,
                    elements=[str(s.specie) for s in structure.sites],
                    coords=[list(s.frac_coords) for s in structure.sites],
                    cartesian=False,
                )
                pred = get_prediction(model_name=self.model_name, atoms=atoms)
            except Exception as exc:
                logger.warning("ALIGNN %s predict failed: %s", self.model_name, exc)
                return None
        if isinstance(pred, (list, tuple)):
            pred = pred[0] if pred else None
        try:
            return float(pred)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# HamGNN backend (accurate, subprocess-orchestrated)
# ---------------------------------------------------------------------------


class HamGNNNotConfigured(RuntimeError):
    """Raised when the HamGNN environment / binaries / checkpoint are missing."""


# OpenMX `.dat` template. {data_path} = DFT_DATA dir; {soc_block} toggles SOC.
# Mirrors HamGNN/DFT_interfaces/openmx/poscar2openmx.yaml (non-SCF: openmx_postprocess
# only reads geometry/basis to build S + H0; SCF knobs are unused for prediction).
_OPENMX_BASIC_COMMAND = """\
System.CurrrentDirectory         ./
System.Name                     openmx
DATA.PATH           {data_path}
level.of.stdout                   1
level.of.fileout                  1
HS.fileout                   on

scf.XcType                  GGA-PBE
{soc_block}
scf.partialCoreCorrection   on
scf.ElectronicTemperature  100.0
scf.energycutoff           {energycutoff}
scf.maxIter                 300
scf.EigenvalueSolver        Band
scf.Kgrid                  {kgrid}
scf.Mixing.Type           rmm-diis
scf.criterion             1.0e-7

MD.Type                      Nomd
"""

_SOC_ON = "scf.SpinPolarization        nc\nscf.SpinOrbit.Coupling      on"
_SOC_OFF = "scf.SpinPolarization        off\nscf.SpinOrbit.Coupling      off"

# band_cal prints e.g. ``band gap = 1.234 eV`` (DFT_interfaces/openmx/band_cal.py).
_BAND_GAP_RE = re.compile(r"band\s*gap\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*eV")


@dataclass
class HamGNNConfig:
    """Paths / parameters for the HamGNN subprocess pipeline.

    Every path defaults from an environment variable so the backend can be
    configured without code changes (see ``scripts/hamgnn/``). All must point
    at a stood-up HamGNN environment.
    """

    # Directory containing the HamGNN env console scripts + python
    # (poscar2openmx, graph_data_gen, band_cal, python). Typically
    # <conda>/envs/<hamgnn_env>/bin.
    env_bin: str = field(
        default_factory=lambda: os.environ.get("HAMGNN_ENV_BIN", "")
    )
    openmx_postprocess: str = field(
        default_factory=lambda: os.environ.get("HAMGNN_OPENMX_POSTPROCESS", "")
    )
    read_openmx: str = field(
        default_factory=lambda: os.environ.get("HAMGNN_READ_OPENMX", "")
    )
    model_pkl: str = field(
        default_factory=lambda: os.environ.get("HAMGNN_MODEL_PKL", "")
    )
    predictor_script: str = field(
        default_factory=lambda: os.environ.get("HAMGNN_PREDICTOR_SCRIPT", "")
    )
    dft_data: str = field(default_factory=lambda: os.environ.get("HAMGNN_DFT_DATA", ""))
    mpirun: str = field(
        default_factory=lambda: os.environ.get("HAMGNN_MPIRUN", "mpirun")
    )
    mpi_ranks: int = field(
        default_factory=lambda: int(os.environ.get("HAMGNN_MPI_RANKS", "1"))
    )
    device: str = field(default_factory=lambda: os.environ.get("HAMGNN_DEVICE", "cpu"))
    nao_max: int = field(
        default_factory=lambda: int(os.environ.get("HAMGNN_NAO_MAX", "26"))
    )
    energycutoff: float = field(
        default_factory=lambda: float(os.environ.get("HAMGNN_ENERGYCUTOFF", "200.0"))
    )
    kgrid: str = field(default_factory=lambda: os.environ.get("HAMGNN_KGRID", "5 5 5"))
    nk: int = field(default_factory=lambda: int(os.environ.get("HAMGNN_NK", "120")))
    # ``mpirun`` extra args (e.g. "--allow-run-as-root") split on spaces.
    mpirun_extra: str = field(
        default_factory=lambda: os.environ.get("HAMGNN_MPIRUN_EXTRA", "")
    )
    timeout: int = field(
        default_factory=lambda: int(os.environ.get("HAMGNN_TIMEOUT", "1800"))
    )
    # Keep the per-structure temp dir for debugging (HAMGNN_KEEP_WORKDIR=1).
    keep_workdir: bool = field(
        default_factory=lambda: os.environ.get("HAMGNN_KEEP_WORKDIR", "")
        not in ("", "0", "false", "False")
    )

    def script(self, name: str) -> str:
        """Path to a HamGNN console script (poscar2openmx / graph_data_gen / band_cal)."""
        return os.path.join(self.env_bin, name) if self.env_bin else name

    def python(self) -> str:
        return os.path.join(self.env_bin, "python") if self.env_bin else "python"

    def missing(self) -> list[str]:
        """Return a checklist of unset / missing required paths."""
        problems: list[str] = []
        checks = {
            "HAMGNN_ENV_BIN (env bin dir)": (self.env_bin, os.path.isdir),
            "HAMGNN_OPENMX_POSTPROCESS": (self.openmx_postprocess, os.path.isfile),
            "HAMGNN_READ_OPENMX": (self.read_openmx, os.path.isfile),
            "HAMGNN_MODEL_PKL (checkpoint)": (self.model_pkl, os.path.isfile),
            "HAMGNN_PREDICTOR_SCRIPT": (self.predictor_script, os.path.isfile),
            "HAMGNN_DFT_DATA (OpenMX pseudopotentials)": (self.dft_data, os.path.isdir),
        }
        for label, (value, check) in checks.items():
            if not value:
                problems.append(f"{label}: not set")
            elif not check(value):
                problems.append(f"{label}: path does not exist -> {value!r}")
        return problems


class HamGNNBandGapBackend(BandGapBackend):
    """HamGNN band gap via the OpenMX -> HamGNN -> band_cal subprocess pipeline.

    The universal SOC model (Uni-HamGNN) is targeted, which needs *both* a
    non-SOC and a SOC ``graph_data.npz`` per structure. Per-structure cost is
    dominated by the OpenMX ``openmx_postprocess`` step (~minutes).

    Configure via :class:`HamGNNConfig` (env vars) or constructor overrides.
    Until everything is in place, :meth:`predict` raises
    :class:`HamGNNNotConfigured` with a concrete checklist.

    .. note::
       The subprocess wiring follows HamGNN's documented CLIs/configs; validate
       end-to-end with ``scripts/hamgnn/smoke_test.py`` once the env is built,
       then tune file conventions here if a step's output layout differs.
    """

    name = "hamgnn"

    def __init__(self, config: HamGNNConfig | None = None, **overrides: Any) -> None:
        self.config = config or HamGNNConfig()
        for key, val in overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown HamGNNConfig field: {key!r}")
            setattr(self.config, key, val)

    def is_available(self) -> bool:  # type: ignore[override]
        return not self.config.missing()

    def check_config(self) -> None:
        problems = self.config.missing()
        if problems:
            raise HamGNNNotConfigured(
                "HamGNN backend is not fully configured. Resolve:\n  - "
                + "\n  - ".join(problems)
                + "\nSee scripts/hamgnn/README.md to stand up the environment."
            )

    # -- pipeline steps -----------------------------------------------------

    def _run(self, cmd: list[str], cwd: str, step: str) -> str:
        logger.debug("HamGNN[%s]: %s", step, " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=self.config.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"HamGNN step {step!r} failed (exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )
        return proc.stdout

    def _write_poscar2openmx_yaml(self, workdir: str, soc: bool) -> str:
        import yaml

        basic = _OPENMX_BASIC_COMMAND.format(
            data_path=self.config.dft_data,
            soc_block=_SOC_ON if soc else _SOC_OFF,
            energycutoff=self.config.energycutoff,
            kgrid=self.config.kgrid,
        )
        cfg = {
            "system_name": "crystal",
            "poscar_path": os.path.join(workdir, "POSCAR"),
            "filepath": workdir,
            "basic_command": basic,
        }
        path = os.path.join(workdir, f"poscar2openmx_{'soc' if soc else 'nosoc'}.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False)
        return path

    def _write_graph_data_gen_yaml(self, workdir: str, scfdir: str, soc: bool) -> str:
        import yaml

        save = os.path.join(workdir, "soc" if soc else "nosoc")
        os.makedirs(save, exist_ok=True)
        cfg = {
            "nao_max": self.config.nao_max,
            "graph_data_save_path": save,
            "read_openmx_path": self.config.read_openmx,
            "max_SCF_skip": 200,
            "scfout_paths": scfdir,
            "dat_file_name": "openmx.dat",
            # YAML null (NOT the string "Null"): tells graph_data_gen there is no
            # openmx.std energy file (we ran openmx_postprocess only, no SCF). A
            # truthy value makes it skip every structure.
            "std_file_name": None,
            "scfout_file_name": "overlap.scfout",
            "soc_switch": bool(soc),
        }
        path = os.path.join(workdir, f"graph_data_gen_{'soc' if soc else 'nosoc'}.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False)
        return os.path.join(save, "graph_data.npz"), path

    def _write_input_yaml(
        self, workdir: str, non_soc_npz: str, soc_npz: str
    ) -> tuple[str, str]:
        import yaml

        out_dir = os.path.join(workdir, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        cfg = {
            "model_pkl_path": self.config.model_pkl,
            "non_soc_data_dir": non_soc_npz,
            "soc_data_dir": soc_npz,
            "output_dir": out_dir,
            "device": self.config.device,
            "calculate_mae": False,
        }
        path = os.path.join(workdir, "Input.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False)
        return os.path.join(out_dir, "hamiltonian.npy"), path

    def _write_band_cal_yaml(
        self, workdir: str, soc_npz: str, hamiltonian: str
    ) -> str:
        import yaml

        cfg = {
            "nao_max": self.config.nao_max,
            "graph_data_path": soc_npz,
            "hamiltonian_path": hamiltonian,
            "nk": self.config.nk,
            "save_dir": os.path.join(workdir, "band"),
            "strcture_name": "crystal",
            "soc_switch": True,
            "spin_colinear": False,
            "auto_mode": True,
            "Ham_type": "openmx",
        }
        path = os.path.join(workdir, "band_cal.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False)
        return path

    def _openmx_postprocess(self, datdir: str) -> None:
        if self.config.mpi_ranks and self.config.mpi_ranks > 1:
            cmd = [self.config.mpirun]
            if self.config.mpirun_extra:
                cmd += self.config.mpirun_extra.split()
            cmd += ["-np", str(self.config.mpi_ranks),
                    self.config.openmx_postprocess, "openmx.dat"]
        else:
            # Single rank: invoke the binary directly. The Intel-MPI-linked
            # openmx_postprocess initialises fine standalone, which avoids
            # depending on a working mpirun/mpiexec launcher (only its runtime
            # libs need to be on LD_LIBRARY_PATH).
            cmd = [self.config.openmx_postprocess, "openmx.dat"]
        self._run(cmd, cwd=datdir, step="openmx_postprocess")

    # -- public API ---------------------------------------------------------

    def predict(self, structure: Any) -> float | None:
        self.check_config()
        try:
            return self._predict_impl(structure)
        except Exception as exc:  # noqa: BLE001 - backend contract: None on failure
            logger.warning("HamGNN predict failed: %s", exc)
            return None

    def _predict_impl(self, structure: Any) -> float | None:
        workdir = tempfile.mkdtemp(prefix="hamgnn_")
        logger.info("HamGNN workdir: %s", workdir)
        try:
            structure.to(filename=os.path.join(workdir, "POSCAR"), fmt="poscar")

            npz_paths: dict[bool, str] = {}
            for soc in (False, True):
                p2o = self._write_poscar2openmx_yaml(workdir, soc)
                self._run(
                    [self.config.script("poscar2openmx"), "--config", p2o],
                    cwd=workdir,
                    step=f"poscar2openmx[{'soc' if soc else 'nosoc'}]",
                )
                # poscar2openmx writes the .dat under `filepath`; locate it and
                # normalise to <dir>/openmx.dat for openmx_postprocess.
                datdir = self._locate_dat(workdir, soc)
                self._openmx_postprocess(datdir)
                npz, gdg = self._write_graph_data_gen_yaml(workdir, datdir, soc)
                self._run(
                    [self.config.script("graph_data_gen"), "--config", gdg],
                    cwd=workdir,
                    step=f"graph_data_gen[{'soc' if soc else 'nosoc'}]",
                )
                npz_paths[soc] = npz

            hamiltonian, input_yaml = self._write_input_yaml(
                workdir, npz_paths[False], npz_paths[True]
            )
            self._run(
                [self.config.python(), self.config.predictor_script,
                 "--config", input_yaml],
                cwd=workdir,
                step="predictor",
            )
            band_yaml = self._write_band_cal_yaml(workdir, npz_paths[True], hamiltonian)
            out = self._run(
                [self.config.script("band_cal"), "--config", band_yaml],
                cwd=workdir,
                step="band_cal",
            )
            return self._parse_gap(out)
        finally:
            if self.config.keep_workdir:
                logger.info("HamGNN workdir kept for inspection: %s", workdir)
            else:
                shutil.rmtree(workdir, ignore_errors=True)

    def _locate_dat(self, workdir: str, soc: bool) -> str:
        """Find the .dat poscar2openmx produced and stage it as openmx.dat.

        poscar2openmx names files from ``system_name``; we copy the single match
        into a per-mode subdir as ``openmx.dat`` so each mode's overlap.scfout is
        isolated.
        """
        import glob

        matches = sorted(glob.glob(os.path.join(workdir, "*.dat")))
        if not matches:
            raise RuntimeError("poscar2openmx produced no .dat file")
        datdir = os.path.join(workdir, "soc" if soc else "nosoc", "dat")
        os.makedirs(datdir, exist_ok=True)
        staged = os.path.join(datdir, "openmx.dat")
        shutil.move(matches[0], staged)
        return datdir

    @staticmethod
    def _parse_gap(stdout: str) -> float | None:
        matches = _BAND_GAP_RE.findall(stdout)
        if not matches:
            logger.warning("HamGNN band_cal produced no parseable 'band gap = ...' line")
            return None
        gap = float(matches[-1])
        return max(0.0, gap)  # clamp tiny negative (band overlap) to metallic 0


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------

register_backend("hamgnn", HamGNNBandGapBackend)
register_backend("alignn", AlignnBandGapBackend)
