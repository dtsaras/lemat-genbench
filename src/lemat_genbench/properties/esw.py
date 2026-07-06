"""Li-exchange-only electrochemical stability window calculations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import tempfile
import traceback
import typing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from monty.json import MontyEncoder
from pymatgen.analysis import structure_matcher as _structure_matcher
from pymatgen.analysis.phase_diagram import (
    GrandPotentialPhaseDiagram,
    GrandPotPDEntry,
    PhaseDiagram,
)
from pymatgen.core import Element, Structure
from pymatgen.entries import compatibility as _entries_compatibility
from pymatgen.entries import computed_entries as _computed_entries
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

try:
    from typing_extensions import NotRequired as _NotRequired
    from typing_extensions import Required as _Required
except ImportError:  # pragma: no cover - py311 envs normally have typing_extensions
    _NotRequired = None
    _Required = None

if _NotRequired is not None and not hasattr(typing, "NotRequired"):
    typing.NotRequired = _NotRequired
if _Required is not None and not hasattr(typing, "Required"):
    typing.Required = _Required

try:
    from emmet.core.vasp.calculation import PotcarSpec as _PotcarSpec
except ImportError:  # pragma: no cover - depends on mp-api/emmet installation
    _PotcarSpec = None

if _PotcarSpec is not None and not hasattr(_PotcarSpec, "get"):

    def _potcar_spec_get(self: Any, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    _PotcarSpec.get = _potcar_spec_get

sys.modules.setdefault("pymatgen.core.entries", _computed_entries)
sys.modules.setdefault("pymatgen.core.structure_matcher", _structure_matcher)
sys.modules.setdefault("pymatgen.analysis.compatibility", _entries_compatibility)

DEFAULT_MACE_MODEL = "/home/cslii/.cache/mace/mace-mpa-0-medium.model"

_MACE_CALCULATORS: dict[tuple[str, str, str], Any] = {}


@dataclass
class RelaxResult:
    used_relaxed_structure: bool
    structure: Structure
    mace_energy: float | None
    optimized_cif_path: str | None
    error: str | None = None


@dataclass
class ESWResult:
    reduction_potential: float | None
    oxidation_potential: float | None
    esw: float | None
    stable_mu_min: float | None
    stable_mu_max: float | None
    ehull: float | None = None
    used_relaxed_structure: bool = False
    mace_energy: float | None = None
    optimized_cif_path: str | None = None
    error: str | None = None


def append_error(existing: str | None, new: str | None) -> str | None:
    if not new:
        return existing
    return new if not existing else f"{existing}\n{new}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_hash(value: str, n: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def structure_hash(structure: Structure) -> str:
    payload = json.dumps(structure.as_dict(), sort_keys=True, cls=MontyEncoder)
    return sha256_bytes(payload.encode("utf-8"))


def cache_name(key: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def chemsys_key(elements: list[str]) -> str:
    return "-".join(sorted(set(elements)))


def load_pickle(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        pickle.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def clean_round(value: float, ndigits: int = 6) -> float:
    rounded = round(float(value), ndigits)
    return 0.0 if abs(rounded) < 10 ** (-ndigits) else rounded


def resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_mace_model(mace_model: str | Path | None = None) -> Path:
    if mace_model:
        return Path(os.path.expandvars(str(mace_model))).expanduser()
    return Path(os.environ.get("MACE_MODEL_PATH", DEFAULT_MACE_MODEL)).expanduser()


def get_mace_calculator(
    mace_model: str | Path | None = None,
    device: str | None = None,
    default_dtype: str = "float64",
) -> Any:
    model_path = resolve_mace_model(mace_model)
    resolved_device = resolve_device(device)
    key = (str(model_path), resolved_device, default_dtype)
    if key not in _MACE_CALCULATORS:
        from mace.calculators import MACECalculator

        _MACE_CALCULATORS[key] = MACECalculator(
            model_paths=str(model_path),
            device=resolved_device,
            default_dtype=default_dtype,
        )
    return _MACE_CALCULATORS[key]


def relaxation_cache_key(
    structure: Structure,
    mace_model: str | Path | None,
    device: str | None,
    default_dtype: str,
    fmax: float,
    max_steps: int,
) -> str:
    payload = {
        "structure_hash": structure_hash(structure),
        "mace_model": str(resolve_mace_model(mace_model)),
        "device": resolve_device(device),
        "default_dtype": default_dtype,
        "fmax": fmax,
        "max_steps": max_steps,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def run_mace_relaxation(
    structure: Structure,
    *,
    cache_dir: str | Path,
    mace_model: str | Path | None = None,
    device: str | None = None,
    default_dtype: str = "float64",
    fmax: float = 0.05,
    max_steps: int = 500,
    refresh: bool = False,
) -> RelaxResult:
    cache_root = Path(cache_dir)
    relax_cache_path = cache_root / "relax_cache.pkl"
    optimized_dir = cache_root / "optimized_cifs"
    key = relaxation_cache_key(
        structure, mace_model, device, default_dtype, fmax, max_steps
    )
    relax_cache = {} if refresh else load_pickle(relax_cache_path, {})
    if key in relax_cache:
        cached = relax_cache[key]
        return RelaxResult(
            used_relaxed_structure=bool(cached["used_relaxed_structure"]),
            structure=Structure.from_dict(cached["structure"]),
            mace_energy=cached.get("mace_energy"),
            optimized_cif_path=cached.get("optimized_cif_path"),
            error=cached.get("error"),
        )

    try:
        from ase.filters import FrechetCellFilter
        from ase.io import write as ase_write
        from ase.optimize import BFGS
        from pymatgen.io.ase import AseAtomsAdaptor

        calc = get_mace_calculator(mace_model, device, default_dtype)
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = calc
        cell_filter = FrechetCellFilter(atoms)
        opt = BFGS(cell_filter, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        energy = float(atoms.get_potential_energy())
        relaxed_structure = AseAtomsAdaptor.get_structure(atoms)
        unique = short_hash(key)
        optimized_cif = optimized_dir / f"esw_relaxed_{unique}.cif"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        ase_write(str(optimized_cif), atoms)
        result = RelaxResult(
            used_relaxed_structure=True,
            structure=relaxed_structure,
            mace_energy=energy,
            optimized_cif_path=str(optimized_cif),
        )
    except Exception:
        result = RelaxResult(
            used_relaxed_structure=False,
            structure=structure,
            mace_energy=None,
            optimized_cif_path=None,
            error="MACE relaxation failed:\n" + traceback.format_exc(),
        )

    relax_cache[key] = {
        "used_relaxed_structure": result.used_relaxed_structure,
        "structure": result.structure.as_dict(),
        "mace_energy": result.mace_energy,
        "optimized_cif_path": result.optimized_cif_path,
        "error": result.error,
    }
    dump_pickle(relax_cache_path, relax_cache)
    return result


def get_entries_for_chemsys(
    elements: list[str],
    *,
    cache_dir: str | Path,
    api_key: str | None = None,
    refresh: bool = False,
) -> list[Any]:
    entries_dir = Path(cache_dir) / "entries_by_chemsys"
    key = chemsys_key(elements)
    path = entries_dir / f"{cache_name(key)}.pkl"
    if path.exists() and not refresh:
        return load_pickle(path, [])

    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("No Materials Project API key provided. Set MP_API_KEY.")

    from pymatgen.ext.matproj import MPRester

    with MPRester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys(elements, compatible_only=True)
    dump_pickle(path, entries)
    return entries


def make_mp2020_compatible_entry(
    entry: ComputedStructureEntry,
) -> ComputedStructureEntry:
    with tempfile.TemporaryDirectory() as tmpdirname:
        relax_set = MPRelaxSet(entry.structure)
        relax_set.write_input(tmpdirname, potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Poscar.from_file(f"{tmpdirname}/POSCAR").structure

    parameters: dict[str, Any] = {"hubbards": {}}
    if "LDAUU" in incar:
        parameters["hubbards"] = dict(
            zip(poscar.site_symbols, incar["LDAUU"], strict=False)
        )
    parameters["is_hubbard"] = bool(
        incar.get("LDAU", True) and sum(parameters["hubbards"].values()) > 0
    )
    parameters["run_type"] = "GGA+U" if parameters["is_hubbard"] else "GGA"

    compatible_entry = ComputedStructureEntry(
        structure=clean_structure,
        energy=entry.uncorrected_energy,
        correction=0.0,
        parameters=parameters,
        data=entry.data,
        entry_id=entry.entry_id,
    )
    processed_entries = MaterialsProject2020Compatibility(
        check_potcar=False
    ).process_entries(
        compatible_entry,
        clean=True,
        inplace=True,
        on_error="raise",
    )
    if not processed_entries:
        raise ValueError(
            f"MP2020 compatibility rejected target entry {entry.entry_id}."
        )
    return processed_entries[0]


def build_phase_diagram_with_target(
    structure: Structure,
    energy: float,
    entry_id: str,
    mp_entries: list[Any],
) -> tuple[ComputedStructureEntry, PhaseDiagram]:
    raw_entry = ComputedStructureEntry(
        structure=structure, energy=energy, entry_id=entry_id
    )
    target_entry = make_mp2020_compatible_entry(raw_entry)
    pd_non_grand = PhaseDiagram(list(mp_entries) + [target_entry])
    return target_entry, pd_non_grand


def relative_mu_grid(mu_min: float, mu_max: float, mu_step: float) -> list[float]:
    if mu_step <= 0:
        raise ValueError("mu_step must be positive")
    if mu_max < mu_min:
        raise ValueError("mu_max must be greater than or equal to mu_min")
    n_steps = int(math.floor((mu_max - mu_min) / mu_step + 1e-12))
    values = [mu_min + i * mu_step for i in range(n_steps + 1)]
    if not values or values[-1] < mu_max - 1e-10:
        values.append(mu_max)
    return values


def sorted_unique_float_values(values: list[float], tol: float = 1e-10) -> list[float]:
    unique: list[float] = []
    for value in sorted(float(v) for v in values):
        if unique and math.isclose(value, unique[-1], abs_tol=tol, rel_tol=0):
            continue
        unique.append(value)
    return unique


def hybrid_mu_edges(
    pd_non_grand: PhaseDiagram,
    open_element: Element,
    reference_mu: float,
    mu_min: float,
    mu_max: float,
    mu_step: float,
) -> list[float]:
    edges = relative_mu_grid(mu_min, mu_max, mu_step)
    absolute_min = reference_mu + mu_min
    absolute_max = reference_mu + mu_max
    for absolute_mu in pd_non_grand.get_transition_chempots(open_element):
        if absolute_min < absolute_mu < absolute_max:
            edges.append(float(absolute_mu - reference_mu))
    return sorted_unique_float_values(edges)


def original_composition(entry: Any):
    original_comp = getattr(entry, "original_comp", None)
    if original_comp is not None:
        return original_comp
    original_entry = getattr(entry, "original_entry", None)
    if original_entry is not None:
        return original_entry.composition
    return entry.composition


def li_exchange_amount(target_entry: Any, decomp: dict[Any, float] | None) -> float:
    if not decomp:
        return 0.0
    li = Element("Li")
    target_non_li_atoms = sum(
        amount for element, amount in target_entry.composition.items() if element != li
    )
    if target_non_li_atoms <= 0:
        return 0.0
    product_li = 0.0
    for entry, amount in decomp.items():
        comp = original_composition(entry)
        non_li_atoms = sum(value for element, value in comp.items() if element != li)
        if non_li_atoms <= 0:
            continue
        product_li += amount * target_non_li_atoms * comp.get(li, 0.0) / non_li_atoms
    return product_li - target_entry.composition.get(li, 0.0)


def zero_li_exchange_at_mu(
    relative_mu: float,
    li: Element,
    li_ref_mu: float,
    entries: list[Any],
    target_entry: Any,
    stability_tol: float,
) -> bool:
    chempots = {li: li_ref_mu + relative_mu}
    gpd = GrandPotentialPhaseDiagram(entries, chempots)
    gp_target = GrandPotPDEntry(target_entry, chempots)
    decomp, _e_above_hull = gpd.get_decomp_and_e_above_hull(
        gp_target, on_error="ignore"
    )
    if decomp is None:
        return False
    evolution = li_exchange_amount(target_entry, decomp)
    return abs(evolution) <= stability_tol


def compute_li_exchange_only_esw(
    target_entry: Any,
    pd_non_grand: PhaseDiagram,
    mu_min: float,
    mu_max: float,
    mu_step: float,
    stability_tol: float,
) -> ESWResult:
    li = Element("Li")
    if li not in pd_non_grand.el_refs:
        return ESWResult(
            None, None, None, None, None, error="Li reference not found in phase diagram."
        )
    li_ref_mu = pd_non_grand.el_refs[li].energy_per_atom
    entries = list(pd_non_grand.all_entries)
    edges = hybrid_mu_edges(pd_non_grand, li, li_ref_mu, mu_min, mu_max, mu_step)
    if len(edges) < 2:
        return ESWResult(None, None, 0.0, None, None)

    intervals: list[tuple[float, float]] = []
    current_left: float | None = None
    current_right: float | None = None

    for left_mu, right_mu in zip(edges[:-1], edges[1:], strict=False):
        if right_mu - left_mu <= 1e-12:
            continue
        sample_mu = (left_mu + right_mu) / 2.0
        is_stable = zero_li_exchange_at_mu(
            relative_mu=sample_mu,
            li=li,
            li_ref_mu=li_ref_mu,
            entries=entries,
            target_entry=target_entry,
            stability_tol=stability_tol,
        )
        if is_stable:
            if current_left is None:
                current_left = float(left_mu)
            current_right = float(right_mu)
        else:
            if current_left is not None and current_right is not None:
                intervals.append((current_left, current_right))
            current_left = None
            current_right = None

    if current_left is not None and current_right is not None:
        intervals.append((current_left, current_right))

    if not intervals:
        return ESWResult(
            reduction_potential=None,
            oxidation_potential=None,
            esw=0.0,
            stable_mu_min=None,
            stable_mu_max=None,
        )

    stable_mu_min, stable_mu_max = max(intervals, key=lambda x: x[1] - x[0])
    reduction_potential = -stable_mu_max
    oxidation_potential = -stable_mu_min
    esw = max(0.0, oxidation_potential - reduction_potential)
    return ESWResult(
        reduction_potential=clean_round(reduction_potential),
        oxidation_potential=clean_round(oxidation_potential),
        esw=clean_round(esw),
        stable_mu_min=clean_round(stable_mu_min),
        stable_mu_max=clean_round(stable_mu_max),
    )


def compute_esw(
    structure: Structure,
    *,
    cache_dir: str | Path = "data/esw_cache",
    api_key: str | None = None,
    mace_model: str | Path | None = None,
    device: str | None = None,
    default_dtype: str = "float64",
    fmax: float = 0.05,
    max_steps: int = 500,
    mu_min: float = -5.0,
    mu_max: float = 0.0,
    mu_step: float = 0.01,
    gpd_stability_tol: float = 1e-4,
    refresh_relax: bool = False,
    refresh_mp: bool = False,
) -> ESWResult:
    cache_root = Path(cache_dir)
    relax_result = run_mace_relaxation(
        structure,
        cache_dir=cache_root,
        mace_model=mace_model,
        device=device,
        default_dtype=default_dtype,
        fmax=fmax,
        max_steps=max_steps,
        refresh=refresh_relax,
    )
    if not relax_result.used_relaxed_structure or relax_result.mace_energy is None:
        return ESWResult(
            None,
            None,
            None,
            None,
            None,
            used_relaxed_structure=relax_result.used_relaxed_structure,
            mace_energy=relax_result.mace_energy,
            optimized_cif_path=relax_result.optimized_cif_path,
            error=relax_result.error,
        )

    try:
        used_structure = relax_result.structure
        elements = sorted(el.symbol for el in used_structure.composition.elements)
        mp_entries = get_entries_for_chemsys(
            elements,
            cache_dir=cache_root,
            api_key=api_key,
            refresh=refresh_mp,
        )
        target_entry, pd_non_grand = build_phase_diagram_with_target(
            used_structure,
            relax_result.mace_energy,
            f"generated_{short_hash(structure_hash(used_structure))}",
            mp_entries,
        )
        ehull = float(pd_non_grand.get_e_above_hull(target_entry))
        result = compute_li_exchange_only_esw(
            target_entry=target_entry,
            pd_non_grand=pd_non_grand,
            mu_min=mu_min,
            mu_max=mu_max,
            mu_step=mu_step,
            stability_tol=gpd_stability_tol,
        )
        result.ehull = ehull
        result.used_relaxed_structure = relax_result.used_relaxed_structure
        result.mace_energy = relax_result.mace_energy
        result.optimized_cif_path = relax_result.optimized_cif_path
        return result
    except Exception:
        return ESWResult(
            None,
            None,
            None,
            None,
            None,
            used_relaxed_structure=relax_result.used_relaxed_structure,
            mace_energy=relax_result.mace_energy,
            optimized_cif_path=relax_result.optimized_cif_path,
            error=append_error(
                relax_result.error, "Ehull/ESW failed:\n" + traceback.format_exc()
            ),
        )


def result_to_properties(result: ESWResult) -> dict[str, Any]:
    return asdict(result)
