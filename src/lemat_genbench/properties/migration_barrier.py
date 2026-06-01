"""BVSE bond-valence site-energy percolation barriers for mobile ions.

Computes the minimum saddle-point energy (eV) a mobile ion (Li+ by default)
must clear to percolate through a crystal structure in 1D / 2D / 3D, using the
bond-valence site-energy (BVSE) method as implemented by the optional
``bvlain`` package (https://github.com/dembart/BVlain). Lower barriers imply
faster ionic transport.

This is a *fast, semi-empirical* proxy (~0.5 s/structure, CPU-only) — not a
substitute for NEB/AIMD, but well suited to high-throughput screening of
generated structures.

Adapted from the battle-tested wrapper in the author's ``matter_evolve``
project: the env-var knobs are replaced by explicit parameters, and the
per-structure wall-clock cap is delegated to the calling metric's
``timeout_context`` (BaseMetric.timeout).

Outcome semantics
-----------------
Distinguishing *missing data* from *definitely-not-a-conductor* matters for the
metric's aggregate statistics:

* mobile ion absent            -> raise :class:`MobileIonAbsent`  (metric -> NaN, excluded)
* ran, but no percolating path -> value == ``no_percolation_sentinel`` (a real
                                  "not a conductor at this dim" answer, counted)
* real barrier                 -> finite float (eV)
* bvlain missing / hard failure -> raise (metric -> failed_indices)

References
----------
Adams, S.; Rao, R. P. "High power lithium ion battery materials by
computational design", *Phys. Status Solidi A* (2011).
Dembart, "BVlain: Bond Valence Site Energy module", GitHub, 2024.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# eV. Returned when bvlain runs but finds no percolating path at the requested
# dimensionality (bvlain reports inf). Chosen well above ``encut`` and any
# physical barrier so downstream "is this a fast conductor?" thresholds treat
# it as a hard no, without contaminating numeric stats with inf/nan.
NO_PERCOLATION_SENTINEL = 10.0

# bvlain.percolation_barriers() returns these keys.
DIM_KEYS = {"1d": "E_1D", "2d": "E_2D", "3d": "E_3D"}


class MobileIonAbsent(Exception):
    """Raised when the configured mobile ion is not present in the structure."""


def is_available() -> bool:
    """True if the optional ``bvlain`` dependency is importable."""
    try:
        import bvlain  # noqa: F401
    except Exception:
        return False
    return True


def _mobile_ion_symbol(mobile_ion: str) -> str:
    """Strip the charge from a pymatgen-style ion string ('Li1+' -> 'Li')."""
    return "".join(c for c in mobile_ion if c.isalpha())


def compute_barriers(
    structure: Any,
    *,
    mobile_ion: str = "Li1+",
    r_cut: float = 10.0,
    resolution: float = 0.2,
    k: int = 100,
    encut: float = 5.0,
    no_percolation_sentinel: float = NO_PERCOLATION_SENTINEL,
) -> dict[str, float]:
    """Compute BVSE percolation barriers for ``mobile_ion`` in ``structure``.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The structure to evaluate. It is never mutated (bvlain operates on a
        defensive copy).
    mobile_ion : str
        Pymatgen-style ion string, e.g. ``"Li1+"``, ``"Na1+"``, ``"Mg2+"``.
    r_cut, resolution, k, encut : float / int
        BVSE grid / search parameters passed through to bvlain.
    no_percolation_sentinel : float
        Value substituted for any non-percolating dimension (bvlain ``inf``).

    Returns
    -------
    dict[str, float]
        ``{"E_1D": float, "E_2D": float, "E_3D": float}`` in eV.

    Raises
    ------
    ImportError
        If ``bvlain`` is not installed.
    MobileIonAbsent
        If ``mobile_ion`` is not present in the structure.
    RuntimeError
        If bvlain fails on every oxidation-state-assignment fallback path.
    """
    # Pre-flight: nothing to migrate if the ion is absent. Done before importing
    # bvlain so an ion-free structure resolves with no optional dependency and no
    # wasted import.
    ion_symbol = _mobile_ion_symbol(mobile_ion)
    try:
        present = {
            str(getattr(sp, "symbol", sp)) for sp in structure.composition.elements
        }
    except Exception:
        present = set()
    if ion_symbol and ion_symbol not in present:
        raise MobileIonAbsent(f"{ion_symbol!r} not present in {sorted(present)}")

    try:
        from bvlain import Lain
    except ImportError as exc:  # pragma: no cover - exercised only without bvlain
        raise ImportError(
            "bvlain is required for the migration-barrier metric. "
            "Install it with:  pip install bvlain   (or: uv sync --extra migration)"
        ) from exc

    # Oxidation-state assignment fallback chain. Generator outputs often trip
    # BVAnalyzer's local BVS minimisation even when the composition is sensible,
    # so we fall back to a composition-level guess. Both operate on a defensive
    # ``.copy()`` so bvlain's in-place mutations never leak back to the caller.
    last_err: Exception | None = None
    for path_name, decorate in (
        ("oxi_check=True", lambda s: s.copy()),
        (
            "add_oxidation_state_by_guess",
            lambda s: (lambda c: (c.add_oxidation_state_by_guess(), c)[1])(s.copy()),
        ),
    ):
        try:
            s_for_bv = decorate(structure)
            calc = Lain(verbose=False)
            calc.read_structure(s_for_bv, oxi_check=(path_name == "oxi_check=True"))
            calc.bvse_distribution(
                mobile_ion=mobile_ion, r_cut=r_cut, resolution=resolution, k=k
            )
            barriers = calc.percolation_barriers(encut=encut)
            if barriers:
                return _coerce_barriers(barriers, no_percolation_sentinel)
            # Empty result -> treat as a failed path and try the next fallback.
        except MobileIonAbsent:
            raise
        except Exception as exc:  # noqa: BLE001 - try the next fallback path
            last_err = exc
            logger.debug("bvlain path %s failed: %s", path_name, exc)
            continue

    raise RuntimeError(f"bvlain failed on all fallback paths (last error: {last_err})")


def _coerce_barriers(barriers: dict, sentinel: float) -> dict[str, float]:
    """Map bvlain's barrier dict to plain floats, ``inf``/``nan`` -> sentinel."""
    out: dict[str, float] = {}
    for key, val in barriers.items():
        # Booleans are ints in Python; force them to the sentinel (not a barrier).
        if isinstance(val, bool):
            out[key] = sentinel
            continue
        try:
            fv = float(val)
        except (TypeError, ValueError):
            fv = float("inf")
        bad = fv == float("inf") or fv == float("-inf") or fv != fv  # inf/-inf/NaN
        out[key] = sentinel if bad else fv
    return out


def is_no_percolation(
    value: float | None, sentinel: float = NO_PERCOLATION_SENTINEL
) -> bool:
    """True if ``value`` is the no-percolation sentinel (tolerant of JSON drift)."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return abs(float(value) - sentinel) < 1e-9
