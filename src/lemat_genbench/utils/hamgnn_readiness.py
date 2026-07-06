"""Readiness helpers for HamGNN/OpenMX band-gap calculations."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pymatgen.core import Element, Structure

HAMGNN_READINESS_RECORD_KEY = "hamgnn_readiness_record"
HAMGNN_READY_KEY = "hamgnn_ready"

_OPENMX_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)(?=[0-9_.-])")


def _valid_element(symbol: str) -> bool:
    try:
        Element(symbol)
    except ValueError:
        return False
    return True


def _elements_from_openmx_dir(path: Path) -> set[str]:
    if not path.is_dir():
        return set()

    elements: set[str] = set()
    for file_path in path.rglob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        match = _OPENMX_ELEMENT_RE.match(file_path.name)
        if match:
            symbol = match.group(1)
            if _valid_element(symbol):
                elements.add(symbol)
    return elements


@lru_cache(maxsize=8)
def load_openmx_supported_elements(
    dft_data_dir: str,
) -> tuple[frozenset[str], dict[str, Any]]:
    """Load elements supported by the installed OpenMX DFT_DATA directory.

    OpenMX needs both a VPS and a PAO entry for an element, so the supported set
    is the intersection of elements found under ``VPS`` and ``PAO``.
    """
    root = Path(os.path.expandvars(dft_data_dir)).expanduser()
    vps_dir = root / "VPS"
    pao_dir = root / "PAO"
    warnings: list[str] = []

    vps_elements = _elements_from_openmx_dir(vps_dir)
    pao_elements = _elements_from_openmx_dir(pao_dir)
    if not vps_elements:
        warnings.append(f"No OpenMX VPS element files found under {vps_dir}")
    if not pao_elements:
        warnings.append(f"No OpenMX PAO element files found under {pao_dir}")

    supported = vps_elements & pao_elements
    info = {
        "dft_data": str(root),
        "vps_dir": str(vps_dir),
        "pao_dir": str(pao_dir),
        "vps_element_count": len(vps_elements),
        "pao_element_count": len(pao_elements),
        "supported_element_count": len(supported),
        "warnings": warnings,
    }
    return frozenset(supported), info


def check_hamgnn_readiness(
    structure: Structure,
    dft_data_dir: str | None,
) -> dict[str, Any]:
    """Check whether all structure elements are present in OpenMX DFT_DATA."""
    elements = sorted({element.symbol for element in structure.composition.elements})
    record = {
        "status": "failed",
        HAMGNN_READY_KEY: False,
        "dft_data": dft_data_dir,
        "elements": elements,
        "unsupported_elements": [],
        "supported_element_count": 0,
        "failure_reasons": [],
        "warnings": [],
    }

    if not dft_data_dir:
        record["failure_reasons"].append(
            "HAMGNN_DFT_DATA/OpenMX DFT_DATA19 path is not configured"
        )
        return record

    root = Path(os.path.expandvars(dft_data_dir)).expanduser()
    record["dft_data"] = str(root)
    if not root.is_dir():
        record["failure_reasons"].append(
            f"HAMGNN_DFT_DATA/OpenMX DFT_DATA19 path does not exist: {root}"
        )
        return record

    supported_elements, support_info = load_openmx_supported_elements(str(root))
    record["support_source"] = support_info
    record["supported_element_count"] = support_info["supported_element_count"]
    record["warnings"].extend(support_info["warnings"])

    if not supported_elements:
        record["failure_reasons"].append(
            "OpenMX DFT_DATA19 supported elements could not be read"
        )
        return record

    unsupported = sorted(set(elements) - set(supported_elements))
    record["unsupported_elements"] = unsupported
    if unsupported:
        record["failure_reasons"].append(
            "elements not supported by OpenMX DFT_DATA19: "
            + ", ".join(unsupported)
        )
        return record

    record["status"] = "success"
    record[HAMGNN_READY_KEY] = True
    return record
