#!/usr/bin/env python
"""End-to-end smoke test for the HamGNN band-gap backend.

Runs ONE structure through the HamGNN subprocess pipeline and prints the gap.
Run from the lemat-genbench environment (the backend shells out to the separate
HamGNN env). Configure via the HAMGNN_* env vars first — see this dir's README.

    python scripts/hamgnn/smoke_test.py            # uses a built-in Si cell
    python scripts/hamgnn/smoke_test.py my.cif     # uses your structure
"""

import sys

from pymatgen.core import Lattice, Structure

from lemat_genbench.properties.band_gap_backends import (
    HamGNNBandGapBackend,
    HamGNNConfig,
)


def _silicon() -> Structure:
    """Diamond-cubic Si (8 atoms, a = 5.43 A). PBE gap ~0.6 eV."""
    a = 5.43
    frac = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00], [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ]
    return Structure(Lattice.cubic(a), ["Si"] * 8, frac)


def main() -> int:
    structure = (
        Structure.from_file(sys.argv[1]) if len(sys.argv) > 1 else _silicon()
    )
    print(f"Structure: {structure.composition.reduced_formula} "
          f"({len(structure)} atoms)")

    config = HamGNNConfig()
    problems = config.missing()
    if problems:
        print("\nHamGNN backend is NOT fully configured:")
        for p in problems:
            print(f"  - {p}")
        print("\nSet the HAMGNN_* env vars (see scripts/hamgnn/README.md), then "
              "re-run.")
        return 2

    backend = HamGNNBandGapBackend(config=config)
    print("Config OK. Running OpenMX -> HamGNN -> band_cal pipeline "
          "(this can take a few minutes)...")
    gap = backend.predict(structure)
    if gap is None:
        print("\nFAILED: backend returned None. Re-run with logging enabled to "
              "see the failing subprocess step:\n"
              "  python -c \"import logging; logging.basicConfig(level=logging.DEBUG); "
              "import runpy; runpy.run_path('scripts/hamgnn/smoke_test.py', run_name='__main__')\"")
        return 1

    print(f"\n✅ Predicted band gap: {gap:.4f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
