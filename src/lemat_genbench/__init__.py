"""LeMat-GenBench: Benchmark suite for generative models for materials.

This package provides a comprehensive benchmarking framework for evaluating 
material generation models across multiple metrics including validity, 
distribution, diversity, novelty, uniqueness, and stability.

This version includes enhanced benchmarks using new implementations:
- novelty_new: Enhanced novelty evaluation using augmented fingerprints
- uniqueness_new: Enhanced uniqueness evaluation using augmented fingerprints  
- sun_new: Enhanced SUN benchmark using augmented fingerprinting

The package includes both legacy and enhanced CLI interfaces:
- cli: Enhanced CLI with new benchmark implementations
- cli_legacy: Legacy CLI implementation
"""

__version__ = "0.2.0"

__all__ = [
    "main",         # Enhanced CLI function with new benchmarks (current)
    "main_legacy",  # Legacy CLI function
]


def __getattr__(name):
    """Lazily expose the CLI entry points (PEP 562).

    Importing the CLI eagerly pulls in ``click`` and every benchmark (and thus
    torch). Loading it lazily keeps ``import lemat_genbench`` — and importing a
    single lightweight metric (e.g. the ``migration`` extra, which needs only
    bvlain) — free of those heavy dependencies. ``from lemat_genbench import
    main`` still works.
    """
    if name == "main":
        from .cli import main

        return main
    if name == "main_legacy":
        from .cli_legacy import main as main_legacy

        return main_legacy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")