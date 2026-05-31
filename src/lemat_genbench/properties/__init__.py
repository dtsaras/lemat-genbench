"""Property predictors for crystal structures.

This subpackage hosts functional-property predictors that complement the
core stability / S.U.N. metrics:

* :mod:`~lemat_genbench.properties.migration_barrier` — BVSE ion migration
  barriers (pure-CPU, via the optional ``bvlain`` dependency).
* :mod:`~lemat_genbench.properties.band_gap_backends` — pluggable band-gap
  predictors (ALIGNN / MatGL now; HamGNN slot for later).

Predictors import their heavy/optional dependencies lazily, so importing this
package never requires those extras to be installed.
"""
