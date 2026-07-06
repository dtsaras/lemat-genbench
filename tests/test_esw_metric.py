import math

from pymatgen.core import Lattice, Structure

from lemat_genbench.benchmarks.esw_benchmark import ESWBenchmark
from lemat_genbench.metrics.esw_metric import ESWMetric


def _structure_with_esw(value):
    structure = Structure(
        Lattice.cubic(4.0),
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    structure.properties["esw"] = value
    return structure


def test_esw_metric_aggregates_valid_values_and_missing_values():
    metric = ESWMetric()
    result = metric([
        _structure_with_esw(2.0),
        _structure_with_esw(None),
        _structure_with_esw(4.0),
    ])

    assert result.metrics["n_evaluated"] == 3
    assert result.metrics["n_valid"] == 2
    assert result.metrics["fraction_esw_valid"] == 2 / 3
    assert result.metrics["mean_esw"] == 3.0
    assert result.metrics["median_esw"] == 3.0
    assert result.primary_metric == "mean_esw"


def test_esw_metric_target_window_becomes_primary_metric():
    metric = ESWMetric(target_min=1.5, target_max=2.5)
    result = metric([
        _structure_with_esw(2.0),
        _structure_with_esw(4.0),
    ])

    assert result.metrics["fraction_in_target_window"] == 0.5
    assert result.primary_metric == "fraction_in_target_window"


def test_esw_benchmark_reads_precomputed_properties():
    benchmark = ESWBenchmark(preprocess=False)
    result = benchmark.evaluate([
        _structure_with_esw(1.0),
        _structure_with_esw(3.0),
    ])

    assert result.final_scores["n_evaluated"] == 2
    assert result.final_scores["n_valid"] == 2
    assert result.final_scores["mean_esw"] == 2.0
    assert math.isnan(result.final_scores["median_esw"]) is False
