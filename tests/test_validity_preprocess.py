from pymatgen.core import Lattice, Structure

from lemat_genbench.preprocess.validity_preprocess import ValidityPreprocessor
from lemat_genbench.utils.oxidation_state import (
    MIGRATION_BARRIER_READY_KEY,
    OXIDATION_STATE_RECORD_KEY,
    assign_oxidation_states_for_bvlain,
)


def _lif_rocksalt() -> Structure:
    return Structure(
        Lattice.cubic(4.026),
        ["Li", "F"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _si() -> Structure:
    return Structure(Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])


def _li() -> Structure:
    return Structure(Lattice.cubic(3.5), ["Li"], [[0.0, 0.0, 0.0]])


def _hg() -> Structure:
    return Structure(Lattice.cubic(3.0), ["Hg"], [[0.0, 0.0, 0.0]])


def _preprocessor(**kwargs) -> ValidityPreprocessor:
    return ValidityPreprocessor(
        assign_oxidation_states=True,
        mobile_ion="Li1+",
        require_mobile_ion=True,
        check_bvlain_parameters=False,
        plausibility_check_format=False,
        plausibility_check_symmetry=False,
        **kwargs,
    )


def test_validity_preprocessor_adds_migration_readiness_record():
    preprocessor = _preprocessor()
    result = preprocessor.run([_lif_rocksalt()], structure_sources=["LiF.cif"])
    structure = result.processed_structures[0]
    record = structure.properties[OXIDATION_STATE_RECORD_KEY]

    assert record["status"] == "success"
    assert record["mobile_ion"] == "Li1+"
    assert structure.properties[MIGRATION_BARRIER_READY_KEY] is True

    benchmark_result = preprocessor.generate_benchmark_result(result)
    assert benchmark_result.final_scores["oxidation_state_success_count"] == 1
    assert benchmark_result.final_scores["migration_barrier_ready_count"] == 1


def test_validity_preprocessor_marks_missing_mobile_ion_not_ready():
    result = _preprocessor().run([_si()], structure_sources=["Si.cif"])
    structure = result.processed_structures[0]
    record = structure.properties[OXIDATION_STATE_RECORD_KEY]

    assert record["status"] == "failed"
    assert structure.properties[MIGRATION_BARRIER_READY_KEY] is False
    assert any("mobile ion Li not present" in reason for reason in record["failure_reasons"])


def test_validity_preprocessor_rejects_forbidden_elements():
    result = ValidityPreprocessor(
        plausibility_check_format=False,
        plausibility_check_symmetry=False,
    ).run([_hg()], structure_sources=["Hg.cif"])
    structure = result.processed_structures[0]

    assert structure.properties["element_valid"] is False
    assert structure.properties["overall_valid"] is False
    assert structure.properties["element_check_details"]["forbidden_elements_present"] == ["Hg"]

    benchmark_result = ValidityPreprocessor().generate_benchmark_result(result)
    assert benchmark_result.final_scores["element_validity_count"] == 0


def test_migration_readiness_requires_minimum_element_count():
    record = assign_oxidation_states_for_bvlain(
        _li(),
        mobile_ion="Li1+",
        require_mobile_ion=True,
        min_num_elements=2,
        check_bvlain_parameters=False,
    )

    assert record["status"] == "failed"
    assert record["migration_barrier_ready"] is False
    assert any("number of elements 1" in reason for reason in record["failure_reasons"])
