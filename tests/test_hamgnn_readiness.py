from pymatgen.core import Lattice, Structure

from lemat_genbench.utils.hamgnn_readiness import (
    HAMGNN_READY_KEY,
    check_hamgnn_readiness,
)


def _dft_data(tmp_path, elements):
    root = tmp_path / "DFT_DATA19"
    vps_dir = root / "VPS"
    pao_dir = root / "PAO"
    vps_dir.mkdir(parents=True)
    pao_dir.mkdir(parents=True)
    for element in elements:
        (vps_dir / f"{element}_PBE19.vps").write_text("")
        (pao_dir / f"{element}7.0-s2p2.pao").write_text("")
    return root


def test_hamgnn_readiness_uses_openmx_dft_data(tmp_path):
    structure = Structure(
        Lattice.cubic(4.0),
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    record = check_hamgnn_readiness(structure, str(_dft_data(tmp_path, ["Li", "O"])))

    assert record["status"] == "success"
    assert record[HAMGNN_READY_KEY] is True
    assert record["supported_element_count"] == 2


def test_hamgnn_readiness_reports_unsupported_elements(tmp_path):
    structure = Structure(
        Lattice.cubic(4.0),
        ["Li", "Fe", "O"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    record = check_hamgnn_readiness(structure, str(_dft_data(tmp_path, ["Li", "O"])))

    assert record["status"] == "failed"
    assert record[HAMGNN_READY_KEY] is False
    assert record["unsupported_elements"] == ["Fe"]
