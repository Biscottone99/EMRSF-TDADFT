from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "examples" / "triplet_benchmark_schreiber2008"


def test_five_external_xyz_geometries_are_complete():
    expected = {
        "ethene": 6,
        "butadiene": 10,
        "hexatriene": 14,
        "cyclopropene": 7,
        "cyclopentadiene": 11,
    }
    files = sorted((BENCHMARK / "geometries").glob("*.xyz"))
    assert {path.stem for path in files} == set(expected)
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert int(lines[0]) == expected[path.stem]
        assert len([line for line in lines[2:] if line.strip()]) == expected[path.stem]


def test_cc3_triplet_targets_cover_all_five_molecules_and_nine_states():
    path = BENCHMARK / "triplet_targets_cc3_tzvp.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert {row["molecule"] for row in rows} == {
        "ethene",
        "butadiene",
        "hexatriene",
        "cyclopropene",
        "cyclopentadiene",
    }
    assert all(float(row["cc3_tzvp_ev"]) > 0.0 for row in rows)
    assert all(row["reference_scope"] == "vertical S0->Tn" for row in rows)


def test_mrsf_core_is_byte_identical_to_v060_release():
    expected = {
        "mrsf.py": "990244f1a1e729440f5e5c985beb07a5a63ff255a1b107fd70d65c81d21371eb",
        "space.py": "157144edd9e5943903cf2e7cf13720c76896aa99eecfdd19b2c381026e17761b",
        "kernels.py": "826943a1519fbcfb89aeb6591764f1f859705baf0ef5e02c6364ceacc36abf63",
    }
    source = ROOT / "src" / "pyscf_emrsf"
    for name, digest in expected.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
