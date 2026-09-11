"""Regression tests for the standalone paper-log extractor."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "paper_ct_benchmark"
    / "extract_log_check.py"
)
SPEC = importlib.util.spec_from_file_location("extract_log_check", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_builtin_targets_need_no_external_csv():
    targets = MODULE.load_targets()
    assert set(targets) == {
        "azulene",
        "n_phenylpyrrole",
        "phthalazine",
        "quinoxaline",
        "twisted_dmabn",
    }
    assert targets["azulene"].paper_term == "2 B2"
    assert targets["azulene"].log_term == "2 B2"
    assert targets["n_phenylpyrrole"].log_term == "3 A1"
    assert targets["phthalazine"].log_term == "1 B1"
    assert targets["quinoxaline"].paper_term == "2 B1"
    assert targets["quinoxaline"].log_term == "2 B2"
    assert targets["twisted_dmabn"].log_term == "1 B1"


def test_bundled_target_csv_preserves_validated_mappings_and_locks():
    csv_path = SCRIPT.parent / "paper_targets.csv"
    targets = MODULE.load_targets(csv_path)
    assert targets == MODULE.load_targets()
    assert targets["n_phenylpyrrole"].requires_locked_reference
    assert targets["twisted_dmabn"].requires_locked_reference
    assert not targets["azulene"].requires_locked_reference


def test_no_universal_cartesian_irrep_conversion_is_assumed():
    assert MODULE.paper_to_pyscf_term("2 B2") == "2 B2"
    assert MODULE.paper_to_pyscf_term("1 B1") == "1 B1"
    assert MODULE.paper_to_pyscf_term("3 A1") == "3 A1"


def test_molecule_detection_from_production_filename():
    targets = MODULE.load_targets()
    path = Path("results/n_phenylpyrrole_v0.3.6.log")
    assert MODULE.detect_molecule(path, targets) == "n_phenylpyrrole"
