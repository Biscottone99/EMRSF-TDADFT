from __future__ import annotations

import io

import numpy as np
import pytest
from pyscf import gto
from pyscf.tools import molden

import pyscf_emrsf.reference as reference_module
from pyscf_emrsf.log import _write_reference
from pyscf_emrsf.reference import (
    analyse_reference,
    build_reference,
    build_reference_from_molden,
    load_molden_reference_seed,
)


def test_two_cycle_monitor_detects_and_averages_only_the_guess():
    monitor = reference_module._SCFHistoryMonitor(
        two_cycle_tolerance=1.0e-10,
        one_cycle_minimum=1.0e-3,
    )
    first = np.zeros((2, 2))
    second = np.eye(2)
    third = np.full((2, 2), 1.0e-12)
    for cycle, density in enumerate((first, second, third)):
        monitor(
            {
                "e_tot": -1.0 - cycle,
                "norm_gorb": 0.1,
                "norm_ddm": 1.0,
                "dm": density,
            }
        )
    assert monitor.oscillation_detected
    np.testing.assert_allclose(
        monitor.averaged_cycle_guess(), 0.5 * (second + third)
    )


def test_robust_reference_reaches_exact_final_grid_with_newton():
    molecule = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mf = build_reference(
        molecule,
        xc=None,
        conv_tol=1.0e-10,
        conv_tol_grad=1.0e-7,
        max_cycle=1,
        verbose=0,
        scf_strategy="robust",
        robust_initial_max_cycle=1,
        robust_stabilization_max_cycle=1,
        robust_newton_max_cycle=30,
    )
    metadata = mf.emrsf_reference_metadata
    assert mf.converged
    assert metadata["scf_strategy"] == "robust"
    assert metadata["scf_rescue_used"] is True
    assert metadata["scf_final_stage"] == "final-grid-newton"
    assert len(metadata["scf_attempt_history"]) == 4
    assert metadata["scf_attempt_history"][-1]["converged"] is True
    reference = analyse_reference(mf)
    assert reference.diagnostics.orbital_gradient_max < 1.0e-7
    report = io.StringIO()
    _write_reference(report, reference)
    text = report.getvalue()
    assert "SCF STRATEGY          robust" in text
    assert "SCF FINAL STAGE       final-grid-newton" in text
    assert "SCF RESCUE USED       True" in text
    assert "SCF ATTEMPT COUNT     4" in text
    assert "Newton/CIAH+MOM" in text


def test_standard_reference_strategy_does_not_invoke_rescue():
    molecule = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    with pytest.raises(RuntimeError, match="SCF attempts: final-grid-standard"):
        build_reference(
            molecule,
            xc=None,
            max_cycle=1,
            verbose=0,
            scf_strategy="standard",
        )


def test_robust_df_roks_newton_has_no_final_stabiliser():
    molecule = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mf = build_reference(
        molecule,
        xc="bhandhlyp",
        density_fit=True,
        conv_tol=1.0e-10,
        conv_tol_grad=1.0e-7,
        max_cycle=1,
        grid_level=1,
        verbose=0,
        scf_strategy="robust",
        robust_initial_max_cycle=1,
        robust_stabilization_max_cycle=1,
        robust_newton_max_cycle=30,
        robust_coarse_grid_level=0,
    )
    assert mf.converged
    assert mf.emrsf_reference_metadata["scf_final_stage"] == "final-grid-newton"
    assert mf.grids.level == 1
    assert getattr(mf, "level_shift", 0.0) == 0.0
    assert getattr(mf, "damp", 0.0) == 0.0
    assert getattr(mf, "with_df", None) is not None
    assert analyse_reference(mf).diagnostics.orbital_gradient_max < 1.0e-7


def _write_spin_resolved_molden(mf, path) -> None:
    occupation = np.asarray(mf.mo_occ)
    alpha = (occupation > 0.5).astype(float)
    beta = (occupation > 1.5).astype(float)
    with path.open("w", encoding="utf-8") as handle:
        molden.header(mf.mol, handle)
        molden.orbital_coeff(
            mf.mol,
            handle,
            mf.mo_coeff,
            spin="Alpha",
            ene=mf.mo_energy,
            occ=alpha,
        )
        molden.orbital_coeff(
            mf.mol,
            handle,
            mf.mo_coeff,
            spin="Beta",
            ene=mf.mo_energy,
            occ=beta,
        )


def test_spin_resolved_molden_import_and_external_reference(h2o_rohf, tmp_path):
    path = tmp_path / "triplet.molden"
    _write_spin_resolved_molden(h2o_rohf, path)
    seed = load_molden_reference_seed(path, target_mol=h2o_rohf.mol)
    assert np.count_nonzero(seed.occupation == 1.0) == 2
    assert seed.orthonormality_error < 1.0e-10

    imported = build_reference_from_molden(path, mol=h2o_rohf.mol, xc=None)
    ref = analyse_reference(imported)
    assert ref.diagnostics.source == "external_molden"
    assert not ref.diagnostics.local_scf_run
    assert ref.diagnostics.orthonormality_error < 1.0e-10
    assert abs(ref.diagnostics.electron_number_error) < 1.0e-10
    assert abs(ref.diagnostics.spin_number_error) < 1.0e-10


def test_mom_reference_preserves_seed_somo_subspace(h2o_rohf, tmp_path):
    path = tmp_path / "triplet.molden"
    _write_spin_resolved_molden(h2o_rohf, path)
    mf = build_reference(
        h2o_rohf.mol,
        xc=None,
        conv_tol=1.0e-11,
        max_cycle=50,
        molden_seed=path,
        minimum_somo_overlap=0.95,
        minimum_occupied_overlap=0.95,
        verbose=0,
    )
    ref = analyse_reference(mf)
    assert ref.diagnostics.source == "mom_molden"
    assert ref.diagnostics.local_scf_run
    assert ref.diagnostics.local_scf_converged
    assert min(ref.diagnostics.seed_somo_singular_values) > 0.95
    assert ref.diagnostics.seed_alpha_occupied_min_singular_value > 0.95
    assert ref.diagnostics.seed_beta_occupied_min_singular_value > 0.95
    assert np.array_equal(
        np.concatenate((ref.closed, ref.open, ref.virtual)),
        np.arange(ref.nmo),
    )


def test_molden_seed_can_be_projected_into_a_larger_basis(h2o_rohf, tmp_path):
    path = tmp_path / "triplet-small-basis.molden"
    _write_spin_resolved_molden(h2o_rohf, path)
    source = h2o_rohf.mol
    target = gto.M(
        atom=[
            (source.atom_symbol(atom), tuple(source.atom_coord(atom)))
            for atom in range(source.natm)
        ],
        unit="Bohr",
        basis="3-21g",
        spin=2,
        verbose=0,
    )

    with pytest.raises(ValueError, match="AO-count mismatch"):
        load_molden_reference_seed(path, target_mol=target)

    seed = load_molden_reference_seed(
        path,
        target_mol=target,
        allow_basis_projection=True,
    )
    assert seed.coeff.shape[0] == target.nao_nr()
    assert seed.coeff.shape[1] == source.nao_nr()
    overlap = target.intor_symmetric("int1e_ovlp")
    metric = seed.coeff.T @ overlap @ seed.coeff
    np.testing.assert_allclose(metric, np.eye(metric.shape[0]), atol=2.0e-10)
    alpha = (seed.coeff * seed.alpha_occupation) @ seed.coeff.T
    beta = (seed.coeff * seed.beta_occupation) @ seed.coeff.T
    np.testing.assert_allclose(
        np.einsum("ij,ji->", overlap, alpha + beta),
        target.nelectron,
        atol=2.0e-9,
    )

    projected_mom = build_reference(
        target,
        xc=None,
        conv_tol=1.0e-11,
        max_cycle=80,
        molden_seed=path,
        allow_basis_projection=True,
        minimum_somo_overlap=0.70,
        minimum_occupied_overlap=0.70,
        verbose=0,
    )
    projected_ref = analyse_reference(projected_mom)
    assert projected_ref.diagnostics.source == "mom_molden"
    assert projected_mom.emrsf_reference_metadata["basis_projection"] is True
    assert min(projected_ref.diagnostics.seed_somo_singular_values) > 0.70
