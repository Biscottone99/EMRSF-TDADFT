from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, scf

from pyscf_emrsf import (
    EMRSFTDA,
    EMRSFResult,
    analyse_ground_coupling,
    analyse_lr_cv_diagonal,
)
from pyscf_emrsf._memory import patch_pyscf_memory_probe


def test_emrsf_block_structure_weights_observables_and_log(h2o_rohf, tmp_path):
    solver = EMRSFTDA(h2o_rohf, nstates=4)
    assert solver.coupling_scale == solver.lr.exact_exchange
    matrix, live, coupling = solver.build_matrix()

    assert matrix.shape == (
        len(live) + solver.lr.space.size,
        len(live) + solver.lr.space.size,
    )
    assert coupling.shape == (len(live), solver.lr.space.size)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-10)
    assert np.linalg.norm(coupling) > 0.0

    result = solver.kernel()
    assert len(result.eigenvalues) == 4
    assert np.all(result.cv_weights >= -1e-14)
    assert np.all(result.cv_weights <= 1.0 + 1e-14)
    assert len(result.mrsf_result.eigenvalues) == 4
    assert len(result.state_symmetries()) == 4
    matches = result.state_matches()
    assert len(matches) == 4
    assert all(match.matched for match in matches)
    assert sorted(match.mrsf_state for match in matches) == list(range(4))
    assert all(match.status.startswith("MATCHED_TERM") for match in matches)

    descriptors = result.charge_transfer_descriptors(states=[0], grid_level=1)
    descriptor = descriptors[0]
    assert descriptor.q >= 0.0
    assert abs(descriptor.integral_delta_rho) < 2.0e-4
    assert abs(descriptor.reference_electrons_grid - h2o_rohf.mol.nelectron) < 2.0e-3
    assert abs(descriptor.state_electrons_grid - h2o_rohf.mol.nelectron) < 2.0e-3

    lr_cv_audit = analyse_lr_cv_diagonal(solver, batch_size=2)
    coupling_audit = analyse_ground_coupling(
        solver,
        result,
        lr_cv_audit=lr_cv_audit,
    )
    log_path = result.write_log(
        tmp_path / "h2o_v0.4.1.log",
        descriptors=descriptors,
        coupling_audit=coupling_audit,
        lr_cv_audit=lr_cv_audit,
        title="pytest water",
    )
    text = log_path.read_text(encoding="utf-8")
    assert "MATRIX-FREE" in text
    assert "gamma_CV" in text
    assert "COUPLING SCALE c_cp" in text
    assert "G SPIN FACTOR        sqrt(2), fixed" in text
    assert "CO1 TABLE COMBINATION S1-S2" in text
    assert "O2V TABLE COMBINATION S1-S2" in text
    assert "ADDED LR-CV DIAGONAL AUDIT" in text
    assert "A_G FROM MRSF OPERATOR" in text
    assert "SEEDS PER IRREP" in text
    assert "REPRESENTATIVE COMPLETE-MATVEC CHECKS" in text
    assert "c_cp * <MRSF|H|CV>" in text
    assert "S42" in text
    assert "S47" in text
    assert "NORMAL TERMINATION" in text


def test_state_matching_uses_terms_and_leaves_missing_terms_unmatched(monkeypatch):
    a1 = SimpleNamespace(irrep_id=0, label="A1", purity=1.0)
    b1 = SimpleNamespace(irrep_id=2, label="B1", purity=1.0)
    mrsf = SimpleNamespace(
        eigenvectors=np.eye(2),
        eigenvalues=np.array([0.0, 1.0]),
        excitation_energies=np.array([0.0, 1.0]),
        state_symmetries=lambda: (a1, b1),
    )
    result = EMRSFResult(
        eigenvalues=np.array([0.0, 0.5, 0.8]),
        eigenvectors=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        matrix=None,
        mrsf_size=2,
        cv_space=SimpleNamespace(),
        mrsf_result=mrsf,
        live_mrsf_indices=np.array([0, 1]),
    )
    monkeypatch.setattr(
        EMRSFResult,
        "state_symmetries",
        lambda self: (a1, b1, b1),
    )

    matches = result.state_matches()
    assert matches[0].matched
    assert matches[0].mrsf_state == 0
    assert matches[1].matched
    assert matches[1].mrsf_state == 1
    assert matches[1].status == "MATCHED_TERM_LOW_OVERLAP"
    assert not matches[2].matched
    assert matches[2].mrsf_state is None
    assert matches[2].candidate_mrsf_state == 1
    assert matches[2].status == "UNMATCHED_TERM_NOT_COMPUTED"
    assert np.isnan(matches[2].raw_energy_shift_hartree)


def test_empty_cv_sector_reduces_exactly_to_mrsf():
    patch_pyscf_memory_probe()
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mol.incore_anyway = True
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    solver = EMRSFTDA(mf, nstates=2)
    matrix, live, coupling = solver.build_matrix()
    expected = solver.mrsf.build_matrix()[np.ix_(live, live)]

    assert solver.lr.space.size == 0
    assert coupling.shape == (len(live), 0)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_triplet_enlarged_space_dense_davidson_audit_and_log(
    h2o_rohf, tmp_path
):
    solver = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=3,
        nstates=4,
        density_fit=False,
        conv_tol=1.0e-10,
    )
    labels = solver.mrsf.space.labels
    ref = solver.mrsf.ref
    live_labels = {labels[index] for index in solver.live}
    assert (ref.o2, ref.o1) not in live_labels
    assert (ref.o1, ref.o2) not in live_labels

    dense = solver.kernel(solver="dense")
    iterative = solver.kernel(solver="davidson")
    assert dense.target_multiplicity == 3
    np.testing.assert_allclose(
        iterative.eigenvalues, dense.eigenvalues, atol=2.0e-8
    )
    np.testing.assert_allclose(
        iterative.cv_weights, dense.cv_weights, atol=2.0e-7
    )
    assert np.all(iterative.converged)
    assert np.max(iterative.residual_norms) < 1.0e-7

    audit = analyse_lr_cv_diagonal(solver, batch_size=2)
    assert audit.maximum_direct_check_error_hartree < 2.0e-8
    np.testing.assert_allclose(
        audit.ground_offset_recomputed_hartree,
        solver.ground_offset,
        atol=2.0e-12,
    )
    with pytest.raises(ValueError, match="S0 diagnostic"):
        analyse_ground_coupling(solver, dense, lr_cv_audit=audit)

    log = dense.write_log(tmp_path / "triplet-emrsf.log", lr_cv_audit=audit)
    text = log.read_text(encoding="utf-8")
    assert "EMRSF-TDDFT/TDA TRIPLET STATES" in text
    assert "G/D CONFIGURATIONS   absent" in text
    assert "OOS SPIN COMBINATION (L+R)/sqrt(2)" in text
    assert "CO1 TABLE COMBINATION S1+S2" in text
    assert "TARGET MULT.         3" in text
    assert "NORMAL TERMINATION" in text


def test_direct_enlarged_space_skips_mrsf_eigensolve_and_closes_energy(
    h2o_rohf, monkeypatch, tmp_path
):
    solver = EMRSFTDA(
        h2o_rohf,
        nstates=4,
        density_fit=False,
        solver="dense",
        solve_mrsf_first=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("standalone MRSF diagonalisation must not run")

    monkeypatch.setattr(solver.mrsf, "kernel", forbidden)
    result = solver.kernel()
    assert result.direct_emrsf
    assert result.mrsf_result is None
    assert result.integral_backend == "direct-four-centre"
    assert len(result.state_symmetries()) == 4
    assert len(result.state_decompositions) == 4
    assert max(
        abs(value.closure_error_hartree)
        for value in result.state_decompositions
    ) < 2.0e-10
    for value in result.state_decompositions:
        np.testing.assert_allclose(
            value.mrsf_weight + value.cv_weight, 1.0, atol=2.0e-10
        )

    log_path = result.write_log(tmp_path / "direct-emrsf.log")
    text = log_path.read_text(encoding="utf-8")
    assert "DIRECT EMRSF MODE    True" in text
    assert "MRSF PRE-DIAGONAL.   False" in text
    assert "INTEGRAL BACKEND     direct-four-centre" in text
    assert "NOT RUN              direct enlarged-space EMRSF mode" in text
    assert "EMRSF EIGENVALUE BLOCK DECOMPOSITION" in text
    assert "\nMRSF-TDDFT/TDA SINGLET STATES\n" not in text
    assert "NORMAL TERMINATION" in text

    with pytest.raises(RuntimeError, match="solve_mrsf_first"):
        result.state_matches()
    with pytest.raises(RuntimeError, match="solve_mrsf_first"):
        result.write_mrsf_nto_molden(tmp_path / "nto")
