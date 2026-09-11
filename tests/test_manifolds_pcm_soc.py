from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import scf

from pyscf_emrsf import (
    EMRSFTDA,
    MRSFTDA,
    PCMConfig,
    build_reference,
    compute_emrsf_soc,
    compute_external_soc,
    compute_mrsf_soc,
    save_soc_archive,
)
from pyscf_emrsf.soc import (
    _emrsf_wavefunction,
    _mrsf_wavefunction,
    _spinor_operator,
)
from pyscf_emrsf.solvent import attach_pcm, pcm_metadata


def test_triplet_manifold_is_computed_and_logged(h2o_rohf, tmp_path):
    result = MRSFTDA(
        h2o_rohf,
        target_multiplicity=3,
        nstates=2,
        density_fit=False,
        solver="dense",
    ).kernel()
    assert result.target_multiplicity == 3
    assert len(result.eigenvalues) == 2
    assert np.all(np.diff(result.eigenvalues) >= 0.0)

    output = result.write_log(tmp_path / "triplets.log")
    text = output.read_text(encoding="utf-8")
    assert "MRSF-TDDFT/TDA TRIPLET STATES" in text
    assert "TARGET MULT.         3" in text


def test_pcm_configuration_and_attachment(h2o_rohf):
    config = PCMConfig(
        epsilon=8.93,
        method="iefpcm",
        lebedev_order=17,
        conv_tol=1.0e-8,
    )
    wrapped = attach_pcm(scf.ROHF(h2o_rohf.mol), config)
    assert wrapped.with_solvent.method == "IEF-PCM"
    assert wrapped.with_solvent.eps == pytest.approx(8.93)
    assert wrapped.with_solvent.lebedev_order == 17
    assert pcm_metadata(wrapped)["scope"] == (
        "self-consistent reference; frozen in response"
    )


def test_reference_pcm_is_solved_self_consistently(h2o_rohf):
    mf = build_reference(
        h2o_rohf.mol,
        xc=None,
        conv_tol=1.0e-10,
        max_cycle=100,
        scf_strategy="standard",
        pcm=PCMConfig(epsilon=8.93, lebedev_order=17),
        verbose=0,
    )
    assert mf.converged
    assert mf.with_solvent.v is not None
    assert np.linalg.norm(mf.with_solvent.v) > 0.0
    assert mf.emrsf_reference_metadata["pcm"]["epsilon"] == pytest.approx(8.93)


@pytest.mark.parametrize("epsilon", [1.0, 0.0, -2.0])
def test_pcm_rejects_nonpolarizable_epsilon(epsilon):
    with pytest.raises(ValueError, match="greater than 1"):
        PCMConfig(epsilon=epsilon)


def test_pcm_rejects_unsupported_lebedev_order(h2o_rohf):
    config = PCMConfig(epsilon=8.93, lebedev_order=18)
    with pytest.raises(ValueError, match="Unsupported PCM Lebedev"):
        attach_pcm(scf.ROHF(h2o_rohf.mol), config)


def test_spinor_soc_operator_is_hermitian():
    spatial = np.zeros((3, 3, 3), dtype=complex)
    spatial[0, 0, 1] = 2j
    spatial[0, 1, 0] = -2j
    spatial[1, 1, 2] = 3j
    spatial[1, 2, 1] = -3j
    spatial[2, 0, 2] = 5j
    spatial[2, 2, 0] = -5j
    operator = _spinor_operator(spatial)
    np.testing.assert_allclose(operator, operator.T.conj(), atol=1.0e-14)


def test_soc_mrsf_one_electron_smoke(h2o_rohf):
    singlets = MRSFTDA(
        h2o_rohf,
        target_multiplicity=1,
        nstates=2,
        density_fit=False,
        solver="dense",
    ).kernel()
    triplets = MRSFTDA(
        h2o_rohf,
        target_multiplicity=3,
        nstates=2,
        density_fit=False,
        solver="dense",
    ).kernel()
    result = compute_mrsf_soc(singlets, triplets, two_electron="none")
    assert result.soc_matrix_hartree.shape == (8, 8)
    assert result.hamiltonian_hartree.shape == (8, 8)
    assert len(result.couplings()) == 4
    assert np.isfinite(result.eigenvalues_hartree).all()
    assert result.hermiticity_error_hartree < 1.0e-10
    np.testing.assert_allclose(
        result.hamiltonian_hartree,
        result.hamiltonian_hartree.T.conj(),
        atol=1.0e-12,
    )
    for root in range(2):
        singlet_wf, singlet_excluded = _mrsf_wavefunction(singlets, root, 0)
        triplet_wf, triplet_excluded = _mrsf_wavefunction(triplets, root, 1)
        assert sum(abs(value) ** 2 for value in singlet_wf.values()) == pytest.approx(
            1.0 - singlet_excluded, abs=1.0e-12
        )
        assert sum(abs(value) ** 2 for value in triplet_wf.values()) == pytest.approx(
            1.0 - triplet_excluded, abs=1.0e-12
        )
    for screening in ("amfi", "full"):
        screened = compute_mrsf_soc(
            singlets, triplets, two_electron=screening
        )
        assert np.isfinite(screened.hamiltonian_hartree).all()
        assert screened.hermiticity_error_hartree < 1.0e-10


def test_soc_requires_identical_reference(h2o_rohf):
    singlets = MRSFTDA(
        h2o_rohf, target_multiplicity=1, nstates=1, solver="dense"
    ).kernel()
    copied = SimpleNamespace(**h2o_rohf.__dict__)
    copied.mol = h2o_rohf.mol
    copied.mo_coeff = h2o_rohf.mo_coeff
    copied.mo_occ = h2o_rohf.mo_occ
    copied.mo_energy = h2o_rohf.mo_energy
    copied.make_rdm1 = h2o_rohf.make_rdm1
    copied.get_fock = h2o_rohf.get_fock
    copied.get_hcore = h2o_rohf.get_hcore
    copied.get_veff = h2o_rohf.get_veff
    copied.get_ovlp = h2o_rohf.get_ovlp
    copied.get_grad = h2o_rohf.get_grad
    copied.max_memory = h2o_rohf.max_memory
    copied.verbose = h2o_rohf.verbose
    # The public guard fires before any integral work.
    triplets = MRSFTDA(
        h2o_rohf, target_multiplicity=3, nstates=1, solver="dense"
    ).kernel()
    object.__setattr__(triplets.space.ref, "mf", copied)
    with pytest.raises(ValueError, match="identical mean-field"):
        compute_mrsf_soc(singlets, triplets, two_electron="none")


def test_emrsf_soc_includes_lr_cv_and_exports_external_archive(
    h2o_rohf, tmp_path
):
    singlets = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=1,
        nstates=2,
        density_fit=False,
        solver="dense",
    ).kernel()
    triplets = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=3,
        nstates=2,
        density_fit=False,
        solver="dense",
    ).kernel()
    result = compute_emrsf_soc(singlets, triplets, two_electron="none")
    assert result.source_model == "EMRSF"
    assert result.soc_matrix_hartree.shape == (8, 8)
    assert result.hermiticity_error_hartree < 1.0e-10
    np.testing.assert_allclose(
        result.included_singlet_lr_cv_weight, singlets.cv_weights, atol=1.0e-12
    )
    np.testing.assert_allclose(
        result.included_triplet_lr_cv_weight, triplets.cv_weights, atol=1.0e-12
    )
    for root in range(2):
        wavefunction, excluded, included = _emrsf_wavefunction(
            singlets, root, 0
        )
        assert included == pytest.approx(singlets.cv_weights[root], abs=1.0e-12)
        assert sum(abs(value) ** 2 for value in wavefunction.values()) == pytest.approx(
            1.0 - excluded, abs=1.0e-11
        )

    archive = save_soc_archive(tmp_path / "emrsf_soc.npz", result)
    with np.load(archive, allow_pickle=False) as saved:
        assert saved["source_model"].item() == "EMRSF"
        assert saved["archive_format_version"].item() == 3
        assert saved["singlet_response_eigenvectors_columns_are_states"].shape == (
            singlets.eigenvectors.shape
        )
        assert saved["triplet_response_eigenvectors_columns_are_states"].shape == (
            triplets.eigenvectors.shape
        )
        assert saved["determinant_state_coefficients"].shape[1] == 8
        assert "one_electron_soc_mo" in saved
        assert "somf_two_electron_soc_mo" in saved

    screened = compute_emrsf_soc(singlets, triplets, two_electron="amfi")
    assert np.isfinite(screened.two_electron_matrix_hartree).all()
    np.testing.assert_allclose(
        screened.soc_matrix_hartree,
        screened.one_electron_matrix_hartree
        + screened.two_electron_matrix_hartree,
        atol=1.0e-14,
    )


def test_external_soc_module_can_reproduce_internal_contraction(
    h2o_rohf, tmp_path
):
    singlets = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=1,
        nstates=1,
        density_fit=False,
        solver="dense",
    ).kernel()
    triplets = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=3,
        nstates=1,
        density_fit=False,
        solver="dense",
    ).kernel()
    module = tmp_path / "external_soc.py"
    module.write_text(
        "def compute_soc(context):\n"
        "    return {\n"
        "        'one_electron_matrix_hartree': "
        "context.contract_one_body(context.integrals.one_electron_mo),\n"
        "        'two_electron_matrix_hartree': "
        "context.contract_one_body(context.integrals.two_electron_mo),\n"
        "        'approximation': 'external regression module',\n"
        "    }\n",
        encoding="utf-8",
    )
    internal = compute_emrsf_soc(singlets, triplets, two_electron="none")
    external = compute_external_soc(
        module, singlets, triplets, two_electron="none"
    )
    np.testing.assert_allclose(
        external.soc_matrix_hartree, internal.soc_matrix_hartree, atol=1.0e-14
    )
    assert external.approximation == "external regression module"


def test_configuration_labelled_eigenvector_csv(h2o_rohf, tmp_path):
    result = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=1,
        nstates=1,
        density_fit=False,
        solver="dense",
    ).kernel()
    output = result.write_eigenvectors_csv(tmp_path / "vectors.csv")
    text = output.read_text(encoding="utf-8")
    assert "basis_row_zero_based" in text
    assert "LR-CV" in text
    assert len(text.splitlines()) == 1 + result.eigenvectors.shape[0]
