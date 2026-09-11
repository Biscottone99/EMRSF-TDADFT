from __future__ import annotations

import numpy as np
from pyscf import ao2mo, gto

from pyscf_emrsf.lr import CorrectedLRTDA
from pyscf_emrsf.reference import analyse_reference, build_reference


def test_hf_cv_block_has_cis_singlet_diagonal(h2o_rohf):
    ref = analyse_reference(h2o_rohf)
    lr = CorrectedLRTDA(ref)
    matrix = lr.build_matrix()
    eri = ao2mo.restore(1, ao2mo.kernel(h2o_rohf.mol, ref.coeff), ref.nmo)

    for k, (i, a) in enumerate(lr.space.labels):
        expected = (
            lr.fock_mo[a, a]
            - lr.fock_mo[i, i]
            + 2.0 * eri[i, a, i, a]
            - eri[i, i, a, a]
        )
        np.testing.assert_allclose(matrix[k, k], expected, atol=2e-9)


def test_hf_cv_block_has_cis_triplet_kernel(h2o_rohf):
    ref = analyse_reference(h2o_rohf)
    lr = CorrectedLRTDA(ref, target_multiplicity=3)
    matrix = lr.build_matrix()
    eri = ao2mo.restore(1, ao2mo.kernel(h2o_rohf.mol, ref.coeff), ref.nmo)

    expected = np.zeros_like(matrix)
    for row, (i, a) in enumerate(lr.space.labels):
        for column, (j, b) in enumerate(lr.space.labels):
            expected[row, column] = -eri[i, j, b, a]
            if i == j:
                expected[row, column] += lr.fock_mo[a, b]
            if a == b:
                expected[row, column] -= lr.fock_mo[i, j]
    np.testing.assert_allclose(matrix, expected, atol=2e-9)


def test_cv_lr_matrix_is_symmetric(h2o_rohf):
    ref = analyse_reference(h2o_rohf)
    matrix = CorrectedLRTDA(ref).build_matrix()
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-10)


def test_cv_diagonal_component_audit_closes_and_recovers_hf_kernel(h2o_rohf):
    ref = analyse_reference(h2o_rohf)
    lr = CorrectedLRTDA(ref)
    components = lr.diagonal_components(batch_size=2)
    matrix = lr.build_matrix()
    eri = ao2mo.restore(1, ao2mo.kernel(h2o_rohf.mol, ref.coeff), ref.nmo)

    np.testing.assert_allclose(
        components["lr_diagonal"], np.diag(matrix), atol=2e-9
    )
    np.testing.assert_allclose(components["s30"], 0.0, atol=2e-12)
    np.testing.assert_allclose(components["xc_kernel"], 0.0, atol=2e-9)
    for index, (i, a) in enumerate(lr.space.labels):
        np.testing.assert_allclose(
            components["coulomb"][index],
            2.0 * eri[i, a, i, a],
            atol=2e-9,
        )
        np.testing.assert_allclose(
            components["exact_exchange"][index],
            -eri[i, i, a, a],
            atol=2e-9,
        )


def test_bhhlyp_cv_diagonal_audit_separates_s30_and_xc_kernel():
    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mol.incore_anyway = True
    mf = build_reference(mol, xc="bhandhlyp", conv_tol=1e-10, verbose=0)
    lr = CorrectedLRTDA(analyse_reference(mf))
    components = lr.diagonal_components(batch_size=2)

    assert np.linalg.norm(components["s30"]) > 1.0e-8
    assert np.linalg.norm(components["xc_kernel"]) > 1.0e-8
    np.testing.assert_allclose(
        components["lr_diagonal"],
        np.diag(lr.build_matrix()),
        atol=2e-8,
    )

    triplet = CorrectedLRTDA(
        analyse_reference(mf), target_multiplicity=3
    )
    triplet_components = triplet.diagonal_components(batch_size=2)
    np.testing.assert_allclose(triplet.exact_exchange, 0.5, atol=1e-14)
    np.testing.assert_allclose(triplet_components["coulomb"], 0.0, atol=0.0)
    np.testing.assert_allclose(triplet_components["xc_kernel"], 0.0, atol=0.0)
    assert np.linalg.norm(triplet_components["exact_exchange"]) > 1.0e-8
    np.testing.assert_allclose(
        triplet_components["lr_diagonal"],
        np.diag(triplet.build_matrix()),
        atol=2e-8,
    )
