from __future__ import annotations

import numpy as np
from pyscf import ao2mo

from pyscf_emrsf import (
    EMRSFTDA,
    HighAccuracyMRSFTDA,
    MRSFTDA,
    analyse_ground_coupling,
    analyse_lr_cv_diagonal,
)
from pyscf_emrsf.coupling import build_coupling
from pyscf_emrsf.solver import davidson_lowest


def test_symmetry_complete_davidson_seeds_recover_low_unseeded_block():
    # A globally chosen diagonal subspace would contain only irrep 0 here,
    # although mixing in irrep 1 creates the true lowest eigenvalue.
    matrix = np.diag(np.arange(12, dtype=float))
    matrix[10, 10] = 10.0
    matrix[11, 11] = 10.0
    matrix[10, 11] = matrix[11, 10] = -20.0
    diagonal = np.diag(matrix).copy()
    irreps = np.array([0] * 10 + [1, 1])

    result = davidson_lowest(
        matrix.__matmul__,
        diagonal,
        nroots=2,
        tol=1.0e-11,
        residual_tol=1.0e-9,
        basis_irreps=irreps,
    )

    np.testing.assert_allclose(result.eigenvalues, [-10.0, 0.0], atol=1e-10)
    assert dict(result.symmetry_seed_counts)[1] == 2


def test_mrsf_davidson_matches_dense(h2o_rohf):
    solver = MRSFTDA(h2o_rohf, nstates=4, conv_tol=1e-9)
    dense = solver.kernel(solver="dense")
    iterative = solver.kernel(solver="davidson")

    np.testing.assert_allclose(iterative.eigenvalues, dense.eigenvalues, atol=2e-8)
    assert iterative.matrix is None
    assert np.all(iterative.converged)
    assert np.max(iterative.residual_norms) < 1e-7


def test_high_accuracy_mrsf_keeps_operator_and_tightens_residual(h2o_rohf):
    ordinary = MRSFTDA(h2o_rohf, nstates=4, density_fit=False)
    tightened = HighAccuracyMRSFTDA(
        h2o_rohf,
        nstates=4,
        density_fit=False,
        conv_tol=1.0e-11,
        residual_tol=1.0e-9,
        max_cycle=160,
    )
    assert HighAccuracyMRSFTDA.matvec is MRSFTDA.matvec
    rng = np.random.default_rng(502)
    vector = rng.normal(size=ordinary.space.size)
    np.testing.assert_allclose(
        tightened.matvec(vector), ordinary.matvec(vector), atol=1.0e-12, rtol=0.0
    )

    dense = ordinary.kernel(solver="dense")
    result = tightened.kernel()
    np.testing.assert_allclose(result.eigenvalues, dense.eigenvalues, atol=2.0e-9)
    assert np.all(result.converged)
    assert np.max(result.residual_norms) <= 1.1e-9


def test_emrsf_davidson_matches_dense(h2o_rohf):
    solver = EMRSFTDA(h2o_rohf, nstates=4, conv_tol=1e-9)
    resources = solver.resource_estimate()
    assert resources.response_dimension == solver.dimension
    assert resources.dense_matrix_gib > 0.0
    assert resources.ao_three_index_gib is not None
    dense = solver.kernel(solver="dense")
    iterative = solver.kernel(solver="davidson")

    np.testing.assert_allclose(iterative.eigenvalues, dense.eigenvalues, atol=2e-8)
    np.testing.assert_allclose(iterative.cv_weights, dense.cv_weights, atol=2e-7)
    assert iterative.matrix is None
    assert np.all(iterative.converged)


def test_ri_coupling_adjoint_and_exact_limit(h2o_rohf):
    # Use a non-unit c_cp even though the fixture is ROHF.  This makes the
    # RI-versus-four-index comparison sensitive to complete-block scaling.
    solver = EMRSFTDA(h2o_rohf, nstates=3, coupling_scale=0.5)
    rng = np.random.default_rng(19)
    x = rng.normal(size=solver.lr.space.size)
    y = rng.normal(size=solver.mrsf.space.size)

    lhs = np.dot(y, solver.coupling.matvec(x))
    rhs = np.dot(solver.coupling.rmatvec(y), x)
    np.testing.assert_allclose(lhs, rhs, atol=2e-12)

    exact = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=solver.coupling_scale,
    )
    eye = np.eye(solver.lr.space.size)
    ri = np.column_stack(
        [solver.coupling.matvec(eye[:, k]) for k in range(solver.lr.space.size)]
    )
    np.testing.assert_allclose(ri, exact, atol=5e-5)


def test_direct_coupling_is_exact_and_adjoint(h2o_rohf):
    solver = EMRSFTDA(
        h2o_rohf,
        nstates=3,
        coupling_scale=0.37,
        density_fit=False,
    )
    assert solver.integral_backend == "direct-four-centre"
    exact = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=solver.coupling_scale,
    )
    eye = np.eye(solver.lr.space.size)
    direct = np.column_stack(
        [solver.coupling.matvec(eye[:, k]) for k in range(eye.shape[1])]
    )
    np.testing.assert_allclose(direct, exact, atol=3.0e-11, rtol=0.0)

    rng = np.random.default_rng(1701)
    x = rng.normal(size=solver.lr.space.size)
    y = rng.normal(size=solver.mrsf.space.size)
    np.testing.assert_allclose(
        np.dot(y, solver.coupling.matvec(x)),
        np.dot(solver.coupling.rmatvec(y), x),
        atol=3.0e-11,
        rtol=0.0,
    )
    forward = solver.coupling.matvec_components(x)
    reverse = solver.coupling.rmatvec_components(y)
    for name in ("fock", "two_electron"):
        np.testing.assert_allclose(
            np.dot(y, forward[name]),
            np.dot(reverse[name], x),
            atol=3.0e-11,
            rtol=0.0,
        )


def test_ri_coupling_component_decomposition_and_adjoint(h2o_rohf):
    solver = EMRSFTDA(h2o_rohf, nstates=3, coupling_scale=0.5)
    rng = np.random.default_rng(23)
    x = rng.normal(size=solver.lr.space.size)
    y = rng.normal(size=solver.mrsf.space.size)

    forward = solver.coupling.matvec_components(x)
    reverse = solver.coupling.rmatvec_components(y)
    np.testing.assert_allclose(
        forward["fock"] + forward["two_electron"],
        solver.coupling.matvec(x),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        reverse["fock"] + reverse["two_electron"],
        solver.coupling.rmatvec(y),
        atol=2e-12,
    )
    for name in ("fock", "two_electron"):
        np.testing.assert_allclose(
            np.dot(y, forward[name]),
            np.dot(reverse[name], x),
            atol=2e-12,
        )


def test_ground_coupling_audit_partitions_the_diagonal_estimate(h2o_rohf):
    solver = EMRSFTDA(h2o_rohf, nstates=4, coupling_scale=0.5)
    result = solver.kernel(solver="dense")
    lr_cv_audit = analyse_lr_cv_diagonal(solver, batch_size=2)
    audit = analyse_ground_coupling(
        solver,
        result,
        lr_cv_audit=lr_cv_audit,
    )

    assert audit.coupling_norm > 0.0
    assert len(audit.families) == 8
    np.testing.assert_allclose(
        sum(value.mrsf_weight for value in audit.families),
        1.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        sum(value.diagonal_partition_shift_hartree for value in audit.families),
        audit.diagonal_second_order_shift_hartree,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        audit.diagonal_fock_partition_hartree
        + audit.diagonal_two_electron_partition_hartree,
        audit.diagonal_second_order_shift_hartree,
        atol=2e-12,
    )


def test_lr_cv_audit_checks_s30_kernel_and_single_ground_offset(h2o_rohf):
    solver = EMRSFTDA(h2o_rohf, nstates=3, coupling_scale=0.5)
    audit = analyse_lr_cv_diagonal(solver, batch_size=2)
    matrix, live, _ = solver.build_matrix()
    cv_diagonal = np.diag(matrix)[len(live) :]

    assert audit.labels == tuple(solver.lr.space.labels)
    assert audit.apply_fock_correction
    assert audit.batch_size == 2
    np.testing.assert_allclose(
        audit.emrsf_diagonal_hartree, cv_diagonal, atol=5e-8
    )
    np.testing.assert_allclose(
        audit.ground_offset_recomputed_hartree,
        solver.ground_offset,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        solver.ground_offset_operator,
        solver.ground_offset,
        atol=2e-12,
    )
    assert audit.maximum_direct_check_error_hartree < 5e-8


def test_coupling_scale_multiplies_the_complete_block(h2o_rohf):
    """The main-text c_cp factor acts on Fock and two-electron terms alike."""

    solver = EMRSFTDA(h2o_rohf, nstates=3, coupling_scale=1.0)
    raw = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=1.0,
    )
    half = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=0.5,
    )
    zero = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=0.0,
    )

    np.testing.assert_allclose(half, 0.5 * raw, atol=2e-13)
    np.testing.assert_allclose(zero, 0.0, atol=0.0)


def test_ri_coupling_vanishes_at_zero_scale(h2o_rohf):
    solver = EMRSFTDA(h2o_rohf, nstates=3, coupling_scale=0.0)
    rng = np.random.default_rng(31)
    x = rng.normal(size=solver.lr.space.size)
    y = rng.normal(size=solver.mrsf.space.size)

    np.testing.assert_allclose(solver.coupling.matvec(x), 0.0, atol=0.0)
    np.testing.assert_allclose(solver.coupling.rmatvec(y), 0.0, atol=0.0)


def test_selected_si_coupling_equations_include_c_cp_on_fock_terms(h2o_rohf):
    """Check G, CO1(p=r), and O2V(q=s) against SI Tables S1--S3."""

    scale = 0.37
    solver = EMRSFTDA(h2o_rohf, nstates=3, coupling_scale=scale)
    ref = solver.mrsf.ref
    matrix = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=scale,
    )
    r = int(ref.closed[0])
    s = int(ref.virtual[0])
    c = ref.coeff

    def eri(p, q, u, v):
        value = ao2mo.general(
            ref.mf.mol,
            (c[:, [p]], c[:, [q]], c[:, [u]], c[:, [v]]),
            compact=False,
        )
        return float(np.asarray(value).reshape(-1)[0])

    jl = solver.lr.space.labels.index((r, s))

    i_g = solver.mrsf.space.labels.index((ref.o2, ref.o1))
    expected_g = scale * np.sqrt(2.0) * solver.lr.fock_mo[r, s]
    np.testing.assert_allclose(matrix[i_g, jl], expected_g, atol=2e-12)

    i_co1 = solver.mrsf.space.labels.index((r, ref.o1))
    expected_co1 = scale * (
        -solver.lr.fock_mo[ref.o2, s] - eri(r, s, ref.o2, r)
    )
    np.testing.assert_allclose(matrix[i_co1, jl], expected_co1, atol=2e-12)

    i_o2v = solver.mrsf.space.labels.index((ref.o2, s))
    expected_o2v = scale * (
        eri(r, s, s, ref.o1) - solver.lr.fock_mo[r, ref.o1]
    )
    np.testing.assert_allclose(matrix[i_o2v, jl], expected_o2v, atol=2e-12)


def test_every_si_table_coupling_branch_against_explicit_four_index_rules():
    """Exercise all conditions in SI Tables S1--S3, including p!=r/q!=s."""

    from pyscf import gto
    from pyscf_emrsf.reference import build_reference

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="3-21g",
        spin=2,
        verbose=0,
    )
    mf = build_reference(mol, xc=None, conv_tol=1.0e-11, verbose=0)
    scale = 0.41
    solver = EMRSFTDA(mf, nstates=3, coupling_scale=scale)
    ref = solver.mrsf.ref
    got = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=scale,
    )
    eri = ao2mo.kernel(mol, ref.coeff, compact=False).reshape((ref.nmo,) * 4)
    closed = set(map(int, ref.closed))
    virtual = set(map(int, ref.virtual))
    o1, o2 = ref.o1, ref.o2

    def integral(p, q, r, s):
        return float(eri[p, q, r, s])

    expected = np.zeros_like(got)
    for im, (i, a) in enumerate(solver.mrsf.space.labels):
        for jl, (r, s) in enumerate(solver.lr.space.labels):
            value = 0.0
            if (i, a) == (o2, o1):
                value = np.sqrt(2.0) * solver.lr.fock_mo[r, s]
            elif (i, a) == (o1, o2):
                value = 0.0
            elif (i, a) == (o1, o1):
                value = integral(r, o1, o2, s) - 2.0 * integral(r, s, o2, o1)
            elif i in closed and a == o1:
                if i == r:
                    value = -solver.lr.fock_mo[o2, s] - integral(r, s, o2, i)
                else:
                    value = integral(r, i, s, o2) - 2.0 * integral(r, s, o2, i)
            elif i in closed and a == o2:
                if i == r:
                    value = -integral(o2, o1, o2, s)
            elif i == o1 and a in virtual:
                if a == s:
                    value = integral(r, o1, o2, o1)
            elif i == o2 and a in virtual:
                if a == s:
                    value = integral(r, s, a, o1) - solver.lr.fock_mo[r, o1]
                else:
                    value = 2.0 * integral(r, s, a, o1) - integral(r, o1, a, s)
            elif i in closed and a in virtual:
                if i == r:
                    value -= integral(a, o1, o2, s)
                if a == s:
                    value += integral(r, o1, o2, i)
            expected[im, jl] = scale * value

    np.testing.assert_allclose(got, expected, atol=3.0e-11, rtol=0.0)

    direct_solver = EMRSFTDA(
        mf,
        nstates=3,
        coupling_scale=scale,
        density_fit=False,
    )
    direct_expected = build_coupling(
        direct_solver.mrsf.space,
        direct_solver.lr.space,
        direct_solver.lr.fock_mo,
        coupling_scale=scale,
    )
    eye = np.eye(direct_solver.lr.space.size)
    direct_action = np.column_stack(
        [direct_solver.coupling.matvec(eye[:, k]) for k in range(eye.shape[1])]
    )
    np.testing.assert_allclose(
        direct_action, direct_expected, atol=3.0e-11, rtol=0.0
    )


def test_every_triplet_coupling_branch_against_explicit_four_index_rules():
    """Exercise the S1+S2 triplet extension for every coupling family."""

    from pyscf import gto
    from pyscf_emrsf.reference import build_reference

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="3-21g",
        spin=2,
        verbose=0,
    )
    mf = build_reference(mol, xc=None, conv_tol=1.0e-11, verbose=0)
    scale = 0.43
    solver = EMRSFTDA(
        mf,
        target_multiplicity=3,
        nstates=3,
        coupling_scale=scale,
        density_fit=False,
    )
    ref = solver.mrsf.ref
    got = build_coupling(
        solver.mrsf.space,
        solver.lr.space,
        solver.lr.fock_mo,
        coupling_scale=scale,
    )
    eri = ao2mo.kernel(mol, ref.coeff, compact=False).reshape((ref.nmo,) * 4)
    closed = set(map(int, ref.closed))
    virtual = set(map(int, ref.virtual))
    o1, o2 = ref.o1, ref.o2

    expected = np.zeros_like(got)
    for im, (i, a) in enumerate(solver.mrsf.space.labels):
        for jl, (r, s) in enumerate(solver.lr.space.labels):
            value = 0.0
            if (i, a) == (o1, o1):
                value = eri[r, o1, o2, s]
            elif i in closed and a == o1:
                if i == r:
                    value = -solver.lr.fock_mo[o2, s] + eri[r, s, o2, i]
                else:
                    value = eri[r, i, s, o2]
            elif i in closed and a == o2:
                if i == r:
                    value = -eri[o2, o1, o2, s]
            elif i == o1 and a in virtual:
                if a == s:
                    value = -eri[r, o1, o2, o1]
            elif i == o2 and a in virtual:
                if a == s:
                    value = solver.lr.fock_mo[r, o1] + eri[r, s, a, o1]
                else:
                    value = eri[r, o1, a, s]
            elif i in closed and a in virtual:
                if i == r:
                    value -= eri[a, o1, o2, s]
                if a == s:
                    value -= eri[r, o1, o2, i]
            expected[im, jl] = scale * value

    np.testing.assert_allclose(got, expected, atol=3.0e-11, rtol=0.0)
    eye = np.eye(solver.lr.space.size)
    action = np.column_stack(
        [solver.coupling.matvec(eye[:, column]) for column in range(eye.shape[1])]
    )
    np.testing.assert_allclose(action, expected, atol=3.0e-11, rtol=0.0)


def test_triplet_ri_matches_direct_limit_and_both_are_adjoint(h2o_rohf):
    scale = 0.39
    ri_solver = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=3,
        nstates=3,
        coupling_scale=scale,
    )
    direct_solver = EMRSFTDA(
        h2o_rohf,
        target_multiplicity=3,
        nstates=3,
        coupling_scale=scale,
        density_fit=False,
    )
    eye = np.eye(ri_solver.lr.space.size)
    ri = np.column_stack(
        [ri_solver.coupling.matvec(eye[:, column]) for column in range(eye.shape[1])]
    )
    direct = np.column_stack(
        [direct_solver.coupling.matvec(eye[:, column]) for column in range(eye.shape[1])]
    )
    np.testing.assert_allclose(ri, direct, atol=5.0e-5, rtol=0.0)

    rng = np.random.default_rng(700)
    x = rng.normal(size=ri_solver.lr.space.size)
    y = rng.normal(size=ri_solver.mrsf.space.size)
    for coupling in (ri_solver.coupling, direct_solver.coupling):
        np.testing.assert_allclose(
            np.dot(y, coupling.matvec(x)),
            np.dot(coupling.rmatvec(y), x),
            atol=3.0e-11,
            rtol=0.0,
        )
