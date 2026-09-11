from __future__ import annotations

import numpy as np

from pyscf_emrsf import MRSFTDA


def test_explicit_matrix_is_finite_and_symmetric(h2o_rohf):
    solver = MRSFTDA(h2o_rohf, target_multiplicity=1, nstates=3)
    matrix = solver.build_matrix()
    assert np.isfinite(matrix).all()
    assert np.max(np.abs(matrix - matrix.T)) < 1.0e-12


def test_kernel_returns_ordered_roots(h2o_rohf):
    result = MRSFTDA(h2o_rohf, target_multiplicity=1, nstates=3).kernel()
    assert len(result.eigenvalues) == 3
    assert np.all(np.diff(result.eigenvalues) >= 0.0)
    assert result.excitation_energies[0] == 0.0

