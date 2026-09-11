"""Numerical cross-check against the public OpenQP H2O example.

The two programs use different grids and ROHF/ROKS canonicalisation, so this is
not expected to be bit-identical.  The tolerance is deliberately much tighter
than chemical accuracy but wide enough to cover the known reference difference.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto

from pyscf_emrsf import MRSFTDA
from pyscf_emrsf.reference import build_reference


def test_h2o_bhhlyp_agrees_with_openqp_public_example():
    mol = gto.M(
        atom="""
        O  0.000000000  0.000000000 -0.041061554
        H -0.533194329  0.533194329 -0.614469223
        H  0.533194329 -0.533194329 -0.614469223
        """,
        basis="6-31g*",
        spin=2,
        unit="Angstrom",
        verbose=0,
    )
    mf = build_reference(mol, xc="bhandhlyp", conv_tol=1e-11, verbose=0)
    got = MRSFTDA(mf, target_multiplicity=1, nstates=3).kernel().eigenvalues
    # OpenQP v1.2.1 public example H2O_BHHLYP-MRSFTDDFT_ENERGY
    reference = np.array([-0.28348967234148004, 0.045304512523271154, 0.10356631483242085])
    assert np.allclose(got, reference, atol=1.5e-3, rtol=0.0)

