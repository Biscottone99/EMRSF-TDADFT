from __future__ import annotations

import pytest
from pyscf import gto, scf

from pyscf_emrsf._memory import patch_pyscf_memory_probe


@pytest.fixture(scope="session")
def h2o_rohf():
    patch_pyscf_memory_probe()
    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mol.incore_anyway = True
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    assert mf.converged
    return mf

