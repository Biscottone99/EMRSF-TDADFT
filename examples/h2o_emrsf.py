"""Small PySCF-only EMRSF example; intended as a smoke test, not a benchmark."""

from pyscf import gto

from pyscf_emrsf import EMRSFTDA, MRSFTDA
from pyscf_emrsf.reference import build_reference


mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="sto-3g",
    spin=2,
    verbose=0,
)
mf = build_reference(mol, xc="bhandhlyp", density_fit=True, verbose=0)

mrsf = MRSFTDA(mf, nstates=4).kernel()
emrsf = EMRSFTDA(mf, nstates=4).kernel()

print("MRSF roots / hartree: ", mrsf.eigenvalues)
print("EMRSF roots / hartree:", emrsf.eigenvalues)
print("EMRSF excitations / eV:", emrsf.excitation_energies_ev)
print("Added LR C->V weights: ", emrsf.cv_weights)
print("Davidson residual norms:", emrsf.residual_norms)
