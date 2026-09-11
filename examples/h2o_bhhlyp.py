"""Reproduce the H2O/BHHLYP MRSF benchmark used during development."""

from pyscf import gto

from pyscf_emrsf import MRSFTDA
from pyscf_emrsf.reference import build_reference


mol = gto.M(
    atom="""
    O  0.000000000  0.000000000 -0.041061554
    H -0.533194329  0.533194329 -0.614469223
    H  0.533194329 -0.533194329 -0.614469223
    """,
    basis="6-31g*",
    spin=2,
    unit="Angstrom",
)

mf = build_reference(mol, xc="bhandhlyp", conv_tol=1e-11)
result = MRSFTDA(mf, target_multiplicity=1, nstates=3).kernel()

print("Triplet ROKS reference / Hartree:", mf.e_tot)
print("MRSF state-interaction roots / Hartree:", result.eigenvalues)
print("MRSF state-interaction roots / eV:", result.energies_ev)
print("Physical excitations relative to S0 / eV:", result.excitation_energies_ev)

