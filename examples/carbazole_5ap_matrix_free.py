"""Matrix-free EMRSF template for a carbazole--5AP geometry in XYZ format."""

from __future__ import annotations

import argparse

import numpy as np
from pyscf import gto, lib

from pyscf_emrsf import EMRSFTDA
from pyscf_emrsf.reference import build_reference


parser = argparse.ArgumentParser()
parser.add_argument("xyz", help="Carbazole--5AP XYZ geometry")
parser.add_argument("--charge", type=int, default=0)
parser.add_argument("--basis", default="def2-tzvp")
parser.add_argument("--auxbasis", default="def2-universal-jkfit")
parser.add_argument("--xc", default="bhandhlyp")
parser.add_argument("--nstates", type=int, default=10)
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--memory-mb", type=int, default=80000)
parser.add_argument("--output", default="cbz_5ap_emrsf.npz")
args = parser.parse_args()

lib.num_threads(args.threads)
mol = gto.M(
    atom=args.xyz,
    basis=args.basis,
    charge=args.charge,
    spin=2,
    unit="Angstrom",
    max_memory=args.memory_mb,
    verbose=4,
)

mf = build_reference(
    mol,
    xc=args.xc,
    density_fit=True,
    conv_tol=1e-10,
    verbose=4,
)
method = EMRSFTDA(
    mf,
    nstates=args.nstates,
    density_fit=True,
    auxbasis=args.auxbasis,
    solver="davidson",
    conv_tol=1e-9,
    max_cycle=100,
)

print("Resource estimate:", method.resource_estimate())
result = method.kernel()
if not np.all(result.converged):
    raise RuntimeError(
        f"EMRSF Davidson did not converge all roots; residuals={result.residual_norms}"
    )

print("Excitation energies / eV:", result.excitation_energies_ev)
print("Added C->V weights:", result.cv_weights)
print("Residual norms:", result.residual_norms)
np.savez(
    args.output,
    roots_hartree=result.eigenvalues,
    excitation_ev=result.excitation_energies_ev,
    eigenvectors=result.eigenvectors,
    cv_weights=result.cv_weights,
    residual_norms=result.residual_norms,
)

