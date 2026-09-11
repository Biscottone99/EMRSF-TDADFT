"""Compare iterative EMRSF solvers on a reproducible small benchmark."""

from time import perf_counter

from pyscf import gto

from pyscf_emrsf import EMRSFTDA
from pyscf_emrsf.reference import build_reference


mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g",
    spin=2,
    verbose=0,
)
mf = build_reference(mol, density_fit=True, verbose=0)
method = EMRSFTDA(mf, nstates=3, conv_tol=1e-9, max_cycle=100)

print(f"EMRSF response dimension: {method.dimension}")
for name in ("davidson", "lobpcg"):
    start = perf_counter()
    result = method.kernel(solver=name)
    elapsed = perf_counter() - start
    print(
        f"{name:8s} time={elapsed:8.3f} s  iterations={result.iterations:3d} "
        f"matvecs={result.matvecs:4d}  max|r|={max(result.residual_norms):.3e}"
    )
    print(" roots / Eh:", result.eigenvalues)

