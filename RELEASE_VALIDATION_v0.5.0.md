# PySCF-EMRSF v0.5.0 release validation

## Implemented invariants

- `run_high_accuracy.py --method mrsf|emrsf` uses one reference-building path
  and one set of basis, grid, integral and eigensolver controls.
- The default `exact-dz` profile is BH&HLYP/cc-pVDZ, grid level 6 without
  pruning, direct integrals, `conv_tol=1e-11`, and `residual_tol=1e-9`.
- `mrsf.py`, `space.py`, and `kernels.py` have the same SHA-256 digests as in
  v0.4.1. `HighAccuracyMRSFTDA.matvec` is the inherited `MRSFTDA.matvec`.
- Direct EMRSF solves the complete MRSF+CV block problem without first solving
  or modifying the standalone MRSF eigenproblem.
- Literature EMRSF energies and TBEs are read only by post-processors, never by
  the calculation driver or Hamiltonian.

## Verification performed

- 42/42 tests pass from the source tree.
- 42/42 tests pass with `pyscf_emrsf` imported from the built wheel in a clean
  virtual environment.
- Dense and matrix-free eigenpairs, direct and density-fitted actions, all
  coupling branches, adjoints, reference controls, S23/S30, S42--S47, symmetry
  and OpenQP regression tests pass.
- A complete small-molecule smoke calculation passes through both
  `--method mrsf` and `--method emrsf`, writes logs and NPZ archives, and gives
  identical triplet fingerprints for the paired calculations.
- Every shipped Python file compiles and every shipped Slurm/shell script
  passes `bash -n`.

## Production acceptance

The five large CT calculations are intentionally supplied as Slurm jobs rather
than claimed as locally completed production benchmarks. Accept a production
result only when all requested roots converge, residuals satisfy the requested
threshold, state symmetry and character are stable, and the strict paired
comparator accepts the MRSF/EMRSF reference. EMRSF additionally requires the
block expectation-value closure check. A smaller error against the five TBEs
must be reported as a result, never assumed from tighter numerical settings.
