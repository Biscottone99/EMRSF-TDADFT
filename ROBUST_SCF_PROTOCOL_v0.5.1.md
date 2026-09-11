# Robust triplet-reference SCF protocol (v0.5.1)

## Scope

This protocol changes only the construction of the high-spin, two-open-shell
ROHF/ROKS determinant around which MRSF and EMRSF response theory is evaluated.
It does not modify the response space, matrix action, coupling equations,
exchange scaling, EMRSF diagonal correction, or Davidson solver.

## Convergence ladder

1. **Final-grid CDIIS.** Run the requested Hamiltonian, quadrature grid, basis,
   integral backend and tight thresholds without stabilisers. A supplied
   restricted Molden seed activates MOM immediately.
2. **Oscillation diagnosis.** Retain only the last three densities and detect
   a period-two cycle from a small `P[n]-P[n-2]` together with a much larger
   `P[n]-P[n-1]`.
3. **Stabilised preconvergence.** Restart with ADIIS, damping, virtual-orbital
   level shifting and at most a level-3 pruned DFT grid. If a two-cycle was
   found, its averaged density is used only as the initial guess.
4. **Exact final-grid refinement.** Rebuild a fresh mean-field object on the
   requested final grid. Remove damping and level shift completely, preserve
   the selected two-SOMO occupation with MOM, and run CDIIS at the final
   thresholds.
5. **Newton/CIAH fallback.** If final-grid CDIIS still fails, use PySCF's
   second-order augmented-Hessian solver on the same final Hamiltonian and
   locked occupation.

Only stages 1, 4, or 5 can be accepted. Stage 3 is a numerical preconditioner,
not a production wavefunction.

## Acceptance conditions

- final `mf.converged` is true;
- exactly two singly occupied orbitals exist;
- electron and spin traces are conserved;
- the final orbitals are orthonormal in the AO metric;
- no damping or level shift enters the accepted final Hamiltonian;
- a Molden-seeded run satisfies the requested SOMO and occupied-subspace
  singular-value thresholds;
- the final orbital gradient is reported and values above `1e-4` are warned.

## Provenance

Every MRSF/EMRSF log reports `SCF STRATEGY`, `SCF FINAL STAGE`, rescue and
oscillation flags, density-average use, and one record for every attempt with
converger, cycle count, energy, gradient norm and density change.

## Important physical limitation

The average of two densities can help leave a numerical two-cycle, but is not
generally idempotent. It is never returned as an MRSF reference. If multiple
converged triplet determinants exist, `auto` still cannot prove that it selected
the same physical reference as another program; use a restricted Molden seed
and inspect the reported subspace overlaps.
