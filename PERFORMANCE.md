# Matrix-free and RI design

## Why v0.2.1 was too slow for the paper molecules

The dense validation path applies the MRSF and LR response kernels once per
matrix column.  For the azulene/cc-pVDZ space this led to a run exceeding 21
hours before any result file was written.  The v0.4.1 default instead applies
the operator only to the trial vectors requested by restarted Davidson, prints
each iteration and never allocates the dense response matrix.  The four-index
coupling setup is replaced by the blockwise RI operator described below.

## Why Davidson is the default

The TDA response matrix is real symmetric and only a small number of its lowest
roots is normally required.  Its orbital-energy diagonal is also a useful,
although imperfect, preconditioner.  This is the regime for which restarted
Davidson is designed.

The implementation oversamples the initial configuration guesses and uses a
stricter residual criterion than the PySCF default.  Both choices are important:
the MRSF ground configuration and the added EMRSF C->V sector can reorder states
relative to the orbital-energy diagonal.

LOBPCG is available as an independent iterative comparison.  On the included
H2O/6-31G EMRSF test (three roots), representative results were:

| Solver | Matvecs | Iterations | Same three roots? |
| --- | ---: | ---: | --- |
| Davidson | 41 | 15 | yes |
| LOBPCG | 312 | 102 | yes |

Wall times depend strongly on BLAS, thread count, disk and the PySCF build, so
the operation counts are the more transferable comparison.

## Resolution of the identity

RI is used in two places:

1. batched J/K builds for the seven MRSF response densities and the LR/CV
   transition densities;
2. construction of the EMRSF coupling intermediates.

The coupling setup streams the whitened AO three-index tensors by auxiliary
block.  A temporary C/V block is transformed, contracted immediately, and
discarded.  The retained objects scale as

`O(nC*nV + nC^2 + nV^2)`

rather than `O(naux*nC*nV)` or `O(nMO^4)`.  The AO three-index data are owned by
PySCF and may be kept out of core.

Call `EMRSFTDA.resource_estimate()` before a large calculation.  It reports the
dense matrix, Davidson vector, retained coupling and AO three-index storage
estimates separately.

## Why there is no custom Fortran kernel yet

The expensive operations already enter compiled code:

- AO integrals and DF J/K: PySCF/Libcint;
- tensor contractions: NumPy linked to BLAS;
- small projected eigensystems: SciPy LAPACK.

The remaining Python work per matvec consists mainly of assembling a small
number of AO transition densities and dispatching batched contractions.  A
custom F90 extension would add compilation and ABI complexity without removing
the current dominant cost.  It becomes justified only if profiling a realistic
carbazole--5AP calculation shows a residual tensor loop—not J/K or XC
quadrature—to dominate.  The clean insertion points would then be the RI setup
in `ri.py` and batched response assembly in `space.py`; the scientific operator
and tests would remain unchanged.
