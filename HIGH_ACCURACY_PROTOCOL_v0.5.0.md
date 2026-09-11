# PySCF-EMRSF v0.5.0: unified high-accuracy protocol

## Scientific objective

Version 0.5.0 provides one driver with `--method mrsf|emrsf`. The default
`exact-dz` profile removes controllable numerical approximations while keeping
the level of theory fixed at BH&HLYP/cc-pVDZ. Consequently, a paired MRSF and
EMRSF run differs only in its response/configuration space. Neither the five
published EMRSF energies nor the TBEs enter either Hamiltonian.

The TBEs of Loos *et al.* are high-level wave-function reference estimates with
basis corrections extending beyond double-zeta, in several cases through
aug-cc-pVQZ. They are not EMRSF parameters and are never read by the calculation
driver. See [Reference Energies for Intramolecular Charge-Transfer
Excitations](https://hal.science/hal-03160172v2/document).

## What is unchanged

The following v0.4.1 source modules are intentionally unchanged:

- `mrsf.py`: MRSF matrix action and standalone solver;
- `space.py`: spin-adapted MRSF basis and redundant-vector mask;
- `kernels.py`: seven AO response components and spin-pair scaling.

The MRSF block is still present in the EMRSF Hamiltonian. Direct mode skips only
its separate eigendecomposition:

\[
\begin{pmatrix}
A_{\mathrm{MRSF}} & c_H C\\
c_H C^\mathrm{T} & A_{\mathrm{CV}}' + A_G I
\end{pmatrix}
X_K=\omega_K X_K.
\]

No MRSF eigenvalue is computed or corrected before this enlarged-space solve.
Standalone high-accuracy MRSF inherits the same matrix action; the new wrapper
only passes the stricter residual threshold to Davidson.

## New numerical controls

1. `--method mrsf|emrsf` selects the response space without changing the
   reference, basis, grid, integral approximation, root count or tolerances.
2. `solve_mrsf_first=False` directly diagonalizes the enlarged EMRSF operator.
3. `density_fit=False` selects exact four-centre J/K contractions and the new
   `DirectCouplingOperator`.
4. The direct coupling retains only the ten MO-integral blocks that occur in SI
   Tables S1--S3; it does not assemble the complete rectangular coupling matrix.
5. Davidson eigenvalue and residual tolerances are independently controlled.
6. The default production grids are unpruned and the DFT small-density cutoff is
   tightened.
7. A cc-pVDZ OpenQP MOM seed can be projected into a larger AO basis. The
   projected occupied and SOMO subspaces are reorthonormalized and checked after
   SCF.
8. Every output stores a reference fingerprint. The paired comparator rejects
   unequal theory settings, triplet energies or AO densities.
9. Every EMRSF eigenvalue is independently reconstructed as

   \[
   \langle X_M|A_M|X_M\rangle+
   \langle X_C|A_C+A_GI|X_C\rangle+
   2\langle X_M|C|X_C\rangle.
   \]

## Predefined profiles

| Profile | Basis | Grid | Roots | Purpose |
|---|---|---:|---:|---|
| `exact-dz` (default) | cc-pVDZ | 6, unpruned | 16 | Equal-level MRSF/EMRSF comparison and numerical-limit check |
| `tbe-dz` | aug-cc-pVDZ | 7, unpruned | 20 | Measure the effect of diffuse functions on CT states |
| `tbe-tz` | aug-cc-pVTZ | 7, unpruned | 24 | Diffuse triple-zeta production calculation |

`tbe-dz` and `tbe-tz` are *TBE-oriented*, not TBE-fitted. A larger basis can
move an excitation either toward or away from the reference because the
remaining functional/model error is not variational in an excitation energy.

## Recommended sequence

Install and validate:

```bash
python -m pip install -e '.[test]'
pytest -q
```

Run the equal-level pair first. Both jobs use exactly the same predefined
settings and state-locked reference protocol:

```bash
cd examples/paper_ct_benchmark
METHOD=mrsf  PROFILE=exact-dz sbatch run_high_accuracy_v050.slurm
METHOD=emrsf PROFILE=exact-dz sbatch run_high_accuracy_v050.slurm
```

Check that the selected integration grid is converged without changing any
other setting (use separate output directories automatically):

```bash
METHOD=emrsf PROFILE=exact-dz SCF_GRID_LEVEL=5 sbatch run_high_accuracy_v050.slurm
METHOD=emrsf PROFILE=exact-dz SCF_GRID_LEVEL=6 sbatch run_high_accuracy_v050.slurm
METHOD=emrsf PROFILE=exact-dz SCF_GRID_LEVEL=7 sbatch run_high_accuracy_v050.slurm
```

The final excitation should be stable between levels 6 and 7 to the reporting
precision required for the study. Grid convergence is a numerical check, not a
fit to the TBE.

Then run the diffuse basis ladder:

```bash
METHOD=emrsf PROFILE=tbe-dz sbatch run_high_accuracy_v050.slurm
METHOD=emrsf PROFILE=tbe-tz sbatch run_high_accuracy_v050.slurm
```

The direct aug-cc-pVTZ step can be very expensive. If it exceeds the available
wall time, use a matching JK-fit basis only after `exact-dz` has quantified the
direct-minus-DF error:

```bash
PROFILE=tbe-tz \
INTEGRAL_BACKEND=df \
AUXBASIS=aug-cc-pvtz-jkfit \
sbatch run_high_accuracy_v050.slurm
```

Compare the paired MRSF/EMRSF results. This command refuses comparisons that
are not at equal settings or do not share the same triplet density:

```bash
python compare_mrsf_emrsf.py \
  results_v0.5.0_mrsf_exact-dz_direct/*.npz \
  results_v0.5.0_emrsf_exact-dz_direct/*.npz \
  --csv mrsf_emrsf_equal_level.csv
```

Compare completed archives without changing any calculation:

```bash
python compare_high_accuracy.py \
  results_v0.5.0_emrsf_*/*.npz \
  --csv emrsf_v050_basis_comparison.csv
```

## Acceptance criteria

For every MRSF or EMRSF calculation require:

- the requested integral backend;
- a state-locked MOM reference and accepted occupied/SOMO overlaps;
- all requested roots converged;
- maximum Davidson residual no greater than `1.1e-9`;
- symmetry purity at least `0.999` for the selected state;
- stable term and dominant configurations across grid changes;
- normal termination.

For EMRSF additionally require `DIRECT EMRSF MODE True`, `MRSF
PRE-DIAGONAL. False`, block-closure error no greater than `5e-9 Eh`, and stable
`gamma_CV`. For an equal-level pair require identical metadata, triplet energy
within `1e-10 Eh`, and AO density difference below `1e-8`.

Basis convergence must be judged from the same physical state, not from a fixed
global root number.

## Deliberate limitations

- The published EMRSF model is retained as TDA with a global-hybrid functional;
  no un-derived B block or range-separated extension is introduced.
- `c_cp=c_H`, S30 and the SI spin factors are fixed; they are not optimization
  variables.
- MRSF S42--S47 observables and MRSF NTOs are disabled in direct enlarged-space
  mode because they would require a separate MRSF eigensolve and are not full
  EMRSF state densities.
- A better MAE against five known TBEs does not by itself establish a more
  transferable method. Report the complete predefined profile, including
  molecules whose error becomes larger.
