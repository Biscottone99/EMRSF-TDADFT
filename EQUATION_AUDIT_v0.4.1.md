# MRSF/EMRSF equation and implementation audit for 0.4.1

## Scope and sources

The audit used the following independent levels of evidence:

1. the EMRSF article and its Supporting Information, especially eqs.
   S18--S23, S30--S39, Tables S1--S3, and eqs. S42--S47;
2. the original collinear MRSF derivation (J. Chem. Phys. 149, 104101,
   DOI 10.1063/1.5044202) and the dimensional-transformation formulation in
   the analytic-gradient paper (J. Chem. Phys. 150, 184111);
3. the Q-Chem 6.4 spin-flip/MRSF formal summary;
4. OpenQP 1.2.1 source, tag `v1.2.1`, commit
   `9bfdfc4f2c0930da98cb9fca4c9fda28a71e91db`, principally
   `source/tdhf_mrsf_lib.F90`;
5. fixed-orbital numerical comparisons with OpenQP output and explicit
   four-index integral tests.

The conclusion is specific: no remaining sign, factor-of-two, transpose, or
`c_H` placement error was found in the implemented singlet MRSF or EMRSF
operator. The dominant benchmark failure in 0.4.0 was selection of a different
triplet reference determinant. Several surrounding implementation defects
could hide that fact and are corrected in 0.4.1.

## Conventional MRSF audit

The implementation uses the restricted two-open-shell triplet and the equal
zeroth-order mixture of its `M_S=+1` and `M_S=-1` components. The response is
TDA and the singlet/triplet spaces are formed with the published dimensional
transformation. The following OpenQP routines were compared line by line:

| Formal operation | OpenQP 1.2.1 | This implementation |
| --- | --- | --- |
| packed MRSF space and redundant OO slots | `mrinivec` | `space.py` |
| seven AO response components | `mrsfcbc` | `MRSFSpace.ao_components` |
| AO-to-MO response action | `mrsfmntoia` | `MRSFSpace.mo_action` |
| one-electron alpha/beta Fock action | `mrsfesum` | `one_electron_action` |
| state/transition 1-RDM | `get_mrsf_transition_density`, `get_trans_den` | `observables.py` |

The non-Hermitian density orientations, the OOS `1/sqrt(2)` expansion, the
extra `sqrt(2)` density-contraction factors, and the singlet signs all match.
The global-hybrid MRSF J/K convention also reproduces OpenQP when both programs
use identical orbitals.

## EMRSF block audit

The implemented operator is

```text
A_EMRSF = [[A_MRSF, c_cp C],
           [c_cp C.T, A_CV' + A_G I]],       c_cp = c_H.
```

Checks and conclusions:

- SI S18--S22: the alpha/beta Fock definitions and MRSF/LR diagonal
  differences are represented in the two response sectors.
- SI S23: `A_G` is built directly as
  `F_beta(O1,O1)-F_alpha(O2,O2)-c_H(O2O2|O1O1)` and independently compared
  with the G,G MRSF action. It is added exactly once.
- SI S30: the Kronecker delta multiplies the full open-orbital Coulomb
  difference, so only diagonal Fock elements are corrected by
  `(1-c_H)[(pq|O2O2)-(pq|O1O1)]`.
- SI S31--S38: the decomposed alpha/beta LR blocks reduce to the conventional
  singlet LR-TDA response used by PySCF. The AO response density orientation
  is intentionally transposed and non-Hermitian.
- SI S39 and Tables S1--S3: `c_cp` multiplies each complete coupling element,
  not only its two-electron part. The G row has `sqrt(2) F_rs`; CO1 and O2V
  use the S1-minus-S2 spin-adapted combinations. Every row and every equality
  condition is checked against explicit MO ERIs in the test suite.
- A range-separated functional is rejected: the published S30 construction
  specifies one global hybrid coefficient and does not define a two-range
  replacement.

Earlier exploratory phase/G-factor switches are absent because changing an
isolated row is neither a consistent determinant rephasing nor part of the
published model.

## S42--S47 audit

The density difference is built from the underlying MRSF one-particle density
relative to the optimized triplet density, exactly as stated in the SI. It is
not built from the squared norm of the EMRSF CV sector. Positive and negative
parts are integrated separately; their average is used as the common finite-
grid charge `q`. Both centroids are divided by that same `q`, after which
`D_CT` and `mu=q D_CT` are calculated.

The following are diagnostics rather than alternate definitions:

- `q_plus-q_minus`;
- integral of the density difference;
- separately integrated reference/state electron counts;
- grid-level and pruning metadata.

The SI does not provide the LR-LR and cross-sector 1-RDM blocks required for a
full coupled-EMRSF state density or NTO. The program therefore reports MRSF
descriptors and MRSF NTOs honestly instead of completing those blocks by an
unsupported assumption.

## Defects found and corrected

| Defect in 0.4.0 workflow | Consequence | Correction in 0.4.1 |
| --- | --- | --- |
| Aufbau ROKS accepted solely because it converged | N-phenylpyrrole and twisted DMABN used different triplet determinants | Molden-seeded MOM and fixed-external reference modes, with overlap and provenance checks |
| Raw Molden coefficient copying | Grouped contractions can have the same AO labels but a different coefficient ordering | Cross-overlap projection into the target AO basis, then symmetric metric orthonormalisation |
| One ambiguous `--grid-level` option | It changed S42--S47 integration but not SCF/response quadrature | Separate SCF and observable grid controls, both logged |
| Forced/assumed molecular symmetry | A wrong point group or axis convention can mislabel roots | Automatic detection, orbital and response purity, and explicit disable option |
| Paper terms interpreted with a global B1/B2 rule | Correct states could be compared with the wrong roots | Per-geometry mappings verified from energy, dominant configurations, and `gamma_CV` |
| Global-only Davidson guesses in an earlier release | A converged Krylov space could omit a lower invariant symmetry block | Up to `nstates` seeds in every Abelian irrep plus global oversampling |

For the bundled coordinates, the paper-to-PySCF mappings are azulene
`2 B2 -> 2 B1` and quinoxaline `2 B1 -> 2 B2`; the other three labels are
unchanged. This is molecule specific, not a universal exchange of axes.

## Numerical evidence and remaining runs

The 0.4.0 automatic references differed from the supplied OpenQP triplets by
about 0.478 eV for N-phenylpyrrole and 0.723 eV for twisted DMABN. Their
conventional-MRSF spectra consequently showed maximum root errors of about
0.584 and 1.246 eV. Phthalazine and quinoxaline, whose reference solutions
matched, already agreed with the OpenQP roots within 0.00160 and 0.00066 eV.

At the fixed supplied OpenQP orbitals, the completed N-phenylpyrrole test
reduced the maximum 12-root discrepancy to about 0.00130 eV. This establishes
that its previous large error was not in the MRSF response algebra. The
twisted-DMABN fixed-reference calculation and the five full EMRSF jobs are
deliberately supplied as cluster inputs rather than represented as completed
local results.

The local release suite passes 38 tests. This proves algebraic consistency on
small systems and catches the identified regressions; it does not substitute
for the supplied molecular benchmark runs.

## Method limitations that remain scientific, not coding errors

- The response space depends on the selected triplet determinant. A T1/T2
  crossing or character exchange can produce discontinuous surfaces.
- Conventional MRSF omits singly excited configurations that keep the
  closed-shell HOMO filled and LUMO empty; EMRSF adds the C-to-V sector aimed
  at this deficiency but remains the published approximate model.
- OpenQP SG2 and PySCF atom-centred grids are not point-for-point identical.
  Fixed-orbital, MOM-relaxed, grid-converged, and paper-rounded values must be
  kept as separate comparisons.
- Density fitting and auxiliary-basis error must be tested independently from
  SCF, response, and descriptor-grid convergence.

These limitations are why the logs make reference state, orbital gaps,
overlaps, symmetry, residuals, and equation diagnostics first-class output.
