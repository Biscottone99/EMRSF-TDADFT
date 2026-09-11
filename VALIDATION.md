# PySCF-EMRSF 0.4.1 validation contract

## Scientific operator

The implemented singlet TDA block is

```text
[ A_MRSF              c_cp C              ]
[ c_cp C.T   A_CV(LR,S30) + A_G I         ]
```

with `c_cp = c_H`, the global exact-exchange fraction, by default. The scale
multiplies the complete Slater--Condon coupling, including Fock and
two-electron parts. Range-separated EMRSF is rejected because SI eq. S30 is
derived for one global `c_H`.

The common offset is evaluated independently from SI eq. S23,

```text
A_G = F_beta[O1,O1] - F_alpha[O2,O2]
      - c_H (O2 O2|O1 O1),
```

and compared with the G,G element of the MRSF action. It is added once to the
entire corrected LR-CV block. SI eq. S30 is diagonal in the MO indices:

```text
F'[p,q] = F_DFT[p,q]
          + delta[p,q] (1-c_H)
            ((p q|O2 O2) - (p q|O1 O1)).
```

The fixed spin-adapted coupling rules include, among all other table rows,

```text
G:                  sqrt(2) F'[r,s]
OOS:                (r O1|O2 s) - 2(r s|O2 O1)
CO1, p != r:        (r p|s O2) - 2(r s|O2 p)
O2V, q != s:        2(r s|q O1) - (r O1|q s)
```

No empirical row-phase or G-factor switch is present. The direct four-index
and streamed RI implementations, `C` and `C.T`, are tested against one
another and against every condition in SI Tables S1--S3.

## Conventional MRSF

MRSF uses the equal mixed `M_S=+1/-1` triplet zeroth-order density, the
spin-adapted singlet dimensional transformation, and TDA. The seven AO
response components and the state/transition-density contraction were checked
line by line against OpenQP 1.2.1 `tdhf_mrsf_lib.F90`. At identical OpenQP
orbitals, the completed N-phenylpyrrole calculation agreed for all 12 roots to
within 0.0013 eV; the supplied Slurm repeats this test for both problematic
references. Differences from an independent Aufbau SCF are not evidence of a
response-equation error.

## Triplet-reference acceptance

SCF convergence alone is insufficient. A calculation is acceptable only if:

- exactly two singly occupied common restricted orbitals are present;
- electron and spin traces and AO-metric orthonormality pass;
- the intended SOMO and occupied subspaces remain locked;
- orbitals are pure in the enabled Abelian point group;
- the reference source and warnings are recorded in the log;
- the orbital gradient is small for a locally optimized reference.

`external_molden` is an operator oracle and may retain a small PySCF orbital
gradient because OpenQP and PySCF do not use identical DFT grids. It must not
be described as a locally optimized PySCF reference.

## S42--S47 observables

For each underlying MRSF root, the spin-summed unrelaxed 1-RDM is constructed
with the published dimensional-transformation factors and the triplet
reference density is subtracted (S42). Positive and negative parts (S43) give
`q_plus` and `q_minus`; their symmetric quadrature estimate

```text
q = (q_plus + q_minus) / 2
```

is the common charge of S44. Both S45 centroids use this same `q`; S46 and S47
then yield `D_CT` and `mu=q D_CT`. The log preserves the charge imbalance,
integrated density trace, and reference/state electron counts. Repeat levels
4, 5, and 6 for a final grid-convergence estimate.

These are MRSF one-particle-density observables, exactly as stated in the SI.
They are not full coupled-EMRSF state densities. Likewise, the optional NTOs
are labelled MRSF NTOs because the paper does not provide the missing LR-LR
and MRSF-LR transition-density blocks.

## Symmetry and target matching

All five supplied SI geometries are detected as C2v by PySCF, but that fact is
not assumed for arbitrary conformers. `--symmetry auto` detects the largest
supported Abelian group; `--symmetry none` disables it. B1/B2 names depend on
the Cartesian axes. For the unmodified bundled coordinates, configuration and
CV-character matching establishes these molecule-specific mappings:

| Molecule | Paper term | PySCF-log term | EMRSF/eV | TBE/eV | gamma_CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Azulene | 2 B2 | 2 B1 | 4.91 | 4.49 | 0.86 |
| N-Phenylpyrrole | 3 A1 | 3 A1 | 6.51 | 5.86 | 0.98 |
| Phthalazine | 1 B1 | 1 B1 | 4.83 | 4.31 | 0.97 |
| Twisted DMABN | 1 B1 | 1 B1 | 5.83 | 4.74 | 1.00 |
| Quinoxaline | 2 B1 | 2 B2 | 6.63 | 6.22 | 0.88 |

This is not a universal B1/B2 swap. Rotating a geometry can change the label
without changing the state.

## Required acceptance checks

- package version exactly 0.4.1 and 38 tests pass;
- intended reference is locked where required and no fatal reference warning;
- all requested MRSF and EMRSF roots converge below the residual threshold;
- every relevant irrep is present in `SEEDS PER IRREP`;
- target term, dominant configurations, purity, and `gamma_CV` agree;
- `A_G SOURCE ERROR` and direct-diagonal audit errors are negligible;
- S42--S47 charge and electron-count grid diagnostics are converged;
- basis, auxiliary basis, SCF grid, and descriptor grid are reported;
- fixed-reference, MOM, and fully relaxed results are not mixed in one table.
