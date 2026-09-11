# Triplet enlarged-space EMRSF-TDA equations (v0.7.0)

## Status and scope

Oh *et al.* publish EMRSF for singlet target states. Version 0.7.0 retains
that singlet implementation and adds a triplet spin adaptation derived from
the same determinant-level Slater--Condon rows. It is therefore labelled a
triplet extension, not a reproduction of equations printed in the paper.
Only the Tamm--Dancoff approximation is implemented.

The conventional MRSF operator is unchanged from v0.6.0. The new code changes
only the added closed-to-virtual LR block and its coupling to the existing
triplet MRSF space.

## Notation

- `C`: doubly occupied orbitals, indexed by `p,r`;
- `O1,O2`: the two singly occupied reference orbitals;
- `V`: virtual orbitals, indexed by `q,s`;
- `F'`: the S30-corrected artificial closed-shell Fock matrix;
- `c_H`: global exact-exchange fraction (`0.5` for BH&HLYP);
- `c_cp`: scale on the complete MRSF--CV coupling (`c_cp=c_H` by default).

The added configurations are the closed-to-virtual pairs `r -> s`.

## Triplet LR-CV block

For a global hybrid, the triplet TDA block is

\[
A^T_{ia,jb}=F'_{ab}\delta_{ij}-F'_{ij}\delta_{ab}
-c_H(ij|ba).
\]

Unlike the singlet block, it has no direct Coulomb term and no semilocal
singlet XC kernel. The diagonal S30 correction is retained exactly as in the
published construction:

\[
F'_{pp}=F_{pp}+(1-c_H)\left[(pp|O_2O_2)-(pp|O_1O_1)\right].
\]

## Triplet MRSF--CV coupling

The triplet spin adaptation combines the two determinant rows as `S1+S2`.
The OOS configuration is `(L+R)/sqrt(2)`. The singlet-only `G` and `D`
configurations are absent from the live triplet MRSF space.

For added configuration `r -> s`, the unscaled nonzero matrix elements are:

| MRSF family | Condition | Triplet matrix element |
|---|---|---|
| OOS | always | `(r O1|O2 s)` |
| CO1, `p -> O1` | `p != r` | `(r p|s O2)` |
| CO1, `p -> O1` | `p = r` | `-F'[O2,s] + (r s|O2 p)` |
| CO2, `p -> O2` | `p = r` | `-(O2 O1|O2 s)` |
| O1V, `O1 -> q` | `q = s` | `-(r O1|O2 O1)` |
| O2V, `O2 -> q` | `q != s` | `(r O1|q s)` |
| O2V, `O2 -> q` | `q = s` | `F'[r,O1] + (r s|q O1)` |
| CV(MRSF), `p -> q` | `p = r` | `-(q O1|O2 s)` |
| CV(MRSF), `p -> q` | `q = s` | `-(r O1|O2 p)` |

When both CV conditions hold, both contributions are added. The complete
coupling, including Fock and two-electron terms, is multiplied once by
`c_cp`. The reverse action is the exact algebraic transpose.

## Common offset and energy origins

The added LR block receives the same explicit S23 offset

\[
A_G=F^\beta_{O_1O_1}-F^\alpha_{O_2O_2}
-c_H(O_2O_2|O_1O_1).
\]

Because `G` is absent from the triplet response space, the independent
operator check of this scalar is evaluated with a temporary singlet MRSF
packing. This does not modify or diagonalise a different triplet operator.

Triplet roots alone define `Omega(Tn)-Omega(T0)`. A vertical literature value
is `Omega(Tn)-Omega(S0)`, so the XYZ driver computes the singlet EMRSF S0 with
the same reference and writes the latter as `common_s0_excitation_ev`.

## Verification boundaries

The automated tests compare all coupling branches with explicit four-index
integrals, check direct and RI actions and adjoints, compare dense and
Davidson roots, decompose the LR diagonal, and verify the independent S23
construction. Agreement with external CC3/TZVP energies remains an accuracy
benchmark, not an equation identity or fitting target.
