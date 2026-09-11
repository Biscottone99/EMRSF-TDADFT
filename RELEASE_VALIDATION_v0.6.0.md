# Release validation: PySCF-EMRSF 0.6.0

## Scope

Version 0.6.0 preserves the v0.5.2 MRSF-TDA and singlet EMRSF operators and
adds three explicitly separated capabilities:

- singlet, triplet or combined MRSF-TDA manifolds;
- self-consistent PCM on the high-spin reference, followed by a frozen
  reaction field in the excitation calculation;
- optional SOC-MRSF state interaction between companion MRSF singlets and
  MRSF triplets.

The published EMRSF extension is used only for singlets. When
`--method emrsf --states both` is requested, the singlet energies are EMRSF
and the triplet energies used for SOC are MRSF-TDA. Full RPA/TDDFT response,
state-specific excited-state PCM and EMRSF-SOC are not claimed.

## Automated tests

The complete source suite passes:

```text
58 passed
```

The original 48 tests cover the unchanged reference, MRSF and EMRSF
operators, dense/matrix-free equivalence, direct/RI branches, spin adaptation,
equation checks, observables and archives. The ten v0.6.0 test cases add:

- triplet-manifold construction and dynamic singlet/triplet log labels;
- PCM option validation and a self-consistent PCM reference calculation;
- rejection of invalid dielectric constants and Lebedev orders;
- Hermiticity of the spin-orbit one-electron spinor operator;
- one-electron, AMFI-SOMF and full-SOMF SOC smoke calculations;
- exact CSF normalization after the documented CV exclusion;
- rejection of SOC calculations built from different references.

The same suite is also run against a clean installation of the release wheel.

## End-to-end XYZ smoke tests

The following workflows complete from an XYZ geometry and produce logs, CSV,
Molden and NPZ output as applicable:

1. gas-phase MRSF singlets plus triplets with one-electron SOC;
2. PCM MRSF-TDA with automatic oscillator strengths;
3. PCM EMRSF singlets plus companion MRSF triplets with AMFI-SOMF SOC.

All MRSF state logs include length-gauge transition dipoles and oscillator
strengths. EMRSF logs label these values as companion-MRSF properties rather
than full enlarged-space EMRSF transition properties.

## Scientific validation boundary

The SOC implementation is validated internally by explicit determinant/CSF
normalization, Hermiticity, zero-coupling and screening-branch tests. A water
smoke calculation also gives the expected sub-wavenumber low-state splitting
scale. This is not a complete independent numerical benchmark against the
same-reference SOC-MRSF implementation in another program. Such a benchmark
is recommended before using new SOC values as publication-grade reference
data.

PCM is equilibrium self-consistent for the triplet reference. Its converged
reaction potential is frozen in the MRSF/EMRSF response. Results must therefore
be labelled `reference-equilibrium/frozen-response PCM`, not state-specific or
linear-response excited-state PCM.
