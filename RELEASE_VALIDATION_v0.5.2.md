# Release validation: PySCF-EMRSF 0.5.2

## Scope

Version 0.5.2 leaves the MRSF and EMRSF Hamiltonians unchanged. Validation
targets the new post-processing archive, S42--S47 driver integration and MRSF
length-gauge transition properties.

## Automated tests

The complete source suite passes:

```text
48 passed
```

The same 48 tests pass against a clean installation of the built wheel.
New tests verify:

- the transition dipole and oscillator strength of an analytically defined
  pure MRSF hole--particle pair;
- vanishing transition-density trace;
- pickle-free archive loading;
- exact preservation of all eigenvector coefficients;
- presence and dimensions of the configuration map, MO coefficients and AO
  position integrals.

## End-to-end XYZ smoke tests

Both `METHOD=mrsf` and `METHOD=emrsf` completed from an XYZ input with:

```text
--write-observables --observables-grid-level 1
--write-oscillator-strengths --save-vectors
```

The runs produced logs, state CSV files, S42--S47 CSV, oscillator-strength CSV,
Molden NTOs, reference Molden, metadata JSON and complete NPZ archives. The
EMRSF run also produced a separate MRSF auxiliary archive and property table.

For the smoke transition, the transition-density trace was
`3.75e-29`, confirming origin independence to numerical precision.

## Scientific labelling

Oscillator strengths and S42--S47 descriptors are MRSF state-to-state
properties. They are not labelled as full EMRSF properties because the
published enlarged-space equations do not provide all one-particle
transition-density blocks. The complete EMRSF coefficient vectors and basis
maps are retained for future post-processing development.
