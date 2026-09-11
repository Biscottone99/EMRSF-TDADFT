# Release validation: PySCF-EMRSF 0.8.0

## Automated verification

- 68 tests pass from the source tree.
- The same 68 tests pass against the built wheel installation.
- MRSF and EMRSF determinant wavefunctions are normalized to one minus the
  explicitly reported four-open projected weight.
- The included LR-CV norm agrees state by state with `EMRSFResult.cv_weights`.
- One-electron, AMFI-SOMF, and total SOC matrices pass Hermiticity checks.
- The example external module reproduces the internal one-body contractions.
- Post-processing and SOC archives load with `allow_pickle=False`.
- Complete eigenvector CSV files include configuration and orbital labels.
- An EMRSF water smoke calculation with AMFI-SOMF SOC, vector export, and a
  second run through the external-module interface terminates normally.
- The three conventional MRSF core files retain their v0.6.0 SHA-256 values.

## Deliberate limits

The result is a projected EMRSF-TDA auxiliary-wavefunction state interaction,
not an analytic quadratic-response property. The added LR-CV sector is
included. Original MRSF C->V rows that require an unpublished four-open spin
completion are excluded without renormalization and their norm is reported.

The five production CC3/TZVP benchmarks remain supplied for execution on the
target hardware and are not claimed as run in this build environment.
