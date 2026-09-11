# Release validation: PySCF-EMRSF 0.7.0

## Automated verification

- 65 tests pass from the source tree.
- The same 65 tests pass against a clean installation of the built wheel.
- Direct triplet coupling equals explicit four-index formulas for every
  configuration family.
- RI and direct coupling actions agree to the auxiliary-basis tolerance and
  both satisfy the algebraic adjoint identity.
- Dense and Davidson triplet EMRSF roots and CV weights agree.
- S30 diagonal closure and the independent S23 `A_G` check pass for triplets.
- An installed-wheel triplet-only XYZ smoke calculation terminates normally
  and automatically constructs the singlet S0 common origin.
- The three conventional MRSF core files have the same SHA-256 values as
  v0.6.0.

## Deliberate limits

The five production CC3/TZVP benchmarks are supplied for the user to run on
their target hardware; they are not claimed as executed in this build
environment. The included `def2-tzvp` example basis is related to, but is not
identical to, the historical Karlsruhe TZVP basis used for the published
targets. No CC3 value is used to fit the implementation.
