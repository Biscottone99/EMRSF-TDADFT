# Schreiber/Thiel vertical triplet benchmark

This directory contains five external MP2/6-31G* ground-state geometries and
nine vertical CC3/TZVP triplet energies transcribed from Tables VIII, X, XI,
XIII and XIV of Schreiber *et al.*, J. Chem. Phys. **128**, 134110 (2008), DOI
10.1063/1.2889385.

The targets are external accuracy references. They are not imported by the
library, used by an operator, or fitted.

## Run

After installing v0.8.0, execute:

```bash
python run_all.py --basis def2-tzvp --xc bhandhlyp --nstates 8
```

To inspect commands without running them:

```bash
python run_all.py --dry-run
```

Run one system with `--only ethene`. The driver computes an EMRSF singlet S0
anchor and the requested enlarged-space triplet roots for each geometry.

## Compare

```bash
python compare_results.py results_v0.8.0 --output triplet_comparison.csv
```

The comparison script matches spectroscopic terms and uses
`common_s0_excitation_ev`, i.e. `Omega(Tn)-Omega(S0)`. Do not compare the
literature vertical energies with `within_manifold_excitation_ev`, which is
`Omega(Tn)-Omega(T0)`.

## Basis warning

The published target label is the historical Karlsruhe `TZVP`. PySCF's
built-in `def2-tzvp` is convenient and similarly sized but is not the same
basis. A strict basis-matched comparison requires loading the exact historical
TZVP contractions. Keep basis-set and method errors separate when interpreting
the MAE.
