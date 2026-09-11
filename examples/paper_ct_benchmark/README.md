# Five-molecule EMRSF benchmark

The XYZ files contain the bohr geometries transcribed from `SI_loos.pdf`.
The publication protocol is BH&HLYP/cc-pVDZ with 12 singlet states. The
scripts keep the SCF grid (`--scf-grid-level 3`) separate from the S42--S47
integration grid (`--observable-grid-level 5`).

## 1. Local equation checks

```bash
python -m pip install -e '.[test]'
bash run_validation_local.sh quick
```

The expected result is a completely passing test suite.

## 2. Fixed-reference conventional-MRSF check

```bash
sbatch run_openqp_mrsf_checks.slurm
```

This array checks N-phenylpyrrole and twisted DMABN at the supplied OpenQP
orbitals. It isolates the MRSF response operator from both SCF root selection
and the EMRSF extension. The job fails if any of the 12 roots differs from the
stored OpenQP 1.2.1 oracle by more than 0.003 eV.

To regenerate an independent OpenQP oracle and Molden for all five systems:

```bash
cd openqp_mrsf
sbatch run_openqp_mrsf_benchmark.slurm
```

## 3. Legacy state-locked EMRSF regression benchmark

```bash
sbatch run_emrsf_benchmark_v041.slurm
```

Array order is azulene, N-phenylpyrrole, phthalazine, twisted DMABN, and
quinoxaline. The default policy uses Molden-seeded MOM for the two systems
known to converge to a different PySCF Aufbau triplet and ordinary optimized
references for the other three.

For an operator-only comparison using OpenQP orbitals for all five, collect
the generated Moldens in one directory and run:

```bash
REFERENCE_POLICY=external_all \
REFERENCE_DIR=/absolute/path/to/moldens \
sbatch run_emrsf_benchmark_v041.slurm
```

For local PySCF optimization while locking all five states, use
`REFERENCE_POLICY=mom_all`. `REFERENCE_POLICY=auto_all` exists only as a
negative control and is explicitly marked unlocked. Different policies write
to different result directories unless `OUTPUT_DIR` is supplied.

Molden orbital and MRSF-NTO exports remain off by default. Enable them only
after energy validation:

```bash
WRITE_MOLDEN=1 WRITE_NTO=1 sbatch run_emrsf_benchmark_v041.slurm
```

## 4. Extract the paper comparison

```bash
python extract_log_check.py \
  --targets-csv paper_targets.csv \
  --csv emrsf_v041_check.csv \
  results_v0.4.1/*_v0.5.1.log
```

`paper_targets.csv` contains the geometry-specific paper-to-PySCF term
mappings. The extractor also reports reference provenance, warnings, orbital
gradient, SOMO overlap, symmetry purity, residuals, S23/S30 audit errors,
`gamma_CV`, and the best CV-character state. Do not compare by global root
number alone.

The full equation audit and acceptance criteria are in
`../../EQUATION_AUDIT_v0.4.1.md` and `../../VALIDATION.md`.

## 5. v0.5.1 equal-level high-accuracy comparison

Use the same `exact-dz` profile for both response spaces:

```bash
METHOD=mrsf  PROFILE=exact-dz sbatch run_high_accuracy_v051.slurm
METHOD=emrsf PROFILE=exact-dz sbatch run_high_accuracy_v051.slurm
```

Then pair the archives with strict reference and settings checks:

```bash
python compare_mrsf_emrsf.py \
  results_v0.5.1_mrsf_exact-dz_direct/*.npz \
  results_v0.5.1_emrsf_exact-dz_direct/*.npz \
  --csv mrsf_emrsf_equal_level.csv
```

`--method mrsf` retains the v0.4.1 MRSF matrix action and tightens only the
Davidson residual. `--method emrsf` directly diagonalizes the enlarged
MRSF+CV Hamiltonian without a preliminary MRSF eigensolve. Full numerical and
basis-convergence instructions are in `../../HIGH_ACCURACY_PROTOCOL_v0.5.0.md`.
