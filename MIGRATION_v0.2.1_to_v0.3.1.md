# Moving the paper benchmark from v0.2.1 to v0.3.1

## Why the old azulene job should be stopped

v0.2.1 is an equation-validation release.  It applies the response operator
once for every column of the complete MRSF and LR matrices, forms the full
EMRSF matrix, and only then diagonalises it.  The observed run time exceeding
21 hours is therefore expected scaling, not a Slurm or PySCF failure.

v0.3.1 keeps the same response equations and S42--S47 definitions but changes
the numerical algorithm:

- restarted Davidson requests only the trial-vector actions needed for 12 roots;
- RI/J-KFIT replaces repeated four-index J/K and coupling transformations;
- the full response matrix is never allocated in the default path;
- the dense solver remains available only for small regression tests;
- every iteration reports elapsed time, matvec count and maximum residual.

## Cluster installation

Keep the v0.2.1 directory as the dense reference and extract v0.3.1 separately:

```bash
tar -xzf pyscf_emrsf_v0.3.1.tar.gz
cd pyscf_emrsf_v0.3.1
conda activate openqp_env
python -m pip install -e '.[test]'
pytest -q
```

The tests compare matrix-free and dense roots on small systems.  Do not launch
the molecular benchmarks unless every test passes.

## Benchmark submission

```bash
cd examples/paper_ct_benchmark
sbatch --array=0 run_all.slurm
```

The single-array command first repeats azulene.  During the run, the Slurm
output must show ROKS iterations followed by lines such as:

```text
[MRSF Davidson] iter=... matvecs=... max|r|=...
[EMRSF Davidson] iter=... matvecs=... max|r|=...
```

Only after azulene terminates normally should the complete array be submitted:

```bash
sbatch run_all.slurm
```

The final `.log` records solver type, convergence flags, residual norms and
matvec counts in addition to energies, symmetries, gamma_CV and S42--S47.  The
`.npz` stores the same diagnostics and the converged eigenpairs, but no dense
matrix is expected in matrix-free mode.
