# OpenQP conventional-MRSF oracle

These inputs are deliberately independent of the EMRSF implementation. They
run conventional singlet MRSF-TDDFT in OpenQP 1.2.1 at BH&HLYP/cc-pVDZ and
save the restricted triplet orbitals.

```bash
sbatch run_openqp_mrsf_benchmark.slurm
```

For one molecule, use for example `sbatch --array=1
run_openqp_mrsf_benchmark.slurm`. The array order is azulene,
N-phenylpyrrole, phthalazine, quinoxaline, and twisted DMABN.

The generated geometry is converted from the bundled bohr XYZ exactly once;
OpenQP input coordinates are angstrom. Symmetry is disabled in OpenQP so that
the numerical oracle does not depend on a B1/B2 axis convention. The saved
Molden can then be passed to `../run_openqp_mrsf_check.py` to compare the
PySCF implementation at identical orbitals.
