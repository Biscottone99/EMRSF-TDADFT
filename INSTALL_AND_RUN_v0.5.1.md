# Install and run PySCF-EMRSF v0.5.1

## Install the verified wheel

```bash
tar -xzf pyscf_emrsf_v0.5.1_complete.tar.gz
cd pyscf_emrsf_v0.5.1

source /hpc/share/tools/miniconda3/etc/profile.d/conda.sh
conda activate openqp_env
python -m pip install --force-reinstall --no-deps \
  dist/pyscf_emrsf-0.5.1-py3-none-any.whl
python -c 'import pyscf_emrsf; print(pyscf_emrsf.__version__)'
```

The last command must print `0.5.1`. `--no-deps` retains the working PySCF,
NumPy and SciPy versions already installed in `openqp_env` and avoids building
SciPy from source.

## Prepare an independent calculation directory

```bash
mkdir -p /hpc/home/matteo.bedogni/test/emrsf/my_calculation
cd /hpc/home/matteo.bedogni/test/emrsf/my_calculation

cp /hpc/home/matteo.bedogni/test/emrsf/pyscf_emrsf_v0.5.1/examples/xyz_job/run_xyz_emrsf.py .
cp /hpc/home/matteo.bedogni/test/emrsf/pyscf_emrsf_v0.5.1/examples/xyz_job/run_xyz_emrsf_robust.slurm .
```

Place the XYZ file in this directory. The Slurm script stages the calculation
below `/hpc/scratch/matteo.bedogni` and copies only final outputs back.

## Automatic robust reference, def2-SVP and density fitting

For five excited transitions (`S0 -> S1` through `S0 -> S5`), request six
total states:

```bash
XYZ_FILE=molecola.xyz METHOD=emrsf NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

The Slurm defaults are BH&HLYP/def2-SVP, density fitting with
def2-universal-jkfit, unpruned level-6 final grid and the robust SCF strategy.

For MRSF only:

```bash
XYZ_FILE=molecola.xyz METHOD=mrsf NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

## OpenQP-seeded reference

The Molden and the requested AO basis must match. For the supplied
6-31G(d,p) OpenQP orbitals, either use direct integrals:

```bash
XYZ_FILE=molecola.xyz METHOD=emrsf NSTATES=6 \
BASIS='6-31g(d,p)' INTEGRAL_BACKEND=direct AUXBASIS='' \
REFERENCE_MOLDEN='mrsf-tddft-openqp_scf_rohf_bhhlyp_6-31g(d,p).molden' \
sbatch run_xyz_emrsf_robust.slurm
```

or enable density fitting and let PySCF select/generate its auxiliary basis:

```bash
XYZ_FILE=molecola.xyz METHOD=emrsf NSTATES=6 \
BASIS='6-31g(d,p)' INTEGRAL_BACKEND=df AUXBASIS='' \
REFERENCE_MOLDEN='mrsf-tddft-openqp_scf_rohf_bhhlyp_6-31g(d,p).molden' \
sbatch run_xyz_emrsf_robust.slurm
```

Do not feed 6-31G(d,p) orbitals into a def2-SVP run unless the explicit
cross-basis projection option has been validated for that calculation.

## Outputs and acceptance

Successful results are copied to a directory named
`results_emrsf_v0.5.1_JOBID`. It contains the main response log, final triplet
reference Molden, state table, metadata, NTO summary and one MRSF
hole/particle Molden file per transition.

Before interpreting excitation energies, verify in the log:

- `SCF FINAL STAGE` is a final-grid stage;
- `LOCAL SCF CONVERGED True` and normal termination are present;
- final orbital gradient is at most `1e-4`;
- all requested response roots converged;
- response residuals are within the requested tolerance;
- the SOMO and occupied-subspace overlaps pass for a Molden-seeded run.

The coarse-grid, damped and level-shifted stage is only a preconditioner. Its
energy is never returned as the production MRSF/EMRSF reference.
