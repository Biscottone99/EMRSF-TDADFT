# Install and run PySCF-EMRSF 0.6.0

## Installation

Python 3.10--3.12 is recommended. In the existing cluster environment:

```bash
source /hpc/share/tools/miniconda3/etc/profile.d/conda.sh
conda activate openqp_env

python --version
python -m pip install --no-deps --force-reinstall \
  pyscf_emrsf-0.6.0-py3-none-any.whl
python -c 'import pyscf_emrsf; print(pyscf_emrsf.__version__)'
```

The final command must print `0.6.0`. `--no-deps` is appropriate only when
NumPy, SciPy and PySCF are already installed and working in that environment.
For a fresh environment omit it:

```bash
python -m pip install pyscf_emrsf-0.6.0-py3-none-any.whl
```

## Calculation folder

Copy these files next to the geometry:

```text
molecule.xyz
run_xyz_emrsf.py
run_xyz_emrsf_robust.slurm
```

The package is imported from the Conda environment; no `src/` folder or
`PROGRAM_DIR` is required in the calculation directory.

## Common submissions

EMRSF singlets with def2-SVP and RI:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=singlets NSTATES=6 \
BASIS=def2-svp INTEGRAL_BACKEND=df AUXBASIS=def2-universal-jkfit \
sbatch run_xyz_emrsf_robust.slurm
```

MRSF singlets and triplets with SOC:

```bash
XYZ_FILE=molecule.xyz METHOD=mrsf STATES=both NSTATES=6 SOC=1 \
SOC_TWO_ELECTRON=amfi BASIS=def2-svp INTEGRAL_BACKEND=df \
AUXBASIS=def2-universal-jkfit \
sbatch run_xyz_emrsf_robust.slurm
```

EMRSF singlets, MRSF triplets, SOC-MRSF and dichloromethane PCM:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=both NSTATES=6 SOC=1 \
SOC_TWO_ELECTRON=amfi PCM_EPSILON=8.93 PCM_METHOD=IEF-PCM \
BASIS=def2-svp INTEGRAL_BACKEND=df AUXBASIS=def2-universal-jkfit \
sbatch run_xyz_emrsf_robust.slurm
```

Use `SOC_TWO_ELECTRON=full` only after checking the AO fourth-power memory
estimate. The program aborts before allocating an unsafe full-SOMF tensor.

## Outputs to inspect

- `*_emrsf_singlet_*.log`: EMRSF singlet energies and gamma_CV;
- `*_mrsf_singlet_*.log`: MRSF singlets and automatic S0-to-Sn oscillator strengths;
- `*_mrsf_triplet_*.log`: MRSF triplets and automatic T0-to-Tn oscillator strengths;
- `*_soc_mrsf_*.log`: complex SOC components, magnitudes and mixed energies;
- `*_triplet_reference_*.molden`: final reference orbitals;
- `*_nto/`: MRSF hole and particle NTO Molden files;
- `*.npz`: complete vectors and matrices when `SAVE_VECTORS=1`.

Every normal result ends with `NORMAL TERMINATION`. Read the metadata JSON to
confirm method, manifold, PCM scope, SOC approximation and exact output paths.
