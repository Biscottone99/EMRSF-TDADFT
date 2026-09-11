# Install and run PySCF-EMRSF 0.7.0

## Installation

Python 3.10 or newer is required. In a clean environment:

```bash
tar -xzf pyscf_emrsf_v0.7.0_complete.tar.gz
cd pyscf_emrsf_v0.7.0
python -m pip install --upgrade pip
python -m pip install dist/pyscf_emrsf-0.7.0-py3-none-any.whl
python -c 'import pyscf_emrsf; print(pyscf_emrsf.__version__)'
```

The final command must print `0.7.0`.

For development and tests:

```bash
python -m pip install -e '.[test]'
pytest -q
```

## One XYZ calculation

Copy `run_xyz_emrsf.py`, the Slurm template and the XYZ into the submission
directory. For gas-phase BH&HLYP/def2-TZVP triplets:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=triplets NSTATES=8 \
BASIS=def2-tzvp INTEGRAL_BACKEND=direct \
sbatch run_xyz_emrsf_robust.slurm
```

For singlets and triplets plus MRSF-level SOC:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=both NSTATES=8 SOC=1 \
SOC_TWO_ELECTRON=amfi BASIS=def2-tzvp INTEGRAL_BACKEND=direct \
sbatch run_xyz_emrsf_robust.slurm
```

For RI, use `INTEGRAL_BACKEND=df` and set a compatible `AUXBASIS`.

## Triplet benchmark

```bash
cd examples/triplet_benchmark_schreiber2008
python run_all.py --basis def2-tzvp --xc bhandhlyp --nstates 8
python compare_results.py results_v0.7.0 --output triplet_comparison.csv
```

Compare literature vertical energies only with `common_s0_excitation_ev`.
`within_manifold_excitation_ev` is relative to T0 and answers a different
question.
