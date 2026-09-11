# Install and run PySCF-EMRSF 0.8.0

## Installation

```bash
tar -xzf pyscf_emrsf_v0.8.0_complete.tar.gz
cd pyscf_emrsf_v0.8.0
python -m pip install -e '.[test]'
python -c 'import pyscf_emrsf; print(pyscf_emrsf.__version__)'
pytest -q
```

The version command must print `0.8.0`.

## EMRSF singlets, triplets, SOC, and vectors

```bash
python examples/xyz_job/run_xyz_emrsf.py molecule.xyz \
  --method emrsf \
  --states both \
  --nstates 5 \
  --basis def2-svp \
  --xc bhandhlyp \
  --integral-backend df \
  --soc \
  --soc-two-electron amfi \
  --write-eigenvectors \
  --output-dir results
```

The SOC output contains a log, coupling CSV, and external-ready NPZ. The
one-electron and mean-field two-electron contributions are stored separately.

For the full molecular SOMF contraction replace `amfi` with `full`. This uses a
three-component four-index AO intermediate and may require substantially more
memory. Use `none` for a one-electron-only comparison.

## External SOC module

```bash
python examples/xyz_job/run_xyz_emrsf.py molecule.xyz \
  --method emrsf --states both --nstates 5 \
  --soc-external-module examples/xyz_job/external_soc_module_example.py \
  --soc-two-electron amfi \
  --write-eigenvectors \
  --output-dir results_external
```

The hook is executed only when the module path is supplied explicitly.

