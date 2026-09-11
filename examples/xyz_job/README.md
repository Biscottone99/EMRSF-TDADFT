# Arbitrary-XYZ workflow (v0.8.0)

Place these three files in the calculation directory:

- the molecular geometry, for example `molecule.xyz`;
- `run_xyz_emrsf.py`;
- `run_xyz_emrsf_robust.slurm`.

The installed `pyscf_emrsf` package supplies the library. The extracted source
tree does not need to be the submission directory.

## Spin-free calculations

MRSF singlets:

```bash
XYZ_FILE=molecule.xyz METHOD=mrsf STATES=singlets NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

MRSF triplets:

```bash
XYZ_FILE=molecule.xyz METHOD=mrsf STATES=triplets NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

MRSF singlets and triplets in one job:

```bash
XYZ_FILE=molecule.xyz METHOD=mrsf STATES=both NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

EMRSF singlets and triplets:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=both NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

EMRSF triplets only:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=triplets NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

The paper publishes the singlet EMRSF construction. Version 0.7.0 extended the
same enlarged space to triplet spin adaptation. For a triplet-only request the
driver automatically computes one singlet EMRSF root as the common vertical
energy origin. The triplet states CSV reports both:

- `within_manifold_excitation_ev`: `Tn-T0`;
- `common_s0_excitation_ev`: `Tn-S0`, the quantity to compare with vertical
  triplet energies in the literature.

`NSTATES` is the number of roots per manifold, including `S0` or `T0`. Thus
`NSTATES=6` gives five transitions from the lowest state.

## PCM

Set the solvent's static dielectric constant to enable self-consistent PCM on
the triplet reference:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=singlets \
PCM_EPSILON=8.93 PCM_METHOD=IEF-PCM \
sbatch run_xyz_emrsf_robust.slurm
```

Available models are `C-PCM`, `COSMO`, `IEF-PCM` and `SS(V)PE`. The defaults
are Lebedev order 29, SWIG surface and van der Waals scale 1.2. The converged
reference reaction field is frozen in the custom MRSF/EMRSF response. This is
not an equilibrium excited-state PCM correction.

## SOC

SOC requires both singlet and triplet manifolds:

```bash
XYZ_FILE=molecule.xyz METHOD=mrsf STATES=both SOC=1 \
SOC_TWO_ELECTRON=amfi NSTATES=6 \
sbatch run_xyz_emrsf_robust.slurm
```

`SOC_TWO_ELECTRON` accepts:

| Value | SOC operator | Cost |
|---|---|---|
| `none` | one-electron Breit-Pauli | low |
| `amfi` | one-electron plus atomic mean-field SOMF | recommended for production |
| `full` | one-electron plus full molecular SOMF | formally tighter, AO fourth-power memory |

With `METHOD=emrsf STATES=both SOC=1`, SOC is evaluated from the EMRSF
singlet and triplet eigenvectors. The added closed-shell LR-CV configurations
are included. The unresolved four-open MRSF C->V auxiliary-wavefunction rows
are projected without hidden renormalization, and their excluded weight is
printed for every state.

The SOC output contains separate one-electron, mean-field two-electron and
total complex couplings for `Ms=-1,0,+1`, their rotationally combined
magnitudes in cm-1, the Hermitian state-interaction Hamiltonian,
SOC-adiabatic energies and mixing coefficients.

To call a custom SOC implementation after the spin-free calculation, place a
module defining `compute_soc(context)` in the submission directory and run:

```bash
XYZ_FILE=molecule.xyz METHOD=emrsf STATES=both \
SOC_EXTERNAL_MODULE=my_soc.py SOC_TWO_ELECTRON=amfi \
sbatch run_xyz_emrsf_robust.slurm
```

The context exposes raw eigenvectors and basis labels, determinant expansions,
orbitals, geometry, and the precomputed one-electron/SOMF integrals. See
`external_soc_module_example.py` for the callback contract.

## Oscillator strengths and NTOs

Length-gauge oscillator strengths are automatic. They appear inside each
MRSF log and in a CSV:

- singlet log: `S0 -> Sn` transitions;
- triplet log: `T0 -> Tn` transitions;
- EMRSF log: the clearly labelled, identically referenced MRSF companion
  values, because a full EMRSF transition 1-RDM is not published.

Disable them only with `OSCILLATOR_STRENGTHS=0`. Reference Molden orbitals and
MRSF hole/particle NTO Molden files are written automatically. Use
`SAVE_VECTORS=1` for complete pickle-free NPZ post-processing archives and
configuration-labelled eigenvector CSV files. Set `EIGENVECTOR_THRESHOLD` to
filter only the readable CSV; the NPZ always remains lossless.

## State locking and RI

An optional restricted triplet Molden seed locks the intended reference:

```bash
XYZ_FILE=molecule.xyz REFERENCE_MOLDEN=openqp_triplet.molden \
BASIS='6-31g(d,p)' METHOD=emrsf STATES=both \
sbatch run_xyz_emrsf_robust.slurm
```

For density fitting with def2-SVP:

```bash
XYZ_FILE=molecule.xyz BASIS=def2-svp INTEGRAL_BACKEND=df \
AUXBASIS=def2-universal-jkfit METHOD=emrsf STATES=singlets \
sbatch run_xyz_emrsf_robust.slurm
```

All temporary files are written below `/hpc/scratch/matteo.bedogni`; final
logs, CSV, Molden, NPZ and metadata are copied back to the submission folder.
