# EMRSF singlet-triplet SOC implementation

## Scope

Version 0.8.0 evaluates spin-orbit couplings between separately converged
singlet and triplet EMRSF-TDA roots. The calculation is a state interaction in
the auxiliary-wavefunction representation of the TDA eigenvectors.

The operator is

\[
\hat H_{\mathrm{SO}}
=\hat H_{\mathrm{SO}}^{1e}+\hat H_{\mathrm{SO}}^{2e,\mathrm{SOMF}}.
\]

The two terms are contracted independently and remain independently available
in the log, CSV, Python result, and NPZ archive.

## EMRSF states

For multiplicity \(Q=1,3\), the response problem is

\[
\begin{pmatrix}
A_{\mathrm{MRSF}}^Q & c_H C^Q\\
c_H C^{Q\dagger} & A_{\mathrm{CV}}^Q+A_G I
\end{pmatrix}
\begin{pmatrix}X_{M,I}^Q\\X_{C,I}^Q\end{pmatrix}
=\Omega_I^Q
\begin{pmatrix}X_{M,I}^Q\\X_{C,I}^Q\end{pmatrix}.
\]

The state-interaction matrix element is

\[
V_{IJ}^{M_S}
=\langle\Psi_I^{S,0}|\hat H_{\mathrm{SO}}|\Psi_J^{T,M_S}\rangle
=X_I^{S\dagger}H_{\mathrm{SO}}^{ST,M_S}X_J^T.
\]

All representable block contractions are included:

\[
V=X_M^{S\dagger}H_{MM}X_M^T
 +X_M^{S\dagger}H_{MC}X_C^T
 +X_C^{S\dagger}H_{CM}X_M^T
 +X_C^{S\dagger}H_{CC}X_C^T.
\]

The added LR-CV configurations are conventional spin-adapted single
excitations from the closed-shell configuration \(G\). They are expanded for
\(M_S=0,\pm1\) and included in the determinant contraction.

The C->V rows belonging to the original MRSF block instead generate four open
orbitals after a spin flip. The published equations do not provide a unique
auxiliary-wavefunction spin completion for these rows. Version 0.8.0 therefore
projects these amplitudes out and reports

\[
w_{4o,I}=\sum_{i\in C,a\in V}|X_{ia,I}^{\mathrm{MRSF}}|^2.
\]

No renormalization is hidden. This keeps the result reproducible and exposes
when the projection may be quantitatively important.

## One-electron operator

The Breit-Pauli one-electron operator is

\[
\hat H_{\mathrm{SO}}^{1e}
=\frac{\alpha^2}{2}\sum_{iA}Z_A
\frac{\mathbf r_{iA}\times\mathbf p_i}{r_{iA}^3}\cdot\mathbf s_i.
\]

With Cartesian MO integrals \(h^x,h^y,h^z\), its spinor matrix is

\[
h_{\mathrm{spin}}=\frac12
\begin{pmatrix}
h^z & h^x-i h^y\\
h^x+i h^y & -h^z
\end{pmatrix}.
\]

For determinant coefficient matrix \(B\), the state matrix is

\[
H_{\mathrm{SO}}^{1e,\mathrm{states}}
=B^\dagger H_{\mathrm{SO}}^{1e,\mathrm{det}}B.
\]

## Mean-field two-electron contribution

The spin-free reference density is constructed from the common high-spin
ROHF/ROKS reference. PySCF `int2e_p1vxp1` integrals are contracted as

\[
h_{\mathrm{SOMF}}^{2e}
=\frac{i\alpha^2}{2}\left[J^{\mathrm{SO}}[D]
-\frac32K^{\mathrm{SO}}[D]\right].
\]

The available choices are:

- `none`: one-electron Breit-Pauli only;
- `amfi`: atomic mean-field contraction of the two-electron screening;
- `full`: full molecular SOMF contraction.

The same spinor transformation and determinant contraction used for the
one-electron term are then applied to \(h_{\mathrm{SOMF}}^{2e}\).

## Reported singlet-triplet coupling

For each pair \(S_I,T_J\), the program retains

\[
V_{IJ}^{-1},\quad V_{IJ}^{0},\quad V_{IJ}^{+1},
\]

and reports

\[
|\mathrm{SOC}_{IJ}|=
\sqrt{|V_{IJ}^{-1}|^2+|V_{IJ}^{0}|^2+|V_{IJ}^{+1}|^2}.
\]

The one-electron, SOMF two-electron, and total norms are printed separately.

## Python API

```python
from pyscf_emrsf import EMRSFTDA, compute_emrsf_soc, save_soc_archive

singlets = EMRSFTDA(
    mf, target_multiplicity=1, nstates=5, solver="davidson"
).kernel()
triplets = EMRSFTDA(
    mf, target_multiplicity=3, nstates=5, solver="davidson"
).kernel()

soc = compute_emrsf_soc(singlets, triplets, two_electron="amfi")
soc.write_log("molecule_soc_emrsf.log")
soc.write_csv("molecule_soc_emrsf_couplings.csv")
save_soc_archive("molecule_soc_emrsf.npz", soc)
```

## Eigenvectors and external modules

```python
singlets.write_eigenvectors_csv("singlet_vectors.csv")
triplets.write_eigenvectors_csv("triplet_vectors.csv")
singlets.save_postprocessing("singlet_vectors.npz")
triplets.save_postprocessing("triplet_vectors.npz")
```

CSV rows identify the state, EMRSF row, MRSF/LR-CV sector, configuration
family, source orbital, target orbital, coefficient, and weight. The NPZ is
lossless and stores all coefficients.

An external module passed with `--soc-external-module path.py` must define

```python
def compute_soc(context):
    one = context.contract_one_body(context.integrals.one_electron_mo)
    two = context.contract_one_body(context.integrals.two_electron_mo)
    return {
        "one_electron_matrix_hartree": one,
        "two_electron_matrix_hartree": two,
        "approximation": "description",
    }
```

It may alternatively return one complete Hermitian SOC matrix in Hartree. The
driver validates dimensions, finite values, and Hermiticity before diagonalizing
the spin-orbit state-interaction Hamiltonian.

## Validation conditions

The implementation checks:

1. common mean-field reference and orbital ordering;
2. singlet/triplet multiplicities;
3. normalization of each explicitly represented CSF;
4. agreement of included LR-CV weights with `EMRSFResult.cv_weights`;
5. Hermiticity of spatial, spinor, and state SOC matrices;
6. equality of internal and example external-module contractions;
7. pickle-free loading of all NPZ archives.

