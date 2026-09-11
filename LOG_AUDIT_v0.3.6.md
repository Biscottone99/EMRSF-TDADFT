# Audit of the supplied v0.3.6 molecular logs

All four calculations reached the normal-termination marker and reported
maximum EMRSF residuals between `6.80e-8` and `9.95e-8` hartree. They are not
SCF or Davidson-residual failures.

| Molecule | EMRSF dimension | Published term read literally | E/eV | gamma_CV | Paper E/eV | Paper gamma_CV |
|---|---:|---:|---:|---:|---:|---:|
| N-phenylpyrrole | 12237 | 3 A1 | 6.0615688 | 0.0028710 | 6.51 | 0.98 |
| Phthalazine | 9249 | 1 B1 | 5.1471845 | 0.9661708 | 4.83 | 0.97 |
| Quinoxaline | 9249 | 2 B1 | 6.9069952 | 0.0070265 | 6.63 | 0.88 |
| Twisted DMABN | 12871 | 1 B1 | 5.2272014 | 0.0012574 | 5.83 | 1.00 |

The low `gamma_CV` values show that a converged root with the requested text
label is not necessarily the intended CT state. Quinoxaline also contains a
`2 B2` root at 6.6956984 eV with `gamma_CV=0.9015844`, which may reflect a
geometry-specific Cartesian-axis convention. That possibility cannot justify
a global B1/B2 swap: the supplied Phthalazine result already identifies its
high-CV target as `1 B1`. Every mapping must therefore be established per
geometry and state character.

The algorithmic defect is independent of that labeling issue. v0.3.6 seeded
only the globally lowest diagonal unit vectors. In an Abelian point group the
operator is block diagonal; Davidson cannot generate an irrep absent from the
initial subspace, and too few independent vectors within an irrep can also
miss a lower mixed state. A small residual only certifies the roots in the
Krylov space that was actually generated.

v0.4.0 corrects this by adding up to `nstates` lowest diagonal seeds in every
irrep for both MRSF and EMRSF, and records those counts in the log. The four
large calculations must be rerun before their disagreement is attributed to
the physical equations.

Separate equation checks found two additional issues in v0.3.6:

- S45 used separate finite-grid denominators `q_plus` and `q_minus`; the SI
  uses the common S44 charge `q` for both centroids.
- `A_G` was obtained only through the response action. v0.4.0 constructs SI
  eq. S23 explicitly and uses the response element as an equality check.

The published `sqrt(2)` G-row factor and S1-S2 spin combinations were verified
and are not fitted. The old sign/factor audit switches are removed.
