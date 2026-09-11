# Scientific and software provenance

## EMRSF equations

M. Oh, N. Kim, Y. Jung, C. H. Choi, and S. Lee, “Extended
Mixed-Reference Spin-Flip Time-Dependent Density Functional Theory for
Charge-Transfer State,” *J. Chem. Theory Comput.* **22** (2026), 6057–6064.
DOI: https://doi.org/10.1021/acs.jctc.6c00454

Supporting Information: https://doi.org/10.1021/acs.jctc.6c00454.s001

Open preprint used for cross-reading the main text:
https://chemrxiv.org/doi/10.26434/chemrxiv.15000818

Code mapping:

- SI eqs. S31–S38: `src/pyscf_emrsf/lr.py`;
- SI eq. S30: `CorrectedLRTDA` in `lr.py`;
- SI eq. S39 and Tables S1–S3: exact reference in
  `src/pyscf_emrsf/coupling.py`, RI operator in `src/pyscf_emrsf/ri.py`;
- matrix-free corrected block action: `src/pyscf_emrsf/emrsf.py`;
- SI eqs. S42–S47: `src/pyscf_emrsf/observables.py`;
- ORCA-like report of all scalar diagnostics: `src/pyscf_emrsf/log.py`.

## CT benchmark geometries

P.-F. Loos, M. Comin, X. Blase, and D. Jacquemin, “Reference Energies for
Intramolecular Charge-Transfer Excitations,” *J. Chem. Theory Comput.* (2021).
The bohr coordinates in `examples/paper_ct_benchmark/geometries` are copied
verbatim from sections S6.3 and S6.15--S6.18 of that work's Supporting
Information.

## Conventional MRSF

S. Lee, M. Filatov, S. Lee, and C. H. Choi, “Eliminating spin-contamination
of spin-flip time dependent density functional theory within linear response
formalism by the use of zeroth-order mixed-reference reduced density matrix,”
*J. Chem. Phys.* **149** (2018), 104101.
DOI: https://doi.org/10.1063/1.5044202

S. Lee, E. E. Kim, H. Nakata, S. Lee, and C. H. Choi, “Efficient
implementations of analytic energy gradient for mixed-reference spin-flip
time-dependent density functional theory,” *J. Chem. Phys.* **150** (2019),
184111. DOI: https://doi.org/10.1063/1.5086895

## Vertical triplet benchmark

M. Schreiber, M. R. Silva-Junior, S. P. A. Sauer, and W. Thiel,
“Benchmarks for electronically excited states: CASPT2, CC2, CCSD, and
CC3,” *J. Chem. Phys.* **128** (2008), 134110.
DOI: https://doi.org/10.1063/1.2889385

The five geometries and nine CC3/TZVP triplet values in
`examples/triplet_benchmark_schreiber2008` are external validation data and
are not used by the runtime implementation.

## Spin-orbit coupling

K. Komarov, W. Park, S. Lee, T. Zeng, and C. H. Choi, “Accurate Spin-Orbit
Coupling and Intersystem Crossing by Relativistic Mixed-Reference Spin-Flip
(MRSF)-TDDFT,” *J. Chem. Theory Comput.* **19** (2023), 1512--1522.
DOI: https://doi.org/10.1021/acs.jctc.2c01036

`src/pyscf_emrsf/soc.py` implements its perturbative Breit--Pauli/SOMF state
interaction from the published equations and explicit spin-adapted MRSF
wavefunctions. CV-type configurations are excluded from SOC transition
densities as required by the paper's time-reversal construction.

The AO SOMF contraction was independently cross-checked against H. Zhai's
GPL-3.0-or-later `fci-siso` implementation:
https://github.com/hczhai/fci-siso

## Continuum solvation

Reference-state PCM uses the public PySCF PCM implementation and its
self-consistent SCF wrapper:
https://pyscf.org/_modules/pyscf/solvent/pcm.html

The reaction field is frozen during the custom MRSF/EMRSF response. This
release does not claim an excited-state PCM response kernel for MRSF.

## Numerical oracle

OpenQP repository: https://github.com/Open-Quantum-Platform/openqp

OpenQP 1.2.1 source snapshot used by the audit:
https://github.com/Open-Quantum-Platform/openqp/tree/v1.2.1

V. Makhnev *et al.*, “OpenQP: A Quantum Chemical Platform Featuring
MRSF-TDDFT with an Emphasis on Open-Source Ecosystem,” *J. Chem. Theory
Comput.* **20** (2024), 9464--9477.
DOI: https://doi.org/10.1021/acs.jctc.4c01117

The automated public-example check records results from OpenQP v1.2.1.  OpenQP
is not a runtime dependency and is not bundled in this project.

## Applicability diagnostics

J. Janos, A. J. Orr-Ewing, B. F. E. Curchod, and P. Slavicek,
“Limitations of MRSF-TDDFT for Applications in Photochemistry,” 2026.
DOI: https://doi.org/10.1021/acs.jpclett.6c01200

This work motivates explicit triplet-reference tracking: it documents missing
singly excited configurations in conventional MRSF and discontinuities when
the selected T1/T2 reference changes character. These are method limitations,
not a license to alter the published response equations.
