"""Perturbative state-interaction SOC for MRSF/EMRSF singlet-triplet roots.

The implementation follows the spin-adapted wavefunctions and Breit--Pauli
SOMF Hamiltonian of Komarov et al., JCTC 2023, DOI 10.1021/acs.jctc.2c01036.
The spin-adapted configuration-state functions are expanded independently into
sparse Slater determinants and contracted with the one-electron spinor
operator.  For EMRSF, the additional closed-shell LR C->V configurations are
included explicitly.  The distinct C->V rows inside the original MRSF block
are four-open-orbital configurations and remain projected out because the
published MRSF/EMRSF equations do not define their auxiliary-wavefunction
spin completion.  Their excluded weight is retained in every result and file.

The AO SOMF contraction follows eqs. 8--11 of that paper.  Its implementation
was cross-checked against the GPL-3.0-or-later ``fci-siso`` reference by Huanchen
Zhai; no OpenQP source code is used here.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping

import numpy as np
from scipy import sparse

from .mrsf import HARTREE_TO_EV

HARTREE_TO_WAVENUMBER = 219474.63136320


@dataclass(frozen=True)
class SOCIntegrals:
    """One-electron and mean-field two-electron SOC integrals."""

    one_electron_ao: np.ndarray
    two_electron_ao: np.ndarray
    one_electron_mo: np.ndarray
    two_electron_mo: np.ndarray
    approximation: str

    @property
    def total_ao(self) -> np.ndarray:
        return self.one_electron_ao + self.two_electron_ao

    @property
    def total_mo(self) -> np.ndarray:
        return self.one_electron_mo + self.two_electron_mo


@dataclass(frozen=True)
class SOCBasisState:
    """One spin component in the state-interaction basis."""

    label: str
    manifold: str
    root: int
    multiplicity: int
    ms: int
    spin_free_energy_hartree: float


@dataclass(frozen=True)
class SOCExternalContext:
    """Stable in-process and on-disk interface for external SOC modules.

    ``state_coefficients`` has one row per determinant and one column per
    spin-free state-interaction basis function.  The columns are ordered as
    all singlets followed by the three Ms components of every triplet root.
    Determinants use the package's alpha-then-beta spin-orbital bit ordering.
    """

    source_model: str
    reference: object
    basis_states: tuple[SOCBasisState, ...]
    determinants: tuple[int, ...]
    state_coefficients: np.ndarray
    integrals: SOCIntegrals
    singlet_count: int
    triplet_count: int
    singlet_eigenvalues_hartree: np.ndarray
    triplet_eigenvalues_hartree: np.ndarray
    singlet_eigenvectors: np.ndarray
    triplet_eigenvectors: np.ndarray
    singlet_basis_sectors: np.ndarray
    triplet_basis_sectors: np.ndarray
    singlet_basis_labels_zero_based: np.ndarray
    triplet_basis_labels_zero_based: np.ndarray
    excluded_singlet_mrsf_four_open_weight: np.ndarray
    excluded_triplet_mrsf_four_open_weight: np.ndarray
    included_singlet_lr_cv_weight: np.ndarray
    included_triplet_lr_cv_weight: np.ndarray

    def contract_one_body(self, spatial_mo: np.ndarray) -> np.ndarray:
        """Contract a Cartesian one-electron SOC operator with all states."""

        return _state_matrix_from_coefficients(
            self.determinants,
            self.state_coefficients,
            _spinor_operator(spatial_mo),
        )


@dataclass(frozen=True)
class SOCCoupling:
    """SOC coupling between one singlet and one triplet root."""

    singlet_state: int
    triplet_state: int
    ms_minus_one_hartree: complex
    ms_zero_hartree: complex
    ms_plus_one_hartree: complex
    one_electron_components_hartree: np.ndarray | None = None
    two_electron_components_hartree: np.ndarray | None = None

    @property
    def components_hartree(self) -> np.ndarray:
        return np.asarray(
            (
                self.ms_minus_one_hartree,
                self.ms_zero_hartree,
                self.ms_plus_one_hartree,
            ),
            dtype=complex,
        )

    @property
    def magnitude_hartree(self) -> float:
        return float(np.linalg.norm(self.components_hartree))

    @property
    def magnitude_wavenumber(self) -> float:
        return self.magnitude_hartree * HARTREE_TO_WAVENUMBER

    @property
    def one_electron_magnitude_wavenumber(self) -> float:
        if self.one_electron_components_hartree is None:
            return float("nan")
        return float(np.linalg.norm(self.one_electron_components_hartree)) * HARTREE_TO_WAVENUMBER

    @property
    def two_electron_magnitude_wavenumber(self) -> float:
        if self.two_electron_components_hartree is None:
            return float("nan")
        return float(np.linalg.norm(self.two_electron_components_hartree)) * HARTREE_TO_WAVENUMBER


@dataclass(frozen=True)
class SOCResult:
    """MRSF/EMRSF spin-orbit state-interaction matrix and adiabatic roots."""

    basis_states: tuple[SOCBasisState, ...]
    one_electron_matrix_hartree: np.ndarray
    two_electron_matrix_hartree: np.ndarray
    hamiltonian_hartree: np.ndarray
    eigenvalues_hartree: np.ndarray
    eigenvectors: np.ndarray
    singlet_count: int
    triplet_count: int
    excluded_singlet_mrsf_four_open_weight: np.ndarray
    excluded_triplet_mrsf_four_open_weight: np.ndarray
    included_singlet_lr_cv_weight: np.ndarray
    included_triplet_lr_cv_weight: np.ndarray
    source_model: str
    approximation: str
    hermiticity_error_hartree: float
    external_context: SOCExternalContext | None = None

    @property
    def excluded_singlet_cv_weight(self) -> np.ndarray:
        """Compatibility alias retained for v0.7.0 readers."""

        return self.excluded_singlet_mrsf_four_open_weight

    @property
    def excluded_triplet_cv_weight(self) -> np.ndarray:
        """Compatibility alias retained for v0.7.0 readers."""

        return self.excluded_triplet_mrsf_four_open_weight

    @property
    def soc_matrix_hartree(self) -> np.ndarray:
        return self.one_electron_matrix_hartree + self.two_electron_matrix_hartree

    @property
    def excitation_energies_ev(self) -> np.ndarray:
        return (self.eigenvalues_hartree - self.eigenvalues_hartree[0]) * HARTREE_TO_EV

    def couplings(self) -> tuple[SOCCoupling, ...]:
        values = []
        offset = self.singlet_count
        matrix = self.soc_matrix_hartree
        for singlet in range(self.singlet_count):
            for triplet in range(self.triplet_count):
                first = offset + 3 * triplet
                values.append(
                    SOCCoupling(
                        singlet_state=singlet,
                        triplet_state=triplet,
                        ms_minus_one_hartree=matrix[singlet, first],
                        ms_zero_hartree=matrix[singlet, first + 1],
                        ms_plus_one_hartree=matrix[singlet, first + 2],
                        one_electron_components_hartree=np.asarray(
                            self.one_electron_matrix_hartree[
                                singlet, first : first + 3
                            ],
                            dtype=complex,
                        ),
                        two_electron_components_hartree=np.asarray(
                            self.two_electron_matrix_hartree[
                                singlet, first : first + 3
                            ],
                            dtype=complex,
                        ),
                    )
                )
        return tuple(values)

    def write_log(self, path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            handle.write("=" * 79 + "\n")
            handle.write(f"                    PySCF-EMRSF SOC-{self.source_model}\n")
            handle.write("=" * 79 + "\n")
            handle.write("FORMULATION          perturbative Breit-Pauli state interaction\n")
            handle.write(f"SOC OPERATOR         {self.approximation}\n")
            handle.write(
                f"SPIN-FREE STATES     {self.source_model}-TDA singlets and triplets\n"
            )
            handle.write(
                "FORMAL STATUS        auxiliary-wavefunction state interaction\n"
            )
            handle.write(
                "ADDED LR-CV SECTOR   included for EMRSF; absent for MRSF\n"
            )
            handle.write(
                "MRSF C->V SECTOR     projected: four-open spin completion undefined\n"
            )
            handle.write(f"SINGLET ROOTS        {self.singlet_count}\n")
            handle.write(f"TRIPLET ROOTS        {self.triplet_count}\n")
            handle.write(
                "MAX EXCLUDED S 4O   "
                f"{float(np.max(self.excluded_singlet_mrsf_four_open_weight)):.8f}\n"
            )
            handle.write(
                "MAX EXCLUDED T 4O   "
                f"{float(np.max(self.excluded_triplet_mrsf_four_open_weight)):.8f}\n"
            )
            handle.write(
                "MAX INCLUDED S LR-CV "
                f"{float(np.max(self.included_singlet_lr_cv_weight)):.8f}\n"
            )
            handle.write(
                "MAX INCLUDED T LR-CV "
                f"{float(np.max(self.included_triplet_lr_cv_weight)):.8f}\n"
            )
            handle.write(
                f"HERMITICITY ERROR    {self.hermiticity_error_hartree:.6e} Eh\n"
            )
            handle.write("\nSINGLET-TRIPLET SOC COUPLINGS (cm^-1)\n")
            handle.write("-" * 79 + "\n")
            handle.write(
                "   S    T        Re(Ms=-1) Im(Ms=-1)"
                "    Re(Ms=0)  Im(Ms=0)   Re(Ms=+1) Im(Ms=+1)"
                "      |1e|      |2e|     |TOTAL|\n"
            )
            for value in self.couplings():
                components = value.components_hartree * HARTREE_TO_WAVENUMBER
                handle.write(
                    f" {value.singlet_state:3d} {value.triplet_state:4d}"
                    f" {components[0].real:11.4f} {components[0].imag:10.4f}"
                    f" {components[1].real:11.4f} {components[1].imag:10.4f}"
                    f" {components[2].real:11.4f} {components[2].imag:10.4f}"
                    f" {value.one_electron_magnitude_wavenumber:9.4f}"
                    f" {value.two_electron_magnitude_wavenumber:9.4f}"
                    f" {value.magnitude_wavenumber:10.4f}\n"
                )
            handle.write("\nSOC-ADIABATIC STATE-INTERACTION ENERGIES\n")
            handle.write("-" * 79 + "\n")
            handle.write(" STATE       ENERGY/Eh       EXCITATION/eV       DOMINANT BASIS      WEIGHT\n")
            for state, energy in enumerate(self.eigenvalues_hartree):
                weights = np.abs(self.eigenvectors[:, state]) ** 2
                dominant = int(np.argmax(weights))
                handle.write(
                    f" {state:5d} {energy:16.10f}"
                    f" {self.excitation_energies_ev[state]:19.8f}"
                    f" {self.basis_states[dominant].label:>20s}"
                    f" {weights[dominant]:11.7f}\n"
                )
            handle.write("\n**** PYSCF-EMRSF SOC NORMAL TERMINATION ****\n")
        return output

    def write_csv(self, path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "singlet_state",
                "triplet_state",
                "soc_ms_minus_1_real_cm-1",
                "soc_ms_minus_1_imag_cm-1",
                "soc_ms_0_real_cm-1",
                "soc_ms_0_imag_cm-1",
                "soc_ms_plus_1_real_cm-1",
                "soc_ms_plus_1_imag_cm-1",
                "soc_1e_magnitude_cm-1",
                "soc_somf_2e_magnitude_cm-1",
                "soc_magnitude_cm-1",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for value in self.couplings():
                component = value.components_hartree * HARTREE_TO_WAVENUMBER
                writer.writerow(
                    {
                        "singlet_state": value.singlet_state,
                        "triplet_state": value.triplet_state,
                        "soc_ms_minus_1_real_cm-1": component[0].real,
                        "soc_ms_minus_1_imag_cm-1": component[0].imag,
                        "soc_ms_0_real_cm-1": component[1].real,
                        "soc_ms_0_imag_cm-1": component[1].imag,
                        "soc_ms_plus_1_real_cm-1": component[2].real,
                        "soc_ms_plus_1_imag_cm-1": component[2].imag,
                        "soc_1e_magnitude_cm-1": value.one_electron_magnitude_wavenumber,
                        "soc_somf_2e_magnitude_cm-1": value.two_electron_magnitude_wavenumber,
                        "soc_magnitude_cm-1": value.magnitude_wavenumber,
                    }
                )
        return output


def _reference_density_ao(ref) -> np.ndarray:
    density = np.asarray(ref.mf.make_rdm1(ref.coeff, ref.occupation))
    if density.ndim == 3:
        density = density[0] + density[1]
    return np.asarray(density.real)


def _somf_jk(mol, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nao = mol.nao_nr()
    integrals = mol.intor("int2e_p1vxp1", comp=3).reshape(
        3, nao, nao, nao, nao
    )
    coulomb = np.einsum("xijkl,lk->xij", integrals, density, optimize=True)
    exchange = np.einsum("xijkl,jk->xil", integrals, density, optimize=True)
    exchange += np.einsum("xijkl,li->xkj", integrals, density, optimize=True)
    return coulomb, exchange


def _amfi_jk(mol, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Atomic mean-field approximation to the two-electron SOC screening."""

    ao_locations = mol.ao_loc_nr()
    nao = int(ao_locations[-1])
    coulomb = np.zeros((3, nao, nao))
    exchange = np.zeros_like(coulomb)
    atom = copy.copy(mol)
    for shell0, shell1, ao0, ao1 in mol.aoslice_by_atom(ao_locations):
        atom._bas = mol._bas[shell0:shell1]
        vj, vk = _somf_jk(atom, density[ao0:ao1, ao0:ao1])
        coulomb[:, ao0:ao1, ao0:ao1] = vj
        exchange[:, ao0:ao1, ao0:ao1] = vk
    return coulomb, exchange


def compute_soc_integrals(ref, *, two_electron: str = "amfi") -> SOCIntegrals:
    """Build Breit-Pauli one-electron and optional SOMF SOC integrals."""

    approximation = str(two_electron).strip().lower()
    if approximation not in {"none", "amfi", "full"}:
        raise ValueError("two_electron must be none, amfi, or full")
    from pyscf.data import nist

    mol = ref.mf.mol
    alpha2_over_two = float(nist.ALPHA) ** 2 / 2.0
    one_real = np.asarray(mol.intor_asymmetric("int1e_pnucxp", comp=3))
    one = 1j * alpha2_over_two * one_real
    if bool(getattr(mol, "has_ecp_soc", lambda: False)()):
        one -= 1j * np.asarray(mol.intor("ECPso"))

    two = np.zeros_like(one)
    if approximation != "none":
        density = _reference_density_ao(ref)
        if approximation == "full":
            nao = mol.nao_nr()
            estimate_mb = 3.0 * nao**4 * 8.0 / 1.0e6
            available_mb = float(getattr(ref.mf, "max_memory", 4000.0))
            if estimate_mb > 0.7 * available_mb:
                raise MemoryError(
                    "Full SOMF intermediate is estimated at "
                    f"{estimate_mb:.0f} MB, above 70% of max_memory; use AMFI "
                    "or increase --memory-mb explicitly"
                )
            coulomb, exchange = _somf_jk(mol, density)
        else:
            coulomb, exchange = _amfi_jk(mol, density)
        two = 1j * alpha2_over_two * (coulomb - 1.5 * exchange)

    # Numerical integral noise must not spoil Hermiticity: i times an
    # antisymmetric real spatial matrix is Hermitian.
    one = 0.5 * (one + one.transpose(0, 2, 1).conj())
    two = 0.5 * (two + two.transpose(0, 2, 1).conj())
    coefficient = np.asarray(ref.coeff)
    one_mo = np.einsum(
        "xuv,up,vq->xpq", one, coefficient.conj(), coefficient, optimize=True
    )
    two_mo = np.einsum(
        "xuv,up,vq->xpq", two, coefficient.conj(), coefficient, optimize=True
    )
    return SOCIntegrals(
        one_electron_ao=one,
        two_electron_ao=two,
        one_electron_mo=one_mo,
        two_electron_mo=two_mo,
        approximation={"none": "Breit-Pauli 1e", "amfi": "Breit-Pauli 1e + AMFI-SOMF", "full": "Breit-Pauli 1e + full SOMF"}[approximation],
    )


def _spinor_operator(spatial: np.ndarray) -> np.ndarray:
    spatial = np.asarray(spatial, dtype=complex)
    if spatial.ndim != 3 or spatial.shape[0] != 3:
        raise ValueError("SOC spatial integrals must have shape (3,nmo,nmo)")
    nmo = spatial.shape[1]
    hx, hy, hz = spatial
    operator = np.zeros((2 * nmo, 2 * nmo), dtype=complex)
    operator[:nmo, :nmo] = 0.5 * hz
    operator[nmo:, nmo:] = -0.5 * hz
    operator[:nmo, nmo:] = 0.5 * (hx - 1j * hy)
    operator[nmo:, :nmo] = 0.5 * (hx + 1j * hy)
    error = float(np.max(np.abs(operator - operator.T.conj())))
    if error > 1.0e-9:
        raise RuntimeError(f"SOC spinor operator is not Hermitian: {error:.3e}")
    return operator


def _determinant(alpha, beta, nmo: int) -> int:
    value = 0
    for orbital in alpha:
        value |= 1 << int(orbital)
    for orbital in beta:
        value |= 1 << (nmo + int(orbital))
    return value


def _add_open_pair(
    state: dict[int, complex],
    *,
    coefficient: complex,
    doubly_occupied,
    first: int,
    second: int,
    multiplicity: int,
    ms: int,
    nmo: int,
) -> None:
    first, second = sorted((int(first), int(second)))
    closed = set(map(int, doubly_occupied))
    if multiplicity == 1:
        scale = coefficient / np.sqrt(2.0)
        determinants = (
            (_determinant(closed | {first}, closed | {second}, nmo), scale),
            (_determinant(closed | {second}, closed | {first}, nmo), scale),
        )
    elif ms == 0:
        scale = coefficient / np.sqrt(2.0)
        determinants = (
            (_determinant(closed | {first}, closed | {second}, nmo), scale),
            (_determinant(closed | {second}, closed | {first}, nmo), -scale),
        )
    elif ms == 1:
        determinants = (
            (_determinant(closed | {first, second}, closed, nmo), coefficient),
        )
    elif ms == -1:
        determinants = (
            (_determinant(closed, closed | {first, second}, nmo), coefficient),
        )
    else:
        raise ValueError("Triplet Ms must be -1, 0, or +1")
    for determinant, value in determinants:
        state[determinant] = state.get(determinant, 0.0j) + value


def _mrsf_vector_wavefunction(
    space, vector: np.ndarray, ms: int
) -> tuple[dict[int, complex], float]:
    """Expand one MRSF vector in the representable SOC determinant basis."""

    multiplicity = int(space.target_multiplicity)
    if multiplicity == 1 and ms != 0:
        raise ValueError("A singlet has only Ms=0")
    vector = space.mask_redundant(np.asarray(vector))
    ref = space.ref
    closed = set(map(int, ref.closed))
    virtual = set(map(int, ref.virtual))
    o1, o2, nmo = ref.o1, ref.o2, ref.nmo
    wavefunction: dict[int, complex] = {}
    excluded_cv_weight = 0.0

    for amplitude, (source, target) in zip(vector, space.labels):
        amplitude = complex(amplitude)
        if abs(amplitude) < 1.0e-14:
            continue
        source, target = int(source), int(target)
        if source in closed and target in virtual:
            excluded_cv_weight += float(abs(amplitude) ** 2)
            continue
        if multiplicity == 1 and source == o2 and target == o1:
            determinant = _determinant(closed | {o1}, closed | {o1}, nmo)
            wavefunction[determinant] = wavefunction.get(determinant, 0.0j) + amplitude
            continue
        if multiplicity == 1 and source == o1 and target == o2:
            determinant = _determinant(closed | {o2}, closed | {o2}, nmo)
            wavefunction[determinant] = wavefunction.get(determinant, 0.0j) + amplitude
            continue
        if source == o1 and target == o1:
            doubly, first, second = closed, o1, o2
        elif source in closed and target == o1:
            doubly, first, second = (closed - {source}) | {o1}, source, o2
        elif source in closed and target == o2:
            doubly, first, second = (closed - {source}) | {o2}, source, o1
        elif source == o1 and target in virtual:
            doubly, first, second = closed, o2, target
        elif source == o2 and target in virtual:
            doubly, first, second = closed, o1, target
        else:
            # Redundant G/D/OO slots are zero after mask_redundant.
            continue
        _add_open_pair(
            wavefunction,
            coefficient=amplitude,
            doubly_occupied=doubly,
            first=first,
            second=second,
            multiplicity=multiplicity,
            ms=ms,
            nmo=nmo,
        )
    return wavefunction, excluded_cv_weight


def _mrsf_wavefunction(
    result, state_index: int, ms: int
) -> tuple[dict[int, complex], float]:
    """Expand one MRSF root in the projected SOC-MRSF determinant basis."""

    return _mrsf_vector_wavefunction(
        result.space, result.eigenvectors[:, int(state_index)], ms
    )


def _excite_determinant(
    determinant: int, source: int, target: int
) -> tuple[int, int]:
    """Apply ``a_target^dagger a_source`` and return determinant and phase."""

    if not determinant & (1 << source):
        raise ValueError("Excitation source is not occupied")
    if determinant & (1 << target):
        raise ValueError("Excitation target is already occupied")
    sign_remove = -1 if (determinant & ((1 << source) - 1)).bit_count() % 2 else 1
    removed = determinant ^ (1 << source)
    sign_add = -1 if (removed & ((1 << target) - 1)).bit_count() % 2 else 1
    return removed | (1 << target), sign_remove * sign_add


def _add_lr_cv_excitation(
    state: dict[int, complex],
    *,
    reference: int,
    coefficient: complex,
    source: int,
    target: int,
    multiplicity: int,
    ms: int,
    nmo: int,
) -> None:
    """Add a spin-adapted closed-shell LR C->V configuration."""

    operations: tuple[tuple[int, int, complex], ...]
    if multiplicity == 1:
        scale = coefficient / np.sqrt(2.0)
        operations = (
            (source, target, scale),
            (nmo + source, nmo + target, scale),
        )
    elif ms == 0:
        scale = coefficient / np.sqrt(2.0)
        operations = (
            (source, target, scale),
            (nmo + source, nmo + target, -scale),
        )
    elif ms == 1:
        operations = ((nmo + source, target, coefficient),)
    elif ms == -1:
        operations = ((source, nmo + target, coefficient),)
    else:
        raise ValueError("Triplet Ms must be -1, 0, or +1")

    for spin_source, spin_target, value in operations:
        determinant, phase = _excite_determinant(
            reference, spin_source, spin_target
        )
        state[determinant] = state.get(determinant, 0.0j) + phase * value


def _lr_cv_vector_wavefunction(
    cv_space,
    vector: np.ndarray,
    *,
    multiplicity: int,
    ms: int,
) -> dict[int, complex]:
    """Expand the added EMRSF LR-CV sector from its closed-shell G state."""

    if multiplicity == 1 and ms != 0:
        raise ValueError("A singlet has only Ms=0")
    vector = np.asarray(vector)
    if vector.size != cv_space.size:
        raise ValueError(f"Expected an LR-CV vector of length {cv_space.size}")
    ref = cv_space.ref
    doubly = set(map(int, ref.closed)) | {int(ref.o1)}
    closed_shell_g = _determinant(doubly, doubly, ref.nmo)
    wavefunction: dict[int, complex] = {}
    for amplitude, (source, target) in zip(vector, cv_space.labels):
        amplitude = complex(amplitude)
        if abs(amplitude) < 1.0e-14:
            continue
        _add_lr_cv_excitation(
            wavefunction,
            reference=closed_shell_g,
            coefficient=amplitude,
            source=int(source),
            target=int(target),
            multiplicity=int(multiplicity),
            ms=int(ms),
            nmo=ref.nmo,
        )
    return wavefunction


def _emrsf_wavefunction(
    result, state_index: int, ms: int
) -> tuple[dict[int, complex], float, float]:
    """Expand one EMRSF root, including its added LR-CV amplitudes.

    The returned weights are respectively the projected-out four-open MRSF
    C->V part and the explicitly included added LR-CV part.
    """

    multiplicity = int(result.target_multiplicity)
    vector = np.asarray(result.eigenvectors[:, int(state_index)])
    full_mrsf = np.zeros(result.mrsf_space.size, dtype=vector.dtype)
    full_mrsf[np.asarray(result.live_mrsf_indices, dtype=int)] = vector[
        : result.mrsf_size
    ]
    wavefunction, excluded = _mrsf_vector_wavefunction(
        result.mrsf_space, full_mrsf, ms
    )
    cv_vector = vector[result.mrsf_size :]
    lr_wavefunction = _lr_cv_vector_wavefunction(
        result.cv_space,
        cv_vector,
        multiplicity=multiplicity,
        ms=ms,
    )
    for determinant, value in lr_wavefunction.items():
        wavefunction[determinant] = wavefunction.get(determinant, 0.0j) + value
    included = float(np.vdot(cv_vector, cv_vector).real)
    return wavefunction, excluded, included


def _state_coefficients(
    wavefunctions: tuple[dict[int, complex], ...]
) -> tuple[tuple[int, ...], np.ndarray]:
    determinants = sorted({key for state in wavefunctions for key in state})
    lookup = {determinant: index for index, determinant in enumerate(determinants)}
    coefficients = np.zeros((len(determinants), len(wavefunctions)), dtype=complex)
    for state_index, state in enumerate(wavefunctions):
        for determinant, value in state.items():
            coefficients[lookup[determinant], state_index] = value
    return tuple(determinants), coefficients


def _state_matrix_from_coefficients(
    determinants: tuple[int, ...], coefficients: np.ndarray, operator: np.ndarray
) -> np.ndarray:
    lookup = {determinant: index for index, determinant in enumerate(determinants)}

    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    nspin = operator.shape[0]
    targets_by_source = [
        np.flatnonzero(np.abs(operator[:, source]) > 1.0e-16)
        for source in range(nspin)
    ]
    for column, determinant in enumerate(determinants):
        occupied = [index for index in range(nspin) if determinant & (1 << index)]
        for source in occupied:
            sign_remove = -1 if (determinant & ((1 << source) - 1)).bit_count() % 2 else 1
            removed = determinant ^ (1 << source)
            for target in targets_by_source[source]:
                target = int(target)
                if removed & (1 << target):
                    continue
                sign_add = -1 if (removed & ((1 << target) - 1)).bit_count() % 2 else 1
                generated = removed | (1 << target)
                row = lookup.get(generated)
                if row is None:
                    continue
                rows.append(row)
                columns.append(column)
                data.append(sign_remove * sign_add * operator[target, source])
    determinant_operator = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(determinants), len(determinants)),
        dtype=complex,
    ).tocsr()
    matrix = coefficients.T.conj() @ (determinant_operator @ coefficients)
    return np.asarray(matrix)


def _state_matrix(
    wavefunctions: tuple[dict[int, complex], ...], operator: np.ndarray
) -> np.ndarray:
    determinants, coefficients = _state_coefficients(wavefunctions)
    return _state_matrix_from_coefficients(determinants, coefficients, operator)


def _result_reference(result):
    return result.mrsf_space.ref if hasattr(result, "mrsf_size") else result.space.ref


def _response_basis_metadata(result) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(result, "mrsf_size"):
        mrsf_labels = np.asarray(result.mrsf_space.labels, dtype=int)[
            np.asarray(result.live_mrsf_indices, dtype=int)
        ]
        cv_labels = np.asarray(result.cv_space.labels, dtype=int).reshape(-1, 2)
        labels = np.vstack((mrsf_labels, cv_labels))
        sectors = np.asarray(
            ["MRSF"] * len(mrsf_labels) + ["LR-CV"] * len(cv_labels),
            dtype="U8",
        )
        return sectors, labels
    labels = np.asarray(result.space.labels, dtype=int).reshape(-1, 2)
    return np.asarray(["MRSF"] * len(labels), dtype="U8"), labels


def _build_soc_external_context(
    singlets,
    triplets,
    *,
    two_electron: str,
    source_model: str,
) -> SOCExternalContext:
    if int(singlets.target_multiplicity) != 1 or int(triplets.target_multiplicity) != 3:
        raise ValueError("SOC requires singlet and triplet result manifolds")
    singlet_ref = _result_reference(singlets)
    triplet_ref = _result_reference(triplets)
    if singlet_ref.mf is not triplet_ref.mf:
        raise ValueError("SOC manifolds must share the identical mean-field reference")

    is_emrsf = source_model == "EMRSF"
    if is_emrsf != hasattr(singlets, "mrsf_size") or is_emrsf != hasattr(
        triplets, "mrsf_size"
    ):
        raise ValueError(f"{source_model} SOC requires two {source_model} results")

    basis_states: list[SOCBasisState] = []
    wavefunctions: list[dict[int, complex]] = []
    excluded_s = np.zeros(len(singlets.eigenvalues))
    excluded_t = np.zeros(len(triplets.eigenvalues))
    included_s = np.zeros(len(singlets.eigenvalues))
    included_t = np.zeros(len(triplets.eigenvalues))

    for root, energy in enumerate(singlets.eigenvalues):
        if is_emrsf:
            wavefunction, excluded, included = _emrsf_wavefunction(
                singlets, root, 0
            )
        else:
            wavefunction, excluded = _mrsf_wavefunction(singlets, root, 0)
            included = 0.0
        wavefunctions.append(wavefunction)
        excluded_s[root] = excluded
        included_s[root] = included
        basis_states.append(
            SOCBasisState(f"S{root}", "singlet", root, 1, 0, float(energy))
        )

    for root, energy in enumerate(triplets.eigenvalues):
        for ms in (-1, 0, 1):
            if is_emrsf:
                wavefunction, excluded, included = _emrsf_wavefunction(
                    triplets, root, ms
                )
            else:
                wavefunction, excluded = _mrsf_wavefunction(triplets, root, ms)
                included = 0.0
            wavefunctions.append(wavefunction)
            excluded_t[root] = max(excluded_t[root], excluded)
            included_t[root] = max(included_t[root], included)
            basis_states.append(
                SOCBasisState(
                    f"T{root}(Ms={ms:+d})",
                    "triplet",
                    root,
                    3,
                    ms,
                    float(energy),
                )
            )

    determinants, coefficients = _state_coefficients(tuple(wavefunctions))
    singlet_sectors, singlet_labels = _response_basis_metadata(singlets)
    triplet_sectors, triplet_labels = _response_basis_metadata(triplets)
    return SOCExternalContext(
        source_model=source_model,
        reference=singlet_ref,
        basis_states=tuple(basis_states),
        determinants=determinants,
        state_coefficients=coefficients,
        integrals=compute_soc_integrals(
            singlet_ref, two_electron=two_electron
        ),
        singlet_count=len(singlets.eigenvalues),
        triplet_count=len(triplets.eigenvalues),
        singlet_eigenvalues_hartree=np.asarray(singlets.eigenvalues, dtype=float),
        triplet_eigenvalues_hartree=np.asarray(triplets.eigenvalues, dtype=float),
        singlet_eigenvectors=np.asarray(singlets.eigenvectors),
        triplet_eigenvectors=np.asarray(triplets.eigenvectors),
        singlet_basis_sectors=singlet_sectors,
        triplet_basis_sectors=triplet_sectors,
        singlet_basis_labels_zero_based=singlet_labels,
        triplet_basis_labels_zero_based=triplet_labels,
        excluded_singlet_mrsf_four_open_weight=excluded_s,
        excluded_triplet_mrsf_four_open_weight=excluded_t,
        included_singlet_lr_cv_weight=included_s,
        included_triplet_lr_cv_weight=included_t,
    )


def _soc_result_from_matrices(
    context: SOCExternalContext,
    one: np.ndarray,
    two: np.ndarray,
    *,
    approximation: str,
) -> SOCResult:
    size = len(context.basis_states)
    one = np.asarray(one, dtype=complex)
    two = np.asarray(two, dtype=complex)
    if one.shape != (size, size) or two.shape != (size, size):
        raise ValueError(
            f"SOC component matrices must both have shape {(size, size)}"
        )
    if not np.isfinite(one).all() or not np.isfinite(two).all():
        raise ValueError("SOC matrices contain non-finite values")
    soc = one + two
    hermiticity_error = float(np.max(np.abs(soc - soc.T.conj())))
    if hermiticity_error > 1.0e-8:
        raise RuntimeError(
            f"SOC state-interaction matrix is not Hermitian: {hermiticity_error:.3e} Eh"
        )
    soc = 0.5 * (soc + soc.T.conj())
    spin_free = np.asarray(
        [state.spin_free_energy_hartree for state in context.basis_states],
        dtype=float,
    )
    hamiltonian = soc + np.diag(spin_free)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return SOCResult(
        basis_states=context.basis_states,
        one_electron_matrix_hartree=one,
        two_electron_matrix_hartree=two,
        hamiltonian_hartree=hamiltonian,
        eigenvalues_hartree=eigenvalues,
        eigenvectors=eigenvectors,
        singlet_count=context.singlet_count,
        triplet_count=context.triplet_count,
        excluded_singlet_mrsf_four_open_weight=(
            context.excluded_singlet_mrsf_four_open_weight
        ),
        excluded_triplet_mrsf_four_open_weight=(
            context.excluded_triplet_mrsf_four_open_weight
        ),
        included_singlet_lr_cv_weight=context.included_singlet_lr_cv_weight,
        included_triplet_lr_cv_weight=context.included_triplet_lr_cv_weight,
        source_model=context.source_model,
        approximation=approximation,
        hermiticity_error_hartree=hermiticity_error,
        external_context=context,
    )


def _compute_soc_from_context(context: SOCExternalContext) -> SOCResult:
    one = context.contract_one_body(context.integrals.one_electron_mo)
    two = context.contract_one_body(context.integrals.two_electron_mo)
    return _soc_result_from_matrices(
        context,
        one,
        two,
        approximation=context.integrals.approximation,
    )


def compute_mrsf_soc(
    singlets, triplets, *, two_electron: str = "amfi"
) -> SOCResult:
    """Compute auxiliary-wavefunction SOC between MRSF manifolds.

    ``singlets`` and ``triplets`` must use the identical triplet reference and
    orbital ordering.
    """

    context = _build_soc_external_context(
        singlets,
        triplets,
        two_electron=two_electron,
        source_model="MRSF",
    )
    return _compute_soc_from_context(context)


def compute_emrsf_soc(
    singlets, triplets, *, two_electron: str = "amfi"
) -> SOCResult:
    """Compute projected EMRSF singlet-triplet SOC including added LR-CV.

    This is an explicitly labelled EMRSF-TDA auxiliary-wavefunction state
    interaction.  The added LR-CV sector is included; the unresolved
    four-open MRSF C->V component is projected out and its weight reported.
    """

    context = _build_soc_external_context(
        singlets,
        triplets,
        two_electron=two_electron,
        source_model="EMRSF",
    )
    return _compute_soc_from_context(context)


def _load_external_module(path) -> ModuleType:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"External SOC module not found: {source}")
    spec = importlib.util.spec_from_file_location(
        f"pyscf_emrsf_external_soc_{abs(hash(source))}", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import external SOC module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_external_soc(
    module_path,
    singlets,
    triplets,
    *,
    two_electron: str = "amfi",
) -> SOCResult:
    """Run ``compute_soc(context)`` from an explicitly supplied Python module.

    The callback may return a :class:`SOCResult`, a complete Hermitian SOC
    matrix in Hartree, or a mapping containing ``soc_matrix_hartree`` or the
    separate ``one_electron_matrix_hartree`` and
    ``two_electron_matrix_hartree`` arrays.
    """

    source_model = "EMRSF" if hasattr(singlets, "mrsf_size") else "MRSF"
    context = _build_soc_external_context(
        singlets,
        triplets,
        two_electron=two_electron,
        source_model=source_model,
    )
    module = _load_external_module(module_path)
    callback = getattr(module, "compute_soc", None)
    if not callable(callback):
        raise AttributeError(
            "External SOC module must define a callable compute_soc(context)"
        )
    returned = callback(context)
    if isinstance(returned, SOCResult):
        return _soc_result_from_matrices(
            context,
            returned.one_electron_matrix_hartree,
            returned.two_electron_matrix_hartree,
            approximation=returned.approximation,
        )
    approximation = f"external module {Path(module_path).name}"
    size = len(context.basis_states)
    zero = np.zeros((size, size), dtype=complex)
    if isinstance(returned, Mapping):
        approximation = str(returned.get("approximation", approximation))
        if "soc_matrix_hartree" in returned:
            one = np.asarray(returned["soc_matrix_hartree"], dtype=complex)
            two = zero
        else:
            one = np.asarray(
                returned.get("one_electron_matrix_hartree", zero), dtype=complex
            )
            two = np.asarray(
                returned.get("two_electron_matrix_hartree", zero), dtype=complex
            )
    else:
        one = np.asarray(returned, dtype=complex)
        two = zero
    return _soc_result_from_matrices(
        context, one, two, approximation=approximation
    )


def save_soc_archive(path, result: SOCResult) -> Path:
    """Save SOC matrices, raw response vectors, and the external-module basis."""

    if result.external_context is None:
        raise ValueError("SOCResult does not retain an external-module context")
    context = result.external_context
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    ref = context.reference
    mol = ref.mf.mol
    np.savez_compressed(
        output,
        archive_format_version=np.asarray(3, dtype=np.int64),
        source_model=np.asarray(context.source_model),
        soc_approximation=np.asarray(result.approximation),
        singlet_count=np.asarray(context.singlet_count, dtype=np.int64),
        triplet_count=np.asarray(context.triplet_count, dtype=np.int64),
        basis_labels=np.asarray([state.label for state in result.basis_states], dtype="U48"),
        basis_multiplicity=np.asarray([state.multiplicity for state in result.basis_states], dtype=int),
        basis_ms=np.asarray([state.ms for state in result.basis_states], dtype=int),
        basis_spin_free_energy_hartree=np.asarray(
            [state.spin_free_energy_hartree for state in result.basis_states], dtype=float
        ),
        determinant_hex=np.asarray([hex(value) for value in context.determinants], dtype="U256"),
        determinant_state_coefficients=context.state_coefficients,
        singlet_response_eigenvalues_hartree=context.singlet_eigenvalues_hartree,
        triplet_response_eigenvalues_hartree=context.triplet_eigenvalues_hartree,
        singlet_response_eigenvectors_columns_are_states=context.singlet_eigenvectors,
        triplet_response_eigenvectors_columns_are_states=context.triplet_eigenvectors,
        singlet_response_basis_sectors=context.singlet_basis_sectors,
        triplet_response_basis_sectors=context.triplet_basis_sectors,
        singlet_response_basis_labels_zero_based=context.singlet_basis_labels_zero_based,
        triplet_response_basis_labels_zero_based=context.triplet_basis_labels_zero_based,
        one_electron_soc_ao=context.integrals.one_electron_ao,
        somf_two_electron_soc_ao=context.integrals.two_electron_ao,
        one_electron_soc_mo=context.integrals.one_electron_mo,
        somf_two_electron_soc_mo=context.integrals.two_electron_mo,
        one_electron_soc_hartree=result.one_electron_matrix_hartree,
        two_electron_soc_hartree=result.two_electron_matrix_hartree,
        total_soc_hartree=result.soc_matrix_hartree,
        spin_orbit_hamiltonian_hartree=result.hamiltonian_hartree,
        soc_adiabatic_energies_hartree=result.eigenvalues_hartree,
        soc_eigenvectors_columns_are_states=result.eigenvectors,
        hermiticity_error_hartree=np.asarray(
            result.hermiticity_error_hartree, dtype=float
        ),
        excluded_singlet_mrsf_four_open_weight=(
            result.excluded_singlet_mrsf_four_open_weight
        ),
        excluded_triplet_mrsf_four_open_weight=(
            result.excluded_triplet_mrsf_four_open_weight
        ),
        included_singlet_lr_cv_weight=result.included_singlet_lr_cv_weight,
        included_triplet_lr_cv_weight=result.included_triplet_lr_cv_weight,
        mo_coeff_ao_by_mo=np.asarray(ref.coeff),
        mo_energies_hartree=np.asarray(ref.energy),
        mo_occupations=np.asarray(ref.occupation),
        closed_orbitals_zero_based=np.asarray(ref.closed, dtype=int),
        open_orbitals_zero_based=np.asarray(ref.open, dtype=int),
        virtual_orbitals_zero_based=np.asarray(ref.virtual, dtype=int),
        atom_symbols=np.asarray(
            [mol.atom_symbol(index) for index in range(mol.natm)], dtype="U8"
        ),
        atom_charges=np.asarray(mol.atom_charges(), dtype=float),
        atom_coordinates_bohr=np.asarray(mol.atom_coords(), dtype=float),
        molecular_charge=np.asarray(mol.charge, dtype=int),
        molecular_spin_2s=np.asarray(mol.spin, dtype=int),
        pyscf_molecule_json=np.asarray(mol.dumps()),
    )
    return output
