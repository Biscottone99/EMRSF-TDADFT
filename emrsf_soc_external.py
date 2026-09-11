#!/usr/bin/env python3
"""Calcolo SOC esterno per autostati PySCF-EMRSF/MRSF v0.8 o successivi.

Il file puo' essere usato in tre modi.

1. Come modulo esterno del driver PySCF-EMRSF::

       python run_xyz_emrsf.py molecola.xyz \
         --method emrsf --states both --write-eigenvectors \
         --soc-external-module emrsf_soc_external.py \
         --soc-two-electron amfi

2. Come programma autonomo, partendo dai due archivi degli autostati::

       python emrsf_soc_external.py states singoletti.npz tripletti.npz \
         --two-electron amfi --output-prefix risultati/soc_emrsf

3. Come post-processore NumPy-only di un archivio SOC external-ready::

       python emrsf_soc_external.py archive calcolo_soc_emrsf.npz \
         --output-prefix risultati/soc_ricalcolato

La contrazione nello spazio dei determinanti e' implementata in questo file e
non richiama le routine SOC interne di ``pyscf_emrsf``. La modalita' ``states``
richiede PySCF per ricalcolare gli integrali AO; ``archive`` richiede soltanto
NumPy. Le configurazioni LR-CV di EMRSF sono incluse. Le configurazioni C->V
del blocco MRSF che richiedono un completamento spin-adattato a quattro
orbitali aperti sono proiettate senza rinormalizzazione e il loro peso viene
riportato nei risultati.
"""

import argparse
import copy
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_WAVENUMBER = 219474.63136320
PROGRAM_VERSION = "1.0.0"


@dataclass(frozen=True)
class SOCInput:
    """Dati completi necessari alla contrazione SOC esterna."""

    source_model: str
    approximation: str
    origin: str
    determinants: tuple[int, ...]
    state_coefficients: np.ndarray
    one_electron_mo: np.ndarray
    two_electron_mo: np.ndarray
    basis_labels: np.ndarray
    basis_multiplicity: np.ndarray
    basis_ms: np.ndarray
    basis_spin_free_energy_hartree: np.ndarray
    excluded_singlet_four_open_weight: np.ndarray
    excluded_triplet_four_open_weight: np.ndarray
    included_singlet_lr_cv_weight: np.ndarray
    included_triplet_lr_cv_weight: np.ndarray
    archived_one_electron_matrix: np.ndarray | None = None
    archived_two_electron_matrix: np.ndarray | None = None
    wavefunction_norm_error: float = float("nan")


@dataclass(frozen=True)
class ExternalSOCResult:
    """Risultato della contrazione e della diagonalizzazione SOC."""

    data: SOCInput
    one_electron_matrix_hartree: np.ndarray
    two_electron_matrix_hartree: np.ndarray
    total_soc_matrix_hartree: np.ndarray
    hamiltonian_hartree: np.ndarray
    eigenvalues_hartree: np.ndarray
    eigenvectors: np.ndarray
    hermiticity_error_hartree: float
    validation_one_electron_error_hartree: float
    validation_two_electron_error_hartree: float


def _spinor_operator(spatial_mo: np.ndarray) -> np.ndarray:
    """Trasforma le tre componenti cartesiane nell'operatore spinoriale."""

    spatial = np.asarray(spatial_mo, dtype=complex)
    if spatial.ndim != 3 or spatial.shape[0] != 3:
        raise ValueError("Gli integrali SOC devono avere forma (3,nmo,nmo)")
    if spatial.shape[1] != spatial.shape[2]:
        raise ValueError("Le matrici SOC spaziali devono essere quadrate")
    nmo = spatial.shape[1]
    hx, hy, hz = spatial
    operator = np.zeros((2 * nmo, 2 * nmo), dtype=complex)
    operator[:nmo, :nmo] = 0.5 * hz
    operator[nmo:, nmo:] = -0.5 * hz
    operator[:nmo, nmo:] = 0.5 * (hx - 1j * hy)
    operator[nmo:, :nmo] = 0.5 * (hx + 1j * hy)
    error = _hermiticity_error(operator)
    if error > 1.0e-9:
        raise RuntimeError(
            f"L'operatore SOC spinoriale non e' hermitiano: {error:.3e} Eh"
        )
    return operator


def _hermiticity_error(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix - matrix.T.conj())))


def _validated_hermitian(
    matrix: np.ndarray, name: str, tolerance: float
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(matrix, dtype=complex)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contiene valori non finiti")
    error = _hermiticity_error(matrix)
    if error > tolerance:
        raise RuntimeError(
            f"{name} non e' hermitiana: errore {error:.3e} Eh "
            f"> tolleranza {tolerance:.3e} Eh"
        )
    return 0.5 * (matrix + matrix.T.conj()), error


def _contract_one_body(
    determinants: tuple[int, ...],
    state_coefficients: np.ndarray,
    spatial_mo: np.ndarray,
) -> np.ndarray:
    """Calcola B^dagger h B applicando esplicitamente Slater--Condon."""

    operator = _spinor_operator(spatial_mo)
    coefficients = np.asarray(state_coefficients, dtype=complex)
    if coefficients.ndim != 2 or coefficients.shape[0] != len(determinants):
        raise ValueError(
            "state_coefficients deve avere una riga per ogni determinante"
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("I coefficienti degli autostati contengono valori non finiti")

    nspin = operator.shape[0]
    if any(value < 0 or value.bit_length() > nspin for value in determinants):
        raise ValueError("Un determinante contiene spin-orbitali fuori intervallo")
    electron_counts = {value.bit_count() for value in determinants}
    if len(electron_counts) != 1:
        raise ValueError("I determinanti non hanno tutti lo stesso numero di elettroni")
    if len(set(determinants)) != len(determinants):
        raise ValueError("La lista dei determinanti contiene duplicati")

    lookup = {determinant: index for index, determinant in enumerate(determinants)}
    applied = np.zeros_like(coefficients, dtype=complex)
    targets_by_source = [
        np.flatnonzero(np.abs(operator[:, source]) > 1.0e-16)
        for source in range(nspin)
    ]

    for column, determinant in enumerate(determinants):
        for source in range(nspin):
            if not determinant & (1 << source):
                continue
            lower_source = determinant & ((1 << source) - 1)
            sign_remove = -1 if lower_source.bit_count() % 2 else 1
            removed = determinant ^ (1 << source)
            for target_value in targets_by_source[source]:
                target = int(target_value)
                if removed & (1 << target):
                    continue
                lower_target = removed & ((1 << target) - 1)
                sign_add = -1 if lower_target.bit_count() % 2 else 1
                generated = removed | (1 << target)
                row = lookup.get(generated)
                if row is None:
                    continue
                factor = sign_remove * sign_add * operator[target, source]
                applied[row, :] += factor * coefficients[column, :]

    return np.asarray(coefficients.T.conj() @ applied)


def compute_soc(context: Any) -> dict[str, Any]:
    """Callback pubblico per ``--soc-external-module``.

    Il driver passa un ``SOCExternalContext``. Questa funzione utilizza solo
    gli attributi pubblici del contesto e svolge localmente la contrazione.
    """

    determinants = tuple(int(value) for value in context.determinants)
    coefficients = np.asarray(context.state_coefficients, dtype=complex)
    one = _contract_one_body(
        determinants, coefficients, context.integrals.one_electron_mo
    )
    two = _contract_one_body(
        determinants, coefficients, context.integrals.two_electron_mo
    )
    return {
        "one_electron_matrix_hartree": one,
        "two_electron_matrix_hartree": two,
        "approximation": (
            "external Slater-Condon contraction: "
            f"{context.integrals.approximation}"
        ),
    }


def _determinant(alpha: set[int], beta: set[int], nmo: int) -> int:
    value = 0
    for orbital in alpha:
        value |= 1 << int(orbital)
    for orbital in beta:
        value |= 1 << (nmo + int(orbital))
    return value


def _add_open_pair(
    state: dict[int, complex],
    coefficient: complex,
    doubly_occupied: set[int],
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
        terms = (
            (_determinant(closed | {first}, closed | {second}, nmo), scale),
            (_determinant(closed | {second}, closed | {first}, nmo), scale),
        )
    elif ms == 0:
        scale = coefficient / np.sqrt(2.0)
        terms = (
            (_determinant(closed | {first}, closed | {second}, nmo), scale),
            (_determinant(closed | {second}, closed | {first}, nmo), -scale),
        )
    elif ms == 1:
        terms = ((_determinant(closed | {first, second}, closed, nmo), coefficient),)
    elif ms == -1:
        terms = ((_determinant(closed, closed | {first, second}, nmo), coefficient),)
    else:
        raise ValueError("Per un tripletto Ms deve essere -1, 0 oppure +1")
    for determinant, value in terms:
        state[determinant] = state.get(determinant, 0.0j) + value


def _mrsf_vector_wavefunction(
    labels: np.ndarray,
    vector: np.ndarray,
    closed_orbitals: np.ndarray,
    open_orbitals: np.ndarray,
    virtual_orbitals: np.ndarray,
    multiplicity: int,
    ms: int,
    nmo: int,
) -> tuple[dict[int, complex], float]:
    """Espande un autovettore MRSF nella base dei determinanti SOC."""

    if multiplicity == 1 and ms != 0:
        raise ValueError("Un singoletto possiede soltanto Ms=0")
    labels = np.asarray(labels, dtype=int).reshape(-1, 2)
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    if len(labels) != len(vector):
        raise ValueError("Il numero di righe MRSF non coincide con l'autovettore")
    closed = set(map(int, closed_orbitals))
    virtual = set(map(int, virtual_orbitals))
    if len(open_orbitals) != 2:
        raise ValueError("Il riferimento MRSF deve avere due orbitali aperti")
    o1, o2 = map(int, open_orbitals)
    wavefunction: dict[int, complex] = {}
    excluded = 0.0

    for amplitude, pair in zip(vector, labels):
        source, target = map(int, pair)
        if (source == o2 and target == o2) or (
            multiplicity == 3
            and ((source == o1 and target == o2) or (source == o2 and target == o1))
        ):
            continue
        if abs(amplitude) < 1.0e-14:
            continue
        if source in closed and target in virtual:
            excluded += float(abs(amplitude) ** 2)
            continue
        if multiplicity == 1 and source == o2 and target == o1:
            det = _determinant(closed | {o1}, closed | {o1}, nmo)
            wavefunction[det] = wavefunction.get(det, 0.0j) + amplitude
            continue
        if multiplicity == 1 and source == o1 and target == o2:
            det = _determinant(closed | {o2}, closed | {o2}, nmo)
            wavefunction[det] = wavefunction.get(det, 0.0j) + amplitude
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
            continue
        _add_open_pair(
            wavefunction,
            amplitude,
            doubly,
            first,
            second,
            multiplicity,
            ms,
            nmo,
        )
    return wavefunction, excluded


def _excite_determinant(determinant: int, source: int, target: int) -> tuple[int, int]:
    if not determinant & (1 << source):
        raise ValueError("La sorgente dell'eccitazione non e' occupata")
    if determinant & (1 << target):
        raise ValueError("Il target dell'eccitazione e' gia' occupato")
    sign_remove = -1 if (determinant & ((1 << source) - 1)).bit_count() % 2 else 1
    removed = determinant ^ (1 << source)
    sign_add = -1 if (removed & ((1 << target) - 1)).bit_count() % 2 else 1
    return removed | (1 << target), sign_remove * sign_add


def _add_lr_cv_excitation(
    state: dict[int, complex],
    reference: int,
    coefficient: complex,
    source: int,
    target: int,
    multiplicity: int,
    ms: int,
    nmo: int,
) -> None:
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
        raise ValueError("Per un tripletto Ms deve essere -1, 0 oppure +1")
    for spin_source, spin_target, value in operations:
        generated, phase = _excite_determinant(reference, spin_source, spin_target)
        state[generated] = state.get(generated, 0.0j) + phase * value


def _state_wavefunction(
    archive: Any, state_index: int, ms: int
) -> tuple[dict[int, complex], float, float]:
    multiplicity = int(np.asarray(archive["target_multiplicity"]).item())
    result_type = str(np.asarray(archive["result_type"]).item()).upper()
    eigenvectors = np.asarray(
        archive["eigenvectors_columns_are_states"], dtype=complex
    )
    vector = eigenvectors[:, int(state_index)]
    labels = np.asarray(archive["mrsf_labels_zero_based"], dtype=int)
    closed = np.asarray(archive["closed_orbitals_zero_based"], dtype=int)
    opened = np.asarray(archive["open_orbitals_zero_based"], dtype=int)
    virtual = np.asarray(archive["virtual_orbitals_zero_based"], dtype=int)
    nmo = int(np.asarray(archive["mo_coeff_ao_by_mo"]).shape[1])

    if result_type == "EMRSF":
        mrsf_size = int(np.asarray(archive["emrsf_mrsf_block_size"]).item())
        live = np.asarray(archive["emrsf_live_mrsf_indices"], dtype=int)
        if mrsf_size != len(live):
            raise ValueError("Dimensione del blocco MRSF EMRSF non coerente")
        full_mrsf = np.zeros(len(labels), dtype=complex)
        full_mrsf[live] = vector[:mrsf_size]
        wavefunction, excluded = _mrsf_vector_wavefunction(
            labels, full_mrsf, closed, opened, virtual, multiplicity, ms, nmo
        )
        cv_labels = np.asarray(archive["emrsf_cv_labels_zero_based"], dtype=int)
        cv_vector = vector[mrsf_size:]
        if len(cv_vector) != len(cv_labels):
            raise ValueError("Dimensione del blocco LR-CV EMRSF non coerente")
        doubly = set(map(int, closed)) | {int(opened[0])}
        ground = _determinant(doubly, doubly, nmo)
        for amplitude, pair in zip(cv_vector, cv_labels):
            if abs(amplitude) < 1.0e-14:
                continue
            source, target = map(int, pair)
            _add_lr_cv_excitation(
                wavefunction,
                ground,
                complex(amplitude),
                source,
                target,
                multiplicity,
                ms,
                nmo,
            )
        included = float(np.vdot(cv_vector, cv_vector).real)
        return wavefunction, excluded, included

    if result_type != "MRSF":
        raise ValueError(f"Tipo di risultato non supportato: {result_type}")
    wavefunction, excluded = _mrsf_vector_wavefunction(
        labels, vector, closed, opened, virtual, multiplicity, ms, nmo
    )
    return wavefunction, excluded, 0.0


def _state_coefficients(
    wavefunctions: list[dict[int, complex]],
) -> tuple[tuple[int, ...], np.ndarray]:
    determinants = tuple(sorted({det for state in wavefunctions for det in state}))
    if not determinants:
        raise ValueError("La base determinantal SOC e' vuota")
    lookup = {determinant: index for index, determinant in enumerate(determinants)}
    coefficients = np.zeros((len(determinants), len(wavefunctions)), dtype=complex)
    for state_index, state in enumerate(wavefunctions):
        for determinant, value in state.items():
            coefficients[lookup[determinant], state_index] = value
    return determinants, coefficients


def _same_array(left: Any, right: Any, key: str, tolerance: float = 1.0e-10) -> None:
    a = np.asarray(left[key])
    b = np.asarray(right[key])
    if a.shape != b.shape or not np.allclose(a, b, atol=tolerance, rtol=tolerance):
        raise ValueError(f"Gli archivi non condividono lo stesso riferimento: {key}")


def _validate_state_archives(singlets: Any, triplets: Any) -> str:
    if int(np.asarray(singlets["archive_format_version"]).item()) < 3:
        raise ValueError("L'archivio dei singoletti deve avere formato >= 3")
    if int(np.asarray(triplets["archive_format_version"]).item()) < 3:
        raise ValueError("L'archivio dei tripletti deve avere formato >= 3")
    if int(np.asarray(singlets["target_multiplicity"]).item()) != 1:
        raise ValueError("Il primo archivio non contiene stati singoletti")
    if int(np.asarray(triplets["target_multiplicity"]).item()) != 3:
        raise ValueError("Il secondo archivio non contiene stati tripletti")
    model_s = str(np.asarray(singlets["result_type"]).item()).upper()
    model_t = str(np.asarray(triplets["result_type"]).item()).upper()
    if model_s != model_t or model_s not in {"MRSF", "EMRSF"}:
        raise ValueError("Gli archivi devono essere entrambi MRSF oppure entrambi EMRSF")
    for archive, name in ((singlets, "singoletti"), (triplets, "tripletti")):
        if not np.all(np.asarray(archive["state_converged"], dtype=bool)):
            raise ValueError(f"Non tutti gli autostati {name} sono convergenti")
    for key in (
        "mo_coeff_ao_by_mo",
        "mo_occupations",
        "closed_orbitals_zero_based",
        "open_orbitals_zero_based",
        "virtual_orbitals_zero_based",
        "atom_charges",
        "atom_coordinates_bohr",
    ):
        _same_array(singlets, triplets, key)
    for key in ("molecular_charge", "molecular_spin_2s", "electron_count"):
        if int(np.asarray(singlets[key]).item()) != int(np.asarray(triplets[key]).item()):
            raise ValueError(f"Gli archivi non sono compatibili: {key}")
    if str(np.asarray(singlets["pyscf_molecule_json"]).item()) != str(
        np.asarray(triplets["pyscf_molecule_json"]).item()
    ):
        raise ValueError("Le definizioni PySCF della molecola non coincidono")
    return model_s


def _somf_jk(mol: Any, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nao = mol.nao_nr()
    integrals = mol.intor("int2e_p1vxp1", comp=3).reshape(3, nao, nao, nao, nao)
    coulomb = np.einsum("xijkl,lk->xij", integrals, density, optimize=True)
    exchange = np.einsum("xijkl,jk->xil", integrals, density, optimize=True)
    exchange += np.einsum("xijkl,li->xkj", integrals, density, optimize=True)
    return coulomb, exchange


def _amfi_jk(mol: Any, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _compute_integrals_from_archive(
    archive: Any, two_electron: str, memory_mb: float
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from pyscf import gto
        from pyscf.data import nist
    except ImportError as exc:
        raise RuntimeError(
            "La modalita' states richiede PySCF nell'ambiente Python"
        ) from exc

    approximation = str(two_electron).lower()
    if approximation not in {"none", "amfi", "full"}:
        raise ValueError("two_electron deve essere none, amfi oppure full")
    mol = gto.loads(str(np.asarray(archive["pyscf_molecule_json"]).item()))
    coefficient = np.asarray(archive["mo_coeff_ao_by_mo"], dtype=complex)
    occupation = np.asarray(archive["mo_occupations"], dtype=float)
    density = np.einsum(
        "up,p,vp->uv", coefficient, occupation, coefficient.conj(), optimize=True
    ).real
    alpha2_over_two = float(nist.ALPHA) ** 2 / 2.0
    one_real = np.asarray(mol.intor_asymmetric("int1e_pnucxp", comp=3))
    one = 1j * alpha2_over_two * one_real
    if bool(getattr(mol, "has_ecp_soc", lambda: False)()):
        one -= 1j * np.asarray(mol.intor("ECPso"))

    two = np.zeros_like(one)
    if approximation != "none":
        if approximation == "full":
            nao = mol.nao_nr()
            estimate_mb = 3.0 * nao**4 * 8.0 / 1.0e6
            if estimate_mb > 0.7 * float(memory_mb):
                raise MemoryError(
                    "L'intermedio full SOMF richiede circa "
                    f"{estimate_mb:.0f} MB, oltre il 70% di --memory-mb; "
                    "aumentare la memoria oppure usare amfi"
                )
            coulomb, exchange = _somf_jk(mol, density)
        else:
            coulomb, exchange = _amfi_jk(mol, density)
        two = 1j * alpha2_over_two * (coulomb - 1.5 * exchange)

    one = 0.5 * (one + one.transpose(0, 2, 1).conj())
    two = 0.5 * (two + two.transpose(0, 2, 1).conj())
    one_mo = np.einsum(
        "xuv,up,vq->xpq", one, coefficient.conj(), coefficient, optimize=True
    )
    two_mo = np.einsum(
        "xuv,up,vq->xpq", two, coefficient.conj(), coefficient, optimize=True
    )
    labels = {
        "none": "Breit-Pauli 1e",
        "amfi": "Breit-Pauli 1e + AMFI-SOMF",
        "full": "Breit-Pauli 1e + full molecular SOMF",
    }
    return one_mo, two_mo, labels[approximation]


def _build_from_state_archives(
    singlet_path: Path,
    triplet_path: Path,
    two_electron: str,
    memory_mb: float,
) -> SOCInput:
    with np.load(singlet_path, allow_pickle=False) as singlets, np.load(
        triplet_path, allow_pickle=False
    ) as triplets:
        source_model = _validate_state_archives(singlets, triplets)
        singlet_energies = np.asarray(singlets["eigenvalues_hartree"], dtype=float)
        triplet_energies = np.asarray(triplets["eigenvalues_hartree"], dtype=float)
        wavefunctions: list[dict[int, complex]] = []
        excluded_s = np.zeros(len(singlet_energies))
        excluded_t = np.zeros(len(triplet_energies))
        included_s = np.zeros(len(singlet_energies))
        included_t = np.zeros(len(triplet_energies))
        basis_labels: list[str] = []
        multiplicities: list[int] = []
        ms_values: list[int] = []
        spin_free_energies: list[float] = []

        for root, energy in enumerate(singlet_energies):
            wavefunction, excluded, included = _state_wavefunction(singlets, root, 0)
            wavefunctions.append(wavefunction)
            excluded_s[root] = excluded
            included_s[root] = included
            basis_labels.append(f"S{root}")
            multiplicities.append(1)
            ms_values.append(0)
            spin_free_energies.append(float(energy))

        for root, energy in enumerate(triplet_energies):
            for ms in (-1, 0, 1):
                wavefunction, excluded, included = _state_wavefunction(
                    triplets, root, ms
                )
                wavefunctions.append(wavefunction)
                excluded_t[root] = max(excluded_t[root], excluded)
                included_t[root] = max(included_t[root], included)
                basis_labels.append(f"T{root}(Ms={ms:+d})")
                multiplicities.append(3)
                ms_values.append(ms)
                spin_free_energies.append(float(energy))

        determinants, coefficients = _state_coefficients(wavefunctions)
        represented_norm = np.real(np.diag(coefficients.T.conj() @ coefficients))
        expected_norm = np.concatenate(
            (excluded_s, np.repeat(excluded_t, 3))
        )
        norm_error = float(np.max(np.abs(represented_norm - (1.0 - expected_norm))))
        if norm_error > 1.0e-7:
            raise RuntimeError(
                "La norma delle funzioni d'onda ricostruite non e' coerente: "
                f"errore massimo {norm_error:.3e}"
            )
        one_mo, two_mo, approximation = _compute_integrals_from_archive(
            singlets, two_electron, memory_mb
        )

    return SOCInput(
        source_model=source_model,
        approximation=approximation,
        origin=f"{singlet_path.resolve()} | {triplet_path.resolve()}",
        determinants=determinants,
        state_coefficients=coefficients,
        one_electron_mo=one_mo,
        two_electron_mo=two_mo,
        basis_labels=np.asarray(basis_labels, dtype="U48"),
        basis_multiplicity=np.asarray(multiplicities, dtype=int),
        basis_ms=np.asarray(ms_values, dtype=int),
        basis_spin_free_energy_hartree=np.asarray(spin_free_energies, dtype=float),
        excluded_singlet_four_open_weight=excluded_s,
        excluded_triplet_four_open_weight=excluded_t,
        included_singlet_lr_cv_weight=included_s,
        included_triplet_lr_cv_weight=included_t,
        wavefunction_norm_error=norm_error,
    )


def _optional_array(archive: Any, key: str) -> np.ndarray | None:
    if key not in archive.files:
        return None
    return np.asarray(archive[key], dtype=complex)


def _build_from_soc_archive(path: Path) -> SOCInput:
    with np.load(path, allow_pickle=False) as archive:
        version = int(np.asarray(archive["archive_format_version"]).item())
        if version < 3:
            raise ValueError("L'archivio SOC deve avere formato >= 3")
        required = {
            "source_model",
            "basis_labels",
            "basis_multiplicity",
            "basis_ms",
            "basis_spin_free_energy_hartree",
            "determinant_hex",
            "determinant_state_coefficients",
            "one_electron_soc_mo",
            "somf_two_electron_soc_mo",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("Array mancanti nell'archivio SOC: " + ", ".join(missing))
        determinants = tuple(int(str(value), 0) for value in archive["determinant_hex"])
        coefficients = np.asarray(
            archive["determinant_state_coefficients"], dtype=complex
        )
        multiplicity = np.asarray(archive["basis_multiplicity"], dtype=int)
        singlet_count = int(np.count_nonzero(multiplicity == 1))
        triplet_count = int(np.count_nonzero(multiplicity == 3) // 3)

        def weight(key: str, length: int) -> np.ndarray:
            if key in archive.files:
                return np.asarray(archive[key], dtype=float)
            return np.zeros(length, dtype=float)

        approximation = (
            str(np.asarray(archive["soc_approximation"]).item())
            if "soc_approximation" in archive.files
            else "Breit-Pauli 1e + stored SOMF"
        )
        return SOCInput(
            source_model=str(np.asarray(archive["source_model"]).item()).upper(),
            approximation=approximation,
            origin=str(path.resolve()),
            determinants=determinants,
            state_coefficients=coefficients,
            one_electron_mo=np.asarray(archive["one_electron_soc_mo"], dtype=complex),
            two_electron_mo=np.asarray(
                archive["somf_two_electron_soc_mo"], dtype=complex
            ),
            basis_labels=np.asarray(archive["basis_labels"], dtype="U48"),
            basis_multiplicity=multiplicity,
            basis_ms=np.asarray(archive["basis_ms"], dtype=int),
            basis_spin_free_energy_hartree=np.asarray(
                archive["basis_spin_free_energy_hartree"], dtype=float
            ),
            excluded_singlet_four_open_weight=weight(
                "excluded_singlet_mrsf_four_open_weight", singlet_count
            ),
            excluded_triplet_four_open_weight=weight(
                "excluded_triplet_mrsf_four_open_weight", triplet_count
            ),
            included_singlet_lr_cv_weight=weight(
                "included_singlet_lr_cv_weight", singlet_count
            ),
            included_triplet_lr_cv_weight=weight(
                "included_triplet_lr_cv_weight", triplet_count
            ),
            archived_one_electron_matrix=_optional_array(
                archive, "one_electron_soc_hartree"
            ),
            archived_two_electron_matrix=_optional_array(
                archive, "two_electron_soc_hartree"
            ),
        )


def _state_layout(data: SOCInput) -> tuple[np.ndarray, list[np.ndarray]]:
    nstates = data.state_coefficients.shape[1]
    arrays = (
        data.basis_labels,
        data.basis_multiplicity,
        data.basis_ms,
        data.basis_spin_free_energy_hartree,
    )
    if any(len(value) != nstates for value in arrays):
        raise ValueError("I metadati della base non coincidono con gli autostati")
    singlets = np.flatnonzero(data.basis_multiplicity == 1)
    triplets = np.flatnonzero(data.basis_multiplicity == 3)
    if len(singlets) == 0 or len(triplets) == 0 or len(triplets) % 3:
        raise ValueError("La base deve contenere singoletti e terne di tripletti")
    groups: list[np.ndarray] = []
    for start in range(0, len(triplets), 3):
        group = triplets[start : start + 3]
        order = np.argsort(data.basis_ms[group])
        group = group[order]
        if tuple(data.basis_ms[group]) != (-1, 0, 1):
            raise ValueError("Ogni tripletto deve contenere Ms=-1,0,+1")
        energies = data.basis_spin_free_energy_hartree[group]
        if float(np.ptp(energies)) > 1.0e-10:
            raise ValueError("I tre sottolivelli Ms non sono degeneri spin-free")
        groups.append(group)
    return singlets, groups


def calculate_soc(
    data: SOCInput,
    *,
    include_two_electron: bool = True,
    tolerance: float = 1.0e-8,
    validate_archived: bool = True,
) -> ExternalSOCResult:
    """Contrae 1e/SOMF e diagonalizza l'Hamiltoniana di state interaction."""

    _state_layout(data)
    one_raw = _contract_one_body(
        data.determinants, data.state_coefficients, data.one_electron_mo
    )
    one, one_herr = _validated_hermitian(one_raw, "Matrice SOC 1e", tolerance)
    if include_two_electron:
        two_raw = _contract_one_body(
            data.determinants, data.state_coefficients, data.two_electron_mo
        )
        two, two_herr = _validated_hermitian(
            two_raw, "Matrice SOC SOMF", tolerance
        )
    else:
        two = np.zeros_like(one)
        two_herr = 0.0
    total_raw = one + two
    total, total_herr = _validated_hermitian(
        total_raw, "Matrice SOC totale", tolerance
    )
    hermiticity_error = max(one_herr, two_herr, total_herr)
    hamiltonian = total + np.diag(data.basis_spin_free_energy_hartree)
    hamiltonian, h_error = _validated_hermitian(
        hamiltonian, "Hamiltoniana di state interaction", tolerance
    )
    hermiticity_error = max(hermiticity_error, h_error)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    validation_one = float("nan")
    validation_two = float("nan")
    if validate_archived and data.archived_one_electron_matrix is not None:
        validation_one = float(
            np.max(np.abs(one - data.archived_one_electron_matrix))
        )
        if validation_one > tolerance:
            raise RuntimeError(
                "La matrice 1e ricalcolata non coincide con quella archiviata: "
                f"{validation_one:.3e} Eh"
            )
    if (
        validate_archived
        and include_two_electron
        and data.archived_two_electron_matrix is not None
    ):
        validation_two = float(
            np.max(np.abs(two - data.archived_two_electron_matrix))
        )
        if validation_two > tolerance:
            raise RuntimeError(
                "La matrice SOMF ricalcolata non coincide con quella archiviata: "
                f"{validation_two:.3e} Eh"
            )
    return ExternalSOCResult(
        data=data,
        one_electron_matrix_hartree=one,
        two_electron_matrix_hartree=two,
        total_soc_matrix_hartree=total,
        hamiltonian_hartree=hamiltonian,
        eigenvalues_hartree=eigenvalues,
        eigenvectors=eigenvectors,
        hermiticity_error_hartree=hermiticity_error,
        validation_one_electron_error_hartree=validation_one,
        validation_two_electron_error_hartree=validation_two,
    )


def _coupling_tensors(
    result: ExternalSOCResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    singlets, triplet_groups = _state_layout(result.data)
    shapes = (len(singlets), len(triplet_groups), 3)
    one = np.zeros(shapes, dtype=complex)
    two = np.zeros(shapes, dtype=complex)
    total = np.zeros(shapes, dtype=complex)
    for singlet_root, singlet_index in enumerate(singlets):
        for triplet_root, group in enumerate(triplet_groups):
            one[singlet_root, triplet_root, :] = (
                result.one_electron_matrix_hartree[singlet_index, group]
            )
            two[singlet_root, triplet_root, :] = (
                result.two_electron_matrix_hartree[singlet_index, group]
            )
            total[singlet_root, triplet_root, :] = (
                result.total_soc_matrix_hartree[singlet_index, group]
            )
    return one, two, total


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if np.asarray(values).size else 0.0


def write_log(path: Path, result: ExternalSOCResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    one, two, total = _coupling_tensors(result)
    data = result.data
    with path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 79 + "\n")
        handle.write("              EXTERNAL MRSF/EMRSF SPIN-ORBIT COUPLING\n")
        handle.write("=" * 79 + "\n")
        handle.write(f"PROGRAM VERSION      {PROGRAM_VERSION}\n")
        handle.write(f"SOURCE MODEL         {data.source_model}\n")
        handle.write(f"OPERATOR             {data.approximation}\n")
        handle.write(f"INPUT                 {data.origin}\n")
        handle.write(f"DETERMINANTS         {len(data.determinants)}\n")
        handle.write(f"STATE BASIS SIZE     {data.state_coefficients.shape[1]}\n")
        handle.write(
            "MAX EXCLUDED S 4O   "
            f"{_safe_max(data.excluded_singlet_four_open_weight):.8f}\n"
        )
        handle.write(
            "MAX EXCLUDED T 4O   "
            f"{_safe_max(data.excluded_triplet_four_open_weight):.8f}\n"
        )
        handle.write(
            "MAX INCLUDED S LR-CV "
            f"{_safe_max(data.included_singlet_lr_cv_weight):.8f}\n"
        )
        handle.write(
            "MAX INCLUDED T LR-CV "
            f"{_safe_max(data.included_triplet_lr_cv_weight):.8f}\n"
        )
        handle.write(
            f"HERMITICITY ERROR    {result.hermiticity_error_hartree:.6e} Eh\n"
        )
        if np.isfinite(data.wavefunction_norm_error):
            handle.write(
                f"WAVEFUNCTION NORM ERR {data.wavefunction_norm_error:.6e}\n"
            )
        if np.isfinite(result.validation_one_electron_error_hartree):
            handle.write(
                "ARCHIVE 1E CHECK     "
                f"{result.validation_one_electron_error_hartree:.6e} Eh\n"
            )
        if np.isfinite(result.validation_two_electron_error_hartree):
            handle.write(
                "ARCHIVE SOMF CHECK   "
                f"{result.validation_two_electron_error_hartree:.6e} Eh\n"
            )
        handle.write("\nSINGLET-TRIPLET SOC COUPLINGS (cm^-1)\n")
        handle.write("-" * 79 + "\n")
        handle.write("   S    T        |1e|       |SOMF|       |TOTAL|\n")
        for s in range(total.shape[0]):
            for t in range(total.shape[1]):
                handle.write(
                    f" {s:3d} {t:4d}"
                    f" {np.linalg.norm(one[s,t])*HARTREE_TO_WAVENUMBER:11.5f}"
                    f" {np.linalg.norm(two[s,t])*HARTREE_TO_WAVENUMBER:12.5f}"
                    f" {np.linalg.norm(total[s,t])*HARTREE_TO_WAVENUMBER:13.5f}\n"
                )
                for component, ms in enumerate((-1, 0, 1)):
                    value = total[s, t, component] * HARTREE_TO_WAVENUMBER
                    handle.write(
                        f"          Ms={ms:+d}: {value.real:14.7f}"
                        f" {value.imag:+14.7f} i\n"
                    )
        excitation_ev = (
            result.eigenvalues_hartree - result.eigenvalues_hartree[0]
        ) * HARTREE_TO_EV
        handle.write("\nSOC-ADIABATIC STATES\n")
        handle.write("-" * 79 + "\n")
        handle.write(" STATE       ENERGY/Eh      EXCITATION/eV       DOMINANT BASIS    WEIGHT\n")
        for state, energy in enumerate(result.eigenvalues_hartree):
            weights = np.abs(result.eigenvectors[:, state]) ** 2
            dominant = int(np.argmax(weights))
            handle.write(
                f" {state:5d} {energy:16.10f} {excitation_ev[state]:18.8f}"
                f" {str(data.basis_labels[dominant]):>20s} {weights[dominant]:9.6f}\n"
            )
        handle.write("\n**** EXTERNAL SOC NORMAL TERMINATION ****\n")
    return path


def write_couplings_csv(path: Path, result: ExternalSOCResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    one, two, total = _coupling_tensors(result)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "source_model",
            "singlet_state",
            "triplet_state",
            "contribution",
            "ms",
            "real_hartree",
            "imag_hartree",
            "real_cm-1",
            "imag_cm-1",
            "rotational_magnitude_hartree",
            "rotational_magnitude_cm-1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for s in range(total.shape[0]):
            for t in range(total.shape[1]):
                for label, tensor in (
                    ("one_electron", one),
                    ("somf_two_electron", two),
                    ("total", total),
                ):
                    magnitude = float(np.linalg.norm(tensor[s, t]))
                    for component, ms in enumerate((-1, 0, 1)):
                        value = tensor[s, t, component]
                        writer.writerow(
                            {
                                "source_model": result.data.source_model,
                                "singlet_state": s,
                                "triplet_state": t,
                                "contribution": label,
                                "ms": ms,
                                "real_hartree": value.real,
                                "imag_hartree": value.imag,
                                "real_cm-1": value.real * HARTREE_TO_WAVENUMBER,
                                "imag_cm-1": value.imag * HARTREE_TO_WAVENUMBER,
                                "rotational_magnitude_hartree": magnitude,
                                "rotational_magnitude_cm-1": (
                                    magnitude * HARTREE_TO_WAVENUMBER
                                ),
                            }
                        )
    return path


def write_states_csv(path: Path, result: ExternalSOCResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    singlets, triplet_groups = _state_layout(result.data)
    triplets = np.concatenate(triplet_groups)
    excitation = (
        result.eigenvalues_hartree - result.eigenvalues_hartree[0]
    ) * HARTREE_TO_EV
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "soc_state",
            "energy_hartree",
            "excitation_ev",
            "singlet_weight",
            "triplet_weight",
            "dominant_basis_label",
            "dominant_basis_weight",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for state, energy in enumerate(result.eigenvalues_hartree):
            weights = np.abs(result.eigenvectors[:, state]) ** 2
            dominant = int(np.argmax(weights))
            writer.writerow(
                {
                    "soc_state": state,
                    "energy_hartree": energy,
                    "excitation_ev": excitation[state],
                    "singlet_weight": float(np.sum(weights[singlets])),
                    "triplet_weight": float(np.sum(weights[triplets])),
                    "dominant_basis_label": result.data.basis_labels[dominant],
                    "dominant_basis_weight": weights[dominant],
                }
            )
    return path


def write_npz(path: Path, result: ExternalSOCResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    one, two, total = _coupling_tensors(result)
    np.savez_compressed(
        path,
        archive_format_version=np.asarray(1, dtype=np.int64),
        program=np.asarray("emrsf_soc_external"),
        program_version=np.asarray(PROGRAM_VERSION),
        source_model=np.asarray(result.data.source_model),
        approximation=np.asarray(result.data.approximation),
        input_origin=np.asarray(result.data.origin),
        basis_labels=result.data.basis_labels,
        basis_multiplicity=result.data.basis_multiplicity,
        basis_ms=result.data.basis_ms,
        basis_spin_free_energy_hartree=(
            result.data.basis_spin_free_energy_hartree
        ),
        one_electron_soc_hartree=result.one_electron_matrix_hartree,
        somf_two_electron_soc_hartree=result.two_electron_matrix_hartree,
        total_soc_hartree=result.total_soc_matrix_hartree,
        state_interaction_hamiltonian_hartree=result.hamiltonian_hartree,
        soc_adiabatic_energies_hartree=result.eigenvalues_hartree,
        soc_eigenvectors_columns_are_states=result.eigenvectors,
        one_electron_st_components_hartree=one,
        somf_two_electron_st_components_hartree=two,
        total_st_components_hartree=total,
        one_electron_st_magnitude_cm_minus_1=(
            np.linalg.norm(one, axis=2) * HARTREE_TO_WAVENUMBER
        ),
        somf_two_electron_st_magnitude_cm_minus_1=(
            np.linalg.norm(two, axis=2) * HARTREE_TO_WAVENUMBER
        ),
        total_st_magnitude_cm_minus_1=(
            np.linalg.norm(total, axis=2) * HARTREE_TO_WAVENUMBER
        ),
        excluded_singlet_mrsf_four_open_weight=(
            result.data.excluded_singlet_four_open_weight
        ),
        excluded_triplet_mrsf_four_open_weight=(
            result.data.excluded_triplet_four_open_weight
        ),
        included_singlet_lr_cv_weight=result.data.included_singlet_lr_cv_weight,
        included_triplet_lr_cv_weight=result.data.included_triplet_lr_cv_weight,
        hermiticity_error_hartree=np.asarray(
            result.hermiticity_error_hartree, dtype=float
        ),
    )
    return path


def write_outputs(prefix: Path, result: ExternalSOCResult) -> tuple[Path, ...]:
    prefix = Path(prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    outputs = (
        write_log(Path(f"{prefix}.log"), result),
        write_couplings_csv(Path(f"{prefix}_couplings.csv"), result),
        write_states_csv(Path(f"{prefix}_states.csv"), result),
        write_npz(Path(f"{prefix}.npz"), result),
    )
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calcola esternamente il SOC 1e e SOMF fra autostati MRSF/EMRSF."
        )
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    states = subparsers.add_parser(
        "states", help="parti dagli archivi NPZ degli autostati singoletti/tripletti"
    )
    states.add_argument("singlets", type=Path, help="archivio postprocessing singoletti")
    states.add_argument("triplets", type=Path, help="archivio postprocessing tripletti")
    states.add_argument(
        "--two-electron",
        choices=("none", "amfi", "full"),
        default="amfi",
        help="nessun 2eSOC, AMFI-SOMF, oppure SOMF molecolare completo",
    )
    states.add_argument(
        "--memory-mb",
        type=float,
        default=4000.0,
        help="memoria disponibile per il controllo del full SOMF",
    )
    states.add_argument("--output-prefix", type=Path, default=None)
    states.add_argument("--tolerance", type=float, default=1.0e-8)

    archive = subparsers.add_parser(
        "archive", help="ricontrai un archivio SOC external-ready"
    )
    archive.add_argument("soc_archive", type=Path)
    archive.add_argument(
        "--one-electron-only",
        action="store_true",
        help="ignora il tensore SOMF conservato nell'archivio",
    )
    archive.add_argument(
        "--no-validate-archived",
        action="store_true",
        help="non confrontare la contrazione con le matrici gia' archiviate",
    )
    archive.add_argument("--output-prefix", type=Path, default=None)
    archive.add_argument("--tolerance", type=float, default=1.0e-8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.tolerance <= 0.0 or not np.isfinite(args.tolerance):
        raise SystemExit("--tolerance deve essere finita e positiva")

    if args.mode == "states":
        if args.memory_mb <= 0.0 or not np.isfinite(args.memory_mb):
            raise SystemExit("--memory-mb deve essere finita e positiva")
        data = _build_from_state_archives(
            args.singlets.resolve(),
            args.triplets.resolve(),
            args.two_electron,
            args.memory_mb,
        )
        prefix = args.output_prefix
        if prefix is None:
            prefix = args.singlets.resolve().parent / f"soc_external_{data.source_model.lower()}"
        result = calculate_soc(data, tolerance=args.tolerance)
    else:
        data = _build_from_soc_archive(args.soc_archive.resolve())
        if args.one_electron_only:
            data = replace(
                data,
                approximation="Breit-Pauli 1e (stored SOMF ignored)",
            )
        prefix = args.output_prefix
        if prefix is None:
            base = args.soc_archive.resolve().with_suffix("")
            prefix = Path(f"{base}_recomputed")
        result = calculate_soc(
            data,
            include_two_electron=not args.one_electron_only,
            tolerance=args.tolerance,
            validate_archived=not args.no_validate_archived,
        )

    outputs = write_outputs(prefix, result)
    print(f"Source model : {result.data.source_model}")
    print(f"Operator     : {result.data.approximation}")
    print(f"Hermiticity : {result.hermiticity_error_hartree:.3e} Eh")
    print("Output:")
    for output in outputs:
        print(f"  {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
