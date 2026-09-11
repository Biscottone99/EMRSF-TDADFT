"""Construction, import, locking, and validation of the triplet reference.

MRSF-TDDFT is a response theory around a *specific* two-open-shell triplet
determinant.  Convergence of a restricted open-shell SCF calculation is not
sufficient: an orbital occupation different from the intended triplet state
defines a different response problem.  This module supports three explicit
workflows: ordinary Aufbau SCF, Molden-seeded maximum-overlap SCF (MOM), and a
fixed external restricted Molden reference for cross-program operator tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import dft, gto, scf
from pyscf.scf import addons
from pyscf.tools import molden

from ._memory import patch_pyscf_memory_probe
from .solvent import PCMConfig, attach_pcm, pcm_metadata


_SCF_STRATEGIES = {"standard", "robust"}


class _SCFHistoryMonitor:
    """Collect compact SCF diagnostics and detect a period-two density cycle.

    Only the last three densities are retained.  A detected oscillation is used
    to form an *initial guess* for the stabilised retry; an averaged density is
    never accepted as the final MRSF reference.
    """

    def __init__(
        self,
        *,
        detect_oscillation: bool = True,
        two_cycle_tolerance: float = 5.0e-3,
        one_cycle_minimum: float = 2.0e-2,
    ) -> None:
        self.detect_oscillation = bool(detect_oscillation)
        self.two_cycle_tolerance = float(two_cycle_tolerance)
        self.one_cycle_minimum = float(one_cycle_minimum)
        self.cycles = 0
        self.last_energy = float("nan")
        self.last_gradient_norm = float("nan")
        self.last_density_change_norm = float("nan")
        self.oscillation_detected = False
        self._densities: deque[np.ndarray] = deque(maxlen=3)

    def __call__(self, environment: dict[str, Any]) -> None:
        self.cycles += 1
        self.last_energy = float(environment.get("e_tot", self.last_energy))
        self.last_gradient_norm = float(
            environment.get("norm_gorb", self.last_gradient_norm)
        )
        self.last_density_change_norm = float(
            environment.get("norm_ddm", self.last_density_change_norm)
        )
        density = environment.get("dm")
        if density is None:
            return
        self._densities.append(np.array(density, copy=True))
        if not self.detect_oscillation or len(self._densities) < 3:
            return
        previous_two, previous, current = self._densities
        one_cycle = float(np.max(np.abs(current - previous)))
        two_cycle = float(np.max(np.abs(current - previous_two)))
        if (
            one_cycle >= self.one_cycle_minimum
            and two_cycle <= self.two_cycle_tolerance
            and one_cycle >= 5.0 * max(two_cycle, 1.0e-14)
        ):
            self.oscillation_detected = True

    def averaged_cycle_guess(self) -> np.ndarray | None:
        if not self.oscillation_detected or len(self._densities) < 2:
            return None
        return 0.5 * (self._densities[-1] + self._densities[-2])


def _last_density(mf) -> np.ndarray | None:
    coefficients = getattr(mf, "mo_coeff", None)
    occupations = getattr(mf, "mo_occ", None)
    if coefficients is None or occupations is None:
        return None
    return np.asarray(mf.make_rdm1(coefficients, occupations))


def _mom_arrays(coefficients: np.ndarray, occupations: np.ndarray):
    occupations = np.asarray(occupations, dtype=float)
    alpha = (occupations > 0.5).astype(float)
    beta = (occupations > 1.5).astype(float)
    return np.asarray(coefficients), np.asarray((alpha, beta))


def _apply_mom(mf, coefficients: np.ndarray, spin_occupations: np.ndarray) -> None:
    coefficients = np.asarray(coefficients)
    spin_occupations = np.asarray(spin_occupations, dtype=float)
    nmo = int(mf.mol.nao_nr())
    if coefficients.shape[1] == nmo:
        addons.mom_occ(mf, coefficients, spin_occupations)
        return
    padded_coefficients = np.zeros((nmo, nmo))
    padded_coefficients[:, : coefficients.shape[1]] = coefficients
    padded_occupations = np.zeros((2, nmo))
    padded_occupations[:, : spin_occupations.shape[1]] = spin_occupations
    addons.mom_occ(mf, padded_coefficients, padded_occupations)


def _attempt_record(
    *,
    name: str,
    converger: str,
    mf,
    monitor: _SCFHistoryMonitor,
) -> dict[str, Any]:
    energy = getattr(mf, "e_tot", monitor.last_energy)
    if energy is None:
        energy = float("nan")
    return {
        "name": str(name),
        "converger": str(converger),
        "converged": bool(getattr(mf, "converged", False)),
        "cycles": int(monitor.cycles),
        "energy_hartree": float(energy),
        "gradient_norm": float(monitor.last_gradient_norm),
        "density_change_norm": float(monitor.last_density_change_norm),
        "oscillation_detected": bool(monitor.oscillation_detected),
    }


def _run_scf_attempt(
    mf,
    *,
    dm0=None,
    mo_coeff=None,
    mo_occ=None,
    name: str,
    converger: str,
    detect_oscillation: bool = True,
) -> tuple[object, _SCFHistoryMonitor, dict[str, Any]]:
    monitor = _SCFHistoryMonitor(detect_oscillation=detect_oscillation)
    mf.callback = monitor
    if mo_coeff is not None and hasattr(mf, "_scf"):
        mf.kernel(mo_coeff=mo_coeff, mo_occ=mo_occ)
    else:
        mf.kernel(dm0=dm0)
    record = _attempt_record(
        name=name,
        converger=converger,
        mf=mf,
        monitor=monitor,
    )
    return mf, monitor, record


@dataclass(frozen=True)
class MoldenReferenceSeed:
    """Restricted orbitals and spin occupations read from a Molden file."""

    path: Path
    mol: object
    coeff: np.ndarray
    energy: np.ndarray
    occupation: np.ndarray
    alpha_occupation: np.ndarray
    beta_occupation: np.ndarray
    input_orthonormality_error: float
    orthonormality_error: float


@dataclass(frozen=True)
class ReferenceDiagnostics:
    """Numerical and provenance checks for the selected triplet determinant."""

    source: str
    source_path: str | None
    local_scf_run: bool
    local_scf_converged: bool
    orthonormality_error: float
    electron_number_error: float
    spin_number_error: float
    orbital_gradient_max: float
    orbital_gradient_rms: float
    minimum_orbital_symmetry_weight: float
    closed_to_o1_gap_hartree: float
    o1_to_o2_gap_hartree: float
    o2_to_virtual_gap_hartree: float
    seed_somo_singular_values: tuple[float, ...]
    seed_alpha_occupied_min_singular_value: float | None
    seed_beta_occupied_min_singular_value: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TripletReference:
    """Canonical data required by the MRSF response equations.

    Array indices are zero-based. ``closed`` contains the doubly occupied
    orbitals, ``open`` the two singly occupied orbitals (O1, O2), and
    ``virtual`` the empty orbitals.
    """

    mf: object
    coeff: np.ndarray
    energy: np.ndarray
    occupation: np.ndarray
    fock_alpha_mo: np.ndarray
    fock_beta_mo: np.ndarray
    closed: np.ndarray
    open: np.ndarray
    virtual: np.ndarray
    diagnostics: ReferenceDiagnostics

    @property
    def nao(self) -> int:
        return self.coeff.shape[0]

    @property
    def nmo(self) -> int:
        return self.coeff.shape[1]

    @property
    def o1(self) -> int:
        return int(self.open[0])

    @property
    def o2(self) -> int:
        return int(self.open[1])


def _new_mean_field(
    mol,
    *,
    xc: str | None,
    density_fit: bool,
    auxbasis,
    conv_tol: float,
    conv_tol_grad: float | None,
    direct_scf_tol: float | None,
    max_cycle: int,
    verbose: int | None,
    grid_level: int | None,
    grid_prune: bool,
    small_rho_cutoff: float | None,
    pcm: PCMConfig | None,
):
    patch_pyscf_memory_probe()
    if mol.spin != 2:
        raise ValueError("MRSF requires a triplet reference: set mol.spin = 2")

    mf = scf.ROHF(mol) if xc is None else dft.ROKS(mol)
    if xc is not None:
        mf.xc = xc
    if density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)
    if pcm is not None:
        mf = attach_pcm(mf, pcm)
    mf.conv_tol = float(conv_tol)
    if conv_tol_grad is not None:
        mf.conv_tol_grad = float(conv_tol_grad)
    if direct_scf_tol is not None:
        mf.direct_scf_tol = float(direct_scf_tol)
    mf.max_cycle = int(max_cycle)
    if verbose is not None:
        mf.verbose = int(verbose)
    if xc is not None and grid_level is not None:
        mf.grids.level = int(grid_level)
        if not grid_prune:
            mf.grids.prune = None
    if xc is not None and small_rho_cutoff is not None:
        mf.small_rho_cutoff = float(small_rho_cutoff)
    return mf


def _occupation_classes(occupation: np.ndarray, tol: float = 1.0e-7):
    occupation = np.asarray(occupation, dtype=float)
    closed = np.flatnonzero(np.abs(occupation - 2.0) < tol)
    opened = np.flatnonzero(np.abs(occupation - 1.0) < tol)
    virtual = np.flatnonzero(np.abs(occupation) < tol)
    classified = np.concatenate((closed, opened, virtual))
    if len(classified) != len(occupation) or len(np.unique(classified)) != len(
        occupation
    ):
        raise ValueError("orbital occupations must be exactly 0, 1, or 2")
    if len(opened) != 2:
        raise ValueError(
            "MRSF requires exactly two singly occupied orbitals; "
            f"found {len(opened)}"
        )
    return closed, opened, virtual


def _class_order(occupation: np.ndarray) -> np.ndarray:
    closed, opened, virtual = _occupation_classes(occupation)
    return np.concatenate((closed, opened, virtual))


def _reorder_reference_orbitals(mf) -> None:
    """Canonicalise within occupancy classes and enforce C/O1/O2/V order."""

    mo_energy, mo_coeff = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    occupation = np.asarray(mf.mo_occ, dtype=float)
    order = _class_order(occupation)
    mf.mo_coeff = np.asarray(mo_coeff)[:, order]
    mf.mo_occ = occupation[order]
    mf.mo_energy = np.asarray(mo_energy)[order]


def _spin_density(
    coeff: np.ndarray,
    alpha_occupation: np.ndarray,
    beta_occupation: np.ndarray,
) -> np.ndarray:
    dma = (coeff * alpha_occupation) @ coeff.T.conj()
    dmb = (coeff * beta_occupation) @ coeff.T.conj()
    return np.asarray((dma, dmb))


def _metric_error(coeff: np.ndarray, overlap: np.ndarray) -> float:
    metric = coeff.T.conj() @ overlap @ coeff
    return float(np.max(np.abs(metric - np.eye(metric.shape[0]))))


def _metric_orthonormalise(
    coeff: np.ndarray, overlap: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Apply the closest (symmetric-polar) correction in the AO metric."""

    coeff = np.asarray(coeff)
    before = _metric_error(coeff, overlap)
    metric = coeff.T.conj() @ overlap @ coeff
    values, vectors = np.linalg.eigh(metric)
    if float(np.min(values)) <= 1.0e-10:
        raise ValueError(
            "Molden orbitals are linearly dependent in the target AO metric"
        )
    inverse_sqrt = (vectors / np.sqrt(values)) @ vectors.T.conj()
    corrected = coeff @ inverse_sqrt
    after = _metric_error(corrected, overlap)
    return np.asarray(corrected.real), before, after


def _normalised_ao_labels(mol) -> list[tuple[int, str, str]]:
    return [
        (int(atom), str(shell), str(component))
        for atom, _symbol, shell, component in mol.ao_labels(fmt=False)
    ]


def _validate_molecule_match(
    source,
    target,
    *,
    coordinate_tol: float,
    require_same_ao_basis: bool = True,
) -> None:
    if source.natm != target.natm:
        raise ValueError(
            f"Molden/target atom-count mismatch: {source.natm} != {target.natm}"
        )
    if not np.array_equal(source.atom_charges(), target.atom_charges()):
        raise ValueError("Molden and target molecule have different atom ordering")
    coordinate_error = float(
        np.max(np.abs(source.atom_coords() - target.atom_coords()))
    )
    if coordinate_error > coordinate_tol:
        raise ValueError(
            "Molden/target geometry mismatch: maximum Cartesian difference "
            f"is {coordinate_error:.3e} bohr"
        )
    if source.cart != target.cart:
        raise ValueError("Molden/target spherical-vs-Cartesian AO mismatch")
    if not require_same_ao_basis:
        return
    if source.nao_nr() != target.nao_nr():
        raise ValueError(
            f"Molden/target AO-count mismatch: {source.nao_nr()} != "
            f"{target.nao_nr()}"
        )
    if _normalised_ao_labels(source) != _normalised_ao_labels(target):
        raise ValueError("Molden and target AO ordering/basis do not match")


def load_molden_reference_seed(
    path,
    *,
    target_mol=None,
    coordinate_tol: float = 1.0e-7,
    restricted_tol: float = 1.0e-7,
    maximum_input_orthonormality_error: float = 5.0e-3,
    maximum_projection_metric_error: float = 2.5e-1,
    allow_basis_projection: bool = False,
) -> MoldenReferenceSeed:
    """Read and validate a common-orbital triplet reference from Molden.

    Alpha and beta coefficients must be identical (ROHF/ROKS form), beta
    occupation must be a subset of alpha occupation, and the file must contain
    exactly two singly occupied orbitals. Decimal truncation in Molden files is
    removed with a symmetric AO-metric orthonormalisation.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"reference Molden file not found: {source_path}")
    loaded_mol, energies, coefficients, occupations, _irreps, _spins = molden.load(
        str(source_path)
    )
    mol = loaded_mol if target_mol is None else target_mol
    if target_mol is not None:
        _validate_molecule_match(
            loaded_mol,
            target_mol,
            coordinate_tol=float(coordinate_tol),
            require_same_ao_basis=not bool(allow_basis_projection),
        )

    if isinstance(coefficients, tuple):
        coeff_a, coeff_b = map(np.asarray, coefficients)
        coefficient_difference = float(np.max(np.abs(coeff_a - coeff_b)))
        if coefficient_difference > restricted_tol:
            raise ValueError(
                "MRSF requires common restricted orbitals, but the Molden "
                f"alpha/beta coefficient difference is {coefficient_difference:.3e}"
            )
        coeff = 0.5 * (coeff_a + coeff_b)
    else:
        coeff = np.asarray(coefficients)

    if target_mol is not None:
        # A Molden file represents generally contracted shells as separate
        # contractions.  A Mole built from a named PySCF basis may group the
        # same contractions in one shell, which changes the AO coefficient
        # ordering even when AO labels and dimensions agree.  Project the
        # physical orbitals through the cross-overlap instead of copying the
        # coefficient array by position.
        if loaded_mol.nao_nr() > target_mol.nao_nr():
            raise ValueError(
                "cannot project a Molden seed into a smaller AO space: "
                f"{loaded_mol.nao_nr()} source AOs > "
                f"{target_mol.nao_nr()} target AOs"
            )
        overlap_target = np.asarray(target_mol.intor_symmetric("int1e_ovlp"))
        cross_overlap = np.asarray(
            gto.intor_cross("int1e_ovlp", target_mol, loaded_mol)
        )
        coeff = np.linalg.solve(overlap_target, cross_overlap @ coeff)

    if isinstance(energies, tuple):
        energy_a, energy_b = map(np.asarray, energies)
        energy_difference = float(np.max(np.abs(energy_a - energy_b)))
        if energy_difference > restricted_tol:
            raise ValueError(
                "MRSF requires common restricted orbital energies, but the "
                f"Molden alpha/beta difference is {energy_difference:.3e} Eh"
            )
        energy = 0.5 * (energy_a + energy_b)
    else:
        energy = np.asarray(energies)

    if isinstance(occupations, tuple):
        alpha_occupation, beta_occupation = map(np.asarray, occupations)
    else:
        raise ValueError(
            "the reference Molden file must contain explicit alpha and beta "
            "orbital occupations"
        )
    alpha_occupation = (alpha_occupation > 0.5).astype(float)
    beta_occupation = (beta_occupation > 0.5).astype(float)
    if np.any(beta_occupation > alpha_occupation):
        raise ValueError("Molden beta occupation is not a subset of alpha occupation")
    occupation = alpha_occupation + beta_occupation
    _occupation_classes(occupation)
    if int(np.sum(occupation)) != int(mol.nelectron):
        raise ValueError(
            "Molden electron count does not match the target molecule: "
            f"{int(np.sum(occupation))} != {mol.nelectron}"
        )
    if int(np.sum(alpha_occupation) - np.sum(beta_occupation)) != 2:
        raise ValueError("Molden occupations do not define an M_S=1 triplet")

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    coeff, input_error, corrected_error = _metric_orthonormalise(coeff, overlap)
    allowed_metric_error = (
        float(maximum_projection_metric_error)
        if allow_basis_projection
        else float(maximum_input_orthonormality_error)
    )
    if input_error > allowed_metric_error:
        raise ValueError(
            "Molden orbitals are too far from AO-metric orthonormality: "
            f"maximum error {input_error:.3e}; allowed {allowed_metric_error:.3e}"
        )

    order = _class_order(occupation)
    return MoldenReferenceSeed(
        path=source_path,
        mol=mol,
        coeff=coeff[:, order],
        energy=np.asarray(energy)[order],
        occupation=occupation[order],
        alpha_occupation=alpha_occupation[order],
        beta_occupation=beta_occupation[order],
        input_orthonormality_error=input_error,
        orthonormality_error=corrected_error,
    )


def _subspace_singular_values(
    left: np.ndarray, right: np.ndarray, overlap: np.ndarray
) -> tuple[float, ...]:
    if left.shape[1] == 0:
        return ()
    values = np.linalg.svd(left.T.conj() @ overlap @ right, compute_uv=False)
    values = np.clip(values.real, 0.0, 1.0)
    return tuple(float(value) for value in np.sort(values)[::-1])


def _attach_provenance(mf, **metadata: Any) -> None:
    current = dict(getattr(mf, "emrsf_reference_metadata", {}))
    current.update(metadata)
    mf.emrsf_reference_metadata = current


def build_reference(
    mol,
    *,
    xc: str | None = None,
    conv_tol: float = 1.0e-10,
    conv_tol_grad: float | None = None,
    direct_scf_tol: float | None = None,
    max_cycle: int = 100,
    density_fit: bool = False,
    auxbasis=None,
    verbose: int | None = None,
    grid_level: int | None = None,
    grid_prune: bool = True,
    small_rho_cutoff: float | None = None,
    molden_seed=None,
    allow_basis_projection: bool = False,
    minimum_somo_overlap: float = 0.80,
    minimum_occupied_overlap: float | None = None,
    scf_strategy: str = "robust",
    robust_initial_max_cycle: int = 40,
    robust_stabilization_max_cycle: int = 80,
    robust_newton_max_cycle: int = 80,
    robust_damping: float = 0.50,
    robust_level_shift: float = 0.30,
    robust_coarse_grid_level: int = 3,
    detect_two_cycle_oscillation: bool = True,
    pcm: PCMConfig | None = None,
):
    """Run the high-spin ROHF/ROKS reference required by MRSF.

    With ``molden_seed=None`` the initial determinant is selected by Aufbau and
    tagged ``auto``. Supplying a :class:`MoldenReferenceSeed` (or its path)
    activates MOM and locks both alpha and beta occupied subspaces to the seed.

    ``scf_strategy="standard"`` preserves the original single-attempt SCF.
    ``scf_strategy="robust"`` first tries the unmodified final Hamiltonian.  If
    that fails, it performs an ADIIS/damping/level-shift preconvergence, locks
    its two SOMOs with MOM, retries the exact final Hamiltonian with all
    stabilisers removed, and finally invokes PySCF's second-order Newton/CIAH
    solver.  A period-two density average may be used only as an initial guess;
    the accepted reference is always a converged, idempotent RO determinant on
    the requested final grid.

    ``minimum_occupied_overlap`` optionally rejects a converged MOM solution
    whose complete alpha or beta occupied subspace has drifted from the seed.
    """
    strategy = str(scf_strategy).strip().lower()
    if strategy not in _SCF_STRATEGIES:
        raise ValueError(
            f"scf_strategy must be one of {sorted(_SCF_STRATEGIES)}; "
            f"received {scf_strategy!r}"
        )
    for label, value in (
        ("max_cycle", max_cycle),
        ("robust_initial_max_cycle", robust_initial_max_cycle),
        ("robust_stabilization_max_cycle", robust_stabilization_max_cycle),
        ("robust_newton_max_cycle", robust_newton_max_cycle),
    ):
        if int(value) < 1:
            raise ValueError(f"{label} must be positive")
    if not 0.0 <= float(robust_damping) < 1.0:
        raise ValueError("robust_damping must lie in [0, 1)")
    if float(robust_level_shift) < 0.0:
        raise ValueError("robust_level_shift must be non-negative")
    if int(robust_coarse_grid_level) < 0:
        raise ValueError("robust_coarse_grid_level must be non-negative")

    seed = None
    if molden_seed is not None:
        if not 0.0 <= float(minimum_somo_overlap) <= 1.0:
            raise ValueError("minimum_somo_overlap must lie between zero and one")
        if minimum_occupied_overlap is not None and not (
            0.0 <= float(minimum_occupied_overlap) <= 1.0
        ):
            raise ValueError(
                "minimum_occupied_overlap must lie between zero and one"
            )
        seed = (
            molden_seed
            if isinstance(molden_seed, MoldenReferenceSeed)
            else load_molden_reference_seed(
                molden_seed,
                target_mol=mol,
                allow_basis_projection=allow_basis_projection,
            )
        )
    seed_coefficients = None if seed is None else np.asarray(seed.coeff)
    seed_spin_occupations = (
        None
        if seed is None
        else np.asarray(
            (seed.alpha_occupation, seed.beta_occupation), dtype=float
        )
    )
    initial_density = (
        None
        if seed is None
        else _spin_density(
            seed.coeff, seed.alpha_occupation, seed.beta_occupation
        )
    )

    def new_mean_field(*, attempt_max_cycle: int, attempt_grid_level):
        return _new_mean_field(
            mol,
            xc=xc,
            density_fit=density_fit,
            auxbasis=auxbasis,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            direct_scf_tol=direct_scf_tol,
            max_cycle=attempt_max_cycle,
            verbose=verbose,
            grid_level=attempt_grid_level,
            grid_prune=grid_prune,
            small_rho_cutoff=small_rho_cutoff,
            pcm=pcm,
        )

    attempt_history: list[dict[str, Any]] = []
    initial_cycles = (
        int(max_cycle)
        if strategy == "standard"
        else min(int(max_cycle), int(robust_initial_max_cycle))
    )
    mf = new_mean_field(
        attempt_max_cycle=initial_cycles,
        attempt_grid_level=grid_level,
    )
    if seed_coefficients is not None:
        _apply_mom(mf, seed_coefficients, seed_spin_occupations)
    mf, initial_monitor, record = _run_scf_attempt(
        mf,
        dm0=initial_density,
        name="final-grid-standard",
        converger="CDIIS",
        detect_oscillation=detect_two_cycle_oscillation,
    )
    attempt_history.append(record)
    final_stage = "final-grid-standard"
    density_averaging_used = False

    if not mf.converged and strategy == "robust":
        retry_density = initial_monitor.averaged_cycle_guess()
        if retry_density is not None:
            density_averaging_used = True
        else:
            retry_density = _last_density(mf)

        if seed_coefficients is not None:
            lock_coefficients = seed_coefficients
            lock_spin_occupations = seed_spin_occupations
        elif getattr(mf, "mo_coeff", None) is not None:
            lock_coefficients, lock_spin_occupations = _mom_arrays(
                mf.mo_coeff, mf.mo_occ
            )
        else:
            lock_coefficients = lock_spin_occupations = None

        coarse_level = grid_level
        if xc is not None and grid_level is not None:
            coarse_level = min(
                int(grid_level), int(robust_coarse_grid_level)
            )
        stabilised = _new_mean_field(
            mol,
            xc=xc,
            density_fit=density_fit,
            auxbasis=auxbasis,
            conv_tol=max(float(conv_tol), 1.0e-7),
            conv_tol_grad=max(
                float(conv_tol_grad) if conv_tol_grad is not None else 0.0,
                1.0e-4,
            ),
            direct_scf_tol=direct_scf_tol,
            max_cycle=int(robust_stabilization_max_cycle),
            verbose=verbose,
            grid_level=coarse_level,
            grid_prune=True,
            small_rho_cutoff=small_rho_cutoff,
            pcm=pcm,
        )
        stabilised.diis = scf.ADIIS()
        stabilised.diis_space = min(int(getattr(stabilised, "diis_space", 8)), 8)
        stabilised.diis_start_cycle = 2
        stabilised.damp = float(robust_damping)
        stabilised.level_shift = float(robust_level_shift)
        if lock_coefficients is not None:
            _apply_mom(
                stabilised, lock_coefficients, lock_spin_occupations
            )
        stabilised, stabilised_monitor, record = _run_scf_attempt(
            stabilised,
            dm0=retry_density,
            name="coarse-grid-stabilised",
            converger="ADIIS+damping+level-shift",
            detect_oscillation=detect_two_cycle_oscillation,
        )
        attempt_history.append(record)

        if seed_coefficients is not None:
            final_lock_coefficients = seed_coefficients
            final_lock_occupations = seed_spin_occupations
        elif getattr(stabilised, "mo_coeff", None) is not None:
            final_lock_coefficients, final_lock_occupations = _mom_arrays(
                stabilised.mo_coeff, stabilised.mo_occ
            )
        else:
            final_lock_coefficients = lock_coefficients
            final_lock_occupations = lock_spin_occupations

        final_density = _last_density(stabilised)
        if final_density is None:
            final_density = retry_density
        final_mf = new_mean_field(
            attempt_max_cycle=int(max_cycle),
            attempt_grid_level=grid_level,
        )
        if final_lock_coefficients is not None:
            _apply_mom(
                final_mf, final_lock_coefficients, final_lock_occupations
            )
        final_mf, final_monitor, record = _run_scf_attempt(
            final_mf,
            dm0=final_density,
            name="final-grid-mom-refinement",
            converger="CDIIS+MOM",
            detect_oscillation=detect_two_cycle_oscillation,
        )
        attempt_history.append(record)
        mf = final_mf
        final_stage = "final-grid-mom-refinement"

        if not mf.converged:
            newton_base = new_mean_field(
                attempt_max_cycle=int(robust_newton_max_cycle),
                attempt_grid_level=grid_level,
            )
            if final_lock_coefficients is not None:
                _apply_mom(
                    newton_base,
                    final_lock_coefficients,
                    final_lock_occupations,
                )
            newton_mf = newton_base.newton()
            newton_mf.max_cycle = int(robust_newton_max_cycle)
            if final_lock_coefficients is not None:
                _apply_mom(
                    newton_mf,
                    final_lock_coefficients,
                    final_lock_occupations,
                )
            newton_mf, _newton_monitor, record = _run_scf_attempt(
                newton_mf,
                mo_coeff=getattr(mf, "mo_coeff", None),
                mo_occ=getattr(mf, "mo_occ", None),
                dm0=_last_density(mf),
                name="final-grid-newton",
                converger="Newton/CIAH+MOM",
                detect_oscillation=False,
            )
            attempt_history.append(record)
            mf = newton_mf
            final_stage = "final-grid-newton"

    if not mf.converged:
        mode = "MOM" if seed is not None else "automatic"
        summary = "; ".join(
            f"{item['name']}={item['converged']}({item['cycles']} cycles)"
            for item in attempt_history
        )
        raise RuntimeError(
            f"The {mode} triplet ROHF/ROKS reference did not converge; "
            f"SCF attempts: {summary}"
        )
    _reorder_reference_orbitals(mf)

    metadata: dict[str, Any] = {
        "source": "mom_molden" if seed is not None else "auto",
        "source_path": str(seed.path) if seed is not None else None,
        "local_scf_run": True,
        "local_scf_converged": bool(mf.converged),
        "input_orthonormality_error": (
            seed.input_orthonormality_error if seed is not None else None
        ),
        "basis_projection": bool(
            seed is not None and seed.coeff.shape[1] != mol.nao_nr()
        ),
        "scf_strategy": strategy,
        "scf_final_stage": final_stage,
        "scf_rescue_used": len(attempt_history) > 1,
        "scf_oscillation_detected": any(
            bool(item["oscillation_detected"]) for item in attempt_history
        ),
        "scf_density_averaging_used": density_averaging_used,
        "scf_attempt_history": tuple(attempt_history),
        "pcm": pcm_metadata(mf),
    }
    if seed is not None:
        overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))
        final_occ = np.asarray(mf.mo_occ)
        final_open = np.flatnonzero(np.abs(final_occ - 1.0) < 1.0e-7)
        final_alpha = np.flatnonzero(final_occ > 0.5)
        final_beta = np.flatnonzero(final_occ > 1.5)
        seed_open = np.flatnonzero(np.abs(seed.occupation - 1.0) < 1.0e-7)
        seed_alpha = np.flatnonzero(seed.alpha_occupation > 0.5)
        seed_beta = np.flatnonzero(seed.beta_occupation > 0.5)
        somo_values = _subspace_singular_values(
            seed.coeff[:, seed_open], mf.mo_coeff[:, final_open], overlap
        )
        alpha_values = _subspace_singular_values(
            seed.coeff[:, seed_alpha], mf.mo_coeff[:, final_alpha], overlap
        )
        beta_values = _subspace_singular_values(
            seed.coeff[:, seed_beta], mf.mo_coeff[:, final_beta], overlap
        )
        metadata.update(
            seed_somo_singular_values=somo_values,
            seed_alpha_occupied_min_singular_value=min(alpha_values),
            seed_beta_occupied_min_singular_value=min(beta_values),
        )
        if not somo_values or min(somo_values) < minimum_somo_overlap:
            raise RuntimeError(
                "MOM converged, but the final SOMO subspace no longer matches "
                f"the seed (singular values={somo_values}; required minimum "
                f"{minimum_somo_overlap:.3f})"
            )
        if minimum_occupied_overlap is not None and (
            min(alpha_values) < minimum_occupied_overlap
            or min(beta_values) < minimum_occupied_overlap
        ):
            raise RuntimeError(
                "MOM converged, but a complete occupied subspace no longer "
                "matches the seed (minimum alpha/beta singular values="
                f"{min(alpha_values):.6f}/{min(beta_values):.6f}; required "
                f"minimum {minimum_occupied_overlap:.3f})"
            )
    _attach_provenance(mf, **metadata)
    return mf


def build_reference_from_molden(
    path,
    *,
    mol=None,
    xc: str | None = None,
    density_fit: bool = False,
    auxbasis=None,
    verbose: int | None = None,
    grid_level: int | None = None,
    grid_prune: bool = True,
    small_rho_cutoff: float | None = None,
    pcm: PCMConfig | None = None,
):
    """Create a PySCF mean-field object from an external restricted Molden.

    No SCF iterations are performed. PySCF evaluates the density, energy, Fock
    matrices, and response kernel with the requested functional. This is the
    clean cross-program test of the MRSF/EMRSF operator at fixed orbitals.
    """

    seed = load_molden_reference_seed(path, target_mol=mol)
    target_mol = seed.mol
    mf = _new_mean_field(
        target_mol,
        xc=xc,
        density_fit=density_fit,
        auxbasis=auxbasis,
        conv_tol=1.0e-10,
        conv_tol_grad=None,
        direct_scf_tol=None,
        max_cycle=0,
        verbose=verbose,
        grid_level=grid_level,
        grid_prune=grid_prune,
        small_rho_cutoff=small_rho_cutoff,
        pcm=pcm,
    )
    mf.mo_coeff = np.asarray(seed.coeff)
    mf.mo_occ = np.asarray(seed.occupation)
    mf.mo_energy = np.asarray(seed.energy)
    dm = _spin_density(
        seed.coeff, seed.alpha_occupation, seed.beta_occupation
    )
    effective_potential = mf.get_veff(target_mol, dm)
    mf.e_tot = float(
        mf.energy_tot(dm=dm, h1e=mf.get_hcore(), vhf=effective_potential)
    )
    mf.converged = True
    _attach_provenance(
        mf,
        source="external_molden",
        source_path=str(seed.path),
        local_scf_run=False,
        local_scf_converged=False,
        external_reference_accepted=True,
        input_orthonormality_error=seed.input_orthonormality_error,
        seed_somo_singular_values=(1.0, 1.0),
        seed_alpha_occupied_min_singular_value=1.0,
        seed_beta_occupied_min_singular_value=1.0,
        scf_strategy="external-fixed",
        scf_final_stage="external-fixed-orbitals",
        scf_rescue_used=False,
        scf_oscillation_detected=False,
        scf_density_averaging_used=False,
        scf_attempt_history=(),
        pcm=pcm_metadata(mf),
    )
    return mf


def _diagnostics(
    mf,
    coeff: np.ndarray,
    energy: np.ndarray,
    occupation: np.ndarray,
    closed: np.ndarray,
    opened: np.ndarray,
    virtual: np.ndarray,
    effective_fock,
) -> ReferenceDiagnostics:
    overlap = np.asarray(mf.get_ovlp())
    orthonormality_error = _metric_error(coeff, overlap)
    dm = np.asarray(mf.make_rdm1(coeff, occupation))
    if dm.ndim == 2:
        total_dm = dm
        spin_dm = np.zeros_like(dm)
    else:
        total_dm = dm[0] + dm[1]
        spin_dm = dm[0] - dm[1]
    electron_error = float(np.einsum("ij,ji->", overlap, total_dm).real)
    electron_error -= float(mf.mol.nelectron)
    spin_error = float(np.einsum("ij,ji->", overlap, spin_dm).real)
    spin_error -= float(mf.mol.spin)
    gradient = np.asarray(mf.get_grad(coeff, occupation, effective_fock))
    gradient_max = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    gradient_rms = (
        float(np.linalg.norm(gradient) / np.sqrt(gradient.size))
        if gradient.size
        else 0.0
    )

    if bool(getattr(mf.mol, "symmetry", False)):
        s_mo = overlap @ coeff
        weights_by_irrep = []
        for symmetry_orbitals in mf.mol.symm_orb:
            projected = symmetry_orbitals.T.conj() @ s_mo
            symmetry_metric = (
                symmetry_orbitals.T.conj() @ overlap @ symmetry_orbitals
            )
            solved = np.linalg.solve(symmetry_metric, projected)
            weights_by_irrep.append(
                np.einsum("pi,pi->i", projected.conj(), solved).real
            )
        weights_by_irrep = np.asarray(weights_by_irrep)
        weights_by_irrep /= np.sum(weights_by_irrep, axis=0)
        minimum_symmetry_weight = float(
            np.min(np.max(weights_by_irrep, axis=0))
        )
    else:
        minimum_symmetry_weight = 1.0

    nan = float("nan")
    closed_o1 = float(energy[opened[0]] - energy[closed[-1]]) if len(closed) else nan
    o1_o2 = float(energy[opened[1]] - energy[opened[0]])
    o2_virtual = (
        float(energy[virtual[0]] - energy[opened[1]]) if len(virtual) else nan
    )
    metadata = dict(getattr(mf, "emrsf_reference_metadata", {}))
    warnings: list[str] = []
    source = str(metadata.get("source", "unknown"))
    if source == "auto":
        warnings.append("AUTOMATIC_REFERENCE_NOT_STATE_LOCKED")
    if orthonormality_error > 1.0e-7:
        warnings.append("ORBITALS_NOT_ORTHONORMAL")
    if abs(electron_error) > 1.0e-7 or abs(spin_error) > 1.0e-7:
        warnings.append("DENSITY_TRACE_MISMATCH")
    if gradient_max > 1.0e-4:
        warnings.append("LARGE_REFERENCE_ORBITAL_GRADIENT")
    if minimum_symmetry_weight < 1.0 - 1.0e-6:
        warnings.append("ORBITALS_NOT_SYMMETRY_PURE")
    somo_values = tuple(metadata.get("seed_somo_singular_values", ()))
    if somo_values and min(somo_values) < 0.90:
        warnings.append("LOW_SOMO_SUBSPACE_OVERLAP_TO_SEED")

    return ReferenceDiagnostics(
        source=source,
        source_path=metadata.get("source_path"),
        local_scf_run=bool(metadata.get("local_scf_run", True)),
        local_scf_converged=bool(
            metadata.get("local_scf_converged", getattr(mf, "converged", False))
        ),
        orthonormality_error=orthonormality_error,
        electron_number_error=electron_error,
        spin_number_error=spin_error,
        orbital_gradient_max=gradient_max,
        orbital_gradient_rms=gradient_rms,
        minimum_orbital_symmetry_weight=minimum_symmetry_weight,
        closed_to_o1_gap_hartree=closed_o1,
        o1_to_o2_gap_hartree=o1_o2,
        o2_to_virtual_gap_hartree=o2_virtual,
        seed_somo_singular_values=somo_values,
        seed_alpha_occupied_min_singular_value=metadata.get(
            "seed_alpha_occupied_min_singular_value"
        ),
        seed_beta_occupied_min_singular_value=metadata.get(
            "seed_beta_occupied_min_singular_value"
        ),
        warnings=tuple(warnings),
    )


def analyse_reference(mf, *, tol: float = 1.0e-7) -> TripletReference:
    """Validate a PySCF ROHF/ROKS object and expose MRSF orbital spaces."""

    coeff = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    energy = np.asarray(mf.mo_energy)
    if coeff.ndim != 2 or occ.ndim != 1:
        raise TypeError("Only restricted open-shell ROHF/ROKS orbitals are supported")

    closed, opened, virtual = _occupation_classes(occ, tol=tol)
    expected = np.arange(coeff.shape[1])
    if not np.array_equal(np.concatenate((closed, opened, virtual)), expected):
        raise ValueError(
            "MRSF requires canonical C/O1/O2/V orbital ordering; use "
            "build_reference() or reorder the imported orbitals"
        )

    dm = mf.make_rdm1(coeff, occ)
    effective_fock = mf.get_fock(dm=dm)
    fock_ao = np.asarray(effective_fock)
    if fock_ao.ndim == 2:
        hcore = mf.get_hcore()
        tagged_veff = mf.get_veff(mf.mol, dm)
        solvent_potential = getattr(tagged_veff, "v_solvent", None)
        veff = np.asarray(tagged_veff)
        if veff.ndim == 3 and veff.shape[0] == 2:
            fock_a_ao = hcore + veff[0]
            fock_b_ao = hcore + veff[1]
            if solvent_potential is not None:
                fock_a_ao = fock_a_ao + solvent_potential
                fock_b_ao = fock_b_ao + solvent_potential
        else:
            fock_a_ao = fock_b_ao = fock_ao
    else:
        fock_a_ao, fock_b_ao = fock_ao

    fa = coeff.T.conj() @ fock_a_ao @ coeff
    fb = coeff.T.conj() @ fock_b_ao @ coeff
    diagnostics = _diagnostics(
        mf,
        coeff,
        energy,
        occ,
        closed,
        opened,
        virtual,
        effective_fock,
    )
    return TripletReference(
        mf=mf,
        coeff=coeff,
        energy=energy,
        occupation=occ,
        fock_alpha_mo=fa,
        fock_beta_mo=fb,
        closed=closed,
        open=opened,
        virtual=virtual,
        diagnostics=diagnostics,
    )
