"""Equation-level diagnostics for the EMRSF MRSF--CV coupling block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mrsf import HARTREE_TO_EV


COUPLING_FAMILIES = (
    "G",
    "D",
    "OOS",
    "CO1",
    "CO2",
    "O1V",
    "O2V",
    "CV(MRSF)",
)


@dataclass(frozen=True)
class CouplingFamilyAudit:
    """Contribution from one MRSF configuration family to ``C.T @ S0``."""

    name: str
    mrsf_weight: float
    coupling_norm: float
    fock_norm: float
    two_electron_norm: float
    diagonal_self_shift_hartree: float
    diagonal_partition_shift_hartree: float

    @property
    def diagonal_self_shift_ev(self) -> float:
        return float(self.diagonal_self_shift_hartree * HARTREE_TO_EV)

    @property
    def diagonal_partition_shift_ev(self) -> float:
        return float(self.diagonal_partition_shift_hartree * HARTREE_TO_EV)


@dataclass(frozen=True)
class GroundCouplingAudit:
    """Ground-state coupling diagnostics evaluated at the MRSF S0 vector."""

    coupling_scale: float
    mrsf_ground_energy_hartree: float
    emrsf_ground_energy_hartree: float
    exact_ground_shift_hartree: float
    gamma_cv_ground: float
    coupling_norm: float
    fock_norm: float
    two_electron_norm: float
    fock_two_electron_dot: float
    diagonal_second_order_shift_hartree: float
    diagonal_fock_partition_hartree: float
    diagonal_two_electron_partition_hartree: float
    families: tuple[CouplingFamilyAudit, ...]

    @property
    def exact_ground_shift_ev(self) -> float:
        return float(self.exact_ground_shift_hartree * HARTREE_TO_EV)

    @property
    def diagonal_second_order_shift_ev(self) -> float:
        return float(self.diagonal_second_order_shift_hartree * HARTREE_TO_EV)

    @property
    def diagonal_fock_partition_ev(self) -> float:
        return float(self.diagonal_fock_partition_hartree * HARTREE_TO_EV)

    @property
    def diagonal_two_electron_partition_ev(self) -> float:
        return float(self.diagonal_two_electron_partition_hartree * HARTREE_TO_EV)


@dataclass(frozen=True)
class LRCVDiagonalAudit:
    """Equation-level decomposition of every added LR-CV diagonal element."""

    labels: tuple[tuple[int, int], ...]
    batch_size: int
    exact_exchange: float
    apply_fock_correction: bool
    raw_fock_hartree: np.ndarray
    s30_hartree: np.ndarray
    coulomb_hartree: np.ndarray
    exact_exchange_hartree: np.ndarray
    xc_kernel_hartree: np.ndarray
    lr_diagonal_hartree: np.ndarray
    ground_offset_hartree: float
    emrsf_diagonal_hartree: np.ndarray
    ground_offset_recomputed_hartree: float
    ground_offset_source_error_hartree: float
    direct_check_indices: np.ndarray
    direct_check_hartree: np.ndarray
    direct_check_error_hartree: np.ndarray

    @property
    def maximum_direct_check_error_hartree(self) -> float:
        if self.direct_check_error_hartree.size == 0:
            return 0.0
        return float(np.max(np.abs(self.direct_check_error_hartree)))

    @property
    def maximum_direct_check_error_ev(self) -> float:
        return float(self.maximum_direct_check_error_hartree * HARTREE_TO_EV)

    @property
    def ground_offset_source_error_ev(self) -> float:
        return float(self.ground_offset_source_error_hartree * HARTREE_TO_EV)


def _representative_cv_indices(components: dict[str, np.ndarray]) -> np.ndarray:
    size = len(components["lr_diagonal"])
    if size == 0:
        return np.empty(0, dtype=int)
    candidates = {0, size // 2, size - 1}
    candidates.add(int(np.argmax(np.abs(components["s30"]))))
    candidates.add(int(np.argmin(components["lr_diagonal"])))
    candidates.add(int(np.argmax(components["lr_diagonal"])))
    return np.asarray(sorted(candidates), dtype=int)


def analyse_lr_cv_diagonal(
    solver,
    *,
    batch_size: int = 32,
    direct_tolerance: float = 2.0e-8,
) -> LRCVDiagonalAudit:
    """Audit S30, the LR kernel, and the single application of ``A_G``.

    All diagonal elements are decomposed in batches.  A small deterministic
    subset is then recomputed through the complete EMRSF ``matvec``.  This
    independently verifies both closure of the component sum and that the
    common ground offset is inserted exactly once in the CV block.
    """

    components = solver.lr.diagonal_components(batch_size=batch_size)
    labels = tuple((int(i), int(a)) for i, a in solver.lr.space.labels)
    ground_offset = float(solver.ground_offset)
    emrsf_diagonal = components["lr_diagonal"] + ground_offset

    ground_recomputed = float(solver.ground_offset_operator)
    ground_error = ground_recomputed - ground_offset

    check_indices = _representative_cv_indices(components)
    direct_values = np.empty(len(check_indices))
    nmrsf = len(solver.live)
    for position, cv_index in enumerate(check_indices):
        vector = np.zeros(solver.dimension)
        vector[nmrsf + int(cv_index)] = 1.0
        direct_values[position] = solver.matvec(vector)[nmrsf + int(cv_index)]
    direct_errors = direct_values - emrsf_diagonal[check_indices]

    tolerance = float(direct_tolerance)
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("direct_tolerance must be finite and positive")
    maximum_error = (
        float(np.max(np.abs(direct_errors))) if direct_errors.size else 0.0
    )
    if abs(ground_error) > tolerance or maximum_error > tolerance:
        raise RuntimeError(
            "LR-CV diagonal audit failed: "
            f"A_G source error={ground_error:.3e} Eh; "
            f"maximum direct error={maximum_error:.3e} Eh"
        )

    return LRCVDiagonalAudit(
        labels=labels,
        batch_size=int(batch_size),
        exact_exchange=float(solver.lr.exact_exchange),
        apply_fock_correction=bool(solver.lr.apply_correction),
        raw_fock_hartree=np.asarray(components["raw_fock"]),
        s30_hartree=np.asarray(components["s30"]),
        coulomb_hartree=np.asarray(components["coulomb"]),
        exact_exchange_hartree=np.asarray(components["exact_exchange"]),
        xc_kernel_hartree=np.asarray(components["xc_kernel"]),
        lr_diagonal_hartree=np.asarray(components["lr_diagonal"]),
        ground_offset_hartree=ground_offset,
        emrsf_diagonal_hartree=np.asarray(emrsf_diagonal),
        ground_offset_recomputed_hartree=ground_recomputed,
        ground_offset_source_error_hartree=float(ground_error),
        direct_check_indices=check_indices,
        direct_check_hartree=direct_values,
        direct_check_error_hartree=direct_errors,
    )


def _family_name(ref, i: int, a: int) -> str:
    closed = set(map(int, ref.closed))
    virtual = set(map(int, ref.virtual))
    if (i, a) == (ref.o2, ref.o1):
        return "G"
    if (i, a) == (ref.o1, ref.o2):
        return "D"
    if (i, a) == (ref.o1, ref.o1):
        return "OOS"
    if (i, a) == (ref.o2, ref.o2):
        return "REDUNDANT"
    if i in closed and a == ref.o1:
        return "CO1"
    if i in closed and a == ref.o2:
        return "CO2"
    if i == ref.o1 and a in virtual:
        return "O1V"
    if i == ref.o2 and a in virtual:
        return "O2V"
    if i in closed and a in virtual:
        return "CV(MRSF)"
    raise ValueError(f"Unclassified MRSF coupling label {(i, a)}")


def family_masks(space) -> dict[str, np.ndarray]:
    """Return mutually exclusive masks for all published MRSF families."""

    names = np.asarray(
        [_family_name(space.ref, int(i), int(a)) for i, a in space.labels]
    )
    masks = {name: names == name for name in COUPLING_FAMILIES}
    coverage = np.sum(np.stack(list(masks.values())), axis=0)
    expected = np.where(names == "REDUNDANT", 0, 1)
    if not np.array_equal(coverage, expected):
        raise RuntimeError("MRSF coupling-family masks are not a partition")
    return masks


def analyse_ground_coupling(
    solver,
    result,
    *,
    lr_cv_audit: LRCVDiagonalAudit | None = None,
) -> GroundCouplingAudit:
    """Decompose the MRSF-S0 coupling to the added LR-CV sector.

    The reported ``diagonal_second_order_shift`` uses the diagonal of the
    shifted LR-CV block as a resolvent.  It is a diagnostic approximation, not
    an alternative EMRSF energy.  Family ``partition`` terms include their
    interference with all other families and therefore sum to the total
    diagonal estimate; ``self`` terms omit that interference.
    """

    if solver.mrsf.target_multiplicity != 1:
        raise ValueError(
            "The ground-coupling audit is an S0 diagnostic and is only "
            "defined for a singlet EMRSF calculation"
        )
    if result.mrsf_result is None:
        raise ValueError(
            "The ground-coupling audit requires solve_mrsf_first=True"
        )

    if result.mrsf_result.space is not solver.mrsf.space:
        raise ValueError("The solver and result do not share the MRSF space")

    psi0 = np.asarray(result.mrsf_result.eigenvectors[:, 0], dtype=float)
    components = solver.coupling.rmatvec_components(psi0)
    fock = components["fock"]
    two_electron = components["two_electron"]
    total = fock + two_electron
    direct = solver.coupling.rmatvec(psi0)
    if not np.allclose(total, direct, atol=2.0e-10, rtol=2.0e-10):
        raise RuntimeError("Fock/two-electron coupling decomposition is inconsistent")

    e0 = float(result.mrsf_result.eigenvalues[0])
    if lr_cv_audit is None:
        diagonal_components = solver.lr.diagonal_components()
        cv_diagonal = np.asarray(
            diagonal_components["lr_diagonal"], dtype=float
        )
        cv_diagonal += float(solver.ground_offset)
    else:
        if lr_cv_audit.labels != tuple(solver.lr.space.labels):
            raise ValueError("The LR-CV audit does not match the solver CV space")
        cv_diagonal = np.asarray(
            lr_cv_audit.emrsf_diagonal_hartree, dtype=float
        )
    denominator = e0 - cv_diagonal
    if np.any(np.abs(denominator) < 1.0e-12):
        raise RuntimeError("Near-zero denominator in diagonal coupling audit")
    response = total / denominator

    masks = family_masks(solver.mrsf.space)
    families = []
    reconstructed = np.zeros_like(total)
    for name in COUPLING_FAMILIES:
        selected = np.where(masks[name], psi0, 0.0)
        part_components = solver.coupling.rmatvec_components(selected)
        part_fock = part_components["fock"]
        part_two = part_components["two_electron"]
        part = part_fock + part_two
        reconstructed += part
        families.append(
            CouplingFamilyAudit(
                name=name,
                mrsf_weight=float(np.dot(selected, selected)),
                coupling_norm=float(np.linalg.norm(part)),
                fock_norm=float(np.linalg.norm(part_fock)),
                two_electron_norm=float(np.linalg.norm(part_two)),
                diagonal_self_shift_hartree=float(
                    np.dot(part, part / denominator)
                ),
                diagonal_partition_shift_hartree=float(np.dot(part, response)),
            )
        )
    if not np.allclose(reconstructed, total, atol=2.0e-10, rtol=2.0e-10):
        raise RuntimeError("Coupling-family decomposition is inconsistent")

    return GroundCouplingAudit(
        coupling_scale=float(solver.coupling_scale),
        mrsf_ground_energy_hartree=e0,
        emrsf_ground_energy_hartree=float(result.eigenvalues[0]),
        exact_ground_shift_hartree=float(result.eigenvalues[0] - e0),
        gamma_cv_ground=float(result.cv_weights[0]),
        coupling_norm=float(np.linalg.norm(total)),
        fock_norm=float(np.linalg.norm(fock)),
        two_electron_norm=float(np.linalg.norm(two_electron)),
        fock_two_electron_dot=float(np.dot(fock, two_electron)),
        diagonal_second_order_shift_hartree=float(np.dot(total, response)),
        diagonal_fock_partition_hartree=float(np.dot(fock, response)),
        diagonal_two_electron_partition_hartree=float(
            np.dot(two_electron, response)
        ),
        families=tuple(families),
    )
