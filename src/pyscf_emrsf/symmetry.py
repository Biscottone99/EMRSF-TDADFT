"""Point-group labels for explicit MRSF and EMRSF response vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateSymmetry:
    """Dominant Abelian irrep and its squared-vector purity."""

    irrep_id: int
    label: str
    purity: float


def orbital_irrep_ids(ref) -> tuple[np.ndarray, str]:
    """Return PySCF XOR irrep IDs for the reference molecular orbitals."""

    mol = ref.mf.mol
    if not bool(getattr(mol, "symmetry", False)):
        return np.zeros(ref.nmo, dtype=int), "C1"

    from pyscf import symm

    labels = symm.label_orb_symm(
        mol,
        mol.irrep_id,
        mol.symm_orb,
        ref.coeff,
        check=False,
    )
    return np.asarray(labels, dtype=int), str(mol.groupname)


def irrep_name(groupname: str, irrep_id: int) -> str:
    """Convert one PySCF irrep ID to its conventional symbol."""

    if groupname == "C1":
        return "A"
    from pyscf import symm

    return str(symm.irrep_id2name(groupname, int(irrep_id)))


def orbital_irrep_names(ref) -> tuple[str, ...]:
    ids, groupname = orbital_irrep_ids(ref)
    return tuple(irrep_name(groupname, value) for value in ids)


def mrsf_basis_irrep_ids(space) -> tuple[np.ndarray, str]:
    """Irrep of every spin-flip determinant in packed MRSF order."""

    orbital_ids, groupname = orbital_irrep_ids(space.ref)
    reference_id = int(orbital_ids[space.ref.o1]) ^ int(
        orbital_ids[space.ref.o2]
    )
    labels = np.asarray(
        [
            reference_id ^ int(orbital_ids[i]) ^ int(orbital_ids[a])
            for i, a in space.labels
        ],
        dtype=int,
    )
    return labels, groupname


def cv_basis_irrep_ids(space) -> tuple[np.ndarray, str]:
    """Irrep of every closed-shell C->V LR determinant."""

    orbital_ids, groupname = orbital_irrep_ids(space.ref)
    labels = np.asarray(
        [int(orbital_ids[i]) ^ int(orbital_ids[a]) for i, a in space.labels],
        dtype=int,
    )
    return labels, groupname


def classify_vector(
    vector: np.ndarray, basis_irreps: np.ndarray, groupname: str
) -> StateSymmetry:
    """Classify a normalized response vector and report symmetry purity."""

    coefficients = np.asarray(vector)
    basis_irreps = np.asarray(basis_irreps, dtype=int)
    if coefficients.ndim != 1 or coefficients.size != basis_irreps.size:
        raise ValueError("Response vector and basis-irrep arrays must have equal length")
    squared = np.abs(coefficients) ** 2
    norm = float(np.sum(squared))
    if norm <= 1.0e-14:
        return StateSymmetry(-1, "UNKNOWN", 0.0)

    weights = {
        int(value): float(np.sum(squared[basis_irreps == value])) / norm
        for value in np.unique(basis_irreps)
    }
    state_id = max(weights, key=weights.get)
    return StateSymmetry(
        irrep_id=state_id,
        label=irrep_name(groupname, state_id),
        purity=weights[state_id],
    )


def term_labels(symmetries) -> tuple[str, ...]:
    """Number energy-ordered roots within each irrep (for example ``2 B2``)."""

    counts: dict[str, int] = {}
    labels = []
    for value in symmetries:
        counts[value.label] = counts.get(value.label, 0) + 1
        labels.append(f"{counts[value.label]} {value.label}")
    return tuple(labels)
