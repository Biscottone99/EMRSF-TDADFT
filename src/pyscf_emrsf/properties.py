"""Electric-dipole transition properties for MRSF state-interaction roots.

The transition density used here is the same spin-adapted state-to-state
one-particle density used for the MRSF NTO export.  The oscillator strengths
are therefore rigorous length-gauge properties of the computed *MRSF* roots.
They must not be relabelled as full EMRSF oscillator strengths: the published
EMRSF equations do not provide the LR--LR and MRSF--LR transition-density
blocks required for that quantity.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .mrsf import HARTREE_TO_EV
from .observables import mrsf_transition_difference_density_mo


@dataclass(frozen=True)
class MRSFTransitionProperty:
    """Length-gauge electric-dipole property for one MRSF transition."""

    initial_state: int
    final_state: int
    excitation_energy_hartree: float
    transition_dipole_e_bohr: np.ndarray
    oscillator_strength: float
    transition_density_trace: float
    transition_density_norm: float

    @property
    def excitation_energy_ev(self) -> float:
        return float(self.excitation_energy_hartree * HARTREE_TO_EV)

    @property
    def transition_dipole_norm_e_bohr(self) -> float:
        return float(np.linalg.norm(self.transition_dipole_e_bohr))

    def as_dict(self) -> dict[str, object]:
        dipole = np.asarray(self.transition_dipole_e_bohr, dtype=float)
        return {
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "excitation_energy_hartree": self.excitation_energy_hartree,
            "excitation_energy_ev": self.excitation_energy_ev,
            "transition_dipole_x_e_bohr": float(dipole[0]),
            "transition_dipole_y_e_bohr": float(dipole[1]),
            "transition_dipole_z_e_bohr": float(dipole[2]),
            "transition_dipole_norm_e_bohr": self.transition_dipole_norm_e_bohr,
            "oscillator_strength_length": self.oscillator_strength,
            "transition_density_trace": self.transition_density_trace,
            "transition_density_norm": self.transition_density_norm,
            "definition": "MRSF state-to-state transition 1-RDM, length gauge",
        }


def mrsf_transition_property(
    result,
    final_state: int,
    *,
    initial_state: int = 0,
) -> MRSFTransitionProperty:
    """Evaluate the length-gauge transition dipole and oscillator strength.

    The electronic charge sign is included in the reported transition dipole.
    It has no effect on the oscillator strength.  The nuclear contribution is
    zero between different orthonormal electronic states.
    """

    initial_state = int(initial_state)
    final_state = int(final_state)
    nstates = result.eigenvectors.shape[1]
    if initial_state < 0 or initial_state >= nstates:
        raise IndexError("Initial state is outside the computed root range")
    if final_state < 0 or final_state >= nstates:
        raise IndexError("Final state is outside the computed root range")
    if initial_state == final_state:
        raise ValueError("Transition properties require two different states")

    energy = float(result.eigenvalues[final_state] - result.eigenvalues[initial_state])
    if energy <= 0.0:
        raise ValueError("Final state must lie above the initial state")

    transition_mo = mrsf_transition_difference_density_mo(
        result.space,
        result.eigenvectors[:, initial_state],
        result.eigenvectors[:, final_state],
    )
    coefficient = np.asarray(result.space.ref.coeff, dtype=float)
    transition_ao = coefficient @ transition_mo @ coefficient.T
    mol = result.space.ref.mf.mol
    dipole_integrals = np.asarray(mol.intor_symmetric("int1e_r", comp=3))
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))

    # The matrix produced by observables.py follows rho(r)=chi D chi.  The
    # Cartesian one-electron integral is symmetric, so this contraction is
    # equivalent to Tr[D r].
    transition_dipole = -np.einsum(
        "xuv,uv->x", dipole_integrals, transition_ao, optimize=True
    )
    density_trace = float(np.einsum("uv,uv->", overlap, transition_ao))
    oscillator_strength = float(
        (2.0 / 3.0) * energy * np.dot(transition_dipole, transition_dipole)
    )
    return MRSFTransitionProperty(
        initial_state=initial_state,
        final_state=final_state,
        excitation_energy_hartree=energy,
        transition_dipole_e_bohr=transition_dipole,
        oscillator_strength=oscillator_strength,
        transition_density_trace=density_trace,
        transition_density_norm=float(np.linalg.norm(transition_mo)),
    )


def analyse_mrsf_transition_properties(
    result,
    *,
    initial_state: int = 0,
    final_states: Iterable[int] | None = None,
) -> tuple[MRSFTransitionProperty, ...]:
    """Evaluate transition properties from one root to selected higher roots."""

    initial_state = int(initial_state)
    if final_states is None:
        final_states = range(initial_state + 1, result.eigenvectors.shape[1])
    return tuple(
        mrsf_transition_property(
            result,
            int(final_state),
            initial_state=initial_state,
        )
        for final_state in final_states
    )


def write_mrsf_transition_properties_csv(properties, path) -> Path:
    """Write a machine-readable table of MRSF transition properties."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [value.as_dict() for value in properties]
    fieldnames = list(rows[0]) if rows else list(MRSFTransitionProperty(
        0, 1, 0.0, np.zeros(3), 0.0, 0.0, 0.0
    ).as_dict())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output
