"""Molden exports and MRSF natural transition orbitals.

The NTOs in this module are state-to-state MRSF NTOs.  For a transition from
state ``I`` to state ``F``, the spin-summed one-particle transition density is
constructed in the orthonormal reference-MO basis and factorized as

``D(F,I) = U diag(s) V.T``.

Columns of ``U`` are particle NTOs, columns of ``V`` are hole NTOs, and the
normalized pair weights are ``s**2``.  This definition must not be confused
with an NTO obtained by treating an EMRSF vector as a conventional LR-TDA
amplitude: the latter would omit state-to-state and cross-sector terms.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from .observables import mrsf_transition_difference_density_mo


@dataclass(frozen=True)
class NaturalTransitionOrbitals:
    """Natural transition orbitals for one MRSF state-to-state transition."""

    initial_state: int
    final_state: int
    transition_norm: float
    weights: np.ndarray
    hole_mo: np.ndarray
    particle_mo: np.ndarray
    hole_ao: np.ndarray
    particle_ao: np.ndarray

    @property
    def cumulative_weights(self) -> np.ndarray:
        return np.cumsum(self.weights)


@dataclass(frozen=True)
class NTOExport:
    """Files produced by a multi-transition NTO export."""

    directory: Path
    molden_files: tuple[Path, ...]
    summary_csv: Path


def _validate_state(result, state: int, label: str) -> int:
    state = int(state)
    nstates = int(result.eigenvectors.shape[1])
    if state < 0 or state >= nstates:
        raise IndexError(f"{label} state {state} is outside [0, {nstates - 1}]")
    return state


def _deterministic_pair_phases(hole: np.ndarray, particle: np.ndarray) -> None:
    """Apply a reproducible simultaneous sign to every real NTO pair."""

    for pair in range(hole.shape[1]):
        h_index = int(np.argmax(np.abs(hole[:, pair])))
        p_index = int(np.argmax(np.abs(particle[:, pair])))
        h_value = float(hole[h_index, pair])
        p_value = float(particle[p_index, pair])
        pivot = h_value if abs(h_value) >= abs(p_value) else p_value
        if pivot < 0.0:
            hole[:, pair] *= -1.0
            particle[:, pair] *= -1.0


def mrsf_natural_transition_orbitals(
    result,
    final_state: int,
    *,
    initial_state: int = 0,
    zero_threshold: float = 1.0e-14,
) -> NaturalTransitionOrbitals:
    """Compute normalized MRSF NTO pairs for ``initial_state -> final_state``.

    The transition-density Frobenius norm is retained separately.  The SVD is
    performed after division by this norm, so the reported pair weights sum to
    one (up to floating-point error) and describe the composition rather than
    the absolute magnitude of the transition density.
    """

    initial_state = _validate_state(result, initial_state, "Initial")
    final_state = _validate_state(result, final_state, "Final")
    if initial_state == final_state:
        raise ValueError("NTOs require two different electronic states")

    left = np.asarray(result.eigenvectors[:, initial_state], dtype=float)
    right = np.asarray(result.eigenvectors[:, final_state], dtype=float)
    transition_mo = mrsf_transition_difference_density_mo(
        result.space, left, right
    )
    transition_norm = float(np.linalg.norm(transition_mo))
    if transition_norm <= float(zero_threshold):
        raise ValueError(
            f"MRSF transition {initial_state}->{final_state} has a zero "
            "one-particle transition density"
        )

    particle_mo, singular_values, hole_transpose = np.linalg.svd(
        transition_mo / transition_norm,
        full_matrices=False,
    )
    hole_mo = hole_transpose.T.copy()
    particle_mo = particle_mo.copy()
    _deterministic_pair_phases(hole_mo, particle_mo)

    weights = np.asarray(singular_values**2, dtype=float)
    coefficient = np.asarray(result.space.ref.coeff, dtype=float)
    return NaturalTransitionOrbitals(
        initial_state=initial_state,
        final_state=final_state,
        transition_norm=transition_norm,
        weights=weights,
        hole_mo=hole_mo,
        particle_mo=particle_mo,
        hole_ao=coefficient @ hole_mo,
        particle_ao=coefficient @ particle_mo,
    )


def _selected_pairs(
    weights: np.ndarray,
    *,
    weight_threshold: float,
    max_pairs: int | None,
) -> np.ndarray:
    threshold = float(weight_threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("weight_threshold must lie between 0 and 1")
    selected = np.flatnonzero(np.asarray(weights) >= threshold)
    if selected.size == 0 and len(weights):
        selected = np.asarray([int(np.argmax(weights))])
    if max_pairs is not None:
        max_pairs = int(max_pairs)
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive when specified")
        selected = selected[:max_pairs]
    return selected


def write_reference_molden(mf, path) -> Path:
    """Write the converged reference orbitals in standard Molden format."""

    from pyscf.tools import molden

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    molden.from_scf(mf, str(output))
    return output


def write_nto_molden(
    result,
    nto: NaturalTransitionOrbitals,
    path,
    *,
    weight_threshold: float = 1.0e-4,
    max_pairs: int | None = None,
) -> tuple[Path, np.ndarray]:
    """Write selected hole/particle pairs to one Molden file.

    Orbitals are interleaved as ``H1, P1, H2, P2, ...``.  Molden's occupancy
    field stores the corresponding normalized NTO weight for both members of a
    pair.  The companion CSV written by :func:`write_mrsf_nto_folder` is the
    authoritative machine-readable record of weights and transition norms.
    """

    from pyscf.tools import molden

    selected = _selected_pairs(
        nto.weights,
        weight_threshold=weight_threshold,
        max_pairs=max_pairs,
    )
    columns = []
    labels = []
    occupations = []
    for pair in selected:
        number = int(pair) + 1
        weight = float(nto.weights[pair])
        columns.extend((nto.hole_ao[:, pair], nto.particle_ao[:, pair]))
        labels.extend((f"H{number}", f"P{number}"))
        occupations.extend((weight, weight))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    coefficients = np.column_stack(columns)
    molden.from_mo(
        result.space.ref.mf.mol,
        str(output),
        coefficients,
        symm=labels,
        ene=np.zeros(len(labels)),
        occ=np.asarray(occupations),
    )
    return output, selected


def write_mrsf_nto_folder(
    result,
    output_dir,
    *,
    initial_state: int = 0,
    final_states: Iterable[int] | None = None,
    weight_threshold: float = 1.0e-4,
    max_pairs: int | None = None,
) -> NTOExport:
    """Write one MRSF NTO Molden file for every selected transition."""

    initial_state = _validate_state(result, initial_state, "Initial")
    if final_states is None:
        final_states = (
            state
            for state in range(result.eigenvectors.shape[1])
            if state != initial_state
        )
    final_states = tuple(_validate_state(result, state, "Final") for state in final_states)
    if initial_state in final_states:
        raise ValueError("final_states cannot contain initial_state")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    rows = []
    for final_state in final_states:
        try:
            nto = mrsf_natural_transition_orbitals(
                result,
                final_state,
                initial_state=initial_state,
            )
        except ValueError as error:
            if "zero one-particle transition density" not in str(error):
                raise
            rows.append(
                {
                    "status": "ZERO_TRANSITION_DENSITY",
                    "initial_state": initial_state,
                    "final_state": final_state,
                    "excitation_energy_ev": float(
                        result.eigenvalues[final_state]
                        - result.eigenvalues[initial_state]
                    )
                    * 27.211386245988,
                    "transition_density_norm": 0.0,
                    "pair": "",
                    "weight": "",
                    "cumulative_weight": "",
                    "molden_file": "",
                }
            )
            continue
        filename = f"mrsf_S{initial_state:03d}_to_S{final_state:03d}.molden"
        path, selected = write_nto_molden(
            result,
            nto,
            directory / filename,
            weight_threshold=weight_threshold,
            max_pairs=max_pairs,
        )
        files.append(path)
        cumulative = np.cumsum(nto.weights)
        for pair in selected:
            rows.append(
                {
                    "status": "WRITTEN",
                    "initial_state": initial_state,
                    "final_state": final_state,
                    "excitation_energy_ev": float(
                        result.eigenvalues[final_state]
                        - result.eigenvalues[initial_state]
                    )
                    * 27.211386245988,
                    "transition_density_norm": nto.transition_norm,
                    "pair": int(pair) + 1,
                    "weight": float(nto.weights[pair]),
                    "cumulative_weight": float(cumulative[pair]),
                    "molden_file": filename,
                }
            )

    summary = directory / "nto_summary.csv"
    fieldnames = [
        "status",
        "initial_state",
        "final_state",
        "excitation_energy_ev",
        "transition_density_norm",
        "pair",
        "weight",
        "cumulative_weight",
        "molden_file",
    ]
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return NTOExport(
        directory=directory,
        molden_files=tuple(files),
        summary_csv=summary,
    )
