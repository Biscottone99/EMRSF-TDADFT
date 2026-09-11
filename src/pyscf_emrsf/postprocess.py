"""Self-contained numerical archives for reproducible post-processing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audit import _family_name
from .symmetry import orbital_irrep_ids, orbital_irrep_names, term_labels


@dataclass(frozen=True)
class PostprocessingExport:
    """Paths produced by :func:`save_postprocessing_archive`."""

    archive: Path
    manifest: Path


def write_eigenvectors_csv(
    result,
    path,
    *,
    coefficient_threshold: float = 0.0,
) -> Path:
    """Write response eigenvectors with an explicit configuration-row map.

    The complete, lossless vectors remain available in the NPZ archive.  This
    CSV is a human-readable interchange table and may optionally omit small
    coefficients.  Orbital, row, and state indices are deliberately zero-based.
    """

    threshold = float(coefficient_threshold)
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError("coefficient_threshold must be finite and non-negative")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    is_emrsf = hasattr(result, "mrsf_size")
    space = result.mrsf_space if is_emrsf else result.space
    if is_emrsf:
        live = np.asarray(result.live_mrsf_indices, dtype=int)
        mrsf_labels = np.asarray(space.labels, dtype=int)[live]
        cv_labels = np.asarray(result.cv_space.labels, dtype=int).reshape(-1, 2)
        labels = np.vstack((mrsf_labels, cv_labels))
        sectors = np.asarray(
            ["MRSF"] * len(mrsf_labels) + ["LR-CV"] * len(cv_labels),
            dtype="U8",
        )
        families = np.concatenate(
            (
                _mrsf_families(space)[live],
                np.asarray(["LR-CV"] * len(cv_labels), dtype="U16"),
            )
        )
    else:
        labels = np.asarray(space.labels, dtype=int).reshape(-1, 2)
        sectors = np.asarray(["MRSF"] * len(labels), dtype="U8")
        families = _mrsf_families(space)
    if len(labels) != result.eigenvectors.shape[0]:
        raise RuntimeError("Eigenvector row count does not match the basis map")

    symmetries = result.state_symmetries()
    terms = term_labels(symmetries)
    prefix = "S" if int(space.target_multiplicity) == 1 else "T"
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "state_index_zero_based",
            "state_label",
            "term",
            "energy_hartree",
            "basis_row_zero_based",
            "sector",
            "family",
            "source_orbital_zero_based",
            "target_orbital_zero_based",
            "coefficient_real",
            "coefficient_imag",
            "weight",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for state in range(result.eigenvectors.shape[1]):
            for row, coefficient in enumerate(result.eigenvectors[:, state]):
                coefficient = complex(coefficient)
                if abs(coefficient) < threshold:
                    continue
                source, target = labels[row]
                writer.writerow(
                    {
                        "state_index_zero_based": state,
                        "state_label": f"{prefix}{state}",
                        "term": terms[state],
                        "energy_hartree": float(result.eigenvalues[state]),
                        "basis_row_zero_based": row,
                        "sector": sectors[row],
                        "family": families[row],
                        "source_orbital_zero_based": int(source),
                        "target_orbital_zero_based": int(target),
                        "coefficient_real": coefficient.real,
                        "coefficient_imag": coefficient.imag,
                        "weight": abs(coefficient) ** 2,
                    }
                )
    return output


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mrsf_families(space) -> np.ndarray:
    return np.asarray(
        [_family_name(space.ref, int(i), int(a)) for i, a in space.labels],
        dtype="U16",
    )


def save_postprocessing_archive(
    result,
    path,
    *,
    descriptors=None,
    transition_properties=None,
    calculation_label: str | None = None,
) -> PostprocessingExport:
    """Save complete roots, basis maps, orbitals and AO integrals.

    The NPZ contains no pickled Python objects and can therefore be loaded with
    ``numpy.load(path, allow_pickle=False)``.  Molecular construction data are
    retained as the JSON string returned by ``Mole.dumps()`` in addition to
    explicit coordinates, charges, orbitals and one-electron matrices.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    manifest_path = output.with_suffix(".manifest.json")

    is_emrsf = hasattr(result, "mrsf_size")
    result_type = "EMRSF" if is_emrsf else "MRSF"
    mrsf_space = result.mrsf_space if is_emrsf else result.space
    ref = mrsf_space.ref
    mol = ref.mf.mol
    state_symmetries = result.state_symmetries()
    state_terms = term_labels(state_symmetries)
    orbital_ids, groupname = orbital_irrep_ids(ref)

    arrays: dict[str, np.ndarray] = {
        "archive_format_version": np.asarray(3, dtype=np.int64),
        "result_type": np.asarray(result_type),
        "target_multiplicity": np.asarray(
            mrsf_space.target_multiplicity, dtype=np.int64
        ),
        "calculation_label": np.asarray(calculation_label or result_type),
        "eigenvalues_hartree": np.asarray(result.eigenvalues, dtype=float),
        "excitation_energies_hartree": np.asarray(
            result.excitation_energies, dtype=float
        ),
        "excitation_energies_ev": np.asarray(
            result.excitation_energies_ev, dtype=float
        ),
        "eigenvectors_columns_are_states": np.asarray(
            result.eigenvectors, dtype=float
        ),
        "state_irrep_ids": np.asarray(
            [value.irrep_id for value in state_symmetries], dtype=int
        ),
        "state_irrep_labels": np.asarray(
            [value.label for value in state_symmetries], dtype="U16"
        ),
        "state_terms": np.asarray(state_terms, dtype="U32"),
        "state_symmetry_purities": np.asarray(
            [value.purity for value in state_symmetries], dtype=float
        ),
        "state_converged": np.asarray(result.converged, dtype=bool),
        "state_residual_norms": np.asarray(result.residual_norms, dtype=float),
        "mo_coeff_ao_by_mo": np.asarray(ref.coeff, dtype=float),
        "mo_energies_hartree": np.asarray(ref.energy, dtype=float),
        "mo_occupations": np.asarray(ref.occupation, dtype=float),
        "fock_alpha_mo": np.asarray(ref.fock_alpha_mo, dtype=float),
        "fock_beta_mo": np.asarray(ref.fock_beta_mo, dtype=float),
        "closed_orbitals_zero_based": np.asarray(ref.closed, dtype=int),
        "open_orbitals_zero_based": np.asarray(ref.open, dtype=int),
        "virtual_orbitals_zero_based": np.asarray(ref.virtual, dtype=int),
        "orbital_irrep_ids": np.asarray(orbital_ids, dtype=int),
        "orbital_irrep_labels": np.asarray(orbital_irrep_names(ref), dtype="U16"),
        "point_group": np.asarray(groupname),
        "atom_symbols": np.asarray(
            [mol.atom_symbol(index) for index in range(mol.natm)], dtype="U8"
        ),
        "atom_charges": np.asarray(mol.atom_charges(), dtype=float),
        "atom_coordinates_bohr": np.asarray(mol.atom_coords(), dtype=float),
        "molecular_charge": np.asarray(mol.charge, dtype=int),
        "molecular_spin_2s": np.asarray(mol.spin, dtype=int),
        "electron_count": np.asarray(mol.nelectron, dtype=int),
        "ao_overlap": np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float),
        "ao_position_integrals_bohr": np.asarray(
            mol.intor_symmetric("int1e_r", comp=3), dtype=float
        ),
        "pyscf_molecule_json": np.asarray(mol.dumps()),
        "mrsf_labels_zero_based": np.asarray(mrsf_space.labels, dtype=int),
        "mrsf_configuration_families": _mrsf_families(mrsf_space),
    }

    if result.matrix is not None:
        arrays["response_matrix_hartree"] = np.asarray(result.matrix, dtype=float)

    if is_emrsf:
        live = np.asarray(result.live_mrsf_indices, dtype=int)
        decompositions = tuple(result.state_decompositions)
        arrays.update(
            {
                "emrsf_mrsf_block_size": np.asarray(result.mrsf_size, dtype=int),
                "emrsf_live_mrsf_indices": live,
                "emrsf_live_mrsf_labels_zero_based": np.asarray(
                    mrsf_space.labels, dtype=int
                )[live],
                "emrsf_live_mrsf_configuration_families": _mrsf_families(
                    mrsf_space
                )[live],
                "emrsf_cv_labels_zero_based": np.asarray(
                    result.cv_space.labels, dtype=int
                ),
                "emrsf_cv_weights_gamma": np.asarray(result.cv_weights, dtype=float),
                "emrsf_mrsf_weights": np.asarray(result.mrsf_weights, dtype=float),
                "emrsf_exact_exchange": np.asarray(
                    np.nan if result.exact_exchange is None else result.exact_exchange,
                    dtype=float,
                ),
                "emrsf_coupling_scale": np.asarray(
                    np.nan if result.coupling_scale is None else result.coupling_scale,
                    dtype=float,
                ),
                "emrsf_ground_offset_hartree": np.asarray(
                    np.nan
                    if result.ground_offset_hartree is None
                    else result.ground_offset_hartree,
                    dtype=float,
                ),
            }
        )
        if decompositions:
            arrays["emrsf_state_decomposition"] = np.asarray(
                [
                    [
                        value.mrsf_weight,
                        value.cv_weight,
                        value.mrsf_block_hartree,
                        value.cv_block_hartree,
                        value.coupling_hartree,
                        value.reconstructed_hartree,
                        value.closure_error_hartree,
                    ]
                    for value in decompositions
                ],
                dtype=float,
            )
            arrays["emrsf_state_decomposition_columns"] = np.asarray(
                [
                    "mrsf_weight",
                    "cv_weight",
                    "mrsf_block_hartree",
                    "cv_block_hartree",
                    "coupling_hartree",
                    "reconstructed_hartree",
                    "closure_error_hartree",
                ],
                dtype="U32",
            )

    descriptor_rows = [] if descriptors is None else [
        value.as_dict() for value in descriptors
    ]
    transition_rows = [] if transition_properties is None else [
        value.as_dict() for value in transition_properties
    ]
    arrays["charge_transfer_descriptors_json"] = np.asarray(
        json.dumps(_jsonable(descriptor_rows), sort_keys=True)
    )
    arrays["mrsf_transition_properties_json"] = np.asarray(
        json.dumps(_jsonable(transition_rows), sort_keys=True)
    )

    metadata = dict(getattr(ref.mf, "emrsf_reference_metadata", {}))
    manifest = {
        "archive_format_version": 3,
        "result_type": result_type,
        "target_multiplicity": int(mrsf_space.target_multiplicity),
        "calculation_label": calculation_label or result_type,
        "array_count": len(arrays),
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "reference_metadata": _jsonable(metadata),
        "indexing": "All orbital and state indices stored in the archive are zero-based.",
        "eigenvector_layout": "Rows are configurations; columns are electronic states.",
        "energy_origin": (
            "Stored excitation_energies are relative to the lowest root in the "
            "same spin manifold. Vertical S0-to-triplet energies require a "
            "separately computed singlet S0 common origin."
        ),
        "mrsf_basis_layout": "mrsf_labels_zero_based gives (source,target) for every full MRSF row.",
        "emrsf_basis_layout": (
            "For EMRSF, rows 0:mrsf_size follow emrsf_live_mrsf_labels_zero_based; "
            "remaining rows follow emrsf_cv_labels_zero_based."
            if is_emrsf
            else None
        ),
        "oscillator_strength_definition": (
            "Length-gauge MRSF state-to-state transition density. Full EMRSF "
            "oscillator strengths are not claimed."
        ),
        "ct_descriptor_definition": (
            "S42-S47 evaluated from MRSF state densities relative to the triplet reference."
        ),
    }

    np.savez_compressed(output, **arrays)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PostprocessingExport(archive=output, manifest=manifest_path)
