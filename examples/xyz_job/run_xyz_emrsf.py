#!/usr/bin/env python3
"""Run production MRSF/EMRSF-TDA manifolds from an XYZ geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib

from pyscf_emrsf import (
    EMRSFTDA,
    HighAccuracyMRSFTDA,
    PCMConfig,
    __version__,
    compute_emrsf_soc,
    compute_external_soc,
    compute_mrsf_soc,
    save_soc_archive,
    write_eigenvectors_csv,
)
from pyscf_emrsf.properties import write_mrsf_transition_properties_csv
from pyscf_emrsf.mrsf import HARTREE_TO_EV
from pyscf_emrsf.reference import build_reference
from pyscf_emrsf.symmetry import term_labels


def xyz_atoms(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {path}")
    natom = int(lines[0].strip())
    atoms = [line.strip() for line in lines[2:] if line.strip()]
    if len(atoms) != natom:
        raise ValueError(
            f"{path} declares {natom} atoms but contains {len(atoms)} atom lines"
        )
    return "\n".join(atoms)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MRSF/EMRSF-TDA singlet/triplet workflow with optional reference "
            "PCM and MRSF/EMRSF spin-orbit state interaction"
        )
    )
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--method", choices=("mrsf", "emrsf"), default="emrsf")
    parser.add_argument(
        "--states",
        "--manifold",
        dest="states",
        choices=("singlets", "triplets", "both"),
        default="singlets",
        help=(
            "spin-free target manifold(s); EMRSF triplets automatically "
            "compute a singlet S0 anchor for vertical energies"
        ),
    )
    parser.add_argument(
        "--response",
        choices=("tda",),
        default="tda",
        help="MRSF is currently derived and validated in the TDA formulation",
    )
    parser.add_argument(
        "--soc",
        action="store_true",
        help="run internal SOC for the selected MRSF/EMRSF model; requires --states both",
    )
    parser.add_argument(
        "--soc-two-electron",
        choices=("none", "amfi", "full"),
        default="amfi",
        help="Breit-Pauli screening: 1e only, AMFI-SOMF, or full SOMF",
    )
    parser.add_argument(
        "--soc-external-module",
        type=Path,
        default=None,
        help=(
            "Python file defining compute_soc(context); receives raw state vectors, "
            "determinant coefficients, and 1e/SOMF integrals"
        ),
    )
    parser.add_argument("--unit", choices=("Angstrom", "Bohr"), default="Angstrom")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=2)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--xc", default="bhandhlyp")
    parser.add_argument(
        "--nstates", type=int, default=5, help="roots per requested manifold"
    )
    parser.add_argument("--grid-level", type=int, default=6)
    parser.add_argument("--pruned-grid", action="store_true")
    parser.add_argument(
        "--integral-backend", choices=("direct", "df"), default="direct"
    )
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument("--reference-molden", type=Path, default=None)
    parser.add_argument("--allow-basis-projection", action="store_true")

    parser.add_argument(
        "--pcm", action="store_true", help="enable self-consistent reference PCM"
    )
    parser.add_argument("--pcm-epsilon", type=float, default=None)
    parser.add_argument(
        "--pcm-method",
        choices=("C-PCM", "COSMO", "IEF-PCM", "SS(V)PE"),
        default="IEF-PCM",
    )
    parser.add_argument("--pcm-lebedev-order", type=int, default=29)
    parser.add_argument("--pcm-vdw-scale", type=float, default=1.2)
    parser.add_argument("--pcm-probe-radius", type=float, default=0.0)
    parser.add_argument(
        "--pcm-surface", choices=("SWIG", "ISWIG"), default="SWIG"
    )
    parser.add_argument("--pcm-max-cycle", type=int, default=50)
    parser.add_argument("--pcm-conv-tol", type=float, default=1.0e-9)

    parser.add_argument(
        "--scf-strategy", choices=("standard", "robust"), default="robust"
    )
    parser.add_argument("--scf-max-cycle", type=int, default=300)
    parser.add_argument("--scf-initial-max-cycle", type=int, default=40)
    parser.add_argument("--scf-stabilization-max-cycle", type=int, default=80)
    parser.add_argument("--scf-newton-max-cycle", type=int, default=80)
    parser.add_argument("--scf-damping", type=float, default=0.50)
    parser.add_argument("--scf-level-shift", type=float, default=0.30)
    parser.add_argument("--scf-coarse-grid-level", type=int, default=3)
    parser.add_argument(
        "--no-scf-oscillation-detection",
        dest="detect_scf_oscillation",
        action="store_false",
    )
    parser.set_defaults(detect_scf_oscillation=True)
    parser.add_argument("--conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--residual-tol", type=float, default=1.0e-7)
    parser.add_argument("--max-cycle", type=int, default=240)
    parser.add_argument("--max-space", type=int, default=120)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=116000)
    parser.add_argument("--nto-weight-threshold", type=float, default=1.0e-4)
    parser.add_argument("--nto-max-pairs", type=int, default=None)
    parser.add_argument("--write-observables", action="store_true")
    parser.add_argument("--observables-grid-level", type=int, default=5)
    parser.add_argument("--observables-unpruned-grid", action="store_true")
    parser.add_argument("--observables-block-size", type=int, default=20000)
    parser.add_argument(
        "--no-oscillator-strengths",
        dest="oscillator_strengths",
        action="store_false",
        help="disable automatic MRSF length-gauge oscillator strengths",
    )
    parser.add_argument(
        "--write-oscillator-strengths",
        dest="oscillator_strengths",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(oscillator_strengths=True)
    parser.add_argument(
        "--save-vectors",
        "--write-eigenvectors",
        dest="save_vectors",
        action="store_true",
        help="write lossless NPZ vectors and a configuration-labelled CSV",
    )
    parser.add_argument(
        "--eigenvector-threshold",
        type=float,
        default=0.0,
        help="omit smaller coefficients only from the readable CSV; NPZ remains complete",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--verbose", type=int, default=4)
    return parser.parse_args()


def check_result(result, label: str, requested_residual: float) -> None:
    if not np.all(result.converged):
        raise RuntimeError(f"{label}: not all roots converged")
    maximum = float(np.max(result.residual_norms))
    if maximum > 1.1 * requested_residual:
        raise RuntimeError(
            f"{label}: maximum residual {maximum:.3e} exceeds "
            f"{requested_residual:.3e}"
        )


def write_states_csv(
    path: Path,
    result,
    method: str,
    manifold: str,
    *,
    common_s0_hartree: float | None = None,
) -> Path:
    symmetries = result.state_symmetries()
    terms = term_labels(symmetries)
    if manifold == "singlets" and common_s0_hartree is None:
        common_s0_hartree = float(result.eigenvalues[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method",
            "manifold",
            "state",
            "response_energy_hartree",
            "excitation_energy_ev",
            "within_manifold_excitation_ev",
            "common_s0_excitation_ev",
            "within_manifold_origin",
            "symmetry",
            "term",
            "purity",
            "gamma_cv",
            "residual_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for state in range(len(result.eigenvalues)):
            within = float(result.excitation_energies_ev[state])
            common = (
                ""
                if common_s0_hartree is None
                else float(
                    (result.eigenvalues[state] - common_s0_hartree)
                    * HARTREE_TO_EV
                )
            )
            writer.writerow(
                {
                    "method": method,
                    "manifold": manifold,
                    "state": state,
                    "response_energy_hartree": float(result.eigenvalues[state]),
                    # Kept for compatibility with v0.6 readers.
                    "excitation_energy_ev": within,
                    "within_manifold_excitation_ev": within,
                    "common_s0_excitation_ev": common,
                    "within_manifold_origin": (
                        "lowest_singlet_root"
                        if manifold == "singlets"
                        else "lowest_triplet_root"
                    ),
                    "symmetry": symmetries[state].label,
                    "term": terms[state],
                    "purity": float(symmetries[state].purity),
                    "gamma_cv": (
                        float(result.cv_weights[state])
                        if method == "emrsf"
                        else ""
                    ),
                    "residual_norm": float(result.residual_norms[state]),
                }
            )
    return path


def write_descriptors_csv(path: Path, descriptors) -> Path:
    rows = [value.as_dict() for value in descriptors]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]) if rows else ["state_index"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def mrsf_solver(mf, args, multiplicity: int):
    return HighAccuracyMRSFTDA(
        mf,
        target_multiplicity=multiplicity,
        nstates=args.nstates,
        density_fit=args.integral_backend == "df",
        auxbasis=args.auxbasis,
        solver="davidson",
        conv_tol=args.conv_tol,
        residual_tol=args.residual_tol,
        max_cycle=args.max_cycle,
        max_space=args.max_space,
        progress=True,
    )


def emrsf_solver(mf, args, multiplicity: int, *, nstates: int | None = None):
    return EMRSFTDA(
        mf,
        target_multiplicity=multiplicity,
        nstates=args.nstates if nstates is None else int(nstates),
        apply_fock_correction=True,
        density_fit=args.integral_backend == "df",
        auxbasis=args.auxbasis,
        solver="davidson",
        conv_tol=args.conv_tol,
        residual_tol=args.residual_tol,
        max_cycle=args.max_cycle,
        max_space=args.max_space,
        progress=True,
        coupling_scale=None,
        solve_mrsf_first=True,
    )


def mrsf_properties(result, args):
    descriptors = ()
    if args.write_observables:
        descriptors = result.charge_transfer_descriptors(
            states=range(len(result.eigenvalues)),
            grid_level=args.observables_grid_level,
            prune=not args.observables_unpruned_grid,
            block_size=args.observables_block_size,
        )
    transitions = ()
    if args.oscillator_strengths and len(result.eigenvalues) > 1:
        transitions = result.transition_properties(
            initial_state=0,
            final_states=range(1, len(result.eigenvalues)),
        )
    return descriptors, transitions


def export_mrsf(
    result,
    args,
    name: str,
    manifold: str,
    *,
    role: str = "primary",
    common_s0_hartree: float | None = None,
) -> dict:
    label = "singlet" if manifold == "singlets" else "triplet"
    descriptors, transitions = mrsf_properties(result, args)
    stem = args.output_dir / f"{name}_mrsf_{label}_{role}_v{__version__}"
    log = result.write_log(
        Path(f"{stem}.log"),
        descriptors=descriptors,
        transition_properties=transitions,
        title=f"MRSF-TDA {label} {role}: {name}",
        top_configurations=12,
    )
    states_csv = write_states_csv(
        Path(f"{stem}_states.csv"),
        result,
        "mrsf",
        manifold,
        common_s0_hartree=common_s0_hartree,
    )
    nto = result.write_nto_molden(
        args.output_dir / f"{name}_mrsf_{label}_nto",
        initial_state=0,
        final_states=range(1, len(result.eigenvalues)),
        weight_threshold=args.nto_weight_threshold,
        max_pairs=args.nto_max_pairs,
    )
    descriptor_csv = None
    if descriptors:
        descriptor_csv = write_descriptors_csv(
            Path(f"{stem}_S42_S47.csv"), descriptors
        )
    oscillator_csv = None
    if transitions:
        oscillator_csv = write_mrsf_transition_properties_csv(
            transitions, Path(f"{stem}_oscillator_strengths.csv")
        )
    archive = None
    eigenvectors_csv = None
    if args.save_vectors:
        archive = result.save_postprocessing(
            Path(f"{stem}_postprocessing.npz"),
            descriptors=descriptors,
            transition_properties=transitions,
            calculation_label=f"MRSF-TDA {label} {role}",
        ).archive
        eigenvectors_csv = write_eigenvectors_csv(
            result,
            Path(f"{stem}_eigenvectors.csv"),
            coefficient_threshold=args.eigenvector_threshold,
        )
    return {
        "log": str(log),
        "states_csv": str(states_csv),
        "nto_directory": str(nto.directory),
        "nto_summary": str(nto.summary_csv),
        "descriptors_csv": (
            None if descriptor_csv is None else str(descriptor_csv)
        ),
        "oscillator_csv": (
            None if oscillator_csv is None else str(oscillator_csv)
        ),
        "postprocessing_archive": None if archive is None else str(archive),
        "eigenvectors_csv": (
            None if eigenvectors_csv is None else str(eigenvectors_csv)
        ),
    }


def export_emrsf(
    result,
    args,
    name: str,
    manifold: str,
    *,
    role: str = "primary",
    common_s0_hartree: float | None = None,
    companion_s0_hartree: float | None = None,
) -> dict:
    label = "singlet" if manifold == "singlets" else "triplet"
    companion = result.mrsf_result
    if companion is None:
        raise RuntimeError("XYZ export requires the companion MRSF roots")
    descriptors, transitions = mrsf_properties(companion, args)
    stem = args.output_dir / f"{name}_emrsf_{label}_{role}_v{__version__}"
    log = result.write_log(
        Path(f"{stem}.log"),
        descriptors=descriptors,
        transition_properties=transitions,
        title=f"EMRSF-TDA {label} {role}: {name}",
        top_configurations=12,
    )
    states_csv = write_states_csv(
        Path(f"{stem}_states.csv"),
        result,
        "emrsf",
        manifold,
        common_s0_hartree=common_s0_hartree,
    )
    archive = None
    eigenvectors_csv = None
    if args.save_vectors:
        archive = result.save_postprocessing(
            Path(f"{stem}_postprocessing.npz"),
            descriptors=descriptors,
            transition_properties=transitions,
            calculation_label=f"EMRSF-TDA {label} {role}",
        ).archive
        eigenvectors_csv = result.write_eigenvectors_csv(
            Path(f"{stem}_eigenvectors.csv"),
            coefficient_threshold=args.eigenvector_threshold,
        )
    companion_output = export_mrsf(
        companion,
        args,
        name,
        manifold,
        role=f"emrsf_{role}_companion",
        common_s0_hartree=companion_s0_hartree,
    )
    return {
        "log": str(log),
        "states_csv": str(states_csv),
        "postprocessing_archive": None if archive is None else str(archive),
        "eigenvectors_csv": (
            None if eigenvectors_csv is None else str(eigenvectors_csv)
        ),
        "oscillator_strength_definition": (
            "corresponding MRSF companion; not a full EMRSF transition 1-RDM"
        ),
        "mrsf_companion": companion_output,
    }


def main() -> None:
    args = arguments()
    args.xyz = args.xyz.resolve()
    if not args.xyz.is_file():
        raise SystemExit(f"XYZ file not found: {args.xyz}")
    if args.spin != 2:
        raise SystemExit(
            "MRSF/EMRSF requires --spin 2 (high-spin triplet reference)"
        )
    if args.nstates < 2:
        raise SystemExit("--nstates must be at least 2")
    if (args.soc or args.soc_external_module is not None) and args.states != "both":
        raise SystemExit("SOC calculation requires --states both")
    if args.eigenvector_threshold < 0.0 or not np.isfinite(
        args.eigenvector_threshold
    ):
        raise SystemExit("--eigenvector-threshold must be finite and non-negative")
    if args.soc_external_module is not None:
        args.soc_external_module = args.soc_external_module.resolve()
        if not args.soc_external_module.is_file():
            raise SystemExit(
                f"External SOC module not found: {args.soc_external_module}"
            )
    density_fit = args.integral_backend == "df"
    if not density_fit and args.auxbasis:
        raise SystemExit(
            "--auxbasis can be used only with --integral-backend df"
        )
    if args.reference_molden is not None:
        args.reference_molden = args.reference_molden.resolve()
        if not args.reference_molden.is_file():
            raise SystemExit(
                f"Reference Molden not found: {args.reference_molden}"
            )
    if args.pcm_epsilon is not None:
        args.pcm = True
    if args.pcm and args.pcm_epsilon is None:
        raise SystemExit(
            "PCM requires --pcm-epsilon (the solvent static dielectric constant)"
        )

    pcm = None
    if args.pcm:
        pcm = PCMConfig(
            epsilon=args.pcm_epsilon,
            method=args.pcm_method,
            lebedev_order=args.pcm_lebedev_order,
            vdw_scale=args.pcm_vdw_scale,
            probe_radius_angstrom=args.pcm_probe_radius,
            surface_method=args.pcm_surface,
            max_cycle=args.pcm_max_cycle,
            conv_tol=args.pcm_conv_tol,
        )

    lib.num_threads(args.threads)
    molecule = gto.M(
        atom=xyz_atoms(args.xyz),
        unit=args.unit,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        symmetry=True,
        verbose=args.verbose,
        max_memory=args.memory_mb,
    )
    mf = build_reference(
        molecule,
        xc=args.xc,
        conv_tol=1.0e-12,
        conv_tol_grad=1.0e-8,
        direct_scf_tol=1.0e-14,
        max_cycle=args.scf_max_cycle,
        density_fit=density_fit,
        auxbasis=args.auxbasis,
        verbose=args.verbose,
        grid_level=args.grid_level,
        grid_prune=args.pruned_grid,
        small_rho_cutoff=1.0e-14,
        molden_seed=args.reference_molden,
        allow_basis_projection=args.allow_basis_projection,
        minimum_somo_overlap=0.80,
        minimum_occupied_overlap=(
            0.80 if args.reference_molden is not None else None
        ),
        scf_strategy=args.scf_strategy,
        robust_initial_max_cycle=args.scf_initial_max_cycle,
        robust_stabilization_max_cycle=args.scf_stabilization_max_cycle,
        robust_newton_max_cycle=args.scf_newton_max_cycle,
        robust_damping=args.scf_damping,
        robust_level_shift=args.scf_level_shift,
        robust_coarse_grid_level=args.scf_coarse_grid_level,
        detect_two_cycle_oscillation=args.detect_scf_oscillation,
        pcm=pcm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.xyz.stem
    outputs: dict[str, object] = {}
    singlet_mrsf = None
    triplet_mrsf = None
    singlet_emrsf = None
    triplet_emrsf = None

    if args.method == "emrsf":
        # Every vertical S0->Tn benchmark needs an energy origin obtained from
        # the same enlarged Hamiltonian.  A triplet-only request therefore
        # solves one singlet root automatically, but does not export a full
        # singlet spectrum unless it was requested.
        singlet_roots = args.nstates if args.states in {"singlets", "both"} else 1
        singlet_emrsf = emrsf_solver(
            mf, args, 1, nstates=singlet_roots
        ).kernel()
        check_result(singlet_emrsf, "EMRSF singlet anchor", args.residual_tol)
        singlet_mrsf = singlet_emrsf.mrsf_result
        check_result(
            singlet_mrsf, "MRSF singlet companion", args.residual_tol
        )
        singlet_origin = float(singlet_emrsf.eigenvalues[0])
        singlet_mrsf_origin = float(singlet_mrsf.eigenvalues[0])

        if args.states in {"singlets", "both"}:
            outputs["emrsf_singlets"] = export_emrsf(
                singlet_emrsf,
                args,
                name,
                "singlets",
                common_s0_hartree=singlet_origin,
                companion_s0_hartree=singlet_mrsf_origin,
            )
        else:
            outputs["emrsf_s0_anchor"] = {
                "response_energy_hartree": singlet_origin,
                "mrsf_companion_energy_hartree": singlet_mrsf_origin,
                "purpose": "common vertical S0 energy origin for triplet roots",
            }

        if args.states in {"triplets", "both"}:
            triplet_emrsf = emrsf_solver(mf, args, 3).kernel()
            check_result(triplet_emrsf, "EMRSF triplets", args.residual_tol)
            triplet_mrsf = triplet_emrsf.mrsf_result
            check_result(
                triplet_mrsf, "MRSF triplet companion", args.residual_tol
            )
            outputs["emrsf_triplets"] = export_emrsf(
                triplet_emrsf,
                args,
                name,
                "triplets",
                common_s0_hartree=singlet_origin,
                companion_s0_hartree=singlet_mrsf_origin,
            )
    else:
        if args.states in {"singlets", "both"}:
            singlet_mrsf = mrsf_solver(mf, args, 1).kernel()
            check_result(singlet_mrsf, "MRSF singlets", args.residual_tol)
        if args.states in {"triplets", "both"}:
            triplet_mrsf = mrsf_solver(mf, args, 3).kernel()
            check_result(triplet_mrsf, "MRSF triplets", args.residual_tol)

        mrsf_s0 = (
            None
            if singlet_mrsf is None
            else float(singlet_mrsf.eigenvalues[0])
        )
        if singlet_mrsf is not None:
            outputs["mrsf_singlets"] = export_mrsf(
                singlet_mrsf,
                args,
                name,
                "singlets",
                common_s0_hartree=mrsf_s0,
            )
        if triplet_mrsf is not None:
            outputs["mrsf_triplets"] = export_mrsf(
                triplet_mrsf,
                args,
                name,
                "triplets",
                common_s0_hartree=mrsf_s0,
            )

    primary = singlet_mrsf if singlet_mrsf is not None else triplet_mrsf
    reference_molden = primary.write_reference_molden(
        args.output_dir / f"{name}_triplet_reference_v{__version__}.molden"
    )
    outputs["reference_molden"] = str(reference_molden)

    if args.soc:
        if args.method == "emrsf":
            soc = compute_emrsf_soc(
                singlet_emrsf,
                triplet_emrsf,
                two_electron=args.soc_two_electron,
            )
        else:
            soc = compute_mrsf_soc(
                singlet_mrsf,
                triplet_mrsf,
                two_electron=args.soc_two_electron,
            )
        soc_stem = args.output_dir / f"{name}_soc_{args.method}_v{__version__}"
        outputs["soc"] = {
            "log": str(soc.write_log(Path(f"{soc_stem}.log"))),
            "couplings_csv": str(
                soc.write_csv(Path(f"{soc_stem}_couplings.csv"))
            ),
            "archive": str(save_soc_archive(Path(f"{soc_stem}.npz"), soc)),
            "operator": soc.approximation,
            "source_model": soc.source_model,
            "external_ready_archive": True,
        }

    if args.soc_external_module is not None:
        source_singlets = singlet_emrsf if args.method == "emrsf" else singlet_mrsf
        source_triplets = triplet_emrsf if args.method == "emrsf" else triplet_mrsf
        external_soc = compute_external_soc(
            args.soc_external_module,
            source_singlets,
            source_triplets,
            two_electron=args.soc_two_electron,
        )
        external_stem = (
            args.output_dir
            / f"{name}_soc_external_{args.method}_v{__version__}"
        )
        outputs["external_soc"] = {
            "module": str(args.soc_external_module),
            "log": str(external_soc.write_log(Path(f"{external_stem}.log"))),
            "couplings_csv": str(
                external_soc.write_csv(Path(f"{external_stem}_couplings.csv"))
            ),
            "archive": str(
                save_soc_archive(Path(f"{external_stem}.npz"), external_soc)
            ),
            "operator": external_soc.approximation,
            "source_model": external_soc.source_model,
        }

    metadata = {
        "program_version": __version__,
        "xyz": str(args.xyz),
        "method": args.method,
        "response": args.response,
        "requested_states": args.states,
        "basis": args.basis,
        "xc": args.xc,
        "roots_per_manifold": args.nstates,
        "integral_backend": args.integral_backend,
        "pcm": None if pcm is None else pcm.as_metadata(),
        "soc_requested": args.soc,
        "soc_external_module": (
            None
            if args.soc_external_module is None
            else str(args.soc_external_module)
        ),
        "soc_two_electron": (
            args.soc_two_electron
            if args.soc or args.soc_external_module is not None
            else None
        ),
        "eigenvectors_saved": args.save_vectors,
        "eigenvector_csv_threshold": (
            args.eigenvector_threshold if args.save_vectors else None
        ),
        "oscillator_strengths_automatic": args.oscillator_strengths,
        "method_scope": (
            "EMRSF singlet and triplet enlarged spaces; SOC includes the "
            "added LR-CV sector and reports projected four-open MRSF C->V weight"
            if args.method == "emrsf"
            else "MRSF-TDA"
        ),
        "outputs": outputs,
    }
    metadata_path = (
        args.output_dir / f"{name}_run_v{__version__}_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("**** XYZ MRSF/EMRSF NORMAL TERMINATION ****", flush=True)
    print(f"Reference Molden : {reference_molden}", flush=True)
    print(f"Metadata         : {metadata_path}", flush=True)
    for key, value in outputs.items():
        if isinstance(value, dict) and value.get("log"):
            print(f"{key:24s}: {value['log']}", flush=True)


if __name__ == "__main__":
    main()
