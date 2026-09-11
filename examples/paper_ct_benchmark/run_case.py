"""Run one matrix-free v0.4.1 calculation from an SI_loos geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf import gto, lib

from pyscf_emrsf import (
    EMRSFTDA,
    __version__,
    analyse_ground_coupling,
    analyse_lr_cv_diagonal,
)
from pyscf_emrsf.reference import build_reference, build_reference_from_molden
from pyscf_emrsf.symmetry import term_labels

CASES = {
    "azulene": "azulene.bohr.xyz",
    "n_phenylpyrrole": "n_phenylpyrrole.bohr.xyz",
    "phthalazine": "phthalazine.bohr.xyz",
    "quinoxaline": "quinoxaline.bohr.xyz",
    "twisted_dmabn": "twisted_dmabn.bohr.xyz",
}

# These two systems were observed to converge to a different Aufbau triplet in
# PySCF than in the OpenQP benchmark.  An unlocked run is therefore rejected by
# default instead of producing a physically unrelated but numerically smooth
# spectrum.
LOCKED_REFERENCE_CASES = {"n_phenylpyrrole", "twisted_dmabn"}


def read_bohr_xyz(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0].strip())
    atoms = [line.strip() for line in lines[2:] if line.strip()]
    if len(atoms) != atom_count:
        raise ValueError(
            f"{path} declares {atom_count} atoms but contains {len(atoms)}"
        )
    return "\n".join(atoms)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--auxbasis", default="cc-pvdz-jkfit")
    parser.add_argument("--xc", default="bhandhlyp")
    parser.add_argument(
        "--symmetry",
        default="auto",
        help=(
            "PySCF Abelian point group; 'auto' detects it from the supplied "
            "geometry and 'none' disables symmetry"
        ),
    )
    parser.add_argument(
        "--reference-mode",
        choices=("auto", "mom", "external"),
        default="auto",
        help=(
            "triplet reference: Aufbau ROKS, Molden-seeded MOM ROKS, or a "
            "fixed external Molden reference"
        ),
    )
    parser.add_argument(
        "--reference-molden",
        type=Path,
        default=None,
        help="restricted alpha/beta Molden seed required by mom/external modes",
    )
    parser.add_argument(
        "--allow-unlocked-reference",
        action="store_true",
        help=(
            "allow unsafe automatic references for cases known to exhibit an "
            "occupation mismatch (diagnostic experiments only)"
        ),
    )
    parser.add_argument("--reference-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--reference-max-cycle", type=int, default=200)
    parser.add_argument("--minimum-somo-overlap", type=float, default=0.80)
    parser.add_argument("--minimum-occupied-overlap", type=float, default=0.80)
    parser.add_argument(
        "--scf-grid-level",
        type=int,
        default=3,
        help="PySCF ROKS and response-kernel grid level",
    )
    parser.add_argument("--unpruned-scf-grid", action="store_true")
    parser.add_argument("--nstates", type=int, default=12)
    parser.add_argument("--solver", choices=("davidson", "lobpcg", "dense"), default="davidson")
    parser.add_argument("--conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--max-space", type=int, default=None)
    parser.add_argument(
        "--observable-grid-level",
        "--grid-level",
        dest="observable_grid_level",
        type=int,
        default=5,
        help="numerical grid used only for observables S42--S47",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--pyscf-verbose", type=int, default=4)
    parser.add_argument(
        "--coupling-scale",
        type=float,
        default=None,
        help=(
            "EMRSF c_cp factor applied to the complete MRSF-CV coupling; "
            "the paper default is the functional's global HF fraction c_H"
        ),
    )
    parser.add_argument(
        "--coupling-audit",
        action="store_true",
        help="decompose C.T|MRSF S0> by configuration family and operator term",
    )
    parser.add_argument(
        "--lr-cv-audit",
        action="store_true",
        help=(
            "decompose every added LR-CV diagonal into raw Fock, S30, "
            "Coulomb, exact exchange, semilocal XC, and A_G"
        ),
    )
    parser.add_argument(
        "--lr-cv-audit-batch-size",
        type=int,
        default=32,
        help="number of AO response densities per LR-CV audit batch",
    )
    parser.add_argument(
        "--skip-observables",
        action="store_true",
        help="skip the S42--S47 grid integration during coupling-only diagnostics",
    )
    parser.add_argument("--unpruned-grid", action="store_true")
    parser.add_argument(
        "--write-molden",
        action="store_true",
        help="write the converged triplet-reference orbitals to Molden format",
    )
    parser.add_argument(
        "--write-nto",
        action="store_true",
        help="write MRSF S0->Sn NTO Molden files and a CSV summary",
    )
    parser.add_argument("--nto-weight-threshold", type=float, default=1.0e-4)
    parser.add_argument("--nto-max-pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    total_start = perf_counter()

    def announce(message: str) -> None:
        elapsed = (perf_counter() - total_start) / 60.0
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"[elapsed {elapsed:.1f} min] {message}",
            flush=True,
        )

    announce(
        "optional exports: "
        f"reference Molden={'ON' if args.write_molden else 'OFF'}, "
        f"MRSF NTO={'ON' if args.write_nto else 'OFF'}"
    )

    here = Path(__file__).resolve().parent
    geometry_path = here / "geometries" / CASES[args.case]
    atoms = read_bohr_xyz(geometry_path)
    lib.num_threads(args.threads)
    symmetry_request = args.symmetry.strip().lower()
    if symmetry_request in {"none", "false", "off", "c1"}:
        symmetry = False
    elif symmetry_request in {"auto", "true", "on"}:
        symmetry = True
    else:
        symmetry = args.symmetry

    if (
        args.reference_mode == "auto"
        and args.case in LOCKED_REFERENCE_CASES
        and not args.allow_unlocked_reference
    ):
        raise SystemExit(
            f"{args.case} requires a state-locked reference. Use "
            "--reference-mode mom (production) or --reference-mode external "
            "(fixed-orbital OpenQP cross-check), together with "
            "--reference-molden. Use --allow-unlocked-reference only to "
            "reproduce the known incorrect Aufbau-reference diagnostic."
        )
    if args.reference_mode in {"mom", "external"} and args.reference_molden is None:
        raise SystemExit(
            f"--reference-mode {args.reference_mode} requires --reference-molden"
        )
    if args.reference_mode == "auto" and args.reference_molden is not None:
        raise SystemExit(
            "--reference-molden is meaningful only with --reference-mode mom "
            "or external"
        )

    announce(f"building {args.case} molecule and {args.basis} basis")
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=args.basis,
        charge=0,
        spin=2,
        symmetry=symmetry,
        verbose=args.pyscf_verbose,
        max_memory=args.memory_mb,
    )
    common_reference = dict(
        xc=args.xc,
        density_fit=True,
        auxbasis=args.auxbasis,
        verbose=args.pyscf_verbose,
        grid_level=args.scf_grid_level,
        grid_prune=not args.unpruned_scf_grid,
    )
    if args.reference_mode == "external":
        announce("loading fixed external restricted triplet Molden reference")
        mf = build_reference_from_molden(
            args.reference_molden,
            mol=mol,
            **common_reference,
        )
        announce(f"external triplet reference accepted: E={mf.e_tot:.14f} Eh")
    else:
        mode_label = "Molden-seeded MOM" if args.reference_mode == "mom" else "Aufbau"
        announce(f"starting density-fitted {mode_label} triplet ROKS reference")
        mf = build_reference(
            mol,
            conv_tol=args.reference_conv_tol,
            max_cycle=args.reference_max_cycle,
            molden_seed=(
                args.reference_molden if args.reference_mode == "mom" else None
            ),
            minimum_somo_overlap=args.minimum_somo_overlap,
            minimum_occupied_overlap=args.minimum_occupied_overlap,
            **common_reference,
        )
        announce(f"triplet ROKS converged: E={mf.e_tot:.14f} Eh")
    announce("building RI intermediates and the matrix-free EMRSF operator")
    solver = EMRSFTDA(
        mf,
        nstates=args.nstates,
        apply_fock_correction=True,
        density_fit=True,
        auxbasis=args.auxbasis,
        solver=args.solver,
        conv_tol=args.conv_tol,
        max_cycle=args.max_cycle,
        max_space=args.max_space,
        progress=True,
        coupling_scale=args.coupling_scale,
    )
    reference_diagnostics = solver.mrsf.ref.diagnostics
    announce(
        "reference audit: "
        f"source={reference_diagnostics.source}; "
        f"O1/O2={solver.mrsf.ref.o1 + 1}/{solver.mrsf.ref.o2 + 1}; "
        f"gradient_max={reference_diagnostics.orbital_gradient_max:.3e}; "
        "warnings="
        + (
            ",".join(reference_diagnostics.warnings)
            if reference_diagnostics.warnings
            else "NONE"
        )
    )
    announce(
        f"EMRSF coupling: c_H={solver.lr.exact_exchange:.8f}; "
        f"c_cp={solver.coupling_scale:.8f}; "
        "G=sqrt(2), CO1/O2V=S1-S2; form=c_cp*<MRSF|H|CV>"
    )
    resources = solver.resource_estimate()
    announce(
        f"operator ready: dimension={solver.dimension}; resources={resources}"
    )
    announce(f"starting {args.solver} diagonalisation")
    result = solver.kernel()
    if not np.all(result.converged):
        raise RuntimeError(
            "EMRSF iterative diagonalisation did not converge all roots; "
            f"residuals={result.residual_norms}"
        )
    announce(
        f"EMRSF converged in {result.iterations} iterations and "
        f"{result.matvecs} matvecs; max residual="
        f"{float(np.max(result.residual_norms)):.3e}"
    )
    lr_cv_audit = None
    if args.lr_cv_audit:
        announce(
            "starting SI S30-S39 LR-CV diagonal audit with batch size "
            f"{args.lr_cv_audit_batch_size}"
        )
        lr_cv_audit = analyse_lr_cv_diagonal(
            solver,
            batch_size=args.lr_cv_audit_batch_size,
        )
        announce(
            "LR-CV diagonal audit completed: A_G source error="
            f"{lr_cv_audit.ground_offset_source_error_hartree:.3e} Eh; "
            "maximum direct error="
            f"{lr_cv_audit.maximum_direct_check_error_hartree:.3e} Eh"
        )

    coupling_audit = None
    if args.coupling_audit:
        announce("starting ground-state MRSF--CV coupling audit")
        coupling_audit = analyse_ground_coupling(
            solver,
            result,
            lr_cv_audit=lr_cv_audit,
        )
        announce(
            "coupling audit completed: exact S0 shift="
            f"{coupling_audit.exact_ground_shift_ev:.8f} eV; "
            "diagonal estimate="
            f"{coupling_audit.diagonal_second_order_shift_ev:.8f} eV"
        )

    descriptors = None
    if args.skip_observables:
        announce("S42--S47 integration skipped by request")
    else:
        announce(
            "starting S42--S47 integration at grid level "
            f"{args.observable_grid_level}"
        )
        descriptors = result.charge_transfer_descriptors(
            grid_level=args.observable_grid_level,
            prune=not args.unpruned_grid,
        )
        announce("S42--S47 integration completed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"{args.case}_v{__version__}"

    molden_path = None
    if args.write_molden:
        announce("writing triplet-reference orbitals in Molden format")
        molden_path = result.write_reference_molden(
            args.output_dir / f"{args.case}_v{__version__}_orbitals.molden"
        )
        announce(f"reference Molden written: {molden_path}")

    nto_export = None
    if args.write_nto:
        announce("computing MRSF S0->Sn natural transition orbitals")
        nto_export = result.write_mrsf_nto_molden(
            args.output_dir / f"{args.case}_v{__version__}_nto",
            initial_state=0,
            weight_threshold=args.nto_weight_threshold,
            max_pairs=args.nto_max_pairs,
        )
        announce(
            f"MRSF NTO export completed: {len(nto_export.molden_files)} "
            f"Molden files; summary={nto_export.summary_csv}"
        )

    log_path = result.write_log(
        Path(f"{stem}.log"),
        descriptors=descriptors,
        coupling_audit=coupling_audit,
        lr_cv_audit=lr_cv_audit,
        title=f"Oh et al. CT benchmark: {args.case}",
    )

    matches = result.state_matches()
    emrsf_symmetries = result.state_symmetries()
    mrsf_symmetries = result.mrsf_result.state_symmetries()
    emrsf_terms = term_labels(emrsf_symmetries)
    mrsf_terms = term_labels(mrsf_symmetries)
    state_densities = [
        result.mrsf_result.state_density(state)
        for state in range(len(result.mrsf_result.eigenvalues))
    ]
    metadata = {
        "program": "pyscf-emrsf",
        "version": __version__,
        "case": args.case,
        "geometry_file": geometry_path.name,
        "geometry_unit": "bohr",
        "basis": args.basis,
        "xc": args.xc,
        "auxbasis": args.auxbasis,
        "solver": result.solver,
        "solver_tolerance": args.conv_tol,
        "exact_exchange_c_h": solver.lr.exact_exchange,
        "coupling_scale_c_cp": solver.coupling_scale,
        "g_spin_factor": "sqrt(2), fixed by SI Table S3",
        "co1_table_combination": "S1-S2, fixed by spin adaptation",
        "o2v_table_combination": "S1-S2, fixed by spin adaptation",
        "coupling_definition": "c_cp * <Phi_MRSF|H|Phi_CV>",
        "requested_symmetry": args.symmetry,
        "pyscf_point_group": getattr(mol, "groupname", "C1"),
        "nstates": args.nstates,
        "reference_mode": args.reference_mode,
        "minimum_somo_overlap": args.minimum_somo_overlap,
        "minimum_occupied_overlap": args.minimum_occupied_overlap,
        "reference_molden_input": (
            str(args.reference_molden.resolve())
            if args.reference_molden is not None
            else None
        ),
        "reference_source": reference_diagnostics.source,
        "reference_diagnostics": asdict(reference_diagnostics),
        "reference_warnings": list(reference_diagnostics.warnings),
        "reference_local_scf_run": reference_diagnostics.local_scf_run,
        "reference_local_scf_converged": (
            reference_diagnostics.local_scf_converged
        ),
        "reference_orthonormality_error": (
            reference_diagnostics.orthonormality_error
        ),
        "reference_orbital_gradient_max": (
            reference_diagnostics.orbital_gradient_max
        ),
        "reference_somo_seed_singular_values": list(
            reference_diagnostics.seed_somo_singular_values
        ),
        "scf_grid_level": args.scf_grid_level,
        "scf_grid_pruned": not args.unpruned_scf_grid,
        "observable_grid_level": args.observable_grid_level,
        "grid_pruned": not args.unpruned_grid,
        "observables_skipped": args.skip_observables,
        "coupling_audit": args.coupling_audit,
        "lr_cv_audit": args.lr_cv_audit,
        "lr_cv_audit_batch_size": args.lr_cv_audit_batch_size,
        "charge": 0,
        "reference_spin_2s": 2,
        "resource_estimate": asdict(resources),
        "write_molden": args.write_molden,
        "reference_molden_file": (
            str(molden_path) if molden_path is not None else None
        ),
        "write_nto": args.write_nto,
        "nto_definition": "MRSF state-to-state S0->Sn transition 1-RDM",
        "nto_weight_threshold": args.nto_weight_threshold,
        "nto_max_pairs": args.nto_max_pairs,
        "nto_directory": (
            str(nto_export.directory) if nto_export is not None else None
        ),
    }

    if descriptors is None:
        descriptor_payload = {
            "ct_q": np.empty(0),
            "ct_q_plus": np.empty(0),
            "ct_q_minus": np.empty(0),
            "ct_r_plus_bohr": np.empty((0, 3)),
            "ct_r_minus_bohr": np.empty((0, 3)),
            "ct_r_plus_angstrom": np.empty((0, 3)),
            "ct_r_minus_angstrom": np.empty((0, 3)),
            "ct_d_bohr": np.empty(0),
            "ct_d_angstrom": np.empty(0),
            "ct_mu_e_bohr": np.empty(0),
            "ct_mu_e_angstrom": np.empty(0),
            "ct_mu_debye": np.empty(0),
            "ct_integral_residual": np.empty(0),
            "ct_reference_electrons_grid": np.empty(0),
            "ct_state_electrons_grid": np.empty(0),
        }
    else:
        descriptor_payload = {
            "ct_q": np.asarray([value.q for value in descriptors]),
            "ct_q_plus": np.asarray([value.q_plus for value in descriptors]),
            "ct_q_minus": np.asarray([value.q_minus for value in descriptors]),
            "ct_r_plus_bohr": np.stack(
                [value.r_plus_bohr for value in descriptors]
            ),
            "ct_r_minus_bohr": np.stack(
                [value.r_minus_bohr for value in descriptors]
            ),
            "ct_r_plus_angstrom": np.stack(
                [value.r_plus_angstrom for value in descriptors]
            ),
            "ct_r_minus_angstrom": np.stack(
                [value.r_minus_angstrom for value in descriptors]
            ),
            "ct_d_bohr": np.asarray([value.d_ct_bohr for value in descriptors]),
            "ct_d_angstrom": np.asarray(
                [value.d_ct_angstrom for value in descriptors]
            ),
            "ct_mu_e_bohr": np.asarray(
                [value.mu_e_bohr for value in descriptors]
            ),
            "ct_mu_e_angstrom": np.asarray(
                [value.mu_e_angstrom for value in descriptors]
            ),
            "ct_mu_debye": np.asarray([value.mu_debye for value in descriptors]),
            "ct_integral_residual": np.asarray(
                [value.integral_delta_rho for value in descriptors]
            ),
            "ct_reference_electrons_grid": np.asarray(
                [value.reference_electrons_grid for value in descriptors]
            ),
            "ct_state_electrons_grid": np.asarray(
                [value.state_electrons_grid for value in descriptors]
            ),
        }

    if coupling_audit is None:
        audit_payload = {
            "coupling_audit_present": np.asarray(False),
            "audit_family_name": np.empty(0, dtype="U1"),
        }
    else:
        families = coupling_audit.families
        audit_payload = {
            "coupling_audit_present": np.asarray(True),
            "audit_exact_ground_shift_hartree": np.asarray(
                coupling_audit.exact_ground_shift_hartree
            ),
            "audit_gamma_cv_ground": np.asarray(coupling_audit.gamma_cv_ground),
            "audit_coupling_norm": np.asarray(coupling_audit.coupling_norm),
            "audit_fock_norm": np.asarray(coupling_audit.fock_norm),
            "audit_two_electron_norm": np.asarray(
                coupling_audit.two_electron_norm
            ),
            "audit_fock_two_electron_dot": np.asarray(
                coupling_audit.fock_two_electron_dot
            ),
            "audit_diagonal_second_order_shift_hartree": np.asarray(
                coupling_audit.diagonal_second_order_shift_hartree
            ),
            "audit_diagonal_fock_partition_hartree": np.asarray(
                coupling_audit.diagonal_fock_partition_hartree
            ),
            "audit_diagonal_two_electron_partition_hartree": np.asarray(
                coupling_audit.diagonal_two_electron_partition_hartree
            ),
            "audit_family_name": np.asarray([value.name for value in families]),
            "audit_family_mrsf_weight": np.asarray(
                [value.mrsf_weight for value in families]
            ),
            "audit_family_coupling_norm": np.asarray(
                [value.coupling_norm for value in families]
            ),
            "audit_family_fock_norm": np.asarray(
                [value.fock_norm for value in families]
            ),
            "audit_family_two_electron_norm": np.asarray(
                [value.two_electron_norm for value in families]
            ),
            "audit_family_diagonal_self_shift_hartree": np.asarray(
                [value.diagonal_self_shift_hartree for value in families]
            ),
            "audit_family_diagonal_partition_shift_hartree": np.asarray(
                [value.diagonal_partition_shift_hartree for value in families]
            ),
        }

    if lr_cv_audit is None:
        lr_cv_audit_payload = {
            "lr_cv_audit_present": np.asarray(False),
            "lr_cv_label_i": np.empty(0, dtype=int),
            "lr_cv_label_a": np.empty(0, dtype=int),
        }
    else:
        lr_cv_audit_payload = {
            "lr_cv_audit_present": np.asarray(True),
            "lr_cv_label_i": np.asarray(
                [i for i, _ in lr_cv_audit.labels], dtype=int
            ),
            "lr_cv_label_a": np.asarray(
                [a for _, a in lr_cv_audit.labels], dtype=int
            ),
            "lr_cv_raw_fock_hartree": lr_cv_audit.raw_fock_hartree,
            "lr_cv_s30_hartree": lr_cv_audit.s30_hartree,
            "lr_cv_coulomb_hartree": lr_cv_audit.coulomb_hartree,
            "lr_cv_exact_exchange_hartree": (
                lr_cv_audit.exact_exchange_hartree
            ),
            "lr_cv_xc_kernel_hartree": lr_cv_audit.xc_kernel_hartree,
            "lr_cv_diagonal_hartree": lr_cv_audit.lr_diagonal_hartree,
            "lr_cv_ground_offset_hartree": np.asarray(
                lr_cv_audit.ground_offset_hartree
            ),
            "lr_cv_emrsf_diagonal_hartree": (
                lr_cv_audit.emrsf_diagonal_hartree
            ),
            "lr_cv_ground_offset_recomputed_hartree": np.asarray(
                lr_cv_audit.ground_offset_recomputed_hartree
            ),
            "lr_cv_ground_offset_source_error_hartree": np.asarray(
                lr_cv_audit.ground_offset_source_error_hartree
            ),
            "lr_cv_direct_check_indices": lr_cv_audit.direct_check_indices,
            "lr_cv_direct_check_hartree": lr_cv_audit.direct_check_hartree,
            "lr_cv_direct_check_error_hartree": (
                lr_cv_audit.direct_check_error_hartree
            ),
        }

    archive_path = Path(f"{stem}.npz")
    np.savez_compressed(
        archive_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        dense_matrix_stored=np.asarray(result.matrix is not None),
        emrsf_eigenvalues=result.eigenvalues,
        emrsf_eigenvectors=result.eigenvectors,
        emrsf_converged=result.converged,
        emrsf_residual_norms=result.residual_norms,
        emrsf_iterations=np.asarray(result.iterations),
        emrsf_matvecs=np.asarray(result.matvecs),
        emrsf_initial_guesses=np.asarray(result.initial_guesses),
        emrsf_symmetry_seed_counts=np.asarray(
            result.symmetry_seed_counts, dtype=int
        ),
        exact_exchange_c_h=np.asarray(solver.lr.exact_exchange),
        coupling_scale_c_cp=np.asarray(solver.coupling_scale),
        ground_offset_hartree=np.asarray(solver.ground_offset),
        ground_offset_operator_hartree=np.asarray(
            solver.ground_offset_operator
        ),
        ground_offset_source_error_hartree=np.asarray(
            solver.ground_offset_source_error
        ),
        mrsf_eigenvalues=result.mrsf_result.eigenvalues,
        mrsf_eigenvectors=result.mrsf_result.eigenvectors,
        mrsf_converged=result.mrsf_result.converged,
        mrsf_residual_norms=result.mrsf_result.residual_norms,
        mrsf_iterations=np.asarray(result.mrsf_result.iterations),
        mrsf_matvecs=np.asarray(result.mrsf_result.matvecs),
        mrsf_initial_guesses=np.asarray(result.mrsf_result.initial_guesses),
        mrsf_symmetry_seed_counts=np.asarray(
            result.mrsf_result.symmetry_seed_counts, dtype=int
        ),
        emrsf_state_symmetry=np.asarray(
            [value.label for value in emrsf_symmetries]
        ),
        emrsf_state_term=np.asarray(emrsf_terms),
        emrsf_state_symmetry_purity=np.asarray(
            [value.purity for value in emrsf_symmetries]
        ),
        mrsf_state_symmetry=np.asarray(
            [value.label for value in mrsf_symmetries]
        ),
        mrsf_state_term=np.asarray(mrsf_terms),
        mrsf_state_symmetry_purity=np.asarray(
            [value.purity for value in mrsf_symmetries]
        ),
        live_mrsf_indices=result.live_mrsf_indices,
        gamma_cv=result.cv_weights,
        matched_mrsf_state=np.asarray(
            [value.mrsf_state if value.matched else -1 for value in matches],
            dtype=int,
        ),
        match_candidate_mrsf_state=np.asarray(
            [
                value.candidate_mrsf_state
                if value.candidate_mrsf_state is not None
                else -1
                for value in matches
            ],
            dtype=int,
        ),
        match_status=np.asarray([value.status for value in matches]),
        match_overlap_squared=np.asarray([value.overlap_squared for value in matches]),
        match_conditional_overlap_squared=np.asarray(
            [value.conditional_overlap_squared for value in matches]
        ),
        raw_energy_shift_hartree=np.asarray(
            [value.raw_energy_shift_hartree for value in matches]
        ),
        excitation_energy_shift_hartree=np.asarray(
            [value.excitation_energy_shift_hartree for value in matches]
        ),
        reference_density_mo=state_densities[0].reference_mo,
        reference_density_ao=state_densities[0].reference_ao,
        difference_density_mo=np.stack(
            [value.difference_mo for value in state_densities]
        ),
        difference_density_ao=np.stack(
            [value.difference_ao for value in state_densities]
        ),
        state_density_mo=np.stack([value.state_mo for value in state_densities]),
        state_density_ao=np.stack([value.state_ao for value in state_densities]),
        mo_coeff=result.mrsf_result.space.ref.coeff,
        mo_energy=result.mrsf_result.space.ref.energy,
        mo_occupation=result.mrsf_result.space.ref.occupation,
        atom_coordinates_bohr=mol.atom_coords(),
        atom_charges=mol.atom_charges(),
        **descriptor_payload,
        **audit_payload,
        **lr_cv_audit_payload,
    )
    announce(f"normal termination; log={log_path}; archive={archive_path}")


if __name__ == "__main__":
    main()
