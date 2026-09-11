"""Unified high-accuracy MRSF/EMRSF driver.

At the default ``exact-dz`` profile, changing ``--method`` is the only theory
change: both calculations use the same BH&HLYP/cc-pVDZ triplet reference,
unpruned integration grid, direct integrals, state-locking and solver
tolerances. No paper or TBE energy enters either Hamiltonian.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf import gto, lib

from pyscf_emrsf import (
    EMRSFTDA,
    HighAccuracyMRSFTDA,
    __version__,
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


@dataclass(frozen=True)
class AccuracyProfile:
    basis: str
    grid_level: int
    nstates: int
    max_space: int
    description: str


PROFILES = {
    "exact-dz": AccuracyProfile(
        basis="cc-pvdz",
        grid_level=6,
        nstates=16,
        max_space=180,
        description="equal-theory BH&HLYP/cc-pVDZ numerical-limit comparison",
    ),
    "tbe-dz": AccuracyProfile(
        basis="aug-cc-pvdz",
        grid_level=7,
        nstates=20,
        max_space=220,
        description="optional diffuse double-zeta CT basis-convergence step",
    ),
    "tbe-tz": AccuracyProfile(
        basis="aug-cc-pvtz",
        grid_level=7,
        nstates=24,
        max_space=260,
        description="optional diffuse triple-zeta basis-convergence step",
    ),
}


def read_bohr_xyz(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    natom = int(lines[0])
    atoms = [line.strip() for line in lines[2:] if line.strip()]
    if len(atoms) != natom:
        raise ValueError(f"{path} declares {natom} atoms but contains {len(atoms)}")
    return "\n".join(atoms)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run MRSF or direct enlarged-space EMRSF with identical numerical "
            "settings, tight unpruned grids and no energy fitting"
        )
    )
    result.add_argument("case", choices=sorted(CASES))
    result.add_argument(
        "--method",
        choices=("mrsf", "emrsf"),
        default="emrsf",
        help="response space to diagonalise (default: emrsf)",
    )
    result.add_argument("--profile", choices=tuple(PROFILES), default="exact-dz")
    result.add_argument("--basis", default=None, help="override profile basis")
    result.add_argument("--xc", default="bhandhlyp")
    result.add_argument(
        "--integral-backend", choices=("direct", "df"), default="direct"
    )
    result.add_argument(
        "--auxbasis",
        default=None,
        help="optional auxiliary basis used only with --integral-backend df",
    )
    result.add_argument("--symmetry", default="auto")
    result.add_argument(
        "--reference-mode", choices=("mom", "auto", "external"), default="mom"
    )
    result.add_argument("--reference-molden", type=Path, default=None)
    result.add_argument("--minimum-somo-overlap", type=float, default=0.80)
    result.add_argument("--minimum-occupied-overlap", type=float, default=0.80)
    result.add_argument("--reference-conv-tol", type=float, default=1.0e-12)
    result.add_argument("--reference-conv-tol-grad", type=float, default=1.0e-8)
    result.add_argument("--direct-scf-tol", type=float, default=1.0e-14)
    result.add_argument("--reference-max-cycle", type=int, default=300)
    result.add_argument("--small-rho-cutoff", type=float, default=1.0e-14)
    result.add_argument("--scf-grid-level", type=int, default=None)
    result.add_argument(
        "--pruned-grid",
        action="store_true",
        help="enable PySCF grid pruning (profiles are unpruned by default)",
    )
    result.add_argument("--nstates", type=int, default=None)
    result.add_argument("--solver", choices=("davidson", "dense"), default="davidson")
    result.add_argument("--conv-tol", type=float, default=1.0e-11)
    result.add_argument("--residual-tol", type=float, default=1.0e-9)
    result.add_argument("--max-cycle", type=int, default=240)
    result.add_argument("--max-space", type=int, default=None)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--memory-mb", type=int, default=120000)
    result.add_argument("--pyscf-verbose", type=int, default=4)
    result.add_argument("--output-dir", type=Path, default=Path("results_v0.5.1"))
    result.add_argument("--write-molden", action="store_true")
    result.add_argument(
        "--lr-cv-audit",
        action="store_true",
        help="EMRSF only: expensive full S30/LR-CV diagonal decomposition",
    )
    result.add_argument("--lr-cv-audit-batch-size", type=int, default=16)
    return result


def resolve_symmetry(value: str):
    request = value.strip().lower()
    if request in {"none", "false", "off", "c1"}:
        return False
    if request in {"auto", "true", "on"}:
        return True
    return value


def reference_fingerprint(coeff, energy, occupation) -> str:
    """Hash the numerical reference so separate MRSF/EMRSF runs can be paired."""

    digest = hashlib.sha256()
    for value in (coeff, energy, occupation):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parser().parse_args()
    profile = PROFILES[args.profile]
    basis = args.basis or profile.basis
    grid_level = (
        profile.grid_level if args.scf_grid_level is None else args.scf_grid_level
    )
    nstates = profile.nstates if args.nstates is None else args.nstates
    max_space = profile.max_space if args.max_space is None else args.max_space
    density_fit = args.integral_backend == "df"
    if not density_fit and args.auxbasis is not None:
        raise SystemExit("--auxbasis is incompatible with the direct backend")
    if args.method == "mrsf" and args.lr_cv_audit:
        raise SystemExit("--lr-cv-audit is available only with --method emrsf")

    start = perf_counter()

    def announce(message: str) -> None:
        elapsed = (perf_counter() - start) / 60.0
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"[elapsed {elapsed:.1f} min] {message}",
            flush=True,
        )

    here = Path(__file__).resolve().parent
    geometry = here / "geometries" / CASES[args.case]
    reference = args.reference_molden
    if reference is None and args.reference_mode in {"mom", "external"}:
        reference = (
            here
            / "reference_orbitals"
            / f"{args.case}_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden"
        )
    if reference is not None and not reference.is_file():
        raise SystemExit(f"reference Molden not found: {reference}")
    if args.reference_mode == "external" and basis.lower() != "cc-pvdz":
        raise SystemExit(
            "external orbitals are fixed in cc-pVDZ; use MOM projection for a "
            "different basis"
        )

    lib.num_threads(args.threads)
    announce(
        f"method={args.method.upper()}; profile={args.profile} "
        f"({profile.description}); basis={basis}; backend={args.integral_backend}; "
        "no TBE/paper value enters the Hamiltonian"
    )
    molecule = gto.M(
        atom=read_bohr_xyz(geometry),
        unit="Bohr",
        basis=basis,
        charge=0,
        spin=2,
        symmetry=resolve_symmetry(args.symmetry),
        verbose=args.pyscf_verbose,
        max_memory=args.memory_mb,
    )

    common = dict(
        xc=args.xc,
        density_fit=density_fit,
        auxbasis=args.auxbasis,
        verbose=args.pyscf_verbose,
        grid_level=grid_level,
        grid_prune=args.pruned_grid,
        small_rho_cutoff=args.small_rho_cutoff,
    )
    if args.reference_mode == "external":
        announce("loading the fixed external triplet reference")
        mf = build_reference_from_molden(reference, mol=molecule, **common)
    else:
        announce(
            "optimising a tightly converged state-locked MOM triplet"
            if args.reference_mode == "mom"
            else "optimising a tightly converged Aufbau triplet"
        )
        mf = build_reference(
            molecule,
            conv_tol=args.reference_conv_tol,
            conv_tol_grad=args.reference_conv_tol_grad,
            direct_scf_tol=args.direct_scf_tol,
            max_cycle=args.reference_max_cycle,
            molden_seed=reference if args.reference_mode == "mom" else None,
            allow_basis_projection=(
                args.reference_mode == "mom" and basis.lower() != "cc-pvdz"
            ),
            minimum_somo_overlap=args.minimum_somo_overlap,
            minimum_occupied_overlap=args.minimum_occupied_overlap,
            **common,
        )
    announce(f"triplet reference ready: E={mf.e_tot:.14f} Eh")

    lr_cv_audit = None
    maximum_closure = None
    if args.method == "mrsf":
        announce("building the unchanged MRSF operator with tight numerics")
        method = HighAccuracyMRSFTDA(
            mf,
            target_multiplicity=1,
            nstates=nstates,
            density_fit=density_fit,
            auxbasis=args.auxbasis,
            solver=args.solver,
            conv_tol=args.conv_tol,
            residual_tol=args.residual_tol,
            max_cycle=args.max_cycle,
            max_space=max_space,
            progress=True,
        )
        response_dimension = method.space.size
        live_dimension = int(
            np.count_nonzero(np.isfinite(method.space.diagonal_guess()))
        )
        resources = {
            "response_dimension": response_dimension,
            "live_response_dimension": live_dimension,
            "dense_matrix_gib": 8.0 * live_dimension**2 / 1024.0**3,
        }
        integral_backend = "density-fitting" if density_fit else "direct-AO-JK"
        ref = method.ref
        result = method.kernel()
    else:
        announce("building the direct enlarged-space EMRSF operator")
        method = EMRSFTDA(
            mf,
            nstates=nstates,
            apply_fock_correction=True,
            density_fit=density_fit,
            auxbasis=args.auxbasis,
            solver=args.solver,
            conv_tol=args.conv_tol,
            residual_tol=args.residual_tol,
            max_cycle=args.max_cycle,
            max_space=max_space,
            progress=True,
            coupling_scale=None,
            solve_mrsf_first=False,
        )
        resource_estimate = method.resource_estimate()
        resources = asdict(resource_estimate)
        response_dimension = method.dimension
        live_dimension = len(method.live) + method.lr.space.size
        integral_backend = method.integral_backend
        ref = method.mrsf.ref
        announce(
            f"operator dimension={method.dimension}; c_H=c_cp="
            f"{method.coupling_scale:.8f}; resources={resource_estimate}"
        )
        result = method.kernel()
        maximum_closure = max(
            abs(value.closure_error_hartree)
            for value in result.state_decompositions
        )
        if maximum_closure > 5.0e-9:
            raise RuntimeError(
                "EMRSF block-decomposition closure failed: "
                f"{maximum_closure:.3e} Eh"
            )
        if args.lr_cv_audit:
            announce("running the full LR-CV/S30 diagonal audit")
            lr_cv_audit = analyse_lr_cv_diagonal(
                method, batch_size=args.lr_cv_audit_batch_size
            )
            announce(
                "LR-CV audit complete: max direct error="
                f"{lr_cv_audit.maximum_direct_check_error_hartree:.3e} Eh"
            )

    if not np.all(result.converged):
        raise RuntimeError(
            f"not all {args.method.upper()} roots converged: "
            f"residuals={np.asarray(result.residual_norms)}"
        )
    maximum_residual = float(np.max(result.residual_norms))
    if maximum_residual > 1.1 * args.residual_tol:
        raise RuntimeError(
            f"maximum residual {maximum_residual:.3e} exceeds requested tolerance"
        )
    closure_message = (
        ""
        if maximum_closure is None
        else f"; max block closure={maximum_closure:.3e} Eh"
    )
    announce(
        f"{args.method.upper()} converged: max residual={maximum_residual:.3e}"
        f"{closure_message}"
    )

    mf.emrsf_run_metadata = {
        "method": args.method.upper(),
        "integral_backend": integral_backend,
        "eigenvalue_tolerance": args.conv_tol,
        "residual_tolerance": args.residual_tol,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"{args.case}_{args.method}_v{__version__}_{args.profile}"
    if args.write_molden:
        result.write_reference_molden(Path(f"{stem}_orbitals.molden"))
    title = (
        f"high-accuracy {args.method.upper()}: {args.case}; "
        f"profile={args.profile}; not fitted to TBE"
    )
    if args.method == "emrsf":
        log_path = result.write_log(
            Path(f"{stem}.log"),
            lr_cv_audit=lr_cv_audit,
            title=title,
            top_configurations=12,
        )
    else:
        log_path = result.write_log(
            Path(f"{stem}.log"), title=title, top_configurations=12
        )

    symmetries = result.state_symmetries()
    terms = term_labels(symmetries)
    fingerprint = reference_fingerprint(ref.coeff, ref.energy, ref.occupation)
    metadata = {
        "program": "pyscf-emrsf",
        "version": __version__,
        "method": args.method,
        "case": args.case,
        "profile": args.profile,
        "profile_description": profile.description,
        "basis": basis,
        "xc": args.xc,
        "integral_backend": args.integral_backend,
        "integral_backend_detail": integral_backend,
        "auxbasis": args.auxbasis,
        "density_fitting": density_fit,
        "direct_emrsf": args.method == "emrsf",
        "mrsf_pre_diagonalisation": False,
        "tbe_used_in_hamiltonian": False,
        "paper_energy_used_in_hamiltonian": False,
        "grid_level": grid_level,
        "grid_pruned": args.pruned_grid,
        "reference_mode": args.reference_mode,
        "reference_molden": str(reference.resolve()) if reference else None,
        "reference_energy_hartree": float(mf.e_tot),
        "reference_fingerprint": fingerprint,
        "reference_diagnostics": asdict(ref.diagnostics),
        "nstates": nstates,
        "response_dimension": response_dimension,
        "live_response_dimension": live_dimension,
        "solver": result.solver,
        "eigenvalue_tolerance": args.conv_tol,
        "residual_tolerance": args.residual_tol,
        "maximum_residual": maximum_residual,
        "maximum_block_closure_hartree": maximum_closure,
        "resource_estimate": resources,
    }
    archive_path = Path(f"{stem}.npz")
    common_archive = dict(
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        response_eigenvalues=result.eigenvalues,
        response_eigenvectors=result.eigenvectors,
        response_excitation_energies_ev=result.excitation_energies_ev,
        response_converged=result.converged,
        response_residual_norms=result.residual_norms,
        response_state_symmetry=np.asarray([value.label for value in symmetries]),
        response_state_term=np.asarray(terms),
        response_state_symmetry_purity=np.asarray(
            [value.purity for value in symmetries]
        ),
        mo_coeff=ref.coeff,
        mo_energy=ref.energy,
        mo_occupation=ref.occupation,
        atom_coordinates_bohr=molecule.atom_coords(),
        atom_charges=molecule.atom_charges(),
    )
    if args.method == "mrsf":
        np.savez_compressed(
            archive_path,
            **common_archive,
            mrsf_eigenvalues=result.eigenvalues,
            mrsf_eigenvectors=result.eigenvectors,
            mrsf_excitation_energies_ev=result.excitation_energies_ev,
            mrsf_converged=result.converged,
            mrsf_residual_norms=result.residual_norms,
            mrsf_state_symmetry=np.asarray([value.label for value in symmetries]),
            mrsf_state_term=np.asarray(terms),
            mrsf_state_symmetry_purity=np.asarray(
                [value.purity for value in symmetries]
            ),
        )
    else:
        decompositions = result.state_decompositions
        np.savez_compressed(
            archive_path,
            **common_archive,
            emrsf_eigenvalues=result.eigenvalues,
            emrsf_eigenvectors=result.eigenvectors,
            emrsf_excitation_energies_ev=result.excitation_energies_ev,
            emrsf_converged=result.converged,
            emrsf_residual_norms=result.residual_norms,
            emrsf_state_symmetry=np.asarray([value.label for value in symmetries]),
            emrsf_state_term=np.asarray(terms),
            emrsf_state_symmetry_purity=np.asarray(
                [value.purity for value in symmetries]
            ),
            gamma_cv=result.cv_weights,
            live_mrsf_indices=result.live_mrsf_indices,
            decomposition_mrsf_weight=np.asarray(
                [value.mrsf_weight for value in decompositions]
            ),
            decomposition_cv_weight=np.asarray(
                [value.cv_weight for value in decompositions]
            ),
            decomposition_mrsf_block_hartree=np.asarray(
                [value.mrsf_block_hartree for value in decompositions]
            ),
            decomposition_cv_block_hartree=np.asarray(
                [value.cv_block_hartree for value in decompositions]
            ),
            decomposition_coupling_hartree=np.asarray(
                [value.coupling_hartree for value in decompositions]
            ),
            decomposition_closure_error_hartree=np.asarray(
                [value.closure_error_hartree for value in decompositions]
            ),
        )
    announce(f"normal termination; log={log_path}; archive={archive_path}")


if __name__ == "__main__":
    main()
