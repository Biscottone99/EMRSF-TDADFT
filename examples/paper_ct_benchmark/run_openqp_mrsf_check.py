#!/usr/bin/env python3
"""Fixed-reference MRSF regression against the supplied OpenQP calculations.

This script isolates the conventional MRSF operator from both triplet-SCF root
selection and the new EMRSF block.  It imports the same OpenQP restricted
triplet orbitals, evaluates the PySCF MRSF-TDA operator, and compares all roots
against excitation energies recorded from OpenQP 1.2.1 calculations.  The two
problematic references are bundled; Molden files generated for the other
three molecules can be supplied with ``--reference-molden``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib

from pyscf_emrsf import MRSFTDA, __version__
from pyscf_emrsf.reference import build_reference, build_reference_from_molden


HERE = Path(__file__).resolve().parent
CASES = {
    "azulene": {
        "geometry": "azulene.bohr.xyz",
        "molden": "azulene_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden",
        "openqp_raw_ev": (
            -1.927549,
            0.467332,
            1.540132,
            2.800860,
            3.691401,
            3.906635,
            4.108387,
            4.310634,
            4.443027,
            4.688409,
            4.930269,
            4.991611,
        ),
    },
    "n_phenylpyrrole": {
        "geometry": "n_phenylpyrrole.bohr.xyz",
        "molden": "n_phenylpyrrole_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden",
        "openqp_raw_ev": (
            -3.961491,
            0.874560,
            1.506128,
            1.737755,
            2.721227,
            3.255483,
            3.266358,
            3.824920,
            4.126622,
            4.270003,
            4.413250,
            4.554365,
        ),
    },
    "phthalazine": {
        "geometry": "phthalazine.bohr.xyz",
        "molden": "phthalazine_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden",
        "openqp_raw_ev": (
            -3.527526,
            1.120331,
            1.314813,
            1.895770,
            2.832258,
            2.898498,
            2.924000,
            3.547935,
            4.071627,
            4.176304,
            4.351722,
            4.537356,
        ),
    },
    "quinoxaline": {
        "geometry": "quinoxaline.bohr.xyz",
        "molden": "quinoxaline_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden",
        "openqp_raw_ev": (
            -3.664498,
            0.566123,
            1.156249,
            1.514701,
            1.723488,
            2.112054,
            3.243935,
            3.973705,
            4.072145,
            4.305367,
            4.538005,
            4.574074,
        ),
    },
    "twisted_dmabn": {
        "geometry": "twisted_dmabn.bohr.xyz",
        "molden": "twisted_dmabn_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden",
        "openqp_raw_ev": (
            -3.886097,
            0.413335,
            0.966551,
            2.280073,
            2.806020,
            3.173521,
            3.686907,
            3.787848,
            4.172405,
            4.299685,
            4.350818,
            4.832627,
        ),
    },
}


def read_bohr_xyz(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    atoms = [line.strip() for line in lines[2:] if line.strip()]
    if len(atoms) != atom_count:
        raise ValueError(f"invalid XYZ atom count in {path}")
    return "\n".join(atoms)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument(
        "--reference-mode",
        choices=("external", "mom"),
        default="external",
        help="external isolates the response operator; mom also tests PySCF SCF",
    )
    parser.add_argument("--reference-molden", type=Path, default=None)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--auxbasis", default="cc-pvdz-jkfit")
    parser.add_argument("--xc", default="bhandhlyp")
    parser.add_argument("--symmetry", default="none")
    parser.add_argument("--solver", choices=("davidson", "dense"), default="davidson")
    parser.add_argument("--nstates", type=int, default=12)
    parser.add_argument("--conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--max-space", type=int, default=None)
    parser.add_argument("--scf-grid-level", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=8000)
    parser.add_argument("--pyscf-verbose", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("reference_checks"))
    parser.add_argument(
        "--tolerance-ev",
        type=float,
        default=3.0e-3,
        help="allowed fixed-reference PySCF/OpenQP root error",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    specification = CASES[args.case]
    molden_path = args.reference_molden
    if molden_path is None:
        molden_path = HERE / "reference_orbitals" / specification["molden"]
    if not molden_path.is_file():
        raise SystemExit(
            f"reference Molden file not found: {molden_path}. Run the OpenQP "
            "cross-check first or pass --reference-molden explicitly."
        )
    symmetry = (
        False
        if args.symmetry.lower() in {"none", "c1", "off", "false"}
        else (True if args.symmetry.lower() == "auto" else args.symmetry)
    )
    mol = gto.M(
        atom=read_bohr_xyz(HERE / "geometries" / specification["geometry"]),
        unit="Bohr",
        basis=args.basis,
        charge=0,
        spin=2,
        symmetry=symmetry,
        verbose=args.pyscf_verbose,
        max_memory=args.memory_mb,
    )
    lib.num_threads(args.threads)
    common = dict(
        mol=mol,
        xc=args.xc,
        density_fit=True,
        auxbasis=args.auxbasis,
        verbose=args.pyscf_verbose,
        grid_level=args.scf_grid_level,
    )
    if args.reference_mode == "external":
        mf = build_reference_from_molden(molden_path, **common)
    else:
        common.pop("mol")
        mf = build_reference(
            mol,
            molden_seed=molden_path,
            conv_tol=1.0e-10,
            max_cycle=200,
            minimum_somo_overlap=0.80,
            minimum_occupied_overlap=0.80,
            **common,
        )

    solver = MRSFTDA(
        mf,
        target_multiplicity=1,
        nstates=args.nstates,
        density_fit=True,
        auxbasis=args.auxbasis,
        solver=args.solver,
        conv_tol=args.conv_tol,
        max_cycle=args.max_cycle,
        max_space=args.max_space,
        progress=True,
    )
    result = solver.kernel()
    if not np.all(result.converged):
        raise RuntimeError(f"unconverged MRSF roots: {result.residual_norms}")

    openqp_raw = np.asarray(specification["openqp_raw_ev"], dtype=float)
    openqp_excitation = openqp_raw - openqp_raw[0]
    count = min(len(result.excitation_energies_ev), len(openqp_excitation))
    error = result.excitation_energies_ev[:count] - openqp_excitation[:count]
    rows = [
        {
            "root": root,
            "pyscf_emrsf_ev": float(result.excitation_energies_ev[root]),
            "openqp_v1_2_1_ev": float(openqp_excitation[root]),
            "difference_ev": float(error[root]),
            "residual": float(result.residual_norms[root]),
        }
        for root in range(count)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / (
        f"{args.case}_{args.reference_mode}_mrsf_v{__version__}_openqp_check"
    )
    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")
    log_path = result.write_log(stem.with_suffix(".log"), title="OpenQP fixed-reference MRSF check")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "program_version": __version__,
        "case": args.case,
        "reference_mode": args.reference_mode,
        "reference_molden": str(molden_path.resolve()),
        "reference_energy_hartree": float(mf.e_tot),
        "reference_source": solver.ref.diagnostics.source,
        "reference_warnings": list(solver.ref.diagnostics.warnings),
        "maximum_absolute_error_ev": float(np.max(np.abs(error))),
        "rms_error_ev": float(np.sqrt(np.mean(error * error))),
        "tolerance_ev": args.tolerance_ev,
        "passed": bool(np.max(np.abs(error)) <= args.tolerance_ev),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("root  PySCF/eV    OpenQP/eV   difference/eV   residual")
    for row in rows:
        print(
            f"{row['root']:4d}  {row['pyscf_emrsf_ev']:10.6f}  "
            f"{row['openqp_v1_2_1_ev']:10.6f}  "
            f"{row['difference_ev']:+13.6f}  {row['residual']:.3e}"
        )
    print(f"maximum |difference| = {payload['maximum_absolute_error_ev']:.6f} eV")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"LOG:  {log_path}")
    if not payload["passed"]:
        print("CHECK FAILED")
        return 1
    print("CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
