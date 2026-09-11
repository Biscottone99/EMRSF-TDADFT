"""Compare v0.5.1 basis profiles with paper values and independent TBEs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=Path(__file__).with_name("paper_targets.csv"),
    )
    parser.add_argument("--csv", type=Path, default=Path("emrsf_v050_comparison.csv"))
    return parser.parse_args()


def targets(path: Path) -> dict[str, dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["molecule"]: {
                "paper_term": row["target_state"],
                "log_term": row["log_target_state"],
                "paper_ev": float(row["emrsf_ev"]),
                "tbe_ev": float(row["tbe_ev"]),
                "paper_gamma": float(row["gamma_cv"]),
            }
            for row in csv.DictReader(handle)
        }


def read_archive(path: Path, target: dict[str, object]) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("method") != "emrsf":
            raise ValueError(f"{path}: literature comparison requires EMRSF output")
        terms = np.asarray(data["emrsf_state_term"]).astype(str)
        matches = np.flatnonzero(terms == target["log_term"])
        if len(matches) != 1:
            raise ValueError(
                f"{path}: expected exactly one TERM={target['log_term']}, "
                f"found {len(matches)}; available={list(terms)}"
            )
        root = int(matches[0])
        energy = float(data["emrsf_excitation_energies_ev"][root])
        gamma = float(data["gamma_cv"][root])
        residual = float(np.max(data["emrsf_residual_norms"]))
        closure = float(
            np.max(np.abs(data["decomposition_closure_error_hartree"]))
        )
    return {
        "molecule": metadata["case"],
        "profile": metadata["profile"],
        "basis": metadata["basis"],
        "backend": metadata["integral_backend"],
        "paper_term": target["paper_term"],
        "log_term": target["log_term"],
        "root": root,
        "energy_ev": energy,
        "paper_ev": target["paper_ev"],
        "delta_paper_ev": energy - float(target["paper_ev"]),
        "tbe_ev": target["tbe_ev"],
        "delta_tbe_ev": energy - float(target["tbe_ev"]),
        "gamma_cv": gamma,
        "paper_gamma": target["paper_gamma"],
        "delta_gamma": gamma - float(target["paper_gamma"]),
        "max_residual": residual,
        "max_closure_eh": closure,
        "archive": str(path),
    }


def main() -> None:
    args = arguments()
    target_table = targets(args.targets_csv)
    rows = []
    for path in args.archives:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
        molecule = metadata["case"]
        if molecule not in target_table:
            raise ValueError(f"no target for {molecule}")
        rows.append(read_archive(path, target_table[molecule]))
    rows.sort(key=lambda row: (str(row["profile"]), str(row["molecule"])))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "profile,molecule,basis,term,EMRSF/eV,paper/eV,dPaper/eV,"
        "TBE/eV,dTBE/eV,gamma_CV"
    )
    for row in rows:
        print(
            f"{row['profile']},{row['molecule']},{row['basis']},"
            f"{row['log_term']},{row['energy_ev']:.8f},{row['paper_ev']:.8f},"
            f"{row['delta_paper_ev']:+.8f},{row['tbe_ev']:.8f},"
            f"{row['delta_tbe_ev']:+.8f},{row['gamma_cv']:.8f}"
        )

    print("\nMAE by complete profile (no value is used to alter the calculation)")
    for profile in sorted({str(row["profile"]) for row in rows}):
        selected = [row for row in rows if row["profile"] == profile]
        if len(selected) != len(target_table):
            print(f"{profile}: incomplete ({len(selected)}/{len(target_table)})")
            continue
        mae_paper = np.mean([abs(float(row["delta_paper_ev"])) for row in selected])
        mae_tbe = np.mean([abs(float(row["delta_tbe_ev"])) for row in selected])
        print(
            f"{profile}: MAE vs paper={mae_paper:.8f} eV; "
            f"MAE vs TBE={mae_tbe:.8f} eV"
        )
    print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()
