"""Compare paired MRSF and EMRSF archives at exactly the same settings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIR_FIELDS = (
    "profile",
    "basis",
    "xc",
    "integral_backend",
    "grid_level",
    "grid_pruned",
    "reference_mode",
    "nstates",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair MRSF/EMRSF NPZ files and compare the same spectroscopic term; "
            "the program rejects pairs with different theory/numerical settings"
        )
    )
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=Path(__file__).with_name("paper_targets.csv"),
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("mrsf_emrsf_equal_level.csv")
    )
    parser.add_argument("--reference-energy-tol", type=float, default=1.0e-10)
    parser.add_argument("--density-tol", type=float, default=1.0e-8)
    return parser.parse_args()


def read_targets(path: Path) -> dict[str, dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["molecule"]: {
                "paper_term": row["target_state"],
                "log_term": row["log_target_state"],
                "paper_ev": float(row["emrsf_ev"]),
                "tbe_ev": float(row["tbe_ev"]),
            }
            for row in csv.DictReader(handle)
        }


def read_archive(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        return {
            "path": path,
            "metadata": metadata,
            "terms": np.asarray(data["response_state_term"]).astype(str),
            "energies": np.asarray(data["response_excitation_energies_ev"]),
            "residual": float(np.max(data["response_residual_norms"])),
            "coeff": np.asarray(data["mo_coeff"]),
            "occupation": np.asarray(data["mo_occupation"]),
        }


def target_state(record: dict[str, object], term: str) -> tuple[int, float]:
    matches = np.flatnonzero(record["terms"] == term)
    if len(matches) != 1:
        raise ValueError(
            f"{record['path']}: expected exactly one TERM={term}; "
            f"found {len(matches)}; available={list(record['terms'])}"
        )
    root = int(matches[0])
    return root, float(record["energies"][root])


def ensure_equal_level(
    mrsf: dict[str, object],
    emrsf: dict[str, object],
    *,
    energy_tol: float,
    density_tol: float,
) -> tuple[float, float, bool]:
    left = mrsf["metadata"]
    right = emrsf["metadata"]
    for field in PAIR_FIELDS:
        if left[field] != right[field]:
            raise ValueError(
                f"{left['case']}: unequal {field}: {left[field]!r} != "
                f"{right[field]!r}"
            )
    energy_difference = abs(
        float(left["reference_energy_hartree"])
        - float(right["reference_energy_hartree"])
    )
    if energy_difference > energy_tol:
        raise ValueError(
            f"{left['case']}: triplet reference energies differ by "
            f"{energy_difference:.3e} Eh"
        )

    density_mrsf = (
        mrsf["coeff"]
        @ np.diag(mrsf["occupation"])
        @ mrsf["coeff"].T
    )
    density_emrsf = (
        emrsf["coeff"]
        @ np.diag(emrsf["occupation"])
        @ emrsf["coeff"].T
    )
    density_difference = float(np.max(np.abs(density_mrsf - density_emrsf)))
    if density_difference > density_tol:
        raise ValueError(
            f"{left['case']}: reference AO densities differ by "
            f"{density_difference:.3e}"
        )
    same_fingerprint = (
        left["reference_fingerprint"] == right["reference_fingerprint"]
    )
    return energy_difference, density_difference, same_fingerprint


def main() -> None:
    args = arguments()
    targets = read_targets(args.targets_csv)
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for path in args.archives:
        record = read_archive(path)
        metadata = record["metadata"]
        key = (str(metadata["case"]), str(metadata["profile"]))
        method = str(metadata["method"])
        if method in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate {method} archive for {key}")
        grouped[key][method] = record

    rows = []
    for (case, profile), pair in sorted(grouped.items()):
        if set(pair) != {"mrsf", "emrsf"}:
            raise ValueError(
                f"{case}/{profile}: need one mrsf and one emrsf archive; "
                f"found {sorted(pair)}"
            )
        if case not in targets:
            raise ValueError(f"no literature target for {case}")
        mrsf, emrsf = pair["mrsf"], pair["emrsf"]
        energy_ref, density_ref, same_hash = ensure_equal_level(
            mrsf,
            emrsf,
            energy_tol=args.reference_energy_tol,
            density_tol=args.density_tol,
        )
        target = targets[case]
        mrsf_root, mrsf_energy = target_state(mrsf, str(target["log_term"]))
        emrsf_root, emrsf_energy = target_state(emrsf, str(target["log_term"]))
        metadata = emrsf["metadata"]
        rows.append(
            {
                "molecule": case,
                "profile": profile,
                "basis": metadata["basis"],
                "xc": metadata["xc"],
                "grid_level": metadata["grid_level"],
                "integral_backend": metadata["integral_backend"],
                "term": target["log_term"],
                "mrsf_root": mrsf_root,
                "mrsf_ev": mrsf_energy,
                "emrsf_root": emrsf_root,
                "emrsf_ev": emrsf_energy,
                "emrsf_minus_mrsf_ev": emrsf_energy - mrsf_energy,
                "paper_emrsf_ev": target["paper_ev"],
                "emrsf_minus_paper_ev": emrsf_energy
                - float(target["paper_ev"]),
                "tbe_ev": target["tbe_ev"],
                "emrsf_minus_tbe_ev": emrsf_energy - float(target["tbe_ev"]),
                "mrsf_max_residual": mrsf["residual"],
                "emrsf_max_residual": emrsf["residual"],
                "reference_energy_difference_eh": energy_ref,
                "reference_density_max_difference": density_ref,
                "identical_reference_fingerprint": same_hash,
                "mrsf_archive": str(mrsf["path"]),
                "emrsf_archive": str(emrsf["path"]),
            }
        )

    if not rows:
        raise ValueError("no complete pairs supplied")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "molecule,profile,term,MRSF/eV,EMRSF/eV,EMRSF-MRSF/eV,"
        "paper/eV,EMRSF-paper/eV,TBE/eV,EMRSF-TBE/eV"
    )
    for row in rows:
        print(
            f"{row['molecule']},{row['profile']},{row['term']},"
            f"{row['mrsf_ev']:.8f},{row['emrsf_ev']:.8f},"
            f"{row['emrsf_minus_mrsf_ev']:+.8f},"
            f"{row['paper_emrsf_ev']:.8f},"
            f"{row['emrsf_minus_paper_ev']:+.8f},{row['tbe_ev']:.8f},"
            f"{row['emrsf_minus_tbe_ev']:+.8f}"
        )
    print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()
