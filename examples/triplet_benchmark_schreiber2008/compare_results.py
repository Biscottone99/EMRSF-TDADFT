#!/usr/bin/env python3
"""Compare vertical EMRSF triplet energies with CC3/TZVP references."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def arguments() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results_v0.8.0"))
    parser.add_argument(
        "--targets", type=Path, default=here / "triplet_targets_cc3_tzvp.csv"
    )
    parser.add_argument("--output", type=Path, default=Path("triplet_comparison.csv"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def locate_states(results: Path, molecule: str) -> Path:
    matches = sorted(
        results.glob(
            f"**/{molecule}_emrsf_triplet_primary_v*_states.csv"
        )
    )
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one triplet states CSV for {molecule}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = arguments()
    targets = read_csv(args.targets.resolve())
    rows = []
    for target in targets:
        molecule = target["molecule"]
        states_path = locate_states(args.results.resolve(), molecule)
        candidates = [
            row for row in read_csv(states_path)
            if row["term"].strip() == target["program_term"].strip()
        ]
        if len(candidates) != 1:
            raise SystemExit(
                f"{molecule}: term {target['program_term']} matched "
                f"{len(candidates)} roots in {states_path}"
            )
        state = candidates[0]
        value_text = state.get("common_s0_excitation_ev", "").strip()
        if not value_text:
            raise SystemExit(
                f"{states_path} has no common-S0 energy for {target['program_term']}"
            )
        calculated = float(value_text)
        reference = float(target["cc3_tzvp_ev"])
        error = calculated - reference
        rows.append(
            {
                **target,
                "calculated_emrsf_ev": f"{calculated:.8f}",
                "signed_error_ev": f"{error:.8f}",
                "absolute_error_ev": f"{abs(error):.8f}",
                "states_csv": str(states_path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mae = sum(float(row["absolute_error_ev"]) for row in rows) / len(rows)
    print(f"Matched states : {len(rows)}")
    print(f"MAE vs CC3/TZVP: {mae:.8f} eV")
    print(f"Comparison CSV : {args.output.resolve()}")


if __name__ == "__main__":
    main()
