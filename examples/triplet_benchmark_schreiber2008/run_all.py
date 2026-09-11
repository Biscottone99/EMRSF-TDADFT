#!/usr/bin/env python3
"""Run the five Schreiber/Thiel triplet benchmarks with the XYZ driver."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MOLECULES = (
    "ethene",
    "butadiene",
    "hexatriene",
    "cyclopropene",
    "cyclopentadiene",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="def2-tzvp")
    parser.add_argument("--xc", default="bhandhlyp")
    parser.add_argument("--nstates", type=int, default=8)
    parser.add_argument(
        "--integral-backend", choices=("direct", "df"), default="direct"
    )
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("results_v0.8.0"))
    parser.add_argument("--only", choices=MOLECULES, action="append")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=116000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    here = Path(__file__).resolve().parent
    driver = here.parent / "xyz_job" / "run_xyz_emrsf.py"
    selected = tuple(args.only) if args.only else MOLECULES
    for name in selected:
        xyz = here / "geometries" / f"{name}.xyz"
        output = args.output_root.resolve() / name
        command = [
            sys.executable,
            str(driver),
            str(xyz),
            "--method",
            "emrsf",
            "--states",
            "triplets",
            "--basis",
            args.basis,
            "--xc",
            args.xc,
            "--nstates",
            str(args.nstates),
            "--integral-backend",
            args.integral_backend,
            "--threads",
            str(args.threads),
            "--memory-mb",
            str(args.memory_mb),
            "--output-dir",
            str(output),
            "--save-vectors",
        ]
        if args.auxbasis:
            command.extend(("--auxbasis", args.auxbasis))
        print(" ".join(command), flush=True)
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
