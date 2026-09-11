#!/usr/bin/env python3
"""Generate reproducible OpenQP 1.2.1 MRSF inputs from the SI geometries."""

from __future__ import annotations

import argparse
from pathlib import Path


BOHR_TO_ANGSTROM = 0.529177210903
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8}
HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parent
CASES = (
    "azulene",
    "n_phenylpyrrole",
    "phthalazine",
    "quinoxaline",
    "twisted_dmabn",
)


def read_geometry(case: str) -> list[tuple[int, float, float, float]]:
    path = BENCHMARK_DIR / "geometries" / f"{case}.bohr.xyz"
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    atoms = []
    for line in lines[2:]:
        if not line.strip():
            continue
        symbol, x, y, z = line.split()
        try:
            atomic_number = ATOMIC_NUMBERS[symbol.capitalize()]
        except KeyError as error:
            raise ValueError(f"unsupported element {symbol!r} in {path}") from error
        atoms.append(
            (
                atomic_number,
                float(x) * BOHR_TO_ANGSTROM,
                float(y) * BOHR_TO_ANGSTROM,
                float(z) * BOHR_TO_ANGSTROM,
            )
        )
    if len(atoms) != atom_count:
        raise ValueError(f"{path}: expected {atom_count} atoms, found {len(atoms)}")
    return atoms


def render(case: str, *, nstates: int, save_molden: bool) -> str:
    geometry = "\n".join(
        f"  {z:2d}  {x: .12f}  {y: .12f}  {zc: .12f}"
        for z, x, y, zc in read_geometry(case)
    )
    molden = "True" if save_molden else "False"
    return f"""# OpenQP 1.2.1 conventional MRSF-TDDFT oracle for {case}.
# Geometry is converted only here from the bundled SI_loos coordinates in bohr.

[input]
system=
{geometry}
charge=0
runtype=energy
basis=cc-pvdz
functional=bhhlyp
method=tdhf
d4=False
perf=1

[guess]
type=huckel
save_mol=False

[scf]
type=rohf
multiplicity=3
maxit=200
forced_attempt=2
conv=1.0e-10
save_molden={molden}

[tdhf]
type=mrsf
multiplicity=1
nstate={nstates}
maxit=100
conv=1.0e-8
nvdav=100
spc_coco=0.5
spc_ovov=0.5
spc_coov=0.5

[symmetry]
enabled=False
point_group=auto
subgroup=auto
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", choices=CASES)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_inputs"))
    parser.add_argument("--nstates", type=int, default=12)
    parser.add_argument(
        "--no-molden",
        action="store_true",
        help="do not save the restricted triplet orbitals",
    )
    args = parser.parse_args()
    if args.nstates < 1:
        parser.error("--nstates must be positive")
    selected = tuple(args.cases) or CASES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in selected:
        output = args.output_dir / f"{case}_openqp_mrsf.inp"
        output.write_text(
            render(case, nstates=args.nstates, save_molden=not args.no_molden),
            encoding="utf-8",
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
