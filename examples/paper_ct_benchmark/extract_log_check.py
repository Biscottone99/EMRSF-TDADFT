"""Extract the compact EMRSF benchmark diagnostics directly from .log files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path
import re
import sys


HARTREE_TO_EV = 27.211386245988
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


@dataclass(frozen=True)
class LogCheck:
    file: str
    molecule: str
    version: str
    point_group: str
    reference_source: str
    reference_local_scf_run: str
    reference_warnings: str
    reference_orthonormality_error: float
    reference_electron_trace_error: float
    reference_spin_trace_error: float
    reference_gradient_max: float
    reference_min_symmetry_weight: float
    reference_somo_min_sv: float
    reference_alpha_occ_min_sv: float
    reference_beta_occ_min_sv: float
    root: int
    term: str
    symmetry_purity: float
    paper_term: str
    energy_ev: float
    paper_energy_ev: float
    energy_error_ev: float
    tbe_ev: float
    tbe_error_ev: float
    gamma_cv: float
    paper_gamma_cv: float
    gamma_error: float
    target_status: str
    best_cv_root: int
    best_cv_term: str
    best_cv_symmetry_purity: float
    best_cv_energy_ev: float
    best_cv_gamma: float
    s0_shift_ev: float
    residual: float
    emrsf_initial_guesses: int
    emrsf_seeds_per_irrep: str
    ag_ev: float
    ag_source_error_eh: float
    max_diagonal_error_eh: float
    diagonal_s0_estimate_ev: float
    g_partition_ev: float
    co1_partition_ev: float
    o2v_partition_ev: float
    normal_termination: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the values needed for the five-molecule EMRSF check"
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=None,
        help=(
            "optional custom paper target table; when omitted, use the five "
            "validated targets embedded in this standalone script"
        ),
    )
    parser.add_argument(
        "--molecule",
        default=None,
        help="override molecule auto-detection (only valid with one log)",
    )
    parser.add_argument("--target-term", default=None)
    parser.add_argument("--paper-energy", type=float, default=None)
    parser.add_argument("--paper-gamma", type=float, default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="also write the compact summary to this CSV file",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print LR-CV component ranges and dominant target configurations",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class TargetSpec:
    molecule: str
    paper_term: str
    log_term: str
    paper_energy_ev: float
    tbe_ev: float
    paper_gamma_cv: float
    requires_locked_reference: bool = False


BUILTIN_TARGETS = (
    # B1/B2 depend on the Cartesian convention.  These two explicit mappings
    # were established for the unmodified SI_loos coordinates by matching the
    # dominant configurations and CV character; they are not a global swap.
    TargetSpec("azulene", "2 B2", "2 B2", 4.91, 4.49, 0.86),
    TargetSpec(
        "n_phenylpyrrole", "3 A1", "3 A1", 6.51, 5.86, 0.98,
        requires_locked_reference=True,
    ),
    TargetSpec("phthalazine", "1 B1", "1 B1", 4.83, 4.31, 0.97),
    TargetSpec(
        "twisted_dmabn", "1 B1", "1 B1", 5.83, 4.74, 1.00,
        requires_locked_reference=True,
    ),
    TargetSpec("quinoxaline", "2 B1", "2 B2", 6.63, 6.22, 0.88),
)


def paper_to_pyscf_term(term: str) -> str:
    """Return a term unchanged; no universal B1/B2 conversion exists.

    Cartesian axes and hence B1/B2 labels depend on the molecular orientation
    and program convention.  A custom ``log_target_state`` may be supplied in
    the CSV only after the mapping has been established for that geometry.
    """

    return " ".join(term.split())


def load_targets(path: Path = None) -> dict[str, TargetSpec]:
    if path is None:
        return {target.molecule: target for target in BUILTIN_TARGETS}

    targets: dict[str, TargetSpec] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            molecule = row["molecule"].strip()
            paper_term = row["target_state"].strip()
            log_term = (row.get("log_target_state") or "").strip()
            targets[molecule] = TargetSpec(
                molecule=molecule,
                paper_term=paper_term,
                log_term=log_term or paper_to_pyscf_term(paper_term),
                paper_energy_ev=float(row["emrsf_ev"]),
                tbe_ev=float(row["tbe_ev"]),
                paper_gamma_cv=float(row["gamma_cv"]),
                requires_locked_reference=(
                    (row.get("requires_locked_reference") or "")
                    .strip()
                    .lower()
                    in {"1", "true", "yes", "on"}
                ),
            )
    if not targets:
        raise ValueError(f"no targets found in {path}")
    return targets


def detect_molecule(path: Path, targets: dict[str, TargetSpec]) -> str:
    name = path.name.lower()
    matches = [key for key in targets if name.startswith(key.lower() + "_")]
    if len(matches) != 1:
        known = ", ".join(sorted(targets))
        raise ValueError(
            f"{path}: cannot identify one molecule from the filename; "
            f"known names: {known}. Use --molecule for a single log."
        )
    return matches[0]


def first_float(text: str, pattern: str, default: float = float("nan")) -> float:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return float(match.group(1)) if match else default


def first_text(text: str, pattern: str, default: str = "--") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def first_int(text: str, pattern: str, default: int = -1) -> int:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return int(match.group(1)) if match else default


STATE_PATTERN = re.compile(
    rf"^STATE\s+(?P<root>\d+):\s+E=\s*(?P<energy>{FLOAT})\s+Eh"
    rf"\s+EXC=\s*(?P<excitation>{FLOAT})\s+eV.*?"
    rf"TERM=(?P<term>\d+\s+\S+)(?P<tail>.*)$",
    flags=re.MULTILINE,
)


def state_records(text: str) -> list[dict[str, object]]:
    records = []
    for match in STATE_PATTERN.finditer(text):
        gamma_match = re.search(
            rf"gamma_CV=\s*({FLOAT})", match.group("tail")
        )
        purity_match = re.search(
            rf"PURITY=\s*({FLOAT})", match.group("tail")
        )
        records.append(
            {
                "root": int(match.group("root")),
                "energy": float(match.group("energy")),
                "excitation": float(match.group("excitation")),
                "term": match.group("term"),
                "gamma": (
                    float(gamma_match.group(1))
                    if gamma_match is not None
                    else None
                ),
                "purity": (
                    float(purity_match.group(1))
                    if purity_match is not None
                    else float("nan")
                ),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return records


def emrsf_residual(text: str) -> float:
    section = re.search(
        r"^EMRSF EIGENSOLVER\s*$\n(?P<body>.*?)(?=\n\s*\n)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not section:
        return float("nan")
    return first_float(
        section.group("body"),
        rf"^MAX RESIDUAL NORM\s+({FLOAT})",
    )


def family_partition(text: str, family: str) -> float:
    pattern = (
        rf"^\s*{re.escape(family)}\s+"
        + rf"{FLOAT}\s+{FLOAT}\s+{FLOAT}\s+{FLOAT}\s+{FLOAT}\s+({FLOAT})\s*$"
    )
    return first_float(text, pattern)


def parse_log(
    path: Path,
    *,
    molecule: str,
    paper_term: str,
    target_term: str,
    paper_energy: float,
    tbe_energy: float,
    paper_gamma: float,
    requires_locked_reference: bool,
) -> tuple[LogCheck, str, dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records = state_records(text)
    emrsf_states = [record for record in records if record["gamma"] is not None]
    targets = [record for record in emrsf_states if record["term"] == target_term]
    if len(targets) > 1:
        available = ", ".join(
            f"{record['term']}[root={record['root']}, "
            f"E={record['excitation']:.6f} eV, gamma={record['gamma']:.6f}]"
            for record in emrsf_states
        )
        raise ValueError(
            f"{path}: expected one EMRSF TERM={target_term}, found {len(targets)}. "
            f"Available EMRSF states: {available or 'none'}"
        )
    if emrsf_states:
        best_cv = max(emrsf_states, key=lambda record: float(record["gamma"]))
    else:
        raise ValueError(f"{path}: no EMRSF states found")
    reference_source = first_text(text, r"^REFERENCE SOURCE\s+(\S+)")
    reference_local_scf_run = first_text(text, r"^LOCAL SCF RUN\s+(\S+)")
    reference_warnings = first_text(text, r"^REFERENCE WARNINGS\s+(.+)$")
    locked_reference = reference_source in {"mom_molden", "external_molden"}
    normal_termination = "PYSCF-EMRSF NORMAL TERMINATION" in text
    residual_value = emrsf_residual(text)
    ag_source_error_value = first_float(
        text, rf"^(?:A_G SOURCE ERROR|A_G CHECK ERROR)\s+({FLOAT})"
    )
    max_diagonal_error_value = first_float(
        text, rf"^MAX DIRECT DIAGONAL ERROR\s+({FLOAT})"
    )
    reference_min_symmetry_weight = first_float(
        text, rf"^MIN ORBITAL SYM WEIGHT\s+({FLOAT})"
    )
    fatal_reference_warnings = {
        "ORBITALS_NOT_ORTHONORMAL",
        "DENSITY_TRACE_MISMATCH",
        "ORBITALS_NOT_SYMMETRY_PURE",
        "LOW_SOMO_SUBSPACE_OVERLAP_TO_SEED",
    }
    reference_audit_failed = any(
        warning in reference_warnings for warning in fatal_reference_warnings
    )

    if targets:
        target = targets[0]
        energy_error = float(target["excitation"]) - paper_energy
        gamma_error = float(target["gamma"]) - paper_gamma
        if not normal_termination:
            target_status = "ABNORMAL_TERMINATION"
        elif requires_locked_reference and not locked_reference:
            target_status = "REFERENCE_NOT_LOCKED"
        elif reference_audit_failed:
            target_status = "REFERENCE_AUDIT_FAILED"
        elif residual_value != residual_value or residual_value > 1.2e-7:
            target_status = "RESPONSE_NOT_CONVERGED"
        elif (
            reference_min_symmetry_weight == reference_min_symmetry_weight
            and reference_min_symmetry_weight < 0.999999
        ):
            target_status = "REFERENCE_SYMMETRY_MIXING"
        elif float(target["purity"]) < 0.999:
            target_status = "SYMMETRY_MIXING"
        elif abs(ag_source_error_value) > 1.0e-8:
            target_status = "A_G_AUDIT_FAILED"
        elif abs(max_diagonal_error_value) > 1.0e-8:
            target_status = "LR_CV_DIAGONAL_AUDIT_FAILED"
        elif abs(gamma_error) > 0.25:
            target_status = "CHARACTER_MISMATCH"
        elif abs(energy_error) > 0.10:
            target_status = "ENERGY_MISMATCH"
        else:
            target_status = "CONSISTENT_WITH_TARGET"
    else:
        target = {
            "root": -1,
            "term": target_term,
            "energy": float("nan"),
            "excitation": float("nan"),
            "gamma": float("nan"),
            "purity": float("nan"),
            "start": 0,
            "end": 0,
        }
        energy_error = float("nan")
        gamma_error = float("nan")
        target_status = "TARGET_TERM_NOT_COMPUTED"

    exact_s0_shift = first_float(
        text,
        rf"^EXACT S0 SHIFT\s+{FLOAT}\s+Eh\s+({FLOAT})\s+eV",
    )
    if exact_s0_shift != exact_s0_shift:
        mrsf_s0 = next(
            (
                record
                for record in records
                if record["root"] == 0 and record["gamma"] is None
            ),
            None,
        )
        emrsf_s0 = next(
            (
                record
                for record in records
                if record["root"] == 0 and record["gamma"] is not None
            ),
            None,
        )
        if mrsf_s0 is not None and emrsf_s0 is not None:
            exact_s0_shift = (
                float(emrsf_s0["energy"]) - float(mrsf_s0["energy"])
            ) * HARTREE_TO_EV

    ag_ev = first_float(
        text,
        rf"^(?:A_G EXPLICIT SI S23|A_G STORED|GROUND OFFSET A_G)"
        rf"\s+{FLOAT}\s+Eh\s+({FLOAT})\s+eV",
    )
    result = LogCheck(
        file=str(path),
        molecule=molecule,
        version=first_text(text, r"release v([^\s]+)"),
        point_group=first_text(text, r"^POINT GROUP\s+(\S+)"),
        reference_source=reference_source,
        reference_local_scf_run=reference_local_scf_run,
        reference_warnings=reference_warnings,
        reference_orthonormality_error=first_float(
            text, rf"^FINAL S-ORTH ERROR\s+({FLOAT})"
        ),
        reference_electron_trace_error=first_float(
            text, rf"^ELECTRON TRACE ERROR\s+({FLOAT})"
        ),
        reference_spin_trace_error=first_float(
            text, rf"^SPIN TRACE ERROR\s+({FLOAT})"
        ),
        reference_gradient_max=first_float(
            text, rf"^ORBITAL GRADIENT MAX\s+({FLOAT})"
        ),
        reference_min_symmetry_weight=reference_min_symmetry_weight,
        reference_somo_min_sv=min(
            (
                float(value)
                for value in first_text(
                    text, r"^SEED SOMO SINGULAR VAL\s+(.+)$", default=""
                ).split()
            ),
            default=float("nan"),
        ),
        reference_alpha_occ_min_sv=first_float(
            text, rf"^SEED ALPHA OCC MIN SV\s+({FLOAT})"
        ),
        reference_beta_occ_min_sv=first_float(
            text, rf"^SEED BETA OCC MIN SV\s+({FLOAT})"
        ),
        root=int(target["root"]),
        term=str(target["term"]),
        symmetry_purity=float(target["purity"]),
        paper_term=paper_term,
        energy_ev=float(target["excitation"]),
        paper_energy_ev=paper_energy,
        energy_error_ev=energy_error,
        tbe_ev=tbe_energy,
        tbe_error_ev=float(target["excitation"]) - tbe_energy,
        gamma_cv=float(target["gamma"]),
        paper_gamma_cv=paper_gamma,
        gamma_error=gamma_error,
        target_status=target_status,
        best_cv_root=int(best_cv["root"]),
        best_cv_term=str(best_cv["term"]),
        best_cv_symmetry_purity=float(best_cv["purity"]),
        best_cv_energy_ev=float(best_cv["excitation"]),
        best_cv_gamma=float(best_cv["gamma"]),
        s0_shift_ev=exact_s0_shift,
        residual=residual_value,
        emrsf_initial_guesses=first_int(
            text,
            r"^EMRSF EIGENSOLVER\s*$[\s\S]*?^INITIAL GUESSES\s+(\d+)",
        ),
        emrsf_seeds_per_irrep=first_text(
            text,
            r"^EMRSF EIGENSOLVER\s*$[\s\S]*?^SEEDS PER IRREP\s+(.+)$",
        ),
        ag_ev=ag_ev,
        ag_source_error_eh=ag_source_error_value,
        max_diagonal_error_eh=max_diagonal_error_value,
        diagonal_s0_estimate_ev=first_float(
            text,
            rf"^DIAG 2ND-ORDER S0 SHIFT\s+{FLOAT}\s+Eh\s+({FLOAT})\s+eV",
        ),
        g_partition_ev=family_partition(text, "G"),
        co1_partition_ev=family_partition(text, "CO1"),
        o2v_partition_ev=family_partition(text, "O2V"),
        normal_termination=normal_termination,
    )
    return result, text, target


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.10g}"
    return str(value)


def write_rows(handle, rows: list[LogCheck], *, delimiter: str) -> None:
    names = [field.name for field in fields(LogCheck)]
    writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
    writer.writerow(names)
    for row in rows:
        writer.writerow([fmt(getattr(row, name)) for name in names])


LR_ROW = re.compile(
    rf"^\s*(?P<index>\d+)\s+(?P<i>\d+)\s+->\s+(?P<a>\d+)\s+"
    rf"(?P<raw>{FLOAT})\s+(?P<s30>{FLOAT})\s+(?P<j>{FLOAT})\s+"
    rf"(?P<x>{FLOAT})\s+(?P<xc>{FLOAT})\s+(?P<ag>{FLOAT})\s+"
    rf"(?P<total>{FLOAT})\s*$",
    flags=re.MULTILINE,
)


def print_details(path: Path, text: str, target: dict[str, object]) -> None:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for match in LR_ROW.finditer(text):
        rows[(int(match.group("i")), int(match.group("a")))] = {
            name: float(match.group(name))
            for name in ("raw", "s30", "j", "x", "xc", "ag", "total")
        }
    print(f"\nDETAILS {path}")
    if int(target["root"]) < 0:
        print(f"  requested term {target['term']} is not among the computed roots")
        return
    if rows:
        print("LR-CV component ranges / Eh")
        for name in ("raw", "s30", "j", "x", "xc", "ag", "total"):
            values = [row[name] for row in rows.values()]
            print(f"  {name:<6s} min={min(values): .9f}  max={max(values): .9f}")
    else:
        print("  LR-CV diagonal table not present in this log")
        return

    next_state = re.search(r"^STATE\s+\d+:", text[int(target["end"]) :], re.MULTILINE)
    block_end = (
        int(target["end"]) + next_state.start()
        if next_state
        else len(text)
    )
    block = text[int(target["end"]) : block_end]
    config_pattern = re.compile(
        rf"^\s*CV\(LR\)\s+(\d+)\s+->\s+(\d+)\s+"
        rf"c=\s*({FLOAT})\s+w=\s*({FLOAT})",
        re.MULTILINE,
    )
    print("Dominant target CV(LR) configurations present in the log")
    print("       C -> V          c           w       S30/Eh      XC/Eh    total/Eh")
    found = False
    for match in config_pattern.finditer(block):
        found = True
        i, a = int(match.group(1)), int(match.group(2))
        diagonal = rows.get((i, a))
        if diagonal is None:
            print(
                f"  {i:6d} -> {a:<6d} {float(match.group(3)): .9f}"
                f" {float(match.group(4)): .9f}   diagonal not found"
            )
            continue
        print(
            f"  {i:6d} -> {a:<6d} {float(match.group(3)): .9f}"
            f" {float(match.group(4)): .9f} {diagonal['s30']: .9f}"
            f" {diagonal['xc']: .9f} {diagonal['total']: .9f}"
        )
    if not found:
        print("  no CV(LR) configuration was included among the printed top amplitudes")


def main() -> int:
    args = parse_arguments()
    if args.molecule is not None and len(args.logs) != 1:
        print(
            "ERROR: --molecule can be used only when exactly one log is given",
            file=sys.stderr,
        )
        return 2
    try:
        targets = load_targets(args.targets_csv)
        parsed = []
        for path in args.logs:
            if not path.is_file():
                raise FileNotFoundError(f"log file not found: {path}")
            molecule = args.molecule or detect_molecule(path, targets)
            if molecule not in targets:
                raise ValueError(f"unknown molecule: {molecule}")
            spec = targets[molecule]
            parsed.append(
                parse_log(
                    path,
                    molecule=molecule,
                    paper_term=spec.paper_term,
                    target_term=args.target_term or spec.log_term,
                    paper_energy=(
                        args.paper_energy
                        if args.paper_energy is not None
                        else spec.paper_energy_ev
                    ),
                    tbe_energy=spec.tbe_ev,
                    paper_gamma=(
                        args.paper_gamma
                        if args.paper_gamma is not None
                        else spec.paper_gamma_cv
                    ),
                    requires_locked_reference=spec.requires_locked_reference,
                )
            )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rows = [item[0] for item in parsed]
    write_rows(sys.stdout, rows, delimiter="\t")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            write_rows(handle, rows, delimiter=",")
        print(f"\nCSV written: {args.csv}", file=sys.stderr)
    if args.details:
        for path, (_, text, target) in zip(args.logs, parsed):
            print_details(path, text, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
