#!/usr/bin/env bash
# Local validation driver. Usage:
#   bash run_validation_local.sh quick       # unit/equation tests only
#   bash run_validation_local.sh reference   # two fixed-OpenQP MRSF checks
#   bash run_validation_local.sh benchmark   # five full EMRSF jobs
#   bash run_validation_local.sh all         # all three stages

set -euo pipefail

mode="${1:-quick}"
threads="${EMRSF_THREADS:-8}"
memory_mb="${EMRSF_MEMORY_MB:-24000}"
benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${benchmark_dir}/../.." && pwd)"
cd "${benchmark_dir}"

export OMP_NUM_THREADS="${threads}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_quick() {
    (cd "${project_dir}" && python -m pytest -q)
}

run_reference() {
    for case_name in n_phenylpyrrole twisted_dmabn; do
        python -u run_openqp_mrsf_check.py "${case_name}" \
            --reference-mode external \
            --basis cc-pvdz \
            --auxbasis cc-pvdz-jkfit \
            --xc bhandhlyp \
            --symmetry none \
            --solver davidson \
            --nstates 12 \
            --conv-tol 1.0e-8 \
            --max-cycle 120 \
            --scf-grid-level 3 \
            --threads "${threads}" \
            --memory-mb "${memory_mb}" \
            --output-dir reference_checks_v0.4.1
    done
}

run_benchmark() {
    for case_name in azulene n_phenylpyrrole phthalazine twisted_dmabn quinoxaline; do
        command=(
            python -u run_case.py "${case_name}"
            --basis cc-pvdz
            --auxbasis cc-pvdz-jkfit
            --xc bhandhlyp
            --symmetry auto
            --solver davidson
            --scf-grid-level 3
            --nstates 12
            --conv-tol 1.0e-8
            --max-cycle 120
            --observable-grid-level 5
            --threads "${threads}"
            --memory-mb "${memory_mb}"
            --output-dir results_v0.4.1
            --coupling-audit
            --lr-cv-audit
        )
        case "${case_name}" in
            n_phenylpyrrole)
                command+=(
                    --reference-mode mom
                    --reference-molden reference_orbitals/n_phenylpyrrole_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden
                    --minimum-somo-overlap 0.80
                    --minimum-occupied-overlap 0.80
                )
                ;;
            twisted_dmabn)
                command+=(
                    --reference-mode mom
                    --reference-molden reference_orbitals/twisted_dmabn_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden
                    --minimum-somo-overlap 0.80
                    --minimum-occupied-overlap 0.80
                )
                ;;
            *)
                command+=(--reference-mode auto)
                ;;
        esac
        "${command[@]}"
    done
    python extract_log_check.py \
        --targets-csv paper_targets.csv \
        --csv results_v0.4.1/emrsf_v041_check.csv \
        results_v0.4.1/*_v0.5.1.log
}

case "${mode}" in
    quick) run_quick ;;
    reference) run_reference ;;
    benchmark) run_benchmark ;;
    all) run_quick; run_reference; run_benchmark ;;
    *) echo "Usage: $0 {quick|reference|benchmark|all}" >&2; exit 2 ;;
esac
