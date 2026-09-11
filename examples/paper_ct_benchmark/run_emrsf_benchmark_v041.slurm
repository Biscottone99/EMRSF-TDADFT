#!/usr/bin/env bash
#SBATCH --job-name=emrsf-v041-ct
#SBATCH --partition=cpu_mmm
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Legacy-protocol regression job shipped with v0.5.1. Submit from
# pyscf_emrsf_v0.5.1/examples/paper_ct_benchmark:
#   sbatch run_emrsf_benchmark_v041.slurm
# Optional exports (off by default):
#   WRITE_MOLDEN=1 WRITE_NTO=1 sbatch run_emrsf_benchmark_v041.slurm
# Disable the expensive equation decomposition after it has passed once:
#   FULL_AUDIT=0 sbatch run_emrsf_benchmark_v041.slurm
# Use OpenQP orbitals for all five fixed-reference operator comparisons:
#   REFERENCE_POLICY=external_all REFERENCE_DIR=/path/to/moldens sbatch ...
# Use every OpenQP Molden only as a MOM state lock, then reoptimise in PySCF:
#   REFERENCE_POLICY=mom_all REFERENCE_DIR=/path/to/moldens sbatch ...

set -eo pipefail
set +u
CONDA_SH="${CONDA_SH:-/hpc/share/tools/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "ERROR: cannot read Conda hook ${CONDA_SH}" >&2
    exit 1
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME:-openqp_env}"
set -u

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${BENCHMARK_DIR}"
if [[ ! -f run_case.py ]]; then
    echo "ERROR: submit from examples/paper_ct_benchmark" >&2
    exit 2
fi

cases=(azulene n_phenylpyrrole phthalazine twisted_dmabn quinoxaline)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not defined}"
if (( task_id < 0 || task_id >= ${#cases[@]} )); then
    echo "ERROR: invalid array index ${task_id}" >&2
    exit 3
fi
case_name="${cases[$task_id]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export TMPDIR="${SLURM_TMPDIR}"
fi

reference_policy="${REFERENCE_POLICY:-benchmark}"
if [[ "${reference_policy}" == "benchmark" ]]; then
    default_output_dir="${BENCHMARK_DIR}/results_v0.4.1"
else
    default_output_dir="${BENCHMARK_DIR}/results_v0.4.1_${reference_policy}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${default_output_dir}}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "PySCF-EMRSF v0.5.1 legacy state-locked CT regression benchmark"
echo "Date/host         : $(date) / $(hostname)"
echo "Job/task          : ${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Case              : ${case_name}"
echo "CPUs/memory       : ${SLURM_CPUS_PER_TASK} / ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Python            : $(command -v python)"
echo "Output            : ${OUTPUT_DIR}"
echo "REFERENCE_POLICY  : ${reference_policy}"
echo "FULL_AUDIT        : ${FULL_AUDIT:-1}"
echo "WRITE_MOLDEN/NTO  : ${WRITE_MOLDEN:-0}/${WRITE_NTO:-0}"
echo "============================================================"

python - <<'PY'
import numpy
import scipy
import pyscf
import pyscf_emrsf
print("NumPy version      :", numpy.__version__)
print("SciPy version      :", scipy.__version__)
print("PySCF version      :", pyscf.__version__)
print("PySCF-EMRSF version:", pyscf_emrsf.__version__)
assert pyscf_emrsf.__version__ == "0.5.1"
PY

command=(
    python -u run_case.py "${case_name}"
    --basis cc-pvdz
    --auxbasis cc-pvdz-jkfit
    --xc bhandhlyp
    --symmetry auto
    --solver davidson
    --reference-conv-tol 1.0e-10
    --reference-max-cycle 200
    --scf-grid-level 3
    --conv-tol 1.0e-8
    --max-cycle 120
    --nstates 12
    --observable-grid-level 5
    --threads "${SLURM_CPUS_PER_TASK}"
    --memory-mb 60000
    --pyscf-verbose 4
    --output-dir "${OUTPUT_DIR}"
)

reference_dir="${REFERENCE_DIR:-${BENCHMARK_DIR}/reference_orbitals}"
reference_file="${reference_dir}/${case_name}_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden"
case "${reference_policy}" in
    benchmark)
        case "${case_name}" in
            n_phenylpyrrole|twisted_dmabn)
                command+=(
                    --reference-mode mom
                    --reference-molden "${reference_file}"
                    --minimum-somo-overlap 0.80
                    --minimum-occupied-overlap 0.80
                )
                ;;
            *)
                command+=(--reference-mode auto)
                ;;
        esac
        ;;
    external_all|mom_all)
        if [[ ! -s "${reference_file}" ]]; then
            echo "ERROR: ${reference_policy} requires ${reference_file}" >&2
            exit 4
        fi
        if [[ "${reference_policy}" == "external_all" ]]; then
            command+=(--reference-mode external --reference-molden "${reference_file}")
        else
            command+=(
                --reference-mode mom
                --reference-molden "${reference_file}"
                --minimum-somo-overlap 0.80
                --minimum-occupied-overlap 0.80
            )
        fi
        ;;
    auto_all)
        command+=(--reference-mode auto --allow-unlocked-reference)
        ;;
    *)
        echo "ERROR: unknown REFERENCE_POLICY=${reference_policy}" >&2
        exit 4
        ;;
esac

if [[ "${FULL_AUDIT:-1}" == "1" ]]; then
    command+=(--coupling-audit --lr-cv-audit --lr-cv-audit-batch-size 32)
fi
if [[ "${WRITE_MOLDEN:-0}" == "1" ]]; then
    command+=(--write-molden)
fi
if [[ "${WRITE_NTO:-0}" == "1" ]]; then
    command+=(--write-nto --nto-weight-threshold 1.0e-4)
fi

"${command[@]}"

log_file="${OUTPUT_DIR}/${case_name}_v0.5.1.log"
npz_file="${OUTPUT_DIR}/${case_name}_v0.5.1.npz"
if [[ ! -s "${log_file}" ]] || [[ ! -s "${npz_file}" ]]; then
    echo "ERROR: missing result files for ${case_name}" >&2
    exit 5
fi
if ! grep -q "PYSCF-EMRSF NORMAL TERMINATION" "${log_file}"; then
    echo "ERROR: normal-termination marker absent from ${log_file}" >&2
    exit 6
fi
if grep -q "REFERENCE WARNINGS.*ORBITALS_NOT_SYMMETRY_PURE" "${log_file}"; then
    echo "ERROR: reference orbitals are not symmetry pure" >&2
    exit 7
fi
if [[ "${reference_policy}" == "benchmark" ]] \
    && [[ "${case_name}" == "n_phenylpyrrole" || "${case_name}" == "twisted_dmabn" ]]; then
    if ! grep -q "REFERENCE SOURCE.*mom_molden" "${log_file}"; then
        echo "ERROR: required MOM reference was not used" >&2
        exit 8
    fi
fi

echo "NORMAL TERMINATION: ${case_name} EMRSF benchmark completed"
