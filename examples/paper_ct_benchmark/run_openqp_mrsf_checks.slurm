#!/usr/bin/env bash
#SBATCH --job-name=emrsf-v041-mrsf-ref
#SBATCH --partition=cpu_mmm
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Submit from pyscf_emrsf_v0.5.1/examples/paper_ct_benchmark:
#   sbatch run_openqp_mrsf_checks.slurm

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
if [[ ! -f run_openqp_mrsf_check.py ]]; then
    echo "ERROR: submit from examples/paper_ct_benchmark" >&2
    exit 2
fi

cases=(n_phenylpyrrole twisted_dmabn)
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

OUTPUT_DIR="${BENCHMARK_DIR}/reference_checks_v0.4.1"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "PySCF-EMRSF v0.5.1 fixed-reference OpenQP MRSF regression check"
echo "Date/host         : $(date) / $(hostname)"
echo "Job/task          : ${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Case              : ${case_name}"
echo "CPUs/memory       : ${SLURM_CPUS_PER_TASK} / ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Python            : $(command -v python)"
echo "Output            : ${OUTPUT_DIR}"
echo "============================================================"

python - <<'PY'
import pyscf
import pyscf_emrsf
print("PySCF version      :", pyscf.__version__)
print("PySCF-EMRSF version:", pyscf_emrsf.__version__)
assert pyscf_emrsf.__version__ == "0.5.1"
PY

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
    --threads "${SLURM_CPUS_PER_TASK}" \
    --memory-mb 60000 \
    --pyscf-verbose 4 \
    --tolerance-ev 3.0e-3 \
    --output-dir "${OUTPUT_DIR}"

json_file="${OUTPUT_DIR}/${case_name}_external_mrsf_v0.5.1_openqp_check.json"
if [[ ! -s "${json_file}" ]] || ! grep -q '"passed": true' "${json_file}"; then
    echo "ERROR: fixed-reference MRSF check failed: ${json_file}" >&2
    exit 4
fi
echo "NORMAL TERMINATION: ${case_name} fixed-reference MRSF check passed"
