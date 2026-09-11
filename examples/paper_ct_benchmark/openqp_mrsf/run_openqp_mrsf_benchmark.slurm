#!/usr/bin/env bash
#SBATCH --job-name=oqp121-mrsf-ct
#SBATCH --partition=cpu_mmm
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Submit from examples/paper_ct_benchmark/openqp_mrsf:
#   sbatch run_openqp_mrsf_benchmark.slurm
# The calculation saves the restricted triplet Molden needed by the fixed-
# reference PySCF operator check.

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

SOURCE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${SOURCE_DIR}"
if [[ ! -f generate_inputs.py ]]; then
    echo "ERROR: submit from examples/paper_ct_benchmark/openqp_mrsf" >&2
    exit 2
fi

cases=(azulene n_phenylpyrrole phthalazine quinoxaline twisted_dmabn)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not defined}"
if (( task_id < 0 || task_id >= ${#cases[@]} )); then
    echo "ERROR: invalid array index ${task_id}" >&2
    exit 3
fi
case_name="${cases[$task_id]}"

if ! command -v openqp >/dev/null 2>&1; then
    echo "ERROR: openqp executable not found in the active environment" >&2
    exit 4
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_STACKSIZE=512M
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export TMPDIR="${SLURM_TMPDIR}"
fi
ulimit -s unlimited

RESULTS_ROOT="${OUTPUT_ROOT:-${SOURCE_DIR}/results_openqp_1.2.1}"
run_dir="${RESULTS_ROOT}/${case_name}_${SLURM_ARRAY_JOB_ID}_${task_id}"
mkdir -p "${run_dir}"
python -u generate_inputs.py "${case_name}" --output-dir "${run_dir}"
input_file="${run_dir}/${case_name}_openqp_mrsf.inp"
cd "${run_dir}"

echo "============================================================"
echo "OpenQP 1.2.1 conventional MRSF benchmark oracle"
echo "Date/host         : $(date) / $(hostname)"
echo "Job/task          : ${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Case              : ${case_name}"
echo "OpenQP            : $(command -v openqp)"
echo "Input             : ${input_file}"
echo "Output            : ${run_dir}"
echo "============================================================"

openqp --nompi --omp "${OMP_NUM_THREADS}" "${input_file}" 2>&1 \
    | tee "${case_name}_openqp_console.out"

log_file="${run_dir}/${case_name}_openqp_mrsf.log"
molden_file="${run_dir}/${case_name}_openqp_mrsf_scf_rohf_bhhlyp_cc-pvdz.molden"
if [[ ! -s "${log_file}" ]]; then
    echo "ERROR: missing OpenQP log ${log_file}" >&2
    exit 5
fi
if ! grep -q "PyOQP: SCF converged" "${log_file}"; then
    echo "ERROR: OpenQP SCF did not report convergence" >&2
    exit 6
fi
if ! grep -q "MRSF-TD-DFT energies converged" "${log_file}"; then
    echo "ERROR: OpenQP MRSF did not report convergence" >&2
    exit 7
fi
if ! grep -q "PyOQP terminated at" "${log_file}"; then
    echo "ERROR: OpenQP normal-termination marker is absent" >&2
    exit 8
fi
if [[ ! -s "${molden_file}" ]]; then
    echo "ERROR: missing OpenQP triplet Molden ${molden_file}" >&2
    exit 9
fi

summary_file="${run_dir}/${case_name}_openqp_mrsf_summary.txt"
awk '
    /PyOQP electronic energies/ {inside=1; next}
    inside && /PyOQP triplet ROHF reference/ {print; next}
    inside && /PyOQP S[0-9]+[[:space:]]/ {print; next}
    inside && /PyOQP dftd correction/ {exit}
' "${log_file}" > "${summary_file}"

echo "NORMAL TERMINATION: ${case_name} OpenQP MRSF oracle completed"
echo "Log     : ${log_file}"
echo "Molden  : ${molden_file}"
echo "Summary : ${summary_file}"
