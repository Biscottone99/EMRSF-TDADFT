#!/usr/bin/env bash
#SBATCH --job-name=mrsf-emrsf-v050
#SBATCH --partition=cpu_mmm
#SBATCH --array=0-4%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Equal-level comparison (recommended):
#   METHOD=mrsf  PROFILE=exact-dz sbatch run_high_accuracy_v051.slurm
#   METHOD=emrsf PROFILE=exact-dz sbatch run_high_accuracy_v051.slurm
# Optional basis-convergence stages: PROFILE=tbe-dz or PROFILE=tbe-tz.

set -eo pipefail
set +u
CONDA_SH="${CONDA_SH:-/hpc/share/tools/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "ERROR: cannot read Conda hook ${CONDA_SH}" >&2
    exit 1
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME:-emrsf_py312}"
set -u

benchmark_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${benchmark_dir}"
if [[ ! -f run_high_accuracy.py ]]; then
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
method="${METHOD:-emrsf}"
profile="${PROFILE:-exact-dz}"
backend="${INTEGRAL_BACKEND:-direct}"
residual_tol="${RESIDUAL_TOL:-1e-9}"
if [[ "${method}" != "mrsf" ]] && [[ "${method}" != "emrsf" ]]; then
    echo "ERROR: METHOD must be mrsf or emrsf" >&2
    exit 3
fi
grid_tag=""
if [[ -n "${SCF_GRID_LEVEL:-}" ]]; then
    grid_tag="_grid${SCF_GRID_LEVEL}"
fi
output_dir="${OUTPUT_DIR:-${benchmark_dir}/results_v0.5.1_${method}_${profile}_${backend}${grid_tag}}"
mkdir -p "${output_dir}"

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

echo "============================================================"
echo "PySCF-EMRSF v0.5.1 unified high-accuracy calculation"
echo "Date/host          : $(date) / $(hostname)"
echo "Case/method        : ${case_name} / ${method}"
echo "Profile/backend    : ${profile} / ${backend}"
echo "Residual tolerance : ${residual_tol}"
echo "CPUs/memory        : ${SLURM_CPUS_PER_TASK} / ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Output             : ${output_dir}"
echo "TBE fitting        : DISABLED"
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
    python -u run_high_accuracy.py "${case_name}"
    --method "${method}"
    --profile "${profile}"
    --integral-backend "${backend}"
    --reference-mode mom
    --solver davidson
    --residual-tol "${residual_tol}"
    --threads "${SLURM_CPUS_PER_TASK}"
    --memory-mb 116000
    --output-dir "${output_dir}"
    --pyscf-verbose 4
)
if [[ "${backend}" == "df" ]] && [[ -n "${AUXBASIS:-}" ]]; then
    command+=(--auxbasis "${AUXBASIS}")
fi
if [[ "${method}" == "emrsf" ]] && [[ "${FULL_LR_CV_AUDIT:-0}" == "1" ]]; then
    command+=(--lr-cv-audit --lr-cv-audit-batch-size 16)
fi
if [[ "${WRITE_MOLDEN:-0}" == "1" ]]; then
    command+=(--write-molden)
fi
if [[ -n "${SCF_GRID_LEVEL:-}" ]]; then
    command+=(--scf-grid-level "${SCF_GRID_LEVEL}")
fi

"${command[@]}"

stem="${output_dir}/${case_name}_${method}_v0.5.1_${profile}"
log_file="${stem}.log"
npz_file="${stem}.npz"
if [[ ! -s "${log_file}" ]] || [[ ! -s "${npz_file}" ]]; then
    echo "ERROR: missing result files" >&2
    exit 4
fi
grep -q "PYSCF-EMRSF NORMAL TERMINATION" "${log_file}"
if [[ "${method}" == "emrsf" ]]; then
    grep -q "DIRECT EMRSF MODE    True" "${log_file}"
    grep -q "MRSF PRE-DIAGONAL.   False" "${log_file}"
fi

python - "${npz_file}" "${method}" <<'PY'
import json
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata_json"]))
    residual = float(np.max(data["response_residual_norms"]))
    assert metadata["method"] == sys.argv[2]
    assert metadata["tbe_used_in_hamiltonian"] is False
    assert residual <= 1.1 * float(metadata["residual_tolerance"]), residual
    if metadata["method"] == "emrsf":
        closure = float(
            np.max(np.abs(data["decomposition_closure_error_hartree"]))
        )
        assert metadata["direct_emrsf"] is True
        assert closure <= 5.0e-9, closure
        print(f"verified residual={residual:.3e}; closure={closure:.3e} Eh")
    else:
        assert metadata["direct_emrsf"] is False
        print(f"verified residual={residual:.3e}")
PY

echo "NORMAL TERMINATION: ${case_name} ${method} ${profile}"
