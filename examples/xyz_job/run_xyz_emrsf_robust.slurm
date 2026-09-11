#!/usr/bin/env bash
#SBATCH --job-name=emrsf-robust
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Conda's Intel-MPI hook reads unset variables, so enable nounset only after
# activation.
set -eo pipefail

source /hpc/share/tools/miniconda3/etc/profile.d/conda.sh
conda activate openqp_env
set -u

XYZ_FILE="${XYZ_FILE:-molecola.xyz}"
REFERENCE_MOLDEN="${REFERENCE_MOLDEN:-}"
METHOD="${METHOD:-emrsf}"
STATES="${STATES:-singlets}"
SOC="${SOC:-0}"
SOC_TWO_ELECTRON="${SOC_TWO_ELECTRON:-amfi}"
SOC_EXTERNAL_MODULE="${SOC_EXTERNAL_MODULE:-}"
CHARGE="${CHARGE:-0}"
SPIN="${SPIN:-2}"
XC="${XC:-bhandhlyp}"
BASIS="${BASIS:-def2-svp}"
INTEGRAL_BACKEND="${INTEGRAL_BACKEND:-df}"
AUXBASIS="${AUXBASIS:-def2-universal-jkfit}"
NSTATES="${NSTATES:-6}"
GRID_LEVEL="${GRID_LEVEL:-6}"
MEMORY_MB="${MEMORY_MB:-140000}"
WRITE_OBSERVABLES="${WRITE_OBSERVABLES:-1}"
OBSERVABLES_GRID_LEVEL="${OBSERVABLES_GRID_LEVEL:-5}"
OBSERVABLES_UNPRUNED_GRID="${OBSERVABLES_UNPRUNED_GRID:-0}"
OSCILLATOR_STRENGTHS="${OSCILLATOR_STRENGTHS:-1}"
SAVE_VECTORS="${SAVE_VECTORS:-1}"
EIGENVECTOR_THRESHOLD="${EIGENVECTOR_THRESHOLD:-0.0}"

# Leave PCM_EPSILON empty for a gas-phase calculation.  Examples: water
# 78.3553, dichloromethane 8.93, toluene 2.3741 (use the value appropriate to
# the experimental temperature and solvent model).
PCM_EPSILON="${PCM_EPSILON:-}"
PCM_METHOD="${PCM_METHOD:-IEF-PCM}"
PCM_LEBEDEV_ORDER="${PCM_LEBEDEV_ORDER:-29}"
PCM_VDW_SCALE="${PCM_VDW_SCALE:-1.2}"
PCM_PROBE_RADIUS="${PCM_PROBE_RADIUS:-0.0}"
PCM_SURFACE="${PCM_SURFACE:-SWIG}"

SCF_STRATEGY="${SCF_STRATEGY:-robust}"
SCF_MAX_CYCLE="${SCF_MAX_CYCLE:-300}"
SCF_INITIAL_MAX_CYCLE="${SCF_INITIAL_MAX_CYCLE:-40}"
SCF_STABILIZATION_MAX_CYCLE="${SCF_STABILIZATION_MAX_CYCLE:-80}"
SCF_NEWTON_MAX_CYCLE="${SCF_NEWTON_MAX_CYCLE:-80}"
SCF_DAMPING="${SCF_DAMPING:-0.50}"
SCF_LEVEL_SHIFT="${SCF_LEVEL_SHIFT:-0.30}"
SCF_COARSE_GRID_LEVEL="${SCF_COARSE_GRID_LEVEL:-3}"

SCRATCH_BASE="/hpc/scratch/matteo.bedogni"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
FINAL_DIR="${SUBMIT_DIR}/results_${METHOD}_${STATES}_v0.8.0_${SLURM_JOB_ID}"
RUN_DIR="${SCRATCH_BASE}/emrsf_${USER}_${SLURM_JOB_ID}"
TMP_DIR="${RUN_DIR}/tmp"
RUN_OUTPUT="${RUN_DIR}/results"
LOCAL_XYZ="${RUN_DIR}/molecule.xyz"
LOCAL_DRIVER="${RUN_DIR}/run_xyz_emrsf.py"
LOCAL_MOLDEN="${RUN_DIR}/reference_seed.molden"
LOCAL_SOC_MODULE="${RUN_DIR}/external_soc_module.py"
SOC_EXTERNAL_SOURCE=""

if [[ ! -f "${SUBMIT_DIR}/${XYZ_FILE}" ]]; then
    echo "ERROR: XYZ not found: ${SUBMIT_DIR}/${XYZ_FILE}" >&2
    exit 2
fi
if [[ ! -f "${SUBMIT_DIR}/run_xyz_emrsf.py" ]]; then
    echo "ERROR: run_xyz_emrsf.py not found in ${SUBMIT_DIR}" >&2
    exit 3
fi
if [[ -n "${REFERENCE_MOLDEN}" && ! -f "${SUBMIT_DIR}/${REFERENCE_MOLDEN}" ]]; then
    echo "ERROR: reference Molden not found: ${SUBMIT_DIR}/${REFERENCE_MOLDEN}" >&2
    exit 4
fi
if [[ -n "${SOC_EXTERNAL_MODULE}" ]]; then
    if [[ "${SOC_EXTERNAL_MODULE}" = /* ]]; then
        SOC_EXTERNAL_SOURCE="${SOC_EXTERNAL_MODULE}"
    else
        SOC_EXTERNAL_SOURCE="${SUBMIT_DIR}/${SOC_EXTERNAL_MODULE}"
    fi
    if [[ ! -f "${SOC_EXTERNAL_SOURCE}" ]]; then
        echo "ERROR: external SOC module not found: ${SOC_EXTERNAL_SOURCE}" >&2
        exit 6
    fi
fi
if [[ ! -d "${SCRATCH_BASE}" || ! -w "${SCRATCH_BASE}" ]]; then
    echo "ERROR: scratch is unavailable: ${SCRATCH_BASE}" >&2
    exit 5
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "${TMP_DIR}" "${RUN_OUTPUT}" "${FINAL_DIR}"
cp "${SUBMIT_DIR}/${XYZ_FILE}" "${LOCAL_XYZ}"
cp "${SUBMIT_DIR}/run_xyz_emrsf.py" "${LOCAL_DRIVER}"
if [[ -n "${REFERENCE_MOLDEN}" ]]; then
    cp "${SUBMIT_DIR}/${REFERENCE_MOLDEN}" "${LOCAL_MOLDEN}"
fi
if [[ -n "${SOC_EXTERNAL_SOURCE}" ]]; then
    cp "${SOC_EXTERNAL_SOURCE}" "${LOCAL_SOC_MODULE}"
fi

export TMPDIR="${TMP_DIR}"
export TMP="${TMP_DIR}"
export TEMP="${TMP_DIR}"

finalize_job()
{
    calculation_status=$?
    copy_status=0
    trap - EXIT
    set +e
    mkdir -p "${FINAL_DIR}" || copy_status=$?
    if [[ -d "${RUN_OUTPUT}" ]]; then
        cp -a "${RUN_OUTPUT}/." "${FINAL_DIR}/" || copy_status=$?
    fi
    if [[ -f "${RUN_DIR}/calculation.stdout.log" ]]; then
        cp "${RUN_DIR}/calculation.stdout.log" "${FINAL_DIR}/" || copy_status=$?
    fi
    {
        echo "calculation_status=${calculation_status}"
        echo "copy_status=${copy_status}"
        echo "scratch_directory=${RUN_DIR}"
        echo "final_directory=${FINAL_DIR}"
        echo "date=$(date --iso-8601=seconds)"
    } > "${FINAL_DIR}/job_status.txt"

    if [[ "${calculation_status}" -eq 0 && "${copy_status}" -eq 0 ]]; then
        expected="${SCRATCH_BASE}/emrsf_${USER}_${SLURM_JOB_ID}"
        if [[ "${RUN_DIR}" == "${expected}" ]]; then
            rm -rf -- "${RUN_DIR}"
        fi
        echo "Results copied to ${FINAL_DIR}"
        exit 0
    fi
    echo "Job failed; retained scratch directory ${RUN_DIR}" >&2
    if [[ "${calculation_status}" -ne 0 ]]; then
        exit "${calculation_status}"
    fi
    exit "${copy_status}"
}
trap finalize_job EXIT

cd "${RUN_DIR}"
echo "Job ID             : ${SLURM_JOB_ID}"
echo "Host               : $(hostname)"
echo "XYZ                : ${XYZ_FILE}"
echo "Reference Molden   : ${REFERENCE_MOLDEN:-NONE}"
echo "Method             : ${METHOD}"
echo "States             : ${STATES}"
echo "SOC                : ${SOC} (${SOC_TWO_ELECTRON})"
echo "External SOC module: ${SOC_EXTERNAL_MODULE:-NONE}"
echo "PCM epsilon        : ${PCM_EPSILON:-NONE}"
echo "Theory             : ${XC}/${BASIS}"
echo "Integral backend   : ${INTEGRAL_BACKEND}"
echo "Auxiliary basis    : ${AUXBASIS}"
echo "SCF strategy       : ${SCF_STRATEGY}"
echo "Scratch            : ${RUN_DIR}"
echo "Final output       : ${FINAL_DIR}"
df -h "${SCRATCH_BASE}"

python - <<'PYCODE'
import pyscf
import pyscf_emrsf

print("PySCF:", pyscf.__version__)
print("PySCF-EMRSF:", pyscf_emrsf.__version__)
print("Loaded from:", pyscf_emrsf.__file__)
if pyscf_emrsf.__version__ != "0.8.0":
    raise SystemExit("PySCF-EMRSF v0.8.0 is required")
PYCODE

command=(
    python -u "${LOCAL_DRIVER}" "${LOCAL_XYZ}"
    --method "${METHOD}"
    --states "${STATES}"
    --unit Angstrom
    --charge "${CHARGE}"
    --spin "${SPIN}"
    --basis "${BASIS}"
    --xc "${XC}"
    --nstates "${NSTATES}"
    --grid-level "${GRID_LEVEL}"
    --integral-backend "${INTEGRAL_BACKEND}"
    --scf-strategy "${SCF_STRATEGY}"
    --scf-max-cycle "${SCF_MAX_CYCLE}"
    --scf-initial-max-cycle "${SCF_INITIAL_MAX_CYCLE}"
    --scf-stabilization-max-cycle "${SCF_STABILIZATION_MAX_CYCLE}"
    --scf-newton-max-cycle "${SCF_NEWTON_MAX_CYCLE}"
    --scf-damping "${SCF_DAMPING}"
    --scf-level-shift "${SCF_LEVEL_SHIFT}"
    --scf-coarse-grid-level "${SCF_COARSE_GRID_LEVEL}"
    --conv-tol 1e-10
    --residual-tol 1e-7
    --threads "${SLURM_CPUS_PER_TASK}"
    --memory-mb "${MEMORY_MB}"
    --output-dir "${RUN_OUTPUT}"
)
if [[ "${INTEGRAL_BACKEND}" == "df" && -n "${AUXBASIS}" ]]; then
    command+=(--auxbasis "${AUXBASIS}")
fi
if [[ -n "${REFERENCE_MOLDEN}" ]]; then
    command+=(--reference-molden "${LOCAL_MOLDEN}")
fi
if [[ "${WRITE_OBSERVABLES}" == "1" ]]; then
    command+=(--write-observables --observables-grid-level "${OBSERVABLES_GRID_LEVEL}")
fi
if [[ "${OBSERVABLES_UNPRUNED_GRID}" == "1" ]]; then
    command+=(--observables-unpruned-grid)
fi
if [[ "${OSCILLATOR_STRENGTHS}" != "1" ]]; then
    command+=(--no-oscillator-strengths)
fi
if [[ "${SAVE_VECTORS}" == "1" ]]; then
    command+=(--save-vectors --eigenvector-threshold "${EIGENVECTOR_THRESHOLD}")
fi
if [[ -n "${PCM_EPSILON}" ]]; then
    command+=(
        --pcm-epsilon "${PCM_EPSILON}"
        --pcm-method "${PCM_METHOD}"
        --pcm-lebedev-order "${PCM_LEBEDEV_ORDER}"
        --pcm-vdw-scale "${PCM_VDW_SCALE}"
        --pcm-probe-radius "${PCM_PROBE_RADIUS}"
        --pcm-surface "${PCM_SURFACE}"
    )
fi
if [[ "${SOC}" == "1" ]]; then
    command+=(--soc)
fi
if [[ -n "${SOC_EXTERNAL_SOURCE}" ]]; then
    command+=(--soc-external-module "${LOCAL_SOC_MODULE}")
fi
if [[ "${SOC}" == "1" || -n "${SOC_EXTERNAL_SOURCE}" ]]; then
    command+=(--soc-two-electron "${SOC_TWO_ELECTRON}")
fi

"${command[@]}" 2>&1 | tee "${RUN_DIR}/calculation.stdout.log"
