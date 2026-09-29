#!/bin/bash
#SBATCH --job-name=calib-evergreen
#SBATCH --account=project_2020170
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/calibrate_evergreen-%j.out
#SBATCH --error=jobs/logs/calibrate_evergreen-%j.err
#
# Evergreen DALEC calibration over 1997-2010, for the structure comparison against
# DALEC2 (docs/evergreen.md). Identical to jobs/calibrate.sbatch except:
#
#   - --model-variant evergreen, so every output is
#     results/calibration_evergreen_<convention>.* and nothing existing is touched;
#   - a PyTensor compile directory and a Numba cache of its own, so it can run
#     beside jobs/calibrate_dalec2.sbatch without either locking the other.
#
#   sbatch jobs/calibrate_evergreen.sbatch         # hemisurface (default)
#   jobs/submit_comparison.sh                      # both models, then the comparison
#
# Preconditions:
#   - data/processed/fi_hyy_calibration_1997_2010.nc present.
#     Built by scripts/01_prepare_data.py; it is not in git.

set -euo pipefail

VARIANT=evergreen
CONVENTION="${1:-hemisurface}"

mkdir -p jobs/logs results

# --- Tykky container, before anything else -----------------------------------
# As jobs/calibrate.sbatch: this is what makes `which g++` resolve to the
# container's own compiler, which populates pytensor.config.cxx. DECISIONS §13.
export PATH=/projappl/project_2020170/dalec-env/bin:"$PATH"

# --- Compile caches of this job's own ----------------------------------------
# One folder per job id, on /scratch rather than $TMPDIR. Two jobs sharing a
# PyTensor compile directory can block on its lock, so each gets its own; the
# price is that each job pays the JIT compile once. Removed when the job ends.
CACHE_DIR=/scratch/project_2020170/dalec-cache/${SLURM_JOB_ID:-local}
export NUMBA_CACHE_DIR="${CACHE_DIR}/numba"
export PYTENSOR_FLAGS="base_compiledir=${CACHE_DIR}/pytensor"
mkdir -p "$NUMBA_CACHE_DIR" "${CACHE_DIR}/pytensor"
trap 'rm -rf "${CACHE_DIR}"' EXIT

# Keep the numeric libraries from oversubscribing: four chains share four CPUs,
# so each chain gets one thread and they do not fight each other for it.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG="results/calibrate_${VARIANT}_${CONVENTION}_${SLURM_JOB_ID:-local}.log"

echo "=== job ==="
echo "job id       ${SLURM_JOB_ID:-<none>}"
echo "variant      ${VARIANT}"
echo "convention   ${CONVENTION}"
echo "cpus         ${SLURM_CPUS_PER_TASK:-?}"
echo "numba cache  ${NUMBA_CACHE_DIR}"
echo "which g++    $(command -v g++ || echo '<not found>')"
echo "g++ version  $(g++ --version 2>/dev/null | head -1 || echo '<none>')"
echo "which python $(command -v python)"
echo "log          ${LOG}"
echo

python scripts/04_calibrate.py \
    --model-variant "${VARIANT}" \
    --convention "${CONVENTION}" \
    --chains 4 \
    --tune 1000 \
    --draws 1000 \
    --cores "${SLURM_CPUS_PER_TASK:-4}" \
    2>&1 | tee "${LOG}"

echo
echo "=== outputs ==="
echo "trace    results/calibration_${VARIANT}_${CONVENTION}.nc"
echo "summary  results/calibration_${VARIANT}_${CONVENTION}_summary.csv"
echo "meta     results/calibration_${VARIANT}_${CONVENTION}_meta.json"
echo "log      ${LOG}"
