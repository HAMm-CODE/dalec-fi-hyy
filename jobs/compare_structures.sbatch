#!/bin/bash
#SBATCH --job-name=compare-structures
#SBATCH --account=project_2020170
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/compare_structures-%j.out
#SBATCH --error=jobs/logs/compare_structures-%j.err
#
# DALEC2 against the evergreen DALEC, from the two finished traces. Samples
# nothing; forward-runs a thinned set of posterior draws. Settings and
# environment as jobs/calibrate.sbatch, except the 30 min limit from
# jobs/timing.sbatch and a compile cache of its own.
#
#   sbatch jobs/compare_structures.sbatch          # hemisurface (default)
#
# jobs/submit_comparison.sh submits this with --dependency=afterok on both
# calibration jobs, so it only starts once both traces exist.
#
# Preconditions:
#   - results/calibration_dalec2_<convention>.nc and
#     results/calibration_evergreen_<convention>.nc
#   - data/processed/fi_hyy_calibration_1997_2010.nc and
#     data/processed/fi_hyy_prediction_2011_2014.nc

set -euo pipefail

CONVENTION="${1:-hemisurface}"

mkdir -p jobs/logs results/comparison

export PATH=/projappl/project_2020170/dalec-env/bin:"$PATH"

CACHE_DIR=/scratch/project_2020170/dalec-cache/${SLURM_JOB_ID:-local}
export NUMBA_CACHE_DIR="${CACHE_DIR}/numba"
export PYTENSOR_FLAGS="base_compiledir=${CACHE_DIR}/pytensor"
mkdir -p "$NUMBA_CACHE_DIR" "${CACHE_DIR}/pytensor"
trap 'rm -rf "${CACHE_DIR}"' EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== job ==="
echo "job id       ${SLURM_JOB_ID:-<none>}"
echo "convention   ${CONVENTION}"
echo "which python $(command -v python)"
echo

python scripts/compare_structures.py \
    "results/calibration_dalec2_${CONVENTION}.nc" \
    "results/calibration_evergreen_${CONVENTION}.nc"
