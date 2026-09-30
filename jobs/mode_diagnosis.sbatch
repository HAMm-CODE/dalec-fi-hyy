#!/bin/bash
#SBATCH --job-name=mode-diagnosis
#SBATCH --account=project_2020170
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/mode_diagnosis-%j.out
#SBATCH --error=jobs/logs/mode_diagnosis-%j.err
#
# Per-chain diagnosis of the two finished traces: which chains group together,
# and how each chain scores on log posterior, held-out skill and annual fluxes.
# Samples nothing; forward-runs the same thinned posterior draws as the
# comparison. Settings and environment copied from jobs/compare_structures.sbatch.
#
#   sbatch jobs/mode_diagnosis.sbatch              # hemisurface (default)
#
# Preconditions:
#   - results/calibration_dalec2_<convention>.nc and
#     results/calibration_evergreen_<convention>.nc
#   - data/processed/fi_hyy_calibration_1997_2010.nc and
#     data/processed/fi_hyy_prediction_2011_2014.nc
#   - results/comparison/skill.csv from jobs/compare_structures.sbatch, if the
#     report is to check its pooled skill against it (optional)

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

python scripts/26_mode_diagnosis.py \
    "results/calibration_dalec2_${CONVENTION}.nc" \
    "results/calibration_evergreen_${CONVENTION}.nc"
