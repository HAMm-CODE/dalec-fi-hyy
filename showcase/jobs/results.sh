#!/bin/bash
#SBATCH --job-name=showcase-results
#SBATCH --account=project_2020170
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=showcase/outputs/%x-%j.out
#SBATCH --error=showcase/outputs/%x-%j.err
#
# Figures and tables from existing posteriors; samples nothing. Settings and
# environment as jobs/calibrate.sbatch, except the 30 min limit from jobs/timing.sbatch.
# Submit from the repository root:
#
#   sbatch showcase/jobs/results.sh                          # the defaults below
#   sbatch showcase/jobs/results.sh <posterior.nc> <twin.nc>

set -euo pipefail

POSTERIOR="${1:-results/calibration_hemisurface.nc}"     # from the full pipeline
TWIN="${2:-showcase/outputs/twin_posterior.nc}"          # skipped if absent

mkdir -p showcase/outputs

export PATH=/projappl/project_2020170/dalec-env/bin:"$PATH"

export NUMBA_CACHE_DIR=/scratch/project_2020170/dalec-numba-cache
mkdir -p "$NUMBA_CACHE_DIR"

export PYTENSOR_FLAGS="base_compiledir=/scratch/project_2020170/dalec-pytensor-cache"
mkdir -p /scratch/project_2020170/dalec-pytensor-cache

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== job ==="
echo "job id       ${SLURM_JOB_ID:-<none>}"
echo "posterior    ${POSTERIOR}"
echo "twin         ${TWIN}"
echo "which python $(command -v python)"
echo

python showcase/make_results.py "${POSTERIOR}" "${TWIN}"
