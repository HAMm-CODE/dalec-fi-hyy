#!/bin/bash
#SBATCH --job-name=showcase-twin
#SBATCH --account=project_2020170
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=showcase/outputs/%x-%j.out
#SBATCH --error=showcase/outputs/%x-%j.err
#
# Synthetic twin over 1997-2010. Settings and environment as jobs/calibrate.sbatch.
# Submit from the repository root:
#
#   sbatch showcase/jobs/twin.sh

set -euo pipefail

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
echo "cpus         ${SLURM_CPUS_PER_TASK:-?}"
echo "numba cache  ${NUMBA_CACHE_DIR}"
echo "which g++    $(command -v g++ || echo '<not found>')"
echo "g++ version  $(g++ --version 2>/dev/null | head -1 || echo '<none>')"
echo "which python $(command -v python)"
echo

python showcase/run_twin.py
