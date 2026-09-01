#!/bin/bash
#SBATCH --job-name=dalec-calib
#SBATCH --account=project_2020170
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=jobs/logs/calibrate-%j.out
#SBATCH --error=jobs/logs/calibrate-%j.err
#
# DALEC2 calibration over 1997-2010, NEE_VUT_REF only. See DECISIONS §1.
#
#   4 chains, 1000 tune, 1000 draws, PyMC defaults throughout.
#   One chain per allocated CPU, which is why --cpus-per-task matches --chains.
#
# The convention is an argument, so the projected run is this same file:
#
#   sbatch jobs/calibrate.sbatch                 # hemisurface (default)
#   sbatch jobs/calibrate.sbatch projected       # the sensitivity run
#
# Both conventions stay live and neither is adopted (DECISIONS §10,
# LIMITATIONS §15); the pair is the sensitivity, so run both and report the
# difference. A GPP number without its convention is not interpretable.
#
# The 12 h wall limit is deliberately generous. The projection is well under an
# hour at treedepth 6 and about 3 h at treedepth 8 (DECISIONS §13), but that
# rescales a laptop measurement by a factor supplied from elsewhere rather than
# resting on a cluster measurement -- which is exactly what this job produces.
#
# Preconditions:
#   - data/processed/fi_hyy_calibration_1997_2010.nc present.
#     Built by scripts/01_prepare_data.py; it is not in git.

set -euo pipefail

CONVENTION="${1:-hemisurface}"

mkdir -p jobs/logs results

# --- Tykky container, before anything else -----------------------------------
# This must come first: it is what makes `which g++` resolve to the container's
# own compiler rather than the system GCC 15.2.0, which in turn is what
# populates pytensor.config.cxx. See DECISIONS §13.
export PATH=/projappl/project_2020170/dalec-env/bin:"$PATH"

# --- Numba cache, somewhere persistent ---------------------------------------
# Without this each of the four workers repays the JIT compile from scratch, and
# every rerun repays it again. The cache must live on a filesystem that survives
# the job, so it goes under /scratch rather than in $TMPDIR.
export NUMBA_CACHE_DIR=/scratch/project_2020170/dalec-numba-cache
mkdir -p "$NUMBA_CACHE_DIR"

# PyTensor's own compile cache benefits from the same treatment.
export PYTENSOR_FLAGS="base_compiledir=/scratch/project_2020170/dalec-pytensor-cache"
mkdir -p /scratch/project_2020170/dalec-pytensor-cache

# Keep the numeric libraries from oversubscribing: four chains share four CPUs,
# so each chain gets one thread and they do not fight each other for it.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG="results/calibrate_${CONVENTION}_${SLURM_JOB_ID:-local}.log"

echo "=== job ==="
echo "job id       ${SLURM_JOB_ID:-<none>}"
echo "convention   ${CONVENTION}"
echo "cpus         ${SLURM_CPUS_PER_TASK:-?}"
echo "numba cache  ${NUMBA_CACHE_DIR}"
echo "which g++    $(command -v g++ || echo '<not found>')"
echo "g++ version  $(g++ --version 2>/dev/null | head -1 || echo '<none>')"
echo "which python $(command -v python)"
echo "log          ${LOG}"
echo

# The script prints interpreter architecture and hardware architecture on
# separate lines, the resolved linker, config.cxx, and one forward pass and one
# gradient at the real step count, all BEFORE sampling starts. That gives the
# cluster re-timing as a by-product of this run rather than as a separate job.
python scripts/04_calibrate.py \
    --convention "${CONVENTION}" \
    --chains 4 \
    --tune 1000 \
    --draws 1000 \
    --cores "${SLURM_CPUS_PER_TASK:-4}" \
    2>&1 | tee "${LOG}"

echo
echo "=== outputs ==="
echo "trace    results/calibration_${CONVENTION}.nc"
echo "summary  results/calibration_${CONVENTION}_summary.csv"
echo "meta     results/calibration_${CONVENTION}_meta.json"
echo "log      ${LOG}"
