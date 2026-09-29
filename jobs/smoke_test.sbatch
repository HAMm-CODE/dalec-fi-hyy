#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --account=project_2020170
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/smoke_test-%j.out
#SBATCH --error=jobs/logs/smoke_test-%j.err
#
# Smoke test of both model variants, before committing 12 h to the real
# calibrations. Settings and environment as jobs/calibrate.sbatch, except the
# 30 min limit and a compile cache of its own.
#
#   sbatch jobs/smoke_test.sbatch
#
# For each of dalec2 and evergreen:
#   1. scripts/25_evergreen_prior_predictive.py with 20 draws
#   2. scripts/04_calibrate.py with 2 chains, 50 tune, 50 draws
#
# The numbers mean nothing -- 100 draws is not a posterior. What it checks is
# that each variant builds, compiles, has a finite initial logp, samples and
# writes a trace. The two variants run side by side, two CPUs each: one after
# the other would not fit in 30 min at the real run's ~7 s per iteration.
#
# Nothing is written into results/ or reports/. Every output goes under
# /scratch/project_2020170/dalec-smoke/<job id>/, which is kept for inspection;
# delete it when done. Each variant's full log is <variant>.log there.

set -euo pipefail

CONVENTION=hemisurface
PRIOR_DRAWS=20
VARIANTS=(dalec2 evergreen)
SMOKE_DIR=/scratch/project_2020170/dalec-smoke/${SLURM_JOB_ID:-local}

mkdir -p jobs/logs "${SMOKE_DIR}"

# --- Tykky container, before anything else (see jobs/calibrate.sbatch) --------
export PATH=/projappl/project_2020170/dalec-env/bin:"$PATH"

# --- Compile caches of this job's own, one subfolder per variant --------------
# The two variants compile at the same time, so they do not share one either.
CACHE_DIR=/scratch/project_2020170/dalec-cache/${SLURM_JOB_ID:-local}
mkdir -p "${CACHE_DIR}"
trap 'rm -rf "${CACHE_DIR}"' EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== job ==="
echo "job id       ${SLURM_JOB_ID:-<none>}"
echo "convention   ${CONVENTION}"
echo "cpus         ${SLURM_CPUS_PER_TASK:-?}"
echo "outputs      ${SMOKE_DIR}"
echo "which g++    $(command -v g++ || echo '<not found>')"
echo "which python $(command -v python)"
echo

# Both steps for one variant. Returns non-zero at the first step that fails.
smoke() {
    local variant=$1
    local out="${SMOKE_DIR}/${variant}"
    export NUMBA_CACHE_DIR="${CACHE_DIR}/${variant}/numba"
    export PYTENSOR_FLAGS="base_compiledir=${CACHE_DIR}/${variant}/pytensor"
    mkdir -p "${out}" "${NUMBA_CACHE_DIR}" "${CACHE_DIR}/${variant}/pytensor"

    echo "--- ${variant}: prior predictive, ${PRIOR_DRAWS} draws ---"
    python scripts/25_evergreen_prior_predictive.py \
        --model-variant "${variant}" \
        --convention "${CONVENTION}" \
        --draws "${PRIOR_DRAWS}" \
        --out "${out}/prior_predictive" || return 1

    echo "--- ${variant}: calibration, 2 chains, 50 tune, 50 draws ---"
    python scripts/04_calibrate.py \
        --model-variant "${variant}" \
        --convention "${CONVENTION}" \
        --chains 2 \
        --tune 50 \
        --draws 50 \
        --cores 2 \
        --outdir "${out}" || return 1

    test -s "${out}/calibration_${variant}_${CONVENTION}.nc"
}

declare -A PIDS STATUS

# If the time limit is reached, Slurm sends TERM first: say which variants
# never finished rather than ending without a verdict.
on_timeout() {
    echo
    echo "=== smoke test: TIME LIMIT REACHED ==="
    for variant in "${VARIANTS[@]}"; do
        echo "SMOKE ${variant}: ${STATUS[${variant}]:-FAILED (did not finish in time)}"
    done
    exit 1
}
trap on_timeout TERM

for variant in "${VARIANTS[@]}"; do
    smoke "${variant}" > "${SMOKE_DIR}/${variant}.log" 2>&1 &
    PIDS[${variant}]=$!
done

for variant in "${VARIANTS[@]}"; do
    if wait "${PIDS[${variant}]}"; then
        STATUS[${variant}]=PASSED
    else
        STATUS[${variant}]=FAILED
    fi
done

for variant in "${VARIANTS[@]}"; do
    echo "=== ${variant}: last lines of ${SMOKE_DIR}/${variant}.log ==="
    tail -n 30 "${SMOKE_DIR}/${variant}.log"
    echo
done

echo "=== smoke test ==="
FAILED=0
for variant in "${VARIANTS[@]}"; do
    echo "SMOKE ${variant}: ${STATUS[${variant}]}"
    [ "${STATUS[${variant}]}" = PASSED ] || FAILED=1
done
exit "${FAILED}"
