#!/bin/bash
# Submit the DALEC2 and evergreen calibrations side by side, then the comparison,
# which waits for both to succeed (--dependency=afterok). Only submits; nothing
# runs on the login node. Works from any directory: it moves to the repository
# root first, because the job scripts use paths relative to it.
#
#   jobs/submit_comparison.sh                 # hemisurface
#   jobs/submit_comparison.sh projected       # the same pair on the other convention

set -euo pipefail

cd "$(dirname "$0")/.."
CONVENTION="${1:-hemisurface}"

# scripts/04_calibrate.py refuses to overwrite a finished trace. Check here too,
# so a clash fails now rather than after hours in the queue -- and so the
# comparison is not left waiting on a job that was always going to stop.
for VARIANT in dalec2 evergreen; do
    TRACE="results/calibration_${VARIANT}_${CONVENTION}.nc"
    if [ -e "${TRACE}" ]; then
        echo "refusing to submit: ${TRACE} already exists. Move it aside first." >&2
        exit 1
    fi
done

# --parsable prints "jobid" or "jobid;cluster"; the dependency needs the id alone.
DALEC2_JOB=$(sbatch --parsable jobs/calibrate_dalec2.sbatch "${CONVENTION}")
DALEC2_JOB=${DALEC2_JOB%%;*}
EVERGREEN_JOB=$(sbatch --parsable jobs/calibrate_evergreen.sbatch "${CONVENTION}")
EVERGREEN_JOB=${EVERGREEN_JOB%%;*}
COMPARE_JOB=$(sbatch --parsable \
    --dependency=afterok:${DALEC2_JOB}:${EVERGREEN_JOB} \
    jobs/compare_structures.sbatch "${CONVENTION}")
COMPARE_JOB=${COMPARE_JOB%%;*}

echo "convention   ${CONVENTION}"
echo "dalec2       ${DALEC2_JOB}"
echo "evergreen    ${EVERGREEN_JOB}"
echo "comparison   ${COMPARE_JOB}   (starts when both succeed)"
