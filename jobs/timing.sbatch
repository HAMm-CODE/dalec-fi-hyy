#!/bin/bash
#SBATCH --job-name=dalec-timing
#SBATCH --output=jobs/logs/timing-%j.out
#SBATCH --error=jobs/logs/timing-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#
# --- FILL THESE IN -----------------------------------------------------------
# Deliberately left unset rather than guessed. A wrong account or partition
# either fails at submit or, worse, runs somewhere that is not the machine the
# numbers are meant to characterise.
##SBATCH --account=project_2020170
##SBATCH --partition=<partition>
# -----------------------------------------------------------------------------
#
# Re-times the FIXED graph at 5113 steps on both backends. See DECISIONS §13.
#
# What this exists to settle:
#   1. ms per forward pass and per gradient, on the graph the sampler runs
#      (NOT timing_spike.py, which still has the closure pattern and is the
#      unfixed graph).
#   2. the gradient-to-forward ratio.
#   3. the wall-clock projection at treedepth 6, 7 and 8 for four chains.
#   4. Numba against the C backend, now that the container supplies a g++ and
#      the C backend genuinely exists here. The pin is NOT changed by this run.
#
# Preconditions:
#   - data/processed/fi_hyy_calibration_1997_2010.nc present on the cluster.
#     Produced by scripts/01_prepare_data.py; it is not in git.
#   - the Tykky environment on PATH, so that `which g++` resolves to the
#     container compiler rather than the system GCC.

set -euo pipefail

mkdir -p jobs/logs

export PATH=/projappl/project_2020170/dalec-env/bin:"$PATH"

echo "=== provenance ==="
echo "host        $(hostname)"
echo "which g++   $(command -v g++ || echo '<not found>')"
echo "g++ version $(g++ --version 2>/dev/null | head -1 || echo '<none>')"
echo "which python $(command -v python)"
echo

# The script prints interpreter architecture and hardware architecture
# separately: the laptop figures were inflated by an emulated x86-64
# interpreter on ARM hardware, and platform.machine() hid it by reporting the
# hardware. If those two lines disagree here, the numbers are not native.
python scripts/21_cluster_timing.py

echo
echo "Report written to reports/sampling/cluster_timing.txt"
