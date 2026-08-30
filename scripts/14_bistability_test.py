#!/usr/bin/env python
"""Is the 24/16/0 split bistability, or monostability split across parameters?

Two readings of the spin-up result in `scripts/13_foliage_fixed_point.py`:

**A -- genuine bistability.** One parameter set has two attractors, and the
measured canopy lies on the separatrix between them.

**B -- monostable per parameter set, split across parameter space.** Each set has
one attractor. 24 draws have foliage loop gain below one and decay to zero, 16
have it above one and grow to saturation, and none has a fixed point at the
measured canopy.

The test distinguishes them directly: take one parameter set from each branch and
integrate it from two very different initial foliar pools, one well below the
measured canopy and one well above.

    same endpoint from both starts  ->  B (single attractor, monostable)
    different endpoints             ->  A (two attractors, bistable)

Reports only. No prior is adjusted.

Usage
-----
    python scripts/14_bistability_test.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    require_year_block,
    resolve_path,
)
from dalec.data_io import SiteData  # noqa: E402
from dalec.diagnostics import sample_reparameterised_parameters  # noqa: E402
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
DEFAULT_DRAWS = 40
MAX_CYCLES = 60
CONVERGENCE_TOLERANCE = 1e-3
COLLAPSE_FLOOR = 5.0

#: Starting foliar pools, g C m-2: well below and well above the measured
#: canopy of roughly 460-770. If a set is monostable both must arrive together.
START_LOW = 25.0
START_HIGH = 2500.0

#: Measured foliar carbon implied by litterfall and 3-5 yr longevity.
MEASURED_C_FOL = (462.0, 770.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    return parser.parse_args()


def integrate(parameters, block, acm, start_foliage=None):
    """Cycle the block to convergence; return (endpoint C_fol, cycles, status)."""
    current = parameters
    if start_foliage is not None:
        current = replace(current, c_fol_0=float(start_foliage))
    previous = None
    for cycle in range(1, MAX_CYCLES + 1):
        output = run_dalec2(
            current, block, gpp_fn=acm, phenology_fn=dalec2_phenology
        )
        final = output.pools[-1]
        if not np.all(np.isfinite(final)):
            return float("nan"), cycle, "diverged"
        foliage = float(final[1])
        if foliage < COLLAPSE_FLOOR:
            return foliage, cycle, "collapsed"
        if previous is not None and abs(foliage - previous) / previous < CONVERGENCE_TOLERANCE:
            return foliage, cycle, "converged"
        previous = foliage
        current = replace(
            current,
            c_lab_0=float(final[0]),
            c_fol_0=foliage,
            c_roo_0=float(final[2]),
            c_woo_0=float(final[3]),
            c_lit_0=float(final[4]),
            c_som_0=float(final[5]),
        )
    return foliage, MAX_CYCLES, "drifting"


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    calibration = require_year_block(config, "calibration")
    site_code = str(config.get("site", {}).get("code", "")).lower().replace("-", "_")
    block = SiteData.load(
        resolve_path(config["paths"]["processed_dir"])
        / f"{site_code}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )
    acm = acm_from_config(config)
    bar = "=" * 74

    print(bar)
    print("  Bistability (A) or monostability split across parameter space (B)?")
    print(bar)
    print(f"  block {calibration[0]}-{calibration[1]}, {args.draws} draws, seed {seed}")
    print(f"  starts: {START_LOW:.0f} and {START_HIGH:.0f} g C m-2 against a "
          f"measured {MEASURED_C_FOL[0]:.0f}-{MEASURED_C_FOL[1]:.0f}")

    params, _frame = sample_reparameterised_parameters(
        args.draws, rng=np.random.default_rng(seed), t_air=block.t_air
    )

    # -- classify each draw from its own derived initial state ---------------
    branches: dict[str, list[int]] = {}
    baseline = []
    for index, parameters in enumerate(params):
        foliage, cycles, status = integrate(parameters, block, acm)
        branches.setdefault(status, []).append(index)
        baseline.append((index, foliage, cycles, status))
    print("\n  classification from the derived initial state:")
    for name, members in sorted(branches.items()):
        print(f"    {name:<11} {len(members):3d}")

    # -- the test -------------------------------------------------------------
    print("\n" + bar)
    print("  The test: same parameter set, two very different starting canopies")
    print(bar)
    rows = []
    for status in ("collapsed", "converged"):
        members = branches.get(status, [])
        if not members:
            print(f"  no draw on the {status} branch; skipping")
            continue
        # a representative from each branch, plus two more for robustness
        for index in members[:3]:
            parameters = params[index]
            low_end, low_cycles, low_status = integrate(
                parameters, block, acm, START_LOW
            )
            high_end, high_cycles, high_status = integrate(
                parameters, block, acm, START_HIGH
            )
            derived = parameters.c_fol_0
            same = (
                low_status == high_status
                and np.isfinite(low_end)
                and np.isfinite(high_end)
                and (
                    abs(low_end - high_end) / max(high_end, 1e-9) < 0.05
                    or (low_status == "collapsed" and high_status == "collapsed")
                )
            )
            rows.append(
                {
                    "draw": index,
                    "branch": status,
                    "c_fol_0_derived": derived,
                    "from_low_end": low_end,
                    "from_low_status": low_status,
                    "from_high_end": high_end,
                    "from_high_status": high_status,
                    "same_endpoint": bool(same),
                }
            )
            print(f"\n  draw {index:3d}  [{status}]   derived c_fol_0 {derived:7.1f}")
            print(f"    from {START_LOW:7.0f}  ->  {low_end:9.1f}  "
                  f"({low_status}, {low_cycles} cycles)")
            print(f"    from {START_HIGH:7.0f}  ->  {high_end:9.1f}  "
                  f"({high_status}, {high_cycles} cycles)")
            print(f"    same endpoint: {'YES -> monostable' if same else 'NO -> two attractors'}")

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no draws to test")

    print("\n" + bar)
    print("  Verdict")
    print(bar)
    same_count = int(table["same_endpoint"].sum())
    print(f"  parameter sets where both starts reach the same endpoint: "
          f"{same_count} / {len(table)}")
    if same_count == len(table):
        print("  -> READING B. Every set tested is MONOSTABLE: one attractor, and")
        print("     the initial condition does not select it. The 24/16 split is")
        print("     across PARAMETER SPACE, not across basins of one parameter set.")
    elif same_count == 0:
        print("  -> READING A. Every set tested has TWO attractors and the initial")
        print("     condition selects between them: genuine bistability.")
    else:
        print("  -> MIXED. Some sets are monostable and some are bistable; report")
        print("     the split rather than a single reading.")

    path = out_dir / "bistability_test.csv"
    table.to_csv(path, index=False)
    print(f"\n  -> {path}")

    # -- what the annual GPP looks like at each endpoint ---------------------
    print("\n  For context, annual GPP at the derived start, by branch:")
    frame = pd.DataFrame(baseline, columns=["draw", "c_fol", "cycles", "status"])
    for status, group in frame.groupby("status"):
        print(f"    {status:<11} n={len(group):3d}   "
              f"median endpoint C_fol {group['c_fol'].median():9.1f} g C m-2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
