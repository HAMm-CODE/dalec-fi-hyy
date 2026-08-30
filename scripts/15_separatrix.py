#!/usr/bin/env python
"""Where does the foliage separatrix sit, relative to the derived c_fol_0 range?

`scripts/14_bistability_test.py` found the 24/16 split is a mixture: the
collapsed branch is monostable at zero, and part of the converged branch is
genuinely bistable with the initial condition selecting the attractor.

For each bistable parameter set this locates the separatrix by bisection on the
initial foliar pool, and compares it against the derived range of 462-770
g C m-2 that `c_fol_0 = 154 / c_lf` spans across the `c_lf` prior.

    separatrix below 462   -> the derived range is entirely on the sustainable
                              side and the sampler never crosses it at the prior
    separatrix inside 462-770 -> moving c_lf walks the initial condition across it

The separatrix location depends on `f_fol`, `lma` and `ceff`, so a clean result
at the prior does not guarantee one at the posterior. Those are reported
alongside.

Reports only. No prior is adjusted.

Usage
-----
    python scripts/15_separatrix.py
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

START_LOW = 25.0
START_HIGH = 2500.0

#: Bisection bracket and precision, g C m-2.
BISECTION_STEPS = 14

#: c_fol_0 = 154 / c_lf across the c_lf prior U(0.20, 0.333).
DERIVED_RANGE = (462.0, 770.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    return parser.parse_args()


def outcome(parameters, block, acm, start_foliage):
    """Return "converged" or "collapsed" for a given initial foliar pool."""
    current = replace(parameters, c_fol_0=float(start_foliage))
    previous = None
    for _cycle in range(1, MAX_CYCLES + 1):
        output = run_dalec2(
            current, block, gpp_fn=acm, phenology_fn=dalec2_phenology
        )
        final = output.pools[-1]
        if not np.all(np.isfinite(final)):
            return "diverged", float("nan")
        foliage = float(final[1])
        if foliage < COLLAPSE_FLOOR:
            return "collapsed", foliage
        if previous is not None and abs(foliage - previous) / previous < CONVERGENCE_TOLERANCE:
            return "converged", foliage
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
    return "drifting", foliage


def bisect(parameters, block, acm, low, high):
    """Narrow the bracket between a collapsing start and a surviving one."""
    for _step in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        status, _ = outcome(parameters, block, acm, middle)
        if status == "converged":
            high = middle
        else:
            low = middle
    return 0.5 * (low + high), high - low


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
    print("  Locating the foliage separatrix")
    print(bar)
    print(f"  block {calibration[0]}-{calibration[1]}, {args.draws} draws, seed {seed}")
    print(f"  derived c_fol_0 range: {DERIVED_RANGE[0]:.0f}-{DERIVED_RANGE[1]:.0f} "
          "g C m-2 (= 154 / c_lf across the c_lf prior)")

    params, _frame = sample_reparameterised_parameters(
        args.draws, rng=np.random.default_rng(seed), t_air=block.t_air
    )

    print("\n  screening for bistable sets (collapse from low, survive from high)...")
    rows = []
    for index, parameters in enumerate(params):
        low_status, _ = outcome(parameters, block, acm, START_LOW)
        if low_status == "converged":
            continue  # survives even from 25: no separatrix above that
        high_status, high_end = outcome(parameters, block, acm, START_HIGH)
        if high_status != "converged":
            continue  # monostable at zero
        separatrix, width = bisect(parameters, block, acm, START_LOW, START_HIGH)
        derived = parameters.c_fol_0
        rows.append(
            {
                "draw": index,
                "separatrix": separatrix,
                "bracket_width": width,
                "c_fol_0_derived": derived,
                "derived_above_separatrix": bool(derived > separatrix),
                "inside_derived_range": bool(
                    DERIVED_RANGE[0] <= separatrix <= DERIVED_RANGE[1]
                ),
                "upper_attractor": high_end,
                "f_fol": parameters.f_fol,
                "f_lab": parameters.f_lab,
                "lma": parameters.lma,
                "ceff": parameters.ceff,
                "c_lf": parameters.c_lf,
            }
        )
        print(f"    draw {index:3d}  separatrix {separatrix:8.1f} "
              f"(+/- {width / 2:.1f})   derived {derived:6.1f}   "
              f"ceff {parameters.ceff:5.2f}  lma {parameters.lma:6.1f}  "
              f"f_fol {parameters.f_fol:.4f}")

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no bistable sets found; nothing to bisect")

    print("\n" + bar)
    print(f"  {len(table)} bistable sets of {args.draws} draws")
    print(bar)
    q = table["separatrix"].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"  separatrix   min {q[0.0]:8.1f}   median {q[0.5]:8.1f}   "
          f"max {q[1.0]:8.1f}")
    print(f"  IQR {q[0.25]:.1f} to {q[0.75]:.1f}")

    lo_d, hi_d = DERIVED_RANGE
    below = int((table["separatrix"] < lo_d).sum())
    inside = int(table["inside_derived_range"].sum())
    above = int((table["separatrix"] > hi_d).sum())
    print(f"\n  relative to the derived range {lo_d:.0f}-{hi_d:.0f}:")
    print(f"    below  {below:3d}   the derived range is entirely sustainable")
    print(f"    inside {inside:3d}   moving c_lf walks the start across it")
    print(f"    above  {above:3d}   the derived range is entirely on the "
          "collapsing side")
    print(f"\n  derived start above its own separatrix: "
          f"{int(table['derived_above_separatrix'].sum())} / {len(table)}")

    print("\n  The separatrix is not a constant -- it moves with the parameters:")
    for name in ("ceff", "lma", "f_fol", "c_lf"):
        if table[name].nunique() > 2:
            corr = table["separatrix"].corr(table[name], method="spearman")
            print(f"    Spearman corr(separatrix, {name:<5}) = {corr:+.3f}")

    path = out_dir / "separatrix.csv"
    table.to_csv(path, index=False)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
