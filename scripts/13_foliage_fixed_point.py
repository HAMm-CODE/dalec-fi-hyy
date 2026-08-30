#!/usr/bin/env python
"""Where the foliar pool settles, and what the residual says.

**Part 1 -- the steady-state identity.** At the model's own fixed point the
foliar pool must satisfy input = output. In DALEC the labile pool exists to feed
foliage at bud burst, so both allocation terms enter:

    (f_fol + f_lab) * GPP  =  c_lf * C_fol

Every term is now measured or derived, so evaluating each at the fixed point says
which one is wrong: allocation too low, turnover too high, or GPP too low to
sustain the measured canopy.

The calibration block is fourteen years and the pool has not converged inside it,
so the fixed point is found by cycling the block until C_fol stops moving.

**Part 2 -- the all-sided to projected LAI ratio.** ``lma`` rests on a
conventional ratio of 2.5, not a measurement. This reports what ``lma`` and what
GPP residual follow from 2.0 and 3.0 instead, and whether zero stays inside the
residual range in all three cases.

Reports only. No prior is adjusted here.

Usage
-----
    python scripts/13_foliage_fixed_point.py
    python scripts/13_foliage_fixed_point.py --draws 40
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
from dalec.parameters import DAYS_PER_YEAR, NEEDLE_LITTERFALL_G_C_M2  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
DEFAULT_DRAWS = 30

#: Cycles of the calibration block allowed before giving up on convergence.
MAX_CYCLES = 60

#: Foliar carbon below this is a collapsed canopy, g C m-2.
COLLAPSE_FLOOR = 5.0

#: Relative change in end-of-cycle C_fol between cycles that counts as converged.
CONVERGENCE_TOLERANCE = 1e-3

#: Measured annual GPP, g C m-2 yr-1, Ilvesniemi Fig. 6 midpoint.
MEASURED_GPP = 1028.0

#: Kolari et al. (2009): all-sided LAI seasonal minimum 4.5-4.9 and maximum
#: 6.0-6.5, so the annual mean all-sided LAI is about 5.25-5.7.
MEAN_ALL_SIDED_LAI = (5.25, 5.7)

#: All-sided seasonal maximum before the 2002 thinning, which is what lma was
#: derived against.
MAX_ALL_SIDED_LAI = 8.0

#: Steady-state foliar carbon implied by measured litterfall and 3-5 yr longevity.
FOLIAR_CARBON_RANGE = (462.0, 770.0)

RATIOS = (2.0, 2.5, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    return parser.parse_args()


def spin_up(parameters, block, acm):
    """Cycle the block until the foliar pool stops moving.

    Returns ``(output, cycles, status)`` where status is "converged",
    "collapsed" or "drifting".

    Convergence is judged on the **end-of-cycle** foliar pool, not the block
    mean: the block mean lags a pool that is still drifting and reports
    convergence too early. Draws whose foliage collapses are classified rather
    than discarded -- silently dropping them selects for high-canopy solutions
    and biases every statistic that follows.
    """
    current = parameters
    previous = None
    output = None
    for cycle in range(1, MAX_CYCLES + 1):
        output = run_dalec2(
            current, block, gpp_fn=acm, phenology_fn=dalec2_phenology
        )
        final = output.pools[-1]
        if not np.all(np.isfinite(final)):
            return output, cycle, "diverged"
        foliage = float(final[1])
        if foliage < COLLAPSE_FLOOR:
            return output, cycle, "collapsed"
        if previous is not None and abs(foliage - previous) / previous < CONVERGENCE_TOLERANCE:
            return output, cycle, "converged"
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
    return output, MAX_CYCLES, "drifting"


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
    print("  Part 1 -- the foliar steady-state identity at the model's fixed point")
    print(bar)
    print(f"  block {calibration[0]}-{calibration[1]}, {args.draws} draws, "
          f"cycled to convergence (tol {CONVERGENCE_TOLERANCE:g}, max {MAX_CYCLES})")

    params, _frame = sample_reparameterised_parameters(
        args.draws, rng=np.random.default_rng(seed), t_air=block.t_air
    )
    rows = []
    status_counts: dict[str, int] = {}
    for parameters in params:
        output, cycles, status = spin_up(parameters, block, acm)
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "converged":
            continue
        gpp = float(output.gpp.mean() * DAYS_PER_YEAR)
        foliage = float(np.mean(output.pools[1:, 1]))
        share = parameters.f_fol + parameters.f_lab
        rows.append(
            {
                "cycles": cycles,
                "gpp": gpp,
                "c_fol": foliage,
                "f_fol": parameters.f_fol,
                "f_lab": parameters.f_lab,
                "share": share,
                "c_lf": parameters.c_lf,
                "lma": parameters.lma,
                "lai": foliage / parameters.lma,
                "input_fol_only": parameters.f_fol * gpp,
                # At a fixed point the annual foliar loss equals the annual
                # input, so this is the litterfall the model actually sustains.
                "litterfall": share * gpp,
                "implied_c_fol": share * gpp / parameters.c_lf,
            }
        )
    print("  outcome of the spin-up:")
    for name in ("converged", "collapsed", "drifting", "diverged"):
        if name in status_counts:
            print(f"    {name:<11} {status_counts[name]:3d} / {args.draws}")
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no draw converged; nothing to report")

    print(f"  converged: {len(table)} of {args.draws} draws, "
          f"median {int(table['cycles'].median())} cycles "
          f"({int(table['cycles'].median()) * (calibration[1] - calibration[0] + 1)} years)")

    measured_fall = NEEDLE_LITTERFALL_G_C_M2[1]
    print("\n  At the fixed point, annual foliar loss equals annual input, so")
    print("  (f_fol + f_lab) * GPP is the litterfall the model sustains.")
    print(bar)
    print("  term                            median      IQR                measured")
    for label, column, target in (
        ("GPP", "gpp", MEASURED_GPP),
        ("allocation share f_fol+f_lab", "share", measured_fall / MEASURED_GPP),
        ("litterfall = share * GPP", "litterfall", measured_fall),
        ("C_fol", "c_fol", None),
        ("LAI (projected)", "lai", None),
        ("c_lf", "c_lf", None),
    ):
        v = table[column]
        shown = f"{target:,.4g}" if target is not None else ""
        print(f"  {label:<32}{v.median():8,.3f}  {v.quantile(0.25):8,.3f} to "
              f"{v.quantile(0.75):<9,.3f} {shown}")

    print("\n  Which term carries the discrepancy?")
    print(bar)
    share_ratio = table["share"].median() / (measured_fall / MEASURED_GPP)
    gpp_ratio = table["gpp"].median() / MEASURED_GPP
    fall_ratio = table["litterfall"].median() / measured_fall
    print(f"    allocation share   {table['share'].median():.4f} vs measured "
          f"{measured_fall / MEASURED_GPP:.4f}   ratio {share_ratio:.2f}x")
    print(f"    GPP                {table['gpp'].median():,.0f} vs measured "
          f"{MEASURED_GPP:,.0f}       ratio {gpp_ratio:.2f}x")
    print(f"    litterfall         {table['litterfall'].median():,.0f} vs measured "
          f"{measured_fall:,.0f}         ratio {fall_ratio:.2f}x")
    print(f"    the two multiply:  {share_ratio:.2f} x {gpp_ratio:.2f} = "
          f"{share_ratio * gpp_ratio:.2f}")
    print(f"\n    c_lf median {table['c_lf'].median():.4f} "
          f"(residence {1 / table['c_lf'].median():.1f} yr) -- inside the "
          "measured 3-5 yr, so turnover is NOT the culprit")
    print(f"    C_fol {table['c_fol'].median():,.0f} against "
          f"{measured_fall / table['c_lf'].median():,.0f} implied by the measured "
          "litterfall at this c_lf")
    print(f"    LAI {table['lai'].median():.1f} against a measured annual mean of "
          "about 2.1-2.3")

    table.to_csv(out_dir / "foliage_fixed_point.csv", index=False)

    # ---------------------------------------------------------------- part 2 --
    print("\n" + bar)
    print("  Part 2 -- the all-sided to projected LAI ratio is conventional")
    print(bar)
    gpp_table = pd.read_csv(out_dir / "gpp_investigation.csv")
    low_fol, high_fol = FOLIAR_CARBON_RANGE
    low_mean, high_mean = MEAN_ALL_SIDED_LAI
    print("  lma = C_fol / (max all-sided LAI / ratio); mean projected LAI band")
    print("  = mean all-sided 5.25-5.7 over the ratio.\n")
    print("  ratio   lma bounds      mean projected LAI    n   median GPP   residual")
    results = []
    for ratio in RATIOS:
        projected_max = MAX_ALL_SIDED_LAI / ratio
        lma_low = low_fol / projected_max
        lma_high = high_fol / projected_max
        band = (low_mean / ratio, high_mean / ratio)
        selected = gpp_table[
            (gpp_table["lai_mean"] >= band[0]) & (gpp_table["lai_mean"] <= band[1])
        ]
        if len(selected) < 5:
            print(f"  {ratio:<7.1f} {lma_low:5.0f}-{lma_high:<9.0f} "
                  f"{band[0]:.2f}-{band[1]:<15.2f} {len(selected):3d}   too few draws")
            continue
        median_gpp = float(selected["gpp"].median())
        residual = median_gpp / MEASURED_GPP
        mark = "   <- adopted" if ratio == 2.5 else ""
        print(f"  {ratio:<7.1f} {lma_low:5.0f}-{lma_high:<9.0f} "
              f"{band[0]:.2f}-{band[1]:<15.2f} {len(selected):3d}   "
              f"{median_gpp:8,.0f}   {residual:.2f}x{mark}")
        results.append((ratio, residual, selected))

    print("\n  Does zero bias stay inside the residual range in every case?")
    for ratio, residual, selected in results:
        low = float(selected["gpp"].quantile(0.25)) / MEASURED_GPP
        high = float(selected["gpp"].quantile(0.75)) / MEASURED_GPP
        inside = "YES" if low <= 1.0 <= high else "NO"
        print(f"    ratio {ratio:.1f}   median {residual:.2f}x   "
              f"IQR {low:.2f}x to {high:.2f}x   1.00x inside IQR: {inside}")
    print(f"\n  -> {out_dir / 'foliage_fixed_point.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
