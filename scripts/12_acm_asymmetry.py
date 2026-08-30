#!/usr/bin/env python
"""Two structural tests prompted by Kolari (2010), Dissertationes Forestales 99.

**Item 3 -- the spring/autumn asymmetry.** Kolari reports October GPP at SMEAR II
averaging 76% of April GPP over 1997-2007, with nearly equal monthly mean
temperatures (3.5 degC April, 4.1 degC October). The asymmetry comes from
delayed temperature acclimation of photosynthetic capacity, which ACM does not
represent. ACM sees two months of similar temperature and similar day length and
should therefore predict a ratio near one.

This computes the modelled April/October GPP ratio and compares it against 0.76,
and against the ratio in the site's own partitioned products as an independent
check on the 0.76 itself.

**Item 4 -- the 2002 thinning.** The stand was thinned in early 2002, inside the
calibration window, removing about 19% of leaf area. DALEC has no management
event. This checks whether the modelled foliar trajectory shows anything at all
around 2002 -- it should not, and its not doing so is the limitation.

Usage
-----
    python scripts/12_acm_asymmetry.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
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
from dalec.diagnostics import (  # noqa: E402
    classify_prior_draw,
    sample_reparameterised_parameters,
)
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
DEFAULT_DRAWS = 150

#: Kolari (2010): October GPP as a fraction of April GPP, 1997-2007.
KOLARI_AUTUMN_SPRING_RATIO = 0.76

#: Kolari (2010) monthly mean temperatures, degrees C.
KOLARI_APRIL_T, KOLARI_OCTOBER_T = 3.5, 4.1

#: The stand was thinned in early 2002.
THINNING_YEAR = 2002


def month_of(block: SiteData) -> np.ndarray:
    return pd.DatetimeIndex(block.time).month.to_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    return parser.parse_args()


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
    month = month_of(block)
    years = pd.DatetimeIndex(block.time).year.to_numpy()
    april, october = month == 4, month == 10

    bar = "=" * 74
    print(bar)
    print("  Item 3 -- the ACM spring/autumn asymmetry test")
    print(bar)
    print(f"  block {calibration[0]}-{calibration[1]}, {args.draws} draws, seed {seed}")
    print(f"  Kolari (2010): October GPP = {KOLARI_AUTUMN_SPRING_RATIO:.2f} x April GPP")
    print(f"  at {KOLARI_APRIL_T} degC April against {KOLARI_OCTOBER_T} degC October")

    print("\n  Drivers in this block, to confirm the two months are comparable:")
    for label, mask in (("April", april), ("October", october)):
        print(f"    {label:<8} TA_F {block.t_air[mask].mean():5.2f} degC   "
              f"SW_IN {block.sw_in[mask].mean():6.2f}   "
              f"n = {int(mask.sum())} days")

    # -- the site's own partitioned products, as a check on the 0.76 -----------
    print("\n  Independent check on 0.76 from the site's partitioned products:")
    for name in ("gpp_nt", "gpp_dt"):
        values = np.asarray(block.partitioned[name], dtype=float)
        ratio = float(np.nanmean(values[october]) / np.nanmean(values[april]))
        print(f"    {name}   April {np.nanmean(values[april]):5.2f}   "
              f"October {np.nanmean(values[october]):5.2f}   ratio {ratio:.3f}")

    # -- the model ------------------------------------------------------------
    params, _ = sample_reparameterised_parameters(
        args.draws, rng=np.random.default_rng(seed), t_air=block.t_air
    )
    ratios, foliar = [], []
    for parameters in params:
        output = run_dalec2(
            parameters, block, gpp_fn=acm, phenology_fn=dalec2_phenology
        )
        if classify_prior_draw(output) is not None:
            continue
        spring = float(output.gpp[april].mean())
        autumn = float(output.gpp[october].mean())
        if spring > 1e-9:
            ratios.append(autumn / spring)
        # pools carries the initial state at index 0, so it is n_days + 1
        # long; drop it to align with the daily driver series.
        foliar.append(output.pools[1:, 1])

    ratios = np.asarray(ratios)
    q = np.percentile(ratios, [5, 25, 50, 75, 95])
    print("\n  Modelled October / April GPP ratio:")
    print(f"    median {q[2]:.3f}    IQR {q[1]:.3f} to {q[3]:.3f}"
          f"    5-95% {q[0]:.3f} to {q[4]:.3f}")
    print(f"    measured (Kolari)  {KOLARI_AUTUMN_SPRING_RATIO:.3f}")
    print(f"    modelled / measured = {q[2] / KOLARI_AUTUMN_SPRING_RATIO:.2f}x")
    print(f"    draws below the measured ratio: "
          f"{100 * np.mean(ratios < KOLARI_AUTUMN_SPRING_RATIO):.1f}%")

    # -- item 4: the 2002 thinning -------------------------------------------
    print("\n" + bar)
    print("  Item 4 -- does the modelled foliage notice the 2002 thinning?")
    print(bar)
    stack = np.vstack(foliar)
    annual = pd.DataFrame(
        {"year": years, "c_fol": np.median(stack, axis=0)}
    ).groupby("year")["c_fol"].mean()
    print("  median modelled c_fol by year, g C m-2:")
    for year, value in annual.items():
        mark = "   <- thinning, ~19% of leaf area removed" if year == THINNING_YEAR else ""
        print(f"    {year}   {value:8.1f}{mark}")
    before = annual.loc[: THINNING_YEAR - 1]
    after = annual.loc[THINNING_YEAR:]
    step = float(after.iloc[0] - before.iloc[-1]) if len(before) and len(after) else np.nan
    typical = float(np.abs(np.diff(annual.to_numpy())).mean())
    print(f"\n  change into {THINNING_YEAR}: {step:+.1f} g C m-2")
    print(f"  mean absolute year-to-year change: {typical:.1f} g C m-2")
    print(f"  ratio {abs(step) / typical:.2f}  "
          "(a real 19% step would stand well clear of the background)")
    print(f"  a 19% drop would be about "
          f"{-0.19 * float(before.iloc[-1]):+.1f} g C m-2")

    frame = pd.DataFrame({"october_over_april": ratios})
    frame.to_csv(out_dir / "acm_asymmetry.csv", index=False)
    print(f"\n  -> {out_dir / 'acm_asymmetry.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
