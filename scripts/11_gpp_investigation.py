#!/usr/bin/env python
"""Why is prior predictive GPP 2.3x the measured value?

Check 3 of the reparameterised respiration prior failed with a wrong-signed
median annual NEE. The respiration prior was not the cause: the Theta
inconsistency accounted for +19.8 g C m-2 yr-1 of it. Modelled GPP was 2,515
g C m-2 yr-1 against Ilvesniemi et al. (2009) Fig. 6's measured 952-1104.

This script reports the prior predictive GPP distribution against that measured
range and attributes the overestimate to parameters. It proposes nothing that
requires fitting: no parameter is tuned to the measured GPP here, and the
script writes no prior.

Usage
-----
    python scripts/11_gpp_investigation.py
    python scripts/11_gpp_investigation.py --draws 400
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
from dalec.parameters import DAYS_PER_YEAR  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
DEFAULT_DRAWS = 400

#: Ilvesniemi et al. (2009) Fig. 6, GPP (EC), g C m-2 yr-1.
MEASURED_GPP = (952.0, 1104.0)

#: Parameters that can plausibly scale annual GPP, for the attribution.
GPP_DRIVERS = ("ceff", "lma", "f_auto", "c_fol_0", "f_fol", "f_lab", "d_onset",
               "cr_onset", "cr_fall", "c_lf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    config = load_config(args.config)
    out_dir = args.out if args.out is not None else Path(REPORT_DIR)
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
    print("  GPP investigation -- prior predictive against the measured range")
    print(bar)
    print(f"  block {calibration[0]}-{calibration[1]}, {block.n_days} days, "
          f"{args.draws} draws, seed {seed}")

    params, frame = sample_reparameterised_parameters(
        args.draws, rng=np.random.default_rng(seed), t_air=block.t_air
    )
    rows = []
    for index, parameters in enumerate(params):
        output = run_dalec2(
            parameters, block, gpp_fn=acm, phenology_fn=dalec2_phenology
        )
        if classify_prior_draw(output) is not None:
            continue
        lai = output.pools[:, 1] / parameters.lma
        rows.append(
            {
                "draw": index,
                "gpp": float(output.gpp.mean() * DAYS_PER_YEAR),
                "lai_mean": float(np.mean(lai)),
                "lai_max": float(np.max(lai)),
                **{name: float(frame[name].iloc[index]) for name in GPP_DRIVERS},
            }
        )
    table = pd.DataFrame(rows)

    lo_m, hi_m = MEASURED_GPP
    gpp = table["gpp"].to_numpy()
    q = np.percentile(gpp, [5, 25, 50, 75, 95])
    print("\n" + bar)
    print("  Prior predictive annual GPP, g C m-2 yr-1")
    print(bar)
    print(f"  median {q[2]:8,.0f}    IQR {q[1]:,.0f} to {q[3]:,.0f}"
          f"    5-95% {q[0]:,.0f} to {q[4]:,.0f}")
    print(f"  measured (Fig. 6)  {lo_m:,.0f} to {hi_m:,.0f}")
    print(f"  median / measured midpoint = {q[2] / np.mean(MEASURED_GPP):.2f}x")
    print(f"  draws inside the measured range: "
          f"{100 * np.mean((gpp >= lo_m) & (gpp <= hi_m)):.1f}%")
    print(f"  draws above it:                  "
          f"{100 * np.mean(gpp > hi_m):.1f}%")

    print("\n" + bar)
    print("  Attribution -- Spearman correlation with annual GPP")
    print(bar)
    corr = (
        table[[*GPP_DRIVERS, "lai_mean"]]
        .corrwith(table["gpp"], method="spearman")
        .sort_values(key=np.abs, ascending=False)
    )
    for name, value in corr.items():
        print(f"    {name:<12} {value:+.3f}")

    print("\n" + bar)
    print("  Modelled LAI, and what it should be")
    print(bar)
    lai_q = np.percentile(table["lai_mean"], [5, 50, 95])
    print(f"  mean LAI   median {lai_q[1]:.2f}   5-95% {lai_q[0]:.2f} to {lai_q[2]:.2f}")
    print(f"  peak LAI   median {np.median(table['lai_max']):.2f}")
    print("  FI-Hyy projected LAI is about 3 (provisional, DECISIONS.md section 6)")

    print("\n" + bar)
    print("  GPP by ceff decile -- is ceff the lever?")
    print(bar)
    table["ceff_bin"] = pd.qcut(table["ceff"], 5, duplicates="drop")
    for interval, group in table.groupby("ceff_bin", observed=True):
        share = 100 * np.mean(
            (group["gpp"] >= lo_m) & (group["gpp"] <= hi_m)
        )
        print(f"    ceff {interval.left:5.1f}-{interval.right:5.1f}   "
              f"median GPP {group['gpp'].median():7,.0f}   "
              f"inside measured {share:5.1f}%")

    inside = table[(table["gpp"] >= lo_m) & (table["gpp"] <= hi_m)]
    if len(inside) > 5:
        print(f"\n  the {len(inside)} draws that land inside the measured range:")
        for name in ("ceff", "lma", "lai_mean", "f_auto"):
            v = inside[name]
            print(f"    {name:<10} median {v.median():8.2f}   "
                  f"IQR {v.quantile(0.25):.2f} to {v.quantile(0.75):.2f}")

    path = out_dir / "gpp_investigation.csv"
    table.drop(columns=["ceff_bin"]).to_csv(path, index=False)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
