#!/usr/bin/env python
"""The foliage sustainability condition, and what drives the collapse rate.

Foliage at annual resolution obeys

    dC/dt = (f_fol + f_lab) * GPP(C / lma) - c_lf * C

so a canopy can grow from small C when the marginal return on leaf area exceeds
the marginal loss:

    R = (f_fol + f_lab) * (dGPP/dLAI) / (lma * c_lf) > 1

Every term is sourced. Taking logs makes the decomposition exact and additive:

    log R = log(f_fol + f_lab) + log(dGPP/dLAI) - log(lma) - log(c_lf)

so the share of Var(log R) carried by each term says which marginal drives the
60% collapse rate -- a real statement about DALEC's generic priors, or a signal
that one of ours is still too wide.

Reports only. **No constraint is added and no prior is adjusted.**

Usage
-----
    python scripts/16_sustainability_condition.py
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
from dalec.diagnostics import sample_reparameterised_parameters  # noqa: E402
from dalec.parameters import DAYS_PER_YEAR  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
DEFAULT_DRAWS = 3000

#: Leaf area indices at which to evaluate the condition. The small one governs
#: whether a bare canopy can grow at all -- the collapse question. The larger is
#: the site's own annual mean canopy.
EVAL_LAI = (0.10, 2.20)

#: Relative step for the finite difference on LAI.
LAI_STEP = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    return parser.parse_args()


def annual_gpp_at_lai(acm, block, lai, lma, ceff):
    """Annual GPP, g C m-2 yr-1, holding the canopy at a fixed LAI.

    Vectorised over days for one parameter set; ``c_fol = lai * lma`` so the
    canopy is held rather than evolved.
    """
    terms = acm.terms(
        doy=block.doy,
        t_max=block.t_max,
        t_min=block.t_min,
        sw_in=block.sw_in,
        co2=block.co2,
        c_fol=np.full(block.n_days, lai * lma),
        lma=lma,
        ceff=ceff,
    )
    return float(np.mean(terms["gpp"]) * DAYS_PER_YEAR)


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
    print("  The foliage sustainability condition")
    print(bar)
    print("  R = (f_fol + f_lab) * (dGPP/dLAI) / (lma * c_lf),  sustainable if R > 1")
    print(f"  block {calibration[0]}-{calibration[1]}, {args.draws} draws, seed {seed}")

    _params, frame = sample_reparameterised_parameters(
        args.draws, rng=np.random.default_rng(seed), t_air=block.t_air
    )
    share = (frame["f_fol"] + frame["f_lab"]).to_numpy()
    lma = frame["lma"].to_numpy()
    ceff = frame["ceff"].to_numpy()
    c_lf = frame["c_lf"].to_numpy()

    for lai in EVAL_LAI:
        step = lai * LAI_STEP
        slope = np.empty(len(frame))
        for index in range(len(frame)):
            high = annual_gpp_at_lai(acm, block, lai + step, lma[index], ceff[index])
            low = annual_gpp_at_lai(acm, block, lai - step, lma[index], ceff[index])
            slope[index] = (high - low) / (2.0 * step)

        ratio = share * slope / (lma * c_lf)
        sustainable = ratio > 1.0
        print("\n" + bar)
        print(f"  Evaluated at LAI = {lai:.2f}")
        print(bar)
        print(f"    dGPP/dLAI   median {np.median(slope):8.1f}   "
              f"IQR {np.percentile(slope, 25):.1f} to {np.percentile(slope, 75):.1f}")
        print(f"    R           median {np.median(ratio):8.3f}   "
              f"IQR {np.percentile(ratio, 25):.3f} to {np.percentile(ratio, 75):.3f}")
        print(f"    prior mass sustainable (R > 1):  {100 * sustainable.mean():5.1f}%")
        print(f"    prior mass collapsing  (R < 1):  {100 * (~sustainable).mean():5.1f}%")

        # -- exact additive decomposition in logs --------------------------
        logs = {
            "log(f_fol+f_lab)": np.log(share),
            "log(dGPP/dLAI)": np.log(np.maximum(slope, 1e-12)),
            "-log(lma)": -np.log(lma),
            "-log(c_lf)": -np.log(c_lf),
        }
        log_ratio = np.log(np.maximum(ratio, 1e-12))
        total = float(np.var(log_ratio))
        print(f"\n    Var(log R) = {total:.4f}; contribution of each term")
        print("    term                 variance   share   corr with log R")
        rows = []
        for name, values in logs.items():
            variance = float(np.var(values))
            corr = float(np.corrcoef(values, log_ratio)[0, 1])
            contribution = float(np.cov(values, log_ratio)[0, 1]) / total
            rows.append((name, variance, contribution, corr))
        for name, variance, contribution, corr in sorted(
            rows, key=lambda r: -abs(r[2])
        ):
            print(f"    {name:<20} {variance:8.4f}  {100 * contribution:5.1f}%   {corr:+.3f}")
        print("    (shares are cov(term, log R)/Var(log R); they sum to 100% exactly)")
        print(f"    check: shares sum to {100 * sum(r[2] for r in rows):.1f}%")

        if lai == EVAL_LAI[0]:
            table = pd.DataFrame(
                {
                    "share": share,
                    "lma": lma,
                    "ceff": ceff,
                    "c_lf": c_lf,
                    "dgpp_dlai": slope,
                    "R": ratio,
                    "sustainable": sustainable,
                }
            )
            table.to_csv(out_dir / "sustainability_condition.csv", index=False)

            print("\n    Collapse rate by tercile of each marginal:")
            for name, values in (
                ("lma", lma), ("ceff", ceff), ("c_lf", c_lf), ("share", share)
            ):
                edges = np.percentile(values, [33.3, 66.7])
                low = (~sustainable)[values <= edges[0]].mean()
                mid = (~sustainable)[
                    (values > edges[0]) & (values <= edges[1])
                ].mean()
                high = (~sustainable)[values > edges[1]].mean()
                print(f"      {name:<6} low {100 * low:5.1f}%   mid {100 * mid:5.1f}%"
                      f"   high {100 * high:5.1f}%   spread {100 * abs(high - low):5.1f} pts")

    print(f"\n  -> {out_dir / 'sustainability_condition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
