#!/usr/bin/env python
"""Source the allocation Dirichlet concentration instead of judging it.

The concentration was set at 20 as a judgement -- "informative without pinning
allocation to a point". It carries **69% of the variance** in the foliage
sustainability condition and is the last unsourced prior in the model.

Route: the measured foliar allocation share is litterfall / GPP, and Ilvesniemi
et al. (2009) Fig. 6 gives ranges for both -- above-ground needle litter 142-204
and GPP 952-1104 g C m-2 yr-1. Propagating those gives an uncertainty on the
share, and the concentration follows from matching it.

    Dirichlet marginal of a component group is Beta(a_group, a0 - a_group)
    Var(w) = p (1 - p) / (a0 + 1)      =>      a0 = p (1 - p) / Var(w) - 1

**The collapse rate is a consequence, not a target.** Nothing here is tuned
toward it, and if the sourced concentration comes out below 20 it is adopted
anyway and the higher collapse rate reported.

Usage
-----
    python scripts/17_dirichlet_concentration.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

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
from dalec.parameters import (  # noqa: E402
    ALLOCATION_CONCENTRATION,
    ALLOCATION_WEIGHT_ORDER,
    DAYS_PER_YEAR,
    MEASURED_ALLOCATION_G_C_M2,
    NEEDLE_LITTERFALL_G_C_M2,
    allocation_concentration,
    canopy_bounds,
    prior_bounds,
)

REPORT_DIR = Path("reports/prior_diagnostics")

#: Ilvesniemi Fig. 6 above-ground needle litter, g C m-2 yr-1.
LITTERFALL_RANGE = (NEEDLE_LITTERFALL_G_C_M2[0], NEEDLE_LITTERFALL_G_C_M2[2])

#: Ilvesniemi Fig. 6 GPP (EC), g C m-2 yr-1.
GPP_RANGE = (952.0, 1104.0)

#: LAI at which the sustainability condition is evaluated.
EVAL_LAI = 0.10
LAI_STEP = 0.02

DEFAULT_DRAWS = 20000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    return parser.parse_args()


def concentration_from_variance(mean_weight: float, variance: float) -> float:
    """Total Dirichlet concentration reproducing a component-group variance."""
    if not 0.0 < mean_weight < 1.0:
        raise ValueError("mean weight must lie strictly in (0, 1)")
    if variance <= 0.0:
        raise ValueError("variance must be positive")
    return mean_weight * (1.0 - mean_weight) / variance - 1.0


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
    print("  Sourcing the allocation Dirichlet concentration")
    print(bar)

    low_fall, high_fall = LITTERFALL_RANGE
    mid_fall = NEEDLE_LITTERFALL_G_C_M2[1]
    low_gpp, high_gpp = GPP_RANGE
    mid_gpp = 0.5 * (low_gpp + high_gpp)

    share_low = low_fall / high_gpp
    share_high = high_fall / low_gpp
    share_mid = mid_fall / mid_gpp
    print(f"  litterfall  {low_fall:.0f}-{high_fall:.0f} (mean {mid_fall:.0f})"
          "  g C m-2 yr-1   Fig. 6")
    print(f"  GPP         {low_gpp:.0f}-{high_gpp:.0f} (mid {mid_gpp:.0f})"
          "   g C m-2 yr-1   Fig. 6")
    print("\n  foliar share of GPP = litterfall / GPP")
    print(f"    low   {low_fall:.0f}/{high_gpp:.0f} = {share_low:.4f}")
    print(f"    mid   {mid_fall:.0f}/{mid_gpp:.0f} = {share_mid:.4f}")
    print(f"    high  {high_fall:.0f}/{low_gpp:.0f} = {share_high:.4f}")

    # -- the Dirichlet is over shares of NPP, so divide out (1 - f_auto) ------
    f_auto_low, f_auto_high = prior_bounds("f_auto")
    npp_fraction = 1.0 - 0.5 * (f_auto_low + f_auto_high)
    total = sum(MEASURED_ALLOCATION_G_C_M2.values())
    mean_weight = MEASURED_ALLOCATION_G_C_M2["foliage"] / total
    print("\n  the simplex is over shares of NPP, so the GPP share is divided by")
    print(f"  (1 - f_auto) = {npp_fraction:.3f} at the prior mean")
    print(f"    measured foliar weight on the simplex: {mean_weight:.4f}")

    print("\n" + bar)
    print("  Implied concentration under two readings of the measured range")
    print(bar)
    readings = {}
    span = share_high - share_low
    for name, sigma_share in (
        ("range as full support (uniform)", span / np.sqrt(12.0)),
        ("range as a 95% interval (+/- 2 sd)", span / 4.0),
    ):
        sigma_weight = sigma_share / npp_fraction
        alpha0 = concentration_from_variance(mean_weight, sigma_weight**2)
        readings[name] = alpha0
        print(f"  {name:<36} sd(share) {sigma_share:.5f}  ->  a0 = {alpha0:6.1f}")

    adopted = min(readings.values())
    print(f"\n  ADOPTED: a0 = {adopted:.1f}, the more conservative reading.")
    print("  The Fig. 6 range is interannual and spatial spread, not a stated")
    print("  confidence interval, and the quantity a fixed parameter needs is the")
    print("  uncertainty in the LONG-RUN MEAN, which is narrower. Treating the")
    print("  full range as the support therefore errs toward a wider prior.")
    print(f"  Judgement value was {ALLOCATION_CONCENTRATION:.0f}; sourced value is "
          f"{adopted / ALLOCATION_CONCENTRATION:.1f}x that.")

    # -- what it does to the allocation share and the collapse rate ----------
    print("\n" + bar)
    print("  Consequences -- reported, not targeted")
    print(bar)
    rng = np.random.default_rng(seed)
    n = args.draws
    lma_low, lma_high = canopy_bounds()["lma"]
    clf_low, clf_high = canopy_bounds()["c_lf"]
    ceff_low, ceff_high = canopy_bounds()["ceff"]
    lma = rng.uniform(lma_low, lma_high, n)
    c_lf = rng.uniform(clf_low, clf_high, n)
    ceff = rng.uniform(ceff_low, ceff_high, n)
    f_auto = rng.uniform(f_auto_low, f_auto_high, n)

    slope_lookup = {}
    for label, alpha0 in (("judgement (20)", ALLOCATION_CONCENTRATION),
                          (f"sourced ({adopted:.0f})", adopted)):
        base = allocation_concentration()
        weights = rng.dirichlet(base / base.sum() * alpha0, size=n)
        lab = weights[:, ALLOCATION_WEIGHT_ORDER.index("f_lab")]
        fol = weights[:, ALLOCATION_WEIGHT_ORDER.index("f_fol")]
        share = (1.0 - f_auto) * (lab + fol)
        q = np.percentile(share, [2.5, 25, 50, 75, 97.5])
        print(f"\n  {label}")
        print(f"    foliar share of GPP  median {q[2]:.4f}   IQR {q[1]:.4f}-{q[3]:.4f}"
              f"   95% {q[0]:.4f}-{q[4]:.4f}")
        print(f"    measured {share_mid:.4f}, range {share_low:.4f}-{share_high:.4f}")
        inside = np.mean((share >= share_low) & (share <= share_high))
        print(f"    inside the measured range: {100 * inside:.1f}%")

        # sustainability condition, sharing one slope sample across both
        key = "slope"
        if key not in slope_lookup:
            sample = np.linspace(0, n - 1, 400).astype(int)
            slopes = np.empty(n)
            step = EVAL_LAI * LAI_STEP
            computed = {}
            for index in sample:
                terms_hi = acm.terms(
                    doy=block.doy, t_max=block.t_max, t_min=block.t_min,
                    sw_in=block.sw_in, co2=block.co2,
                    c_fol=np.full(block.n_days, (EVAL_LAI + step) * lma[index]),
                    lma=lma[index], ceff=ceff[index],
                )
                terms_lo = acm.terms(
                    doy=block.doy, t_max=block.t_max, t_min=block.t_min,
                    sw_in=block.sw_in, co2=block.co2,
                    c_fol=np.full(block.n_days, (EVAL_LAI - step) * lma[index]),
                    lma=lma[index], ceff=ceff[index],
                )
                computed[index] = (
                    (np.mean(terms_hi["gpp"]) - np.mean(terms_lo["gpp"]))
                    * DAYS_PER_YEAR / (2.0 * step)
                )
            # nearest-sample slope; the slope varies slowly with ceff and lma
            keys = np.array(sorted(computed))
            values = np.array([computed[k] for k in keys])
            slopes = np.interp(np.arange(n), keys, values)
            slope_lookup[key] = slopes
        slopes = slope_lookup[key]

        ratio = share * slopes / (lma * c_lf)
        collapse = float(np.mean(ratio < 1.0))
        print(f"    collapse rate (R < 1):     {100 * collapse:.1f}%")

    print("\n  Reference: the previously reported collapse rate was 64.2%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
