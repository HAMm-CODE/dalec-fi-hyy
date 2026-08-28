#!/usr/bin/env python
"""Reparameterised heterotrophic respiration: the three acceptance checks.

Sampled: ``rh_ref``, ``f_som``, ``theta_lit``, ``theta_som``.
Derived: ``c_lit_0 = (1 - f_som) * rh_ref / theta_lit``
         ``c_som_0 = f_som * rh_ref / theta_som``

See DECISIONS.md section 7 for the sources and the derivation. This script only
reports; it changes nothing and fits nothing.

    1. do the derived c_som_0 land in the published 5,000-10,000 g C m-2 range?
    2. Task 1's failure rate and coverage, against 83.7% and 0.651
    3. prior predictive median annual NEE, against the observed value

Usage
-----
    python scripts/10_reparameterised_prior.py
    python scripts/10_reparameterised_prior.py --draws 200
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
    reparameterised_bounds,
    sample_prior_parameters,
    sample_reparameterised_parameters,
)
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402
from dalec.parameters import (  # noqa: E402
    ANNUAL_LITTER_INPUT_G_C_M2,
    ANNUAL_RH_G_C_M2,
    DAYS_PER_YEAR,
    LITTER_RESIDENCE_TIME_YEARS,
    SOIL_CARBON_STOCK_G_C_M2,
    SOM_RESIDENCE_TIME_YEARS,
    decomposition_multiplier,
)

REPORT_DIR = Path("reports/prior_diagnostics")
DEFAULT_DRAWS = 1000
COVERAGE_INTERVAL = (5.0, 95.0)

#: Task 1's numbers, for the comparison the checks are stated against.
TASK1_FAILURE_RATE = 0.837
TASK1_COVERAGE = 0.651

#: Published soil carbon inventory range for the site, g C m-2. Retained as the
#: comparison Check 1 was originally stated against.
SOIL_INVENTORY_RANGE = (5000.0, 10000.0)

#: Large enough that the derived-stock quantiles are not sampling noise; these
#: are closed-form draws, no forward model involved.
ANALYTIC_DRAWS = 60_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def run_block(params, block, acm, n_draws):
    """Forward-run every draw, returning latent NEE for the usable ones."""
    latent = np.full((n_draws, block.n_days), np.nan)
    reasons: list[str | None] = []
    for index, parameters in enumerate(params):
        output = run_dalec2(parameters, block, gpp_fn=acm, phenology_fn=dalec2_phenology)
        reason = classify_prior_draw(output)
        reasons.append(reason)
        if reason is None:
            latent[index] = output.nee
        if (index + 1) % 100 == 0:
            failed = sum(r is not None for r in reasons)
            print(f"    {index + 1:>5} / {n_draws}   failed so far: {failed}")
    ok = np.array([r is None for r in reasons])
    return latent[ok], ok, reasons


def coverage_of(latent, block, rng):
    sigma = np.where(block.nee_mask, block.nee_unc, np.nan)
    noise = np.nan_to_num(sigma, nan=float(np.nanmedian(sigma)))
    observed_level = latent + rng.normal(0.0, noise, size=latent.shape)
    low, high = np.percentile(observed_level, COVERAGE_INTERVAL, axis=0)
    inside = (block.nee_obs >= low) & (block.nee_obs <= high)
    return float(inside[block.nee_mask].mean()), observed_level


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    config = load_config(args.config)
    out_dir = args.out if args.out is not None else Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    calibration = require_year_block(config, "calibration")
    site_code = str(config.get("site", {}).get("code", "")).lower().replace("-", "_")
    block_path = (
        resolve_path(config["paths"]["processed_dir"])
        / f"{site_code}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )
    if not block_path.exists():
        raise SystemExit(f"no processed calibration block at {block_path}")

    block = SiteData.load(block_path)
    acm = acm_from_config(config)
    t_air = block.t_air

    bar = "=" * 74
    print(bar)
    print("  Reparameterised heterotrophic respiration -- three checks")
    print(bar)
    print(f"  calibration block  {calibration[0]}-{calibration[1]}, "
          f"{block.n_days} days, {block.n_assimilated} assimilable")
    print(f"  draws              {args.draws}")
    print(f"  master seed        {seed}")

    # -- construction --------------------------------------------------------
    multiplier = decomposition_multiplier(t_air)
    naive = float(np.exp(0.0366 * t_air.mean()))
    bounds = reparameterised_bounds(t_air)
    print("\n" + bar)
    print("  Construction of the rh_ref prior")
    print(bar)
    print(f"  annual Rh (Fig. 6)       {ANNUAL_RH_G_C_M2[0]:.0f}-"
          f"{ANNUAL_RH_G_C_M2[1]:.0f} g C m-2 yr-1   <- the prior basis")
    print(f"  annual litter input      {ANNUAL_LITTER_INPUT_G_C_M2[0]:.0f}-"
          f"{ANNUAL_LITTER_INPUT_G_C_M2[2]:.0f}   <- corroboration only, "
          "steady-state upper bound")
    print(f"  soil carbon stock        {SOIL_CARBON_STOCK_G_C_M2:.0f} g C m-2 "
          "(Fig. 6, measured)")
    print(f"  mean TA_F                {t_air.mean():.3f} degC "
          f"(sd {t_air.std():.3f})")
    print(f"  M = mean(exp(0.0366 T))  {multiplier:.4f}   <- from the drivers")
    print(f"  exp(0.0366 mean T)       {naive:.4f}   <- NOT used, "
          f"{100 * (naive / multiplier - 1):+.1f}% (Jensen)")
    print("  rh_ref = rh_annual / (M(Theta_draw) * 365.25), M computed PER DRAW")
    print(f"  residence times          litter "
          f"{LITTER_RESIDENCE_TIME_YEARS[0]:g}-{LITTER_RESIDENCE_TIME_YEARS[1]:g} yr, "
          f"SOM {SOM_RESIDENCE_TIME_YEARS[0]:g}-{SOM_RESIDENCE_TIME_YEARS[1]:g} yr")
    for name, (lo, hi) in bounds.items():
        print(f"    {name:<10} U({lo:.6g}, {hi:.6g})")

    # -- CHECK 1 -------------------------------------------------------------
    _, analytic = sample_reparameterised_parameters(
        ANALYTIC_DRAWS, rng=np.random.default_rng(seed), t_air=t_air
    )
    _, analytic_old = sample_prior_parameters(
        ANALYTIC_DRAWS, rng=np.random.default_rng(seed)
    )
    lo_t, hi_t = SOIL_INVENTORY_RANGE
    print("\n" + bar)
    print(f"  CHECK 1 -- derived c_som_0 against the {lo_t:,.0f}-{hi_t:,.0f} "
          "g C m-2 inventory")
    print(bar)
    for name in ("c_som_0", "c_lit_0"):
        v = analytic[name].to_numpy()
        q = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        print(f"  {name}   min {v.min():8,.0f}   2.5% {q[0]:8,.0f}   "
              f"median {q[2]:8,.0f}   97.5% {q[4]:8,.0f}   max {v.max():8,.0f}")
        print(f"  {'':8}  IQR {q[1]:,.0f} to {q[3]:,.0f}")
    total = (analytic["c_lit_0"] + analytic["c_som_0"]).to_numpy()
    q_tot = np.percentile(total, [25, 50, 75])
    print(f"  total soil C  median {q_tot[1]:,.0f}   IQR {q_tot[0]:,.0f} to "
          f"{q_tot[2]:,.0f}   against the measured "
          f"{SOIL_CARBON_STOCK_G_C_M2:,.0f}")
    som = analytic["c_som_0"].to_numpy()
    inside = float(((som >= lo_t) & (som <= hi_t)).mean())
    print(f"\n  prior mass inside the inventory range   {100 * inside:.1f}%")
    print(f"  below {lo_t:,.0f}: {100 * (som < lo_t).mean():.1f}%     "
          f"above {hi_t:,.0f}: {100 * (som > hi_t).mean():.1f}%")
    old_som = analytic_old["c_som_0"].to_numpy()
    old_inside = float(((old_som >= lo_t) & (old_som <= hi_t)).mean())
    print("\n  superseded independent-uniform prior, for contrast:")
    print(f"    median {np.median(old_som):,.0f}   "
          f"inside the range {100 * old_inside:.1f}%")

    for label, f in (("reparameterised", analytic), ("superseded", analytic_old)):
        rh = f["theta_lit"] * f["c_lit_0"] + f["theta_som"] * f["c_som_0"]
        print(f"  Rh at T=0, {label:<16} median {rh.median():8.2f}   "
              f"max {rh.max():10.2f}  g C m-2 d-1")

    # -- CHECK 2 -------------------------------------------------------------
    rng = np.random.default_rng(seed)
    params, frame = sample_reparameterised_parameters(
        args.draws, rng=rng, t_air=t_air
    )
    print("\n" + bar)
    print("  CHECK 2 -- failure rate and coverage")
    print(bar)
    print(f"  running {args.draws} forward integrations...")
    latent, ok, reasons = run_block(params, block, acm, args.draws)
    frame["failure_reason"] = reasons
    n_failed = int((~ok).sum())
    rate = n_failed / args.draws

    counts = frame["failure_reason"].value_counts(dropna=False)
    print()
    for reason, count in counts.items():
        label = "usable" if reason is None or pd.isna(reason) else str(reason)
        print(f"    {label:<24} {count:>5}   {100 * count / args.draws:5.1f}%")
    frame[~ok].to_csv(out_dir / "failed_draws_reparameterised.csv", index=False)

    print(f"\n  failure rate   {n_failed} / {args.draws} = {100 * rate:.1f}%")
    print(f"  Task 1 was     {100 * TASK1_FAILURE_RATE:.1f}%")
    print(f"  change         {100 * (rate - TASK1_FAILURE_RATE):+.1f} points")

    if not ok.any():
        raise SystemExit("every draw failed; no coverage to report")

    coverage, _observed_level = coverage_of(latent, block, rng)
    print(f"\n  coverage       {coverage:.3f}")
    print(f"  Task 1 was     {TASK1_COVERAGE:.3f}")
    print(f"  change         {coverage - TASK1_COVERAGE:+.3f}   (target ~0.90)")

    # -- CHECK 3 -------------------------------------------------------------
    print("\n" + bar)
    print("  CHECK 3 -- prior predictive median annual NEE")
    print(bar)
    annual = latent.mean(axis=1) * DAYS_PER_YEAR
    q = np.percentile(annual, [5, 25, 50, 75, 95])
    obs_assim = block.nee_obs[block.nee_mask]
    observed_annual = float(obs_assim.mean() * DAYS_PER_YEAR)

    print(f"  prior predictive median   {q[2]:+9.1f} g C m-2 yr-1")
    print(f"  IQR                       {q[1]:+9.1f} to {q[3]:+.1f}")
    print(f"  5-95%                     {q[0]:+9.1f} to {q[4]:+.1f}")
    print(f"\n  observed, assimilable days in block   {observed_annual:+9.1f}")
    print(f"  quoted target                          {-206.0:+9.1f}")
    print(f"  difference, median - observed          "
          f"{q[2] - observed_annual:+9.1f}")
    print(f"  observed inside the prior 5-95%?       "
          f"{'yes' if q[0] <= observed_annual <= q[4] else 'NO'}")

    frame.to_csv(out_dir / "reparameterised_draws.csv", index=False)
    print(f"\n  draws -> {out_dir / 'reparameterised_draws.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
