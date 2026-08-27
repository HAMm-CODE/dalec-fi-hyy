#!/usr/bin/env python
"""Task 1, reduced: prior predictive check on the calibration block.

Two outputs only, per the amended specification:

    fig02_prior_pred_seasonal   mean seasonal cycle, observed against prior
    coverage                    fraction of observed days inside the 90%
                                observation-level prior predictive band

Plus the amendment A1 requirement -- the decade-by-decade prior mass of every
parameter spanning more than two orders of magnitude -- and the failed-draw
accounting of section 1.5.

No PyMC and no posterior. Prior draws come from the Parameter registry through
numpy, per amendment A2; the guarantee that these are the priors that will later
be sampled comes from reading the same registry objects, so no bound is retyped
anywhere here. fig01, fig03 and fig04 are deliberately not built, and no prior is
adjusted -- if the check fails, that is the finding.

Usage
-----
    python scripts/08_prior_predictive.py
    python scripts/08_prior_predictive.py --draws 200
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from a checkout without `pip install -e .`.
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
    prior_decade_mass,
    sample_prior_parameters,
)
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402
from dalec.plotting import OKABE_ITO, apply_style, save_figure  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
FLUX_UNIT = r"g C m$^{-2}$ d$^{-1}$"

#: Section 1.1 specifies 1000 draws.
DEFAULT_DRAWS = 1000

#: The band the coverage number is computed against.
COVERAGE_INTERVAL = (5.0, 95.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def day_of_year_mean(values: np.ndarray, doy: np.ndarray) -> pd.Series:
    """Collapse a daily series onto day of year, averaging across years."""
    return pd.DataFrame({"doy": doy, "v": values}).groupby("doy")["v"].mean()


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
        raise SystemExit(
            f"no processed calibration block at {block_path}. "
            "Run scripts/01_prepare_data.py first."
        )

    block = SiteData.load(block_path)
    acm = acm_from_config(config)
    rng = np.random.default_rng(seed)

    print("=" * 74)
    print("  Task 1 -- prior predictive check")
    print("=" * 74)
    print(f"  calibration block  {calibration[0]}-{calibration[1]}, "
          f"{block.n_days} days, {block.n_assimilated} assimilable")
    print(f"  draws              {args.draws}")
    print(f"  master seed        {seed}  (from config, per A2)")

    # -- A1: decade mass of the wide priors ---------------------------------
    print("\n" + "=" * 74)
    print("  A1 -- prior mass by decade, parameters spanning >2 orders")
    print("=" * 74)
    decades = prior_decade_mass()
    for name, group in decades.groupby("parameter", sort=False):
        span = float(group["orders"].iloc[0])
        pieces = "  ".join(
            f"[{row.decade_low:g}, {row.decade_high:g}) {100 * row.mass_fraction:5.1f}%"
            for row in group.itertuples()
        )
        print(f"  {name:<22} {span:.2f} orders   {pieces}")
    decades.to_csv(out_dir / "prior_decade_mass.csv", index=False)

    # -- draw, run, classify -------------------------------------------------
    params, draw_frame = sample_prior_parameters(args.draws, rng=rng)

    latent = np.full((args.draws, block.n_days), np.nan)
    reasons: list[str | None] = []
    print(f"\n  running {args.draws} forward integrations...")
    for index, parameters in enumerate(params):
        output = run_dalec2(
            parameters, block, gpp_fn=acm, phenology_fn=dalec2_phenology
        )
        reason = classify_prior_draw(output)
        reasons.append(reason)
        if reason is None:
            latent[index] = output.nee
        if (index + 1) % 100 == 0:
            failed = sum(r is not None for r in reasons)
            print(f"    {index + 1:>5} / {args.draws}   failed so far: {failed}")

    draw_frame["failure_reason"] = reasons
    ok = np.array([r is None for r in reasons])
    latent = latent[ok]

    print("\n" + "=" * 74)
    print("  Section 1.5 -- failed draws")
    print("=" * 74)
    counts = draw_frame["failure_reason"].value_counts(dropna=False)
    for reason, count in counts.items():
        label = "usable" if reason is None or pd.isna(reason) else str(reason)
        print(f"  {label:<24} {count:>5}   {100 * count / args.draws:5.1f}%")
    n_failed = int((~ok).sum())
    print(f"\n  failure rate  {n_failed} / {args.draws} "
          f"= {100 * n_failed / args.draws:.1f}%")
    failed_path = out_dir / "failed_draws.csv"
    draw_frame[~ok].to_csv(failed_path, index=False)
    print(f"  offending parameter vectors -> {failed_path}")

    if not ok.any():
        raise SystemExit("every prior draw failed; nothing to plot")

    # -- observation-level band: one noise draw per prior sample -------------
    sigma = np.where(block.nee_mask, block.nee_unc, np.nan)
    noise_sigma = np.nan_to_num(sigma, nan=float(np.nanmedian(sigma)))
    observed_level = latent + rng.normal(0.0, noise_sigma, size=latent.shape)

    # -- the coverage number -------------------------------------------------
    low, high = np.percentile(observed_level, COVERAGE_INTERVAL, axis=0)
    inside = (block.nee_obs >= low) & (block.nee_obs <= high)
    coverage = float(inside[block.nee_mask].mean())

    print("\n" + "=" * 74)
    print("  Coverage")
    print("=" * 74)
    print(f"  observed days inside the 90% observation-level band: "
          f"{int(inside[block.nee_mask].sum())} / {int(block.nee_mask.sum())}")
    print(f"  fraction = {coverage:.3f}")
    print("  a well-specified prior gives roughly 0.90 or a little above;")
    print("  substantially below means the priors and the data disagree.")

    # -- fig02 ---------------------------------------------------------------
    apply_style()
    import matplotlib.pyplot as plt

    doy = block.doy
    observed_masked = np.where(block.nee_mask, block.nee_obs, np.nan)
    frame = pd.DataFrame({"doy": doy, "v": observed_masked})
    observed_mean = frame.groupby("doy")["v"].mean()
    observed_q25 = frame.groupby("doy")["v"].quantile(0.25)
    observed_q75 = frame.groupby("doy")["v"].quantile(0.75)

    latent_doy = np.vstack([day_of_year_mean(row, doy).to_numpy() for row in latent])
    obs_doy = np.vstack(
        [day_of_year_mean(row, doy).to_numpy() for row in observed_level]
    )
    axis_doy = observed_mean.index.to_numpy()

    figure, ax = plt.subplots(figsize=(10.0, 5.4))
    obs_low, obs_high = np.percentile(obs_doy, COVERAGE_INTERVAL, axis=0)
    ax.fill_between(axis_doy, obs_low, obs_high, color=OKABE_ITO[5], alpha=0.30, lw=0,
                    label="prior predictive 5-95%, observation level")
    lat_low, lat_high = np.percentile(latent_doy, COVERAGE_INTERVAL, axis=0)
    ax.fill_between(axis_doy, lat_low, lat_high, color=OKABE_ITO[0], alpha=0.35, lw=0,
                    label="prior predictive 5-95%, latent")
    ax.plot(axis_doy, np.median(latent_doy, axis=0), color=OKABE_ITO[0], lw=2.0,
            label="prior predictive median")

    ax.fill_between(axis_doy, observed_q25, observed_q75, color="#2f3b3a", alpha=0.30,
                    lw=0, label="observed interquartile range")
    ax.plot(axis_doy, observed_mean, color="black", lw=2.0, label="observed mean")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlim(1, 366)
    ax.set_xlabel("day of year")
    ax.set_ylabel(f"NEE  ({FLUX_UNIT})")
    ax.set_title(
        f"Prior predictive against observed, {calibration[0]}-{calibration[1]}  "
        f"({int(ok.sum())} usable draws of {args.draws}; "
        f"90% coverage {coverage:.3f})",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8)
    figure.suptitle(
        "fig02 -- mean seasonal cycle, prior predictive check",
        fontsize=11, fontweight="semibold",
    )
    for path in save_figure(figure, out_dir, "fig02_prior_pred_seasonal"):
        print(f"\n  -> {path}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
