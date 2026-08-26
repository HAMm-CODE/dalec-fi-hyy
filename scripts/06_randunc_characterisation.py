#!/usr/bin/env python
"""Characterise NEE_VUT_REF_RANDUNC. Purely descriptive; no model involved.

RANDUNC supplies the per-day standard deviation of the Gaussian likelihood, so
what it actually *is* decides what the likelihood actually asserts. This script
audits the column, then fits the descriptive relationships that answer the
question -- above all whether it scales as ``1/sqrt(n)`` with the number of
contributing half-hours, which would make it an estimate of within-day
variability rather than a known measurement error.

Nothing here is fitted back into the likelihood. See the circularity caveat in
the emitted findings.

Usage
-----
    python scripts/06_randunc_characterisation.py
    python scripts/06_randunc_characterisation.py --audit-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from a checkout without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    require_year_block,
    resolve_path,
)
from dalec.data_io import load_fluxnet_dd  # noqa: E402
from dalec.diagnostics import (  # noqa: E402
    HALFHOURS_PER_DAY,
    binned_median_iqr,
    randunc_audit,
    randunc_relationships,
)
from dalec.plotting import SEASON_COLOURS, apply_style, save_figure, season_of  # noqa: E402

#: Tracked output, per amendment A9: anything a supervisor would be shown.
REPORT_DIR = Path("reports/prior_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="print the audit and stop, without building the figure",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def print_audit(audit, fits) -> None:
    """The §4.1 audit, printed for the record."""
    line = "=" * 74
    print(line)
    print("  RANDUNC audit -- FLUXNET2015 FULLSET DD, FI-Hyy")
    print(line)
    print(f"  total days in record                      {audit.total_days:>6}")
    print(f"  days with valid NEE                       {audit.valid_nee:>6}")
    print(f"  days passing QC >= 0.75                   {audit.qc_pass:>6}")
    print(f"  valid NEE but RANDUNC missing             {audit.valid_nee_missing_unc:>6}")
    print(f"  valid NEE but RANDUNC == 0                {audit.valid_nee_zero_unc:>6}")
    print()
    print(f"  QC-passing days with NO usable sigma      {audit.qc_pass_no_sigma:>6}"
          "   <- cannot enter the likelihood")
    print(f"    of which inside calibration "
          f"{audit.calibration_years[0]}-{audit.calibration_years[1]}     "
          f"{audit.qc_pass_no_sigma_in_block:>6}   <- the operative number")
    print()
    print(f"  distinct QC values                        {audit.qc_levels:>6}")
    print(f"  every QC value a multiple of 1/48         "
          f"{audit.qc_is_multiple_of_halfhour!s:>6}"
          "   -> QC*48 is an exact half-hour count")
    print()
    print(f"  contiguous blocks with no usable sigma: {len(audit.missing_blocks)}")
    if len(audit.missing_blocks):
        blocks = audit.missing_blocks.sort_values("days", ascending=False)
        print(blocks.head(12).to_string(index=False))
        if len(blocks) > 12:
            print(f"    ... {len(blocks) - 12} shorter block(s) not shown")
        print(f"  longest block: {int(blocks['days'].iloc[0])} days; "
              f"total days in blocks: {int(blocks['days'].sum())}")
    print()
    print(line)
    print("  Descriptive fits")
    print(line)
    a, b = fits.on_abs_nee, fits.on_inv_sqrt_n
    print(f"  RANDUNC on |NEE|      slope {a.slope:8.4f}  intercept {a.intercept:7.4f}"
          f"  R2 {a.r_squared:6.4f}  n {a.n}")
    print(f"  RANDUNC on 1/sqrt(n)  slope {b.slope:8.4f}  intercept {b.intercept:7.4f}"
          f"  R2 {b.r_squared:6.4f}  n {b.n}")
    print(f"  RANDUNC / |NEE|       median {fits.relative_median:.3f}   "
          f"90th pct {fits.relative_p90:.3f}")
    print()
    print(f"  log-log slope, log(RANDUNC) on log(n)     {fits.log_log_slope:+7.4f}"
          "   <- -0.5 if sigma were the SE of a mean")
    print(f"  partial r vs 1/sqrt(n), given |NEE|       "
          f"{fits.partial_r_given_abs_nee:+7.4f}")
    print(line)


def build_figure(frame, audit, fits, out_dir: Path) -> list[Path]:
    """The five-panel RANDUNC figure."""
    import matplotlib.pyplot as plt

    apply_style()

    nee = frame["NEE_VUT_REF"].to_numpy(dtype=float)
    unc = frame["NEE_VUT_REF_RANDUNC"].to_numpy(dtype=float)
    qc = frame["NEE_VUT_REF_QC"].to_numpy(dtype=float)
    doy = frame.index.dayofyear.to_numpy()
    month = frame.index.month.to_numpy()

    with np.errstate(invalid="ignore"):
        usable = np.isfinite(nee) & np.isfinite(unc) & (unc > 0.0)

    abs_nee = np.abs(nee[usable])
    sigma = unc[usable]
    seasons = season_of(month[usable])

    figure = plt.figure(figsize=(11.0, 6.6))
    grid = figure.add_gridspec(2, 3)
    unit = r"g C m$^{-2}$ d$^{-1}$"

    # -- Panel 1: RANDUNC vs |NEE| -----------------------------------------
    ax = figure.add_subplot(grid[0, 0])
    for label, colour in SEASON_COLOURS.items():
        pick = seasons == label
        ax.scatter(abs_nee[pick], sigma[pick], s=2, alpha=0.15, color=colour,
                   label=label, linewidths=0)
    edges = np.linspace(0.0, np.percentile(abs_nee, 99.5), 21)
    binned = binned_median_iqr(abs_nee, sigma, edges)
    ax.plot(binned["centre"], binned["median"], color="black", lw=1.6, label="binned median")
    ax.fill_between(binned["centre"], binned["q25"], binned["q75"],
                    color="black", alpha=0.15, lw=0)
    fit = fits.on_abs_nee
    ax.set_title("RANDUNC vs |NEE|")
    ax.set_xlabel(f"|NEE|  ({unit})")
    ax.set_ylabel(f"RANDUNC  ({unit})")
    ax.set_xlim(0, edges[-1])
    ax.set_ylim(0, np.percentile(sigma, 99.5))
    ax.text(0.03, 0.96,
            f"slope {fit.slope:.3f}\nintercept {fit.intercept:.3f}\n$R^2$ {fit.r_squared:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
    ax.legend(loc="lower right", markerscale=3, fontsize=7)

    # -- Panel 2: RANDUNC vs day of year ------------------------------------
    ax = figure.add_subplot(grid[0, 1])
    binned = binned_median_iqr(doy[usable].astype(float), sigma,
                               np.arange(0.5, 367.5, 10.0))
    ax.plot(binned["centre"], binned["median"], color=SEASON_COLOURS["JJA"], lw=1.6)
    ax.fill_between(binned["centre"], binned["q25"], binned["q75"],
                    color=SEASON_COLOURS["JJA"], alpha=0.20, lw=0)
    ax.set_title("RANDUNC vs day of year")
    ax.set_xlabel("day of year")
    ax.set_ylabel(f"RANDUNC  ({unit})")
    ax.set_xlim(1, 366)

    # -- Panel 3: RANDUNC vs QC ---------------------------------------------
    ax = figure.add_subplot(grid[0, 2])
    qc_usable = qc[usable]
    qc_edges = np.linspace(0.0, 1.0, 11)
    groups, labels = [], []
    for i in range(len(qc_edges) - 1):
        pick = (qc_usable >= qc_edges[i]) & (
            qc_usable < qc_edges[i + 1] if i < len(qc_edges) - 2 else qc_usable <= 1.0
        )
        if pick.sum() >= 5:
            groups.append(sigma[pick])
            labels.append(f"{qc_edges[i]:.1f}")
    ax.boxplot(groups, tick_labels=labels, showfliers=False,
               medianprops={"color": SEASON_COLOURS["DJF"]})
    ax.set_title("RANDUNC vs QC flag")
    ax.set_xlabel("NEE_VUT_REF_QC (bin lower edge)")
    ax.set_ylabel(f"RANDUNC  ({unit})")
    ax.tick_params(axis="x", rotation=45)

    # -- Panel 4: RANDUNC vs 1/sqrt(n) -- the panel that settles it ----------
    ax = figure.add_subplot(grid[1, 0])
    n_halfhours = qc_usable * HALFHOURS_PER_DAY
    measured = n_halfhours > 0
    inv_sqrt_n = 1.0 / np.sqrt(n_halfhours[measured])
    ax.scatter(inv_sqrt_n, sigma[measured], s=2, alpha=0.12,
               color=SEASON_COLOURS["MAM"], linewidths=0)
    # 73.5% of days sit at exactly QC = 1, so equal-count bins collapse and
    # evenly spaced ones leave the right-hand tail swinging on a handful of
    # points. Fixed bins, then drop any holding fewer than 30 days.
    binned = binned_median_iqr(
        inv_sqrt_n, sigma[measured],
        np.linspace(inv_sqrt_n.min(), inv_sqrt_n.max(), 15),
    )
    binned = binned[binned["n"] >= 30]
    ax.plot(binned["centre"], binned["median"], color="black", lw=1.6,
            marker="o", ms=3.5, label=r"binned median ($n \geq$ 30 days)")
    fit = fits.on_inv_sqrt_n
    xs = np.linspace(0.0, inv_sqrt_n.max(), 50)
    ax.plot(xs, fit.intercept + fit.slope * xs, color=SEASON_COLOURS["JJA"],
            lw=1.4, ls="--", label="OLS (fitted)")
    # What the data would look like if RANDUNC were the standard error of a
    # daily mean over n half-hours: proportional to 1/sqrt(n), through the
    # origin. Anchored on the median sigma of complete days so the two are
    # directly comparable.
    complete = inv_sqrt_n <= np.min(inv_sqrt_n) * 1.001
    anchor = float(np.median(sigma[measured][complete]))
    ax.plot(xs, anchor * xs / float(np.min(inv_sqrt_n)), color="black", lw=1.2,
            ls=":", label=r"if $\sigma \propto 1/\sqrt{n}$")
    ax.set_title(r"RANDUNC vs $1/\sqrt{n}$,  $n = $ QC $\times$ 48")
    ax.set_xlabel(r"$1/\sqrt{n}$")
    ax.set_ylabel(f"RANDUNC  ({unit})")
    ax.set_ylim(0, np.percentile(sigma[measured], 99.5))
    ax.text(0.03, 0.96,
            f"slope {fit.slope:.3f}\nintercept {fit.intercept:.3f}\n$R^2$ {fit.r_squared:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=7)

    # -- Panel 5: relative uncertainty --------------------------------------
    ax = figure.add_subplot(grid[1, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = sigma / abs_nee
    relative = relative[np.isfinite(relative)]
    ax.hist(np.clip(relative, 0, 5), bins=80, color=SEASON_COLOURS["SON"])
    ax.axvline(fits.relative_median, color="black", lw=1.4,
               label=f"median {fits.relative_median:.2f}")
    ax.axvline(fits.relative_p90, color="black", lw=1.0, ls="--",
               label=f"90th pct {fits.relative_p90:.2f}")
    ax.set_title("relative uncertainty")
    ax.set_xlabel("RANDUNC / |NEE|  (clipped at 5)")
    ax.set_ylabel("days")
    ax.legend(loc="upper right", fontsize=7)

    # -- Panel 6: the audit, as text ----------------------------------------
    ax = figure.add_subplot(grid[1, 2])
    ax.axis("off")
    ax.set_title("audit")
    summary = (
        f"total days                 {audit.total_days}\n"
        f"valid NEE                  {audit.valid_nee}\n"
        f"passing QC $\\geq$ 0.75           {audit.qc_pass}\n"
        f"RANDUNC missing            {audit.valid_nee_missing_unc}\n"
        f"RANDUNC zero               {audit.valid_nee_zero_unc}\n\n"
        f"QC-passing, no sigma       {audit.qc_pass_no_sigma}\n"
        f"  in {audit.calibration_years[0]}-{audit.calibration_years[1]} "
        f"calibration      {audit.qc_pass_no_sigma_in_block}\n\n"
        f"distinct QC values         {audit.qc_levels}\n"
        f"all multiples of 1/48      {audit.qc_is_multiple_of_halfhour}"
    )
    ax.text(0.0, 0.95, summary, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, family="monospace")

    figure.suptitle(
        "NEE_VUT_REF_RANDUNC, FI-Hyy 1996-2014 -- descriptive characterisation",
        fontsize=11, fontweight="semibold",
    )
    written = save_figure(figure, out_dir, "fig11_randunc")
    plt.close(figure)
    return written


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    fluxnet_file = resolve_path(config["paths"]["fluxnet_file"])
    qc_threshold = float(config["data"]["qc_threshold"])
    calibration = require_year_block(config, "calibration")
    out_dir = args.out if args.out is not None else Path(REPORT_DIR)

    frame = load_fluxnet_dd(fluxnet_file)
    audit = randunc_audit(frame, qc_threshold=qc_threshold, calibration_years=calibration)
    fits = randunc_relationships(frame)
    print_audit(audit, fits)

    if args.audit_only:
        return 0

    for path in build_figure(frame, audit, fits, out_dir):
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
