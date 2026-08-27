#!/usr/bin/env python
"""Exploratory figures for the FI-Hyy record. Plain EDA -- no model is run.

Five figures describing what the data is, before anything is fitted to it:

    eda01  the NEE record and what survives screening, year by year
    eda02  the mean seasonal cycle of NEE, and how much it varies between years
    eda03  the drivers, as seasonal climatologies
    eda04  a day-by-day availability map, and whether losses cluster seasonally
    eda05  NEE against temperature and radiation, split winter / growing season

Everything is read from the FLUXNET2015 FULLSET daily file plus the derived
daily extremes; nothing here touches the forward model, PyTensor or a parameter.

Usage
-----
    python scripts/07_eda.py
    python scripts/07_eda.py --figures eda04 eda05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import stats

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
from dalec.data_io import DRIVER_COLUMNS, load_daily_extremes, load_fluxnet_dd  # noqa: E402
from dalec.diagnostics import binned_median_iqr  # noqa: E402
from dalec.plotting import (  # noqa: E402
    OKABE_ITO,
    SEASON_COLOURS,
    apply_style,
    save_figure,
    season_of,
)

REPORT_DIR = Path("reports/prior_diagnostics")
FLUX_UNIT = r"g C m$^{-2}$ d$^{-1}$"

#: Meteorological winter and growing season, as used throughout the project.
WINTER_MONTHS = (11, 12, 1, 2, 3)
GROWING_MONTHS = (5, 6, 7, 8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--figures", nargs="+", default=None,
        help="subset to build, e.g. eda04 eda05; default is all five",
    )
    return parser.parse_args()


class Record:
    """Everything the five figures read, assembled once."""

    def __init__(self, config) -> None:
        self.frame = load_fluxnet_dd(resolve_path(config["paths"]["fluxnet_file"]))
        self.qc_threshold = float(config["data"]["qc_threshold"])
        self.calibration = require_year_block(config, "calibration")
        self.prediction = require_year_block(config, "prediction")

        f = self.frame
        self.time = pd.DatetimeIndex(f.index)
        self.year = np.asarray(self.time.year)
        self.doy = np.asarray(self.time.dayofyear)
        self.month = np.asarray(self.time.month)

        self.nee = f["NEE_VUT_REF"].to_numpy(dtype=float)
        self.qc = f["NEE_VUT_REF_QC"].to_numpy(dtype=float)
        self.unc = f["NEE_VUT_REF_RANDUNC"].to_numpy(dtype=float)
        self.t_air = f[DRIVER_COLUMNS["t_air"]].to_numpy(dtype=float)
        self.sw_in = f[DRIVER_COLUMNS["sw_in"]].to_numpy(dtype=float)
        self.co2 = f[DRIVER_COLUMNS["co2"]].to_numpy(dtype=float)

        with np.errstate(invalid="ignore"):
            self.qc_pass = np.isfinite(self.nee) & np.isfinite(self.qc) & (
                self.qc >= self.qc_threshold
            )
            self.has_sigma = np.isfinite(self.unc) & (self.unc > 0.0)
        self.assimilable = self.qc_pass & self.has_sigma
        self.driver_gap = (
            f[list(DRIVER_COLUMNS.values())].isna().any(axis=1).to_numpy()
        )

        self.winter = np.isin(self.month, WINTER_MONTHS)
        self.growing = np.isin(self.month, GROWING_MONTHS)

        # True daily temperature range, where it has been derived.
        extremes_path = (
            resolve_path(config["paths"]["processed_dir"]) / "fi_hyy_tminmax.csv"
        )
        if extremes_path.exists():
            extremes = load_daily_extremes(extremes_path).reindex(self.time.normalize())
            self.t_range = (extremes["t_max"] - extremes["t_min"]).to_numpy(dtype=float)
        else:
            self.t_range = np.full(len(f), np.nan)

    def in_block(self, block: tuple[int, int]) -> np.ndarray:
        return (self.year >= block[0]) & (self.year <= block[1])

    def climatology(self, values: np.ndarray, mask: np.ndarray | None = None):
        """Per-day-of-year median and quartiles across years."""
        keep = np.isfinite(values) if mask is None else (np.isfinite(values) & mask)
        return binned_median_iqr(
            self.doy[keep].astype(float), values[keep], np.arange(0.5, 367.5, 5.0)
        )


def _shade_blocks(ax, record: Record) -> None:
    for block, colour, label in (
        (record.calibration, OKABE_ITO[0], "calibration"),
        (record.prediction, OKABE_ITO[1], "prediction"),
    ):
        ax.axvspan(
            pd.Timestamp(block[0], 1, 1), pd.Timestamp(block[1], 12, 31),
            color=colour, alpha=0.10, lw=0, label=f"{label} {block[0]}-{block[1]}",
        )


# ---------------------------------------------------------------------------
# eda01 -- the record, and what survives screening
# ---------------------------------------------------------------------------


def eda01(record: Record, out_dir: Path):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11.0, 6.0))
    grid = figure.add_gridspec(2, 1, height_ratios=[2.0, 1.3])

    ax = figure.add_subplot(grid[0])
    _shade_blocks(ax, record)
    ax.scatter(record.time, record.nee, s=1.2, alpha=0.35, color="#3a4a48", linewidths=0)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel(f"NEE  ({FLUX_UNIT})")
    ax.set_title("Daily net ecosystem exchange, whole record")
    ax.legend(loc="upper left", ncol=2)
    ax.text(0.995, 0.04, "negative = net uptake", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="#5E6E6C")

    ax = figure.add_subplot(grid[1])
    years = np.arange(record.year.min(), record.year.max() + 1)
    categories = {
        "assimilable": (record.assimilable, OKABE_ITO[2]),
        "QC below threshold": (~record.qc_pass & ~record.driver_gap, OKABE_ITO[4]),
        "no usable sigma": (record.qc_pass & ~record.has_sigma, OKABE_ITO[1]),
        "driver gap": (record.driver_gap, "#4a4a4a"),
    }
    bottom = np.zeros(len(years))
    for label, (flag, colour) in categories.items():
        counts = np.array([int((flag & (record.year == y)).sum()) for y in years])
        ax.bar(years, counts, bottom=bottom, color=colour, label=label, width=0.8)
        bottom += counts
    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    ax.set_xticks(years[::2])
    ax.set_ylim(0, 470)
    ax.set_ylabel("days")
    ax.set_xlabel("year")
    ax.set_title("Days per year by screening outcome")
    # Above the bars: at lower left it sat on top of 1996's stack.
    ax.legend(loc="upper center", ncol=4, fontsize=7.5)

    figure.suptitle(
        "eda01 -- the FI-Hyy NEE record and what survives screening",
        fontsize=11, fontweight="semibold",
    )
    paths = save_figure(figure, out_dir, "eda01_record")
    plt.close(figure)
    return paths


# ---------------------------------------------------------------------------
# eda02 -- the mean seasonal cycle
# ---------------------------------------------------------------------------


def eda02(record: Record, out_dir: Path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # QC-passing days only. Without this the 1996 winter contributes a repeating
    # three-value gap-fill cycle (-4.66, -5.38, -5.11 g C m-2 d-1, QC = 0.0)
    # which reads as ten weeks of February net uptake and drags the onset
    # estimate to doy 46. It is fill, not flux.
    screened = np.where(record.qc_pass, record.nee, np.nan)
    per_year = pd.DataFrame({"year": record.year, "doy": record.doy, "nee": screened})
    pivot = per_year.pivot_table(index="doy", columns="year", values="nee")
    smooth = pivot.rolling(11, center=True, min_periods=4).mean()

    ax = axes[0]
    for column in smooth.columns:
        ax.plot(smooth.index, smooth[column], color="#8fa3a0", lw=0.6, alpha=0.55)
    mean = smooth.mean(axis=1)
    sd = smooth.std(axis=1)
    ax.fill_between(smooth.index, mean - sd, mean + sd, color=OKABE_ITO[0],
                    alpha=0.25, lw=0, label="$\\pm$1 SD between years")
    ax.plot(smooth.index, mean, color=OKABE_ITO[0], lw=2.0, label="mean of all years")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlim(1, 366)
    ax.set_xlabel("day of year")
    ax.set_ylabel(f"NEE  ({FLUX_UNIT})")
    ax.set_title("Mean seasonal cycle, QC-passing days (11-day smoothed, one line per year)")
    ax.legend(loc="lower left")

    # "Sustained" has to mean something: a single cold-snap dip below zero in
    # March is not spring onset. Require 10 consecutive days of net uptake.
    sustained_days = 10
    min_spring_coverage = 0.5
    crossings, skipped = [], []
    for column in smooth.columns:
        # A year with no usable spring cannot date its spring. 1996 has zero
        # QC-passing days between doy 60 and 150, so the first negative run it
        # can offer is in mid-June -- that is a coverage gap, not a late onset.
        window = (record.doy >= 45) & (record.doy < 220) & (record.year == column)
        coverage = (record.qc_pass & window).sum() / max(window.sum(), 1)
        if coverage < min_spring_coverage:
            skipped.append((int(column), coverage))
            continue
        series = smooth[column].dropna()
        spring = series[(series.index > 45) & (series.index < 220)]
        negative = (spring < 0).to_numpy()
        run = 0
        for position, is_negative in enumerate(negative):
            run = run + 1 if is_negative else 0
            if run == sustained_days:
                crossings.append(int(spring.index[position - sustained_days + 1]))
                break

    ax = axes[1]
    lo = 4 * (min(crossings) // 4)
    edges = np.arange(lo, max(crossings) + 5, 4)
    ax.hist(crossings, bins=edges, color=OKABE_ITO[2])
    ax.axvline(float(np.median(crossings)), color="black", lw=1.4,
               label=f"median doy {np.median(crossings):.0f}")
    ax.set_xlabel(f"day of year of first {sustained_days} consecutive days of net uptake")
    ax.set_ylabel("years")
    ax.yaxis.get_major_locator().set_params(integer=True)
    excluded = "" if not skipped else (
        "; " + ", ".join(str(y) for y, _ in skipped) + " excluded, spring too sparse"
    )
    ax.set_title(f"Spring onset, doy {min(crossings)}-{max(crossings)} "
                 f"(spread {max(crossings) - min(crossings)} days, "
                 f"n = {len(crossings)}{excluded})", fontsize=9)
    ax.legend(loc="upper right")

    figure.suptitle(
        "eda02 -- the seasonal cycle of NEE, and how much it moves between years",
        fontsize=11, fontweight="semibold",
    )
    paths = save_figure(figure, out_dir, "eda02_seasonal_cycle")
    plt.close(figure)
    return paths, crossings, skipped


# ---------------------------------------------------------------------------
# eda03 -- the drivers
# ---------------------------------------------------------------------------


def eda03(record: Record, out_dir: Path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 6.0))

    panels = [
        (axes[0, 0], record.t_air, "TA_F, daily mean air temperature", "$^\\circ$C",
         OKABE_ITO[1]),
        (axes[0, 1], record.sw_in, "SW_IN_F, incoming shortwave", "W m$^{-2}$",
         OKABE_ITO[4]),
        (axes[1, 0], record.t_range, "daily temperature range, from half-hourly",
         "$^\\circ$C", OKABE_ITO[2]),
    ]
    for ax, values, title, unit, colour in panels:
        table = record.climatology(values)
        ax.plot(table["centre"], table["median"], color=colour, lw=1.8)
        ax.fill_between(table["centre"], table["q25"], table["q75"], color=colour,
                        alpha=0.22, lw=0)
        ax.set_xlim(1, 366)
        ax.set_xlabel("day of year")
        ax.set_ylabel(unit)
        ax.set_title(title)
        if values is record.t_air:
            ax.axhline(0.0, color="black", lw=0.8)
            ax.axhline(-2.0, color=OKABE_ITO[0], lw=1.0, ls="--",
                       label="frost cutoff $-2^\\circ$C")
            ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1, 1]
    ax.plot(record.time, record.co2, color=OKABE_ITO[3], lw=0.7)
    _shade_blocks(ax, record)
    ax.set_xlabel("year")
    ax.set_ylabel("$\\mu$mol mol$^{-1}$")
    ax.set_title("CO2_F_MDS, atmospheric CO$_2$")
    ax.legend(loc="upper left", fontsize=7.5)

    figure.suptitle(
        "eda03 -- the drivers, as seasonal climatologies (median and interquartile range)",
        fontsize=11, fontweight="semibold",
    )
    paths = save_figure(figure, out_dir, "eda03_drivers")
    plt.close(figure)
    return paths


# ---------------------------------------------------------------------------
# eda04 -- availability, and whether losses cluster seasonally
# ---------------------------------------------------------------------------


def eda04(record: Record, out_dir: Path):
    import matplotlib.pyplot as plt

    years = np.arange(record.year.min(), record.year.max() + 1)
    codes = np.full((len(years), 366), np.nan)
    for row, year in enumerate(years):
        pick = record.year == year
        column = record.doy[pick] - 1
        value = np.zeros(int(pick.sum()))
        value[~record.qc_pass[pick]] = 1.0
        value[record.qc_pass[pick] & ~record.has_sigma[pick]] = 2.0
        value[record.driver_gap[pick]] = 3.0
        codes[row, column] = value

    figure = plt.figure(figsize=(11.0, 6.2))
    grid = figure.add_gridspec(2, 1, height_ratios=[2.2, 1.0])

    ax = figure.add_subplot(grid[0])
    cmap = ListedColormap([OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[1], "#4a4a4a"])
    ax.imshow(codes, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
              interpolation="nearest",
              extent=(0.5, 366.5, years[-1] + 0.5, years[0] - 0.5))
    ax.set_yticks(years[::2])
    ax.set_xlabel("day of year")
    ax.set_ylabel("year")
    ax.set_title("Every day of the record, by screening outcome")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in (OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[1], "#4a4a4a")
    ]
    ax.legend(handles, ["assimilable", "QC below threshold", "no usable sigma",
                        "driver gap"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8)
    ax.grid(False)

    ax = figure.add_subplot(grid[1])
    months = np.arange(1, 13)
    fail_rate = np.array([
        100.0 * (~record.qc_pass & (record.month == m)).sum() / (record.month == m).sum()
        for m in months
    ])
    nosigma_rate = np.array([
        100.0 * (record.qc_pass & ~record.has_sigma & (record.month == m)).sum()
        / (record.month == m).sum()
        for m in months
    ])
    ax.bar(months - 0.19, fail_rate, width=0.38, color=OKABE_ITO[4],
           label="QC below threshold")
    ax.bar(months + 0.19, nosigma_rate, width=0.38, color=OKABE_ITO[1],
           label="no usable sigma")
    ax.set_xticks(months)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("month")
    ax.set_ylabel("% of days")
    ax.set_title("Losses by month -- both concentrate in winter")
    ax.legend(loc="upper right", ncol=2)

    figure.suptitle(
        "eda04 -- data availability, and where in the year it is lost",
        fontsize=11, fontweight="semibold",
    )
    paths = save_figure(figure, out_dir, "eda04_availability")
    plt.close(figure)
    return paths, fail_rate, nosigma_rate


# ---------------------------------------------------------------------------
# eda05 -- NEE against its drivers
# ---------------------------------------------------------------------------


def eda05(record: Record, out_dir: Path):
    import matplotlib.pyplot as plt

    usable = record.assimilable
    seasons = season_of(record.month[usable])
    nee, t_air, sw_in = record.nee[usable], record.t_air[usable], record.sw_in[usable]
    winter, growing = record.winter[usable], record.growing[usable]

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))

    # -- NEE vs temperature, binned separately by season ---------------------
    ax = axes[0]
    for label, colour in SEASON_COLOURS.items():
        pick = seasons == label
        ax.scatter(t_air[pick], nee[pick], s=2, alpha=0.14, color=colour,
                   label=label, linewidths=0)
    fits = {}
    for label, pick, colour in (
        ("winter Nov-Mar", winter, OKABE_ITO[0]),
        ("growing May-Sep", growing, OKABE_ITO[1]),
    ):
        table = binned_median_iqr(
            t_air[pick], nee[pick],
            np.linspace(np.percentile(t_air[pick], 1), np.percentile(t_air[pick], 99), 14),
        )
        table = table[table["n"] >= 20]
        ax.plot(table["centre"], table["median"], color=colour, lw=2.0, marker="o",
                ms=3.5, label=label)
        fit = stats.linregress(t_air[pick], nee[pick])
        fits[label] = fit
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("TA_F  ($^\\circ$C)")
    ax.set_ylabel(f"NEE  ({FLUX_UNIT})")
    ax.set_title("NEE vs temperature, binned by season")
    ax.legend(loc="lower left", fontsize=7, markerscale=3)

    # -- NEE vs radiation ----------------------------------------------------
    ax = axes[1]
    for label, colour in SEASON_COLOURS.items():
        pick = seasons == label
        ax.scatter(sw_in[pick], nee[pick], s=2, alpha=0.14, color=colour,
                   label=label, linewidths=0)
    table = binned_median_iqr(sw_in, nee, np.linspace(0, np.percentile(sw_in, 99), 16))
    ax.plot(table["centre"], table["median"], color="black", lw=1.8,
            label="binned median")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("SW_IN_F  (W m$^{-2}$)")
    ax.set_ylabel(f"NEE  ({FLUX_UNIT})")
    ax.set_title("NEE vs incoming shortwave")
    ax.legend(loc="lower left", fontsize=7, markerscale=3)

    # -- the winter branch, on a log axis: an empirical Theta -----------------
    ax = axes[2]
    release = winter & (nee > 0.0)
    x, y = t_air[release], np.log(nee[release])
    fit = stats.linregress(x, y)
    ax.scatter(x, np.exp(y), s=3, alpha=0.22, color=SEASON_COLOURS["DJF"], linewidths=0)
    grid_x = np.linspace(x.min(), x.max(), 50)
    ax.plot(grid_x, np.exp(fit.intercept + fit.slope * grid_x), color=OKABE_ITO[1],
            lw=2.0, label="exponential fit")
    ax.set_yscale("log")
    ax.set_xlabel("TA_F  ($^\\circ$C)")
    ax.set_ylabel(f"NEE, release days only  ({FLUX_UNIT})")
    ax.set_title("Winter respiration response")
    half = 1.96 * fit.stderr
    ax.text(
        0.03, 0.97,
        f"$\\Theta$ = {fit.slope:.4f} $^\\circ$C$^{{-1}}$\n"
        f"95% CI [{fit.slope - half:.4f}, {fit.slope + half:.4f}]\n"
        f"$R^2$ {fit.rvalue**2:.3f},  n = {release.sum()}\n"
        f"$Q_{{10}}$ = {np.exp(10 * fit.slope):.2f}\n"
        f"prior [0.018, 0.08]",
        transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
    )
    ax.legend(loc="lower right", fontsize=7)

    figure.suptitle(
        "eda05 -- NEE against its drivers; the winter branch is a near-pure "
        "respiration response",
        fontsize=11, fontweight="semibold",
    )
    paths = save_figure(figure, out_dir, "eda05_driver_response")
    plt.close(figure)
    return paths, fits, fit


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    out_dir = args.out if args.out is not None else Path(REPORT_DIR)
    wanted = set(args.figures) if args.figures else {
        "eda01", "eda02", "eda03", "eda04", "eda05"
    }

    apply_style()
    record = Record(config)
    print(f"record: {len(record.frame)} days, {record.year.min()}-{record.year.max()}   "
          f"assimilable {int(record.assimilable.sum())}")

    written = []
    if "eda01" in wanted:
        written += eda01(record, out_dir)
    if "eda02" in wanted:
        paths, crossings, skipped = eda02(record, out_dir)
        written += paths
        print(f"  eda02  spring onset: median doy {np.median(crossings):.0f}, "
              f"range {min(crossings)}-{max(crossings)}, n={len(crossings)}"
              + (f", excluded {[y for y, _ in skipped]}" if skipped else ""))
    if "eda03" in wanted:
        written += eda03(record, out_dir)
    if "eda04" in wanted:
        paths, _fail_rate, _nosigma_rate = eda04(record, out_dir)
        written += paths
        winter_fail = 100.0 * (~record.qc_pass & record.winter).sum() / record.winter.sum()
        grow_fail = 100.0 * (~record.qc_pass & record.growing).sum() / record.growing.sum()
        print(f"  eda04  QC-fail rate winter {winter_fail:.1f}% vs growing "
              f"{grow_fail:.1f}%  ({winter_fail / grow_fail:.2f}x)")
    if "eda05" in wanted:
        paths, fits, winter_fit = eda05(record, out_dir)
        written += paths
        for label, fit in fits.items():
            print(f"  eda05  {label:<16} NEE~T slope {fit.slope:+.4f}  "
                  f"R2 {fit.rvalue**2:.3f}")
        print(f"  eda05  winter ln(NEE)~T slope {winter_fit.slope:.4f} degC-1 "
              f"(prior 0.018-0.08)")

    for path in written:
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
