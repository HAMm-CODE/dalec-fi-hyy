#!/usr/bin/env python
"""Do the Gaussian likelihood's assumptions hold? A residual diagnostic.

**Diagnostic only.** Nothing here changes the likelihood, adds an error-model
parameter, or fits an AR model. It takes one well-fitting parameter set, runs the
forward model once, and asks what the residuals say about the three assumptions
the Gaussian likelihood rests on:

    log L = -0.5 * sum[ ((y - f)/sigma)^2 + log(2*pi*sigma^2) ]

1. **Scale.** ``sigma`` = ``NEE_VUT_REF_RANDUNC`` is the right size, so
   standardised residuals have sd 1. (Figure 1.)
2. **Shape.** The errors are Gaussian. (Figure 2.)
3. **Independence.** Each day contributes fresh information. (Figure 3.)

Figure 4 adds a fourth question the likelihood does not assume but the model
should answer: is the residual free of seasonal structure?

Why each matters for the posterior
----------------------------------
Assumptions 1 and 3 both control how much information the likelihood believes it
has. If standardised residuals have sd > 1, ``sigma`` is too small and every day
is over-weighted. If residuals are autocorrelated, consecutive days repeat
information the likelihood counts as independent. **Either one makes the
posterior narrower than it should be**, and they compound. Assumption 2 matters
less for the width and more for the shape of the tails.

Which parameter set, and why one
--------------------------------
The posterior is multimodal. This script does **not** assume which chain found
the better region: it compares the mean ``lp`` across chains, takes the best one,
and within it the single highest-``lp`` draw (the MAP draw). Chain and draw index
are printed so the choice is citable.

Using one draw rather than the posterior mean is deliberate. A residual
diagnostic asks whether the *error model* fits, which needs a single coherent
parameter set that the model can actually be run at; an average of parameters is
not itself a parameter set the model ever visited.

Usage
-----
    python scripts/24_residual_diagnostics.py

    # skip steps 1-3 and reuse an existing residual table
    python scripts/24_residual_diagnostics.py --residuals reports/residuals/residuals.csv
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
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402
from dalec.parameters import PARAMETER_NAMES, DalecParameters  # noqa: E402
from dalec.plotting import OKABE_ITO, apply_style, save_figure  # noqa: E402

DEFAULT_TRACE = Path("results/calibration_hemisurface.nc")
OUT_DIR = Path("reports/residuals")
MAX_LAG = 40

COL_MAIN = OKABE_ITO[0]     # blue
COL_ALERT = OKABE_ITO[1]    # vermillion
COL_ALT = OKABE_ITO[2]      # green
COL_REF = "0.25"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--residuals",
        type=Path,
        default=OUT_DIR / "residuals.csv",
        help="Residual table. If it already exists, steps 1-3 are skipped and "
             "the analysis starts from step 4 using this file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the residual table even if it already exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# STEP 1 -- choose the parameter set
# ---------------------------------------------------------------------------
def select_map_draw(idata) -> tuple[int, int, float, np.ndarray]:
    """Find the MAP draw in the best-fitting mode.

    ``lp`` is the log-posterior density PyMC records per draw. Averaging it
    within each chain separates the modes: chains in a better region sit at a
    systematically higher ``lp``, and because the modes here are thousands of
    log-units apart the separation is unambiguous.

    Returns ``(chain, draw, lp, chain_mean_lp)``. No chain index is assumed.
    """
    lp = idata["sample_stats"]["lp"].values          # (chain, draw)
    chain_mean = lp.mean(axis=1)
    best_chain = int(np.argmax(chain_mean))
    best_draw = int(np.argmax(lp[best_chain]))
    return best_chain, best_draw, float(lp[best_chain, best_draw]), chain_mean


def parameters_from_draw(idata, chain: int, draw: int) -> DalecParameters:
    """Rebuild a full parameter set from one posterior draw.

    Every field of :class:`DalecParameters` is stored in the trace -- the
    sampled scalars directly, and the allocation fractions and initial pools as
    deterministics -- so the set is read out rather than reconstructed, and no
    derivation is repeated here where it could drift.
    """
    post = idata["posterior"]
    missing = [name for name in PARAMETER_NAMES if name not in post]
    if missing:
        raise SystemExit(
            f"trace is missing {missing}; cannot rebuild a parameter set. "
            "Was it written by a different version of sampler.py?"
        )
    values = {name: float(post[name].values[chain, draw]) for name in PARAMETER_NAMES}
    return DalecParameters(**values)


# ---------------------------------------------------------------------------
# STEPS 2 and 3 -- forward run and residual table
# ---------------------------------------------------------------------------
def build_residual_table(config, trace_path: Path) -> pd.DataFrame:
    """Run the model once at the MAP draw and tabulate residuals.

    The likelihood filter is **not** re-derived here. ``SiteData.nee_mask`` is
    the same boolean the sampler used, built by ``data_io._likelihood_mask``:
    NEE present, ``NEE_VUT_REF_QC`` >= the configured threshold (0.75), and
    ``NEE_VUT_REF_RANDUNC`` present and strictly positive. Re-implementing that
    rule here would be a second copy that could disagree with the first.
    """
    import arviz as az

    calibration = require_year_block(config, "calibration")
    slug = str(config.get("site", {}).get("code", "")).lower().replace("-", "_")
    block = SiteData.load(
        resolve_path(config["paths"]["processed_dir"])
        / f"{slug}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )

    # -- STEP 1 ----------------------------------------------------------
    idata = az.from_netcdf(str(trace_path))
    chain, draw, lp, chain_mean = select_map_draw(idata)

    print("STEP 1 -- parameter set")
    print(f"  trace                {trace_path}")
    print(f"  chains x draws       {idata['posterior'].sizes['chain']} x "
          f"{idata['posterior'].sizes['draw']}")
    print("  mean lp by chain:")
    for c, value in enumerate(chain_mean):
        marker = "  <- selected" if c == chain else ""
        print(f"    chain {c}: {value:12.1f}{marker}")
    gap = float(chain_mean[chain] - np.delete(chain_mean, chain).max())
    print(f"  gap to next best     {gap:,.1f} log units")
    print(f"  CITE THIS: chain {chain}, draw {draw}, lp = {lp:,.2f}")

    params = parameters_from_draw(idata, chain, draw)

    # -- STEP 2 ----------------------------------------------------------
    print("\nSTEP 2 -- forward run")
    acm = acm_from_config(config)
    output = run_dalec2(params, block, gpp_fn=acm, phenology_fn=dalec2_phenology)
    print(f"  integrated {block.n_days} days, {calibration[0]}-{calibration[1]}")

    # -- STEP 3 ----------------------------------------------------------
    mask = block.nee_mask.astype(bool)
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(block.time),
            "nee_obs": block.nee_obs,
            "nee_mod": output.nee,
            "randunc": block.nee_unc,
            "qc": block.nee_qc,
            "in_likelihood": mask,
        }
    )
    frame = frame.loc[mask].reset_index(drop=True)
    print("\nSTEP 3 -- residual table")
    print(f"  {block.n_days} days integrated, {len(frame)} entered the likelihood "
          f"({100 * len(frame) / block.n_days:.1f}%)")
    frame.attrs["chain"] = chain
    frame.attrs["draw"] = draw
    frame.attrs["lp"] = lp
    return frame


# ---------------------------------------------------------------------------
# STEP 4 -- residuals
# ---------------------------------------------------------------------------
def add_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """residual = observed - modelled; standardised = residual / RANDUNC.

    The sign convention matters for reading figure 4: NEE is negative for
    uptake, so a **positive** residual means the observation is less negative
    than the model, i.e. the model takes up more carbon than was observed.
    """
    frame = frame.copy()
    frame["residual"] = frame["nee_obs"] - frame["nee_mod"]
    frame["residual_std"] = frame["residual"] / frame["randunc"]
    return frame


# ---------------------------------------------------------------------------
# Gap-aware partial autocorrelation
# ---------------------------------------------------------------------------
def gap_aware_acf(dates: pd.DatetimeIndex, values: np.ndarray, max_lag: int):
    """Autocorrelation at each lag using only genuinely k-days-apart pairs.

    The residual series has roughly 700 missing days inside 1997-2010, and those
    days are absent because they failed QC, not because time stopped. Treating
    the surviving rows as consecutive would compress the gaps and shift every
    lag. So the series is placed on a complete daily calendar first, and lag `k`
    uses only pairs whose dates differ by exactly `k` days and where both values
    are present.

    Returns ``(acf, counts)`` with ``acf[0] = 1``.
    """
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    series = pd.Series(np.nan, index=calendar, dtype=float)
    series.loc[dates] = values
    array = series.to_numpy()

    centred = array - np.nanmean(array)
    variance = np.nanvar(array)
    acf = np.ones(max_lag + 1)
    counts = np.zeros(max_lag + 1, dtype=int)
    counts[0] = int(np.isfinite(array).sum())
    for k in range(1, max_lag + 1):
        a, b = centred[:-k], centred[k:]
        both = np.isfinite(a) & np.isfinite(b)
        counts[k] = int(both.sum())
        acf[k] = float(np.mean(a[both] * b[both]) / variance) if both.sum() > 2 else np.nan
    return acf, counts


def pacf_from_acf(acf: np.ndarray, max_lag: int) -> np.ndarray:
    """Partial autocorrelation by the Durbin-Levinson recursion.

    The PACF at lag k is the correlation between residuals k days apart with the
    intervening lags regressed out -- which is what distinguishes a genuine lag-4
    effect from lag 1 echoing through four steps.

    A pairwise-complete ACF is not guaranteed positive definite, so the
    recursion can in principle stray outside [-1, 1]. That is reported rather
    than silently clipped.
    """
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    phi = np.zeros((max_lag + 1, max_lag + 1))
    if max_lag >= 1:
        phi[1, 1] = acf[1]
        pacf[1] = acf[1]
    for k in range(2, max_lag + 1):
        numerator = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
        phi[k, k] = numerator / denominator if denominator != 0 else np.nan
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf[k] = phi[k, k]
    return pacf


# ---------------------------------------------------------------------------
# STEP 5 -- figures
# ---------------------------------------------------------------------------
def figure_scale(frame, out_dir) -> None:
    """Figure 1: does dividing by RANDUNC remove the funnel?"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    ax.scatter(frame["nee_mod"], frame["residual"], s=7, alpha=0.25,
               color=COL_MAIN, edgecolors="none")
    ax.axhline(0.0, color=COL_REF, lw=1.2)
    ax.set_xlabel("modelled NEE, g C m$^{-2}$ d$^{-1}$")
    ax.set_ylabel("residual (observed - modelled), g C m$^{-2}$ d$^{-1}$")
    ax.set_title("Raw residual: expect a funnel", loc="left")

    ax = axes[1]
    std = frame["residual_std"]
    ax.scatter(frame["nee_mod"], std, s=7, alpha=0.25, color=COL_MAIN,
               edgecolors="none")
    ax.axhline(0.0, color=COL_REF, lw=1.2)
    for level in (-2.0, 2.0):
        ax.axhline(level, color=COL_ALERT, lw=1.3, ls="--")
    ax.set_xlabel("modelled NEE, g C m$^{-2}$ d$^{-1}$")
    ax.set_ylabel("standardised residual (residual / RANDUNC)")
    beyond = float(np.mean(np.abs(std) > 2.0))
    ax.set_title(
        f"Standardised: sd {std.std():.2f}, "
        f"{100 * beyond:.1f}% beyond $\\pm$2 (Gaussian: 4.6%)",
        loc="left",
    )

    fig.suptitle(
        "Figure 1. Is RANDUNC the right scale?  If it is, the right panel is a "
        "band of constant width with sd 1.",
        x=0.005, ha="left", y=0.999,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    print("  ", save_figure(fig, out_dir, "fig25_residual_scale"))
    plt.close(fig)


def figure_distribution(frame, out_dir) -> None:
    """Figure 2: Gaussian against Laplace on the standardised residuals."""
    import matplotlib.pyplot as plt
    from scipy import stats

    std = frame["residual_std"].to_numpy()
    grid = np.linspace(np.percentile(std, 0.2), np.percentile(std, 99.8), 800)

    # Both fitted to the same data by maximum likelihood, so the comparison is
    # like for like. Laplace MLE: loc = median, scale = mean |x - median|.
    n_loc, n_scale = stats.norm.fit(std)
    l_loc, l_scale = stats.laplace.fit(std)
    ll_norm = float(np.sum(stats.norm.logpdf(std, n_loc, n_scale)))
    ll_lap = float(np.sum(stats.laplace.logpdf(std, l_loc, l_scale)))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.hist(std, bins=140, density=True, color="0.78", zorder=1,
            label=f"standardised residuals (n = {len(std):,})")
    ax.plot(grid, stats.norm.pdf(grid, n_loc, n_scale), lw=2.0, color=COL_MAIN,
            zorder=3, label=f"Gaussian fit  ($\\mu$ {n_loc:.2f}, $\\sigma$ "
                            f"{n_scale:.2f}),  logL {ll_norm:,.0f}")
    ax.plot(grid, stats.laplace.pdf(grid, l_loc, l_scale), lw=2.0,
            color=COL_ALERT, zorder=3,
            label=f"Laplace fit  (loc {l_loc:.2f}, scale {l_scale:.2f}),  "
                  f"logL {ll_lap:,.0f}")
    ax.plot(grid, stats.norm.pdf(grid, 0.0, 1.0), lw=1.6, ls="--",
            color=COL_REF, zorder=2, label="standard normal (what the likelihood assumes)")
    ax.set_xlabel("standardised residual")
    ax.set_ylabel("density")
    winner = "Laplace" if ll_lap > ll_norm else "Gaussian"
    ax.set_title(
        f"Figure 2. Shape of the error. Excess kurtosis {stats.kurtosis(std):.2f} "
        f"(Gaussian: 0). Better fit: {winner} by "
        f"{abs(ll_lap - ll_norm):,.0f} log units.",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    print("  ", save_figure(fig, out_dir, "fig26_residual_distribution"))
    plt.close(fig)


def figure_pacf(frame, out_dir) -> None:
    """Figure 3: partial autocorrelation of the raw residuals, gaps respected."""
    import matplotlib.pyplot as plt

    dates = pd.DatetimeIndex(frame["date"])
    acf, counts = gap_aware_acf(dates, frame["residual"].to_numpy(), MAX_LAG)
    pacf = pacf_from_acf(acf, MAX_LAG)
    lags = np.arange(MAX_LAG + 1)

    # 95% band under the null of white noise, using the pairs actually available
    # at each lag rather than a single nominal n.
    band = 1.96 / np.sqrt(np.maximum(counts, 1))

    fig, ax = plt.subplots(figsize=(11, 5.4))
    significant = np.abs(pacf) > band
    significant[0] = False
    ax.vlines(lags, 0, pacf, lw=2.4,
              color=[COL_ALERT if s else COL_MAIN for s in significant], zorder=3)
    ax.scatter(lags, pacf, s=22,
               color=[COL_ALERT if s else COL_MAIN for s in significant], zorder=4)
    ax.fill_between(lags, -band, band, color=COL_REF, alpha=0.16, zorder=1,
                    label="95% band under white noise")
    ax.axhline(0.0, color=COL_REF, lw=1.0, zorder=2)
    ax.set_xlabel("lag, days")
    ax.set_ylabel("partial autocorrelation")
    flagged = [int(k) for k in lags[significant]]
    ax.set_title(
        "Figure 3. Partial autocorrelation of raw residuals, gaps respected.\n"
        f"Significant lags: {flagged if flagged else 'none'}",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(-0.6, MAX_LAG + 0.6)
    fig.text(0.005, 0.004,
             f"Pairs available: lag 1 n={counts[1]:,}, lag 40 n={counts[40]:,}. "
             "Lag k uses only pairs exactly k calendar days apart.",
             fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    print("  ", save_figure(fig, out_dir, "fig27_residual_pacf"))
    plt.close(fig)

    if np.nanmax(np.abs(pacf[1:])) > 1.0:
        print("   WARNING: |PACF| > 1 at some lag. The pairwise-complete ACF is "
              "not positive definite; treat high lags as indicative only.")
    return flagged, pacf, band


def figure_seasonality(frame, out_dir) -> None:
    """Figure 4: mean residual by day of year."""
    import matplotlib.pyplot as plt

    dates = pd.DatetimeIndex(frame["date"])
    doy = dates.dayofyear.to_numpy()
    residual = frame["residual"].to_numpy()

    days = np.arange(1, 366)
    mean = np.full(days.size, np.nan)
    sem = np.full(days.size, np.nan)
    for i, d in enumerate(days):
        sel = doy == d
        if sel.sum() >= 3:
            mean[i] = residual[sel].mean()
            sem[i] = residual[sel].std(ddof=1) / np.sqrt(sel.sum())

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.fill_between(days, mean - 1.96 * sem, mean + 1.96 * sem, color=COL_MAIN,
                    alpha=0.22, zorder=2, label="95% interval on the mean")
    ax.plot(days, mean, lw=1.7, color=COL_MAIN, zorder=3,
            label="mean residual by day of year")
    ax.axhline(0.0, color=COL_REF, lw=1.4, zorder=4)
    ax.set_xlabel("day of year")
    ax.set_ylabel("mean residual (observed - modelled), g C m$^{-2}$ d$^{-1}$")
    ax.set_xlim(1, 365)
    ax.set_title(
        "Figure 4. Seasonal structure in the residual.\n"
        "Positive = the model takes up more carbon than was observed "
        "(NEE is negative for uptake).",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    print("  ", save_figure(fig, out_dir, "fig28_residual_seasonality"))
    plt.close(fig)
    return mean, days


# ---------------------------------------------------------------------------
def main() -> int:
    for category in (RuntimeWarning, UserWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)

    args = parse_args()
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_path = Path(args.residuals)

    if residual_path.exists() and not args.force:
        print(f"Using existing residual table: {residual_path}")
        print("  (steps 1-3 skipped; pass --force to rebuild)")
        frame = pd.read_csv(residual_path, parse_dates=["date"])
    else:
        config = load_config(args.config)
        frame = build_residual_table(config, Path(args.trace))
        frame = add_residuals(frame)
        frame.to_csv(residual_path, index=False)
        print(f"  wrote {residual_path}")

    if "residual" not in frame.columns:
        frame = add_residuals(frame)

    # -- STEP 4 ----------------------------------------------------------
    std = frame["residual_std"]
    print("\nSTEP 4 -- residual summary")
    print(f"  n days                     {len(frame):,}")
    print(f"  raw residual   mean {frame['residual'].mean():+.4f}  "
          f"sd {frame['residual'].std():.4f}")
    print(f"  standardised   mean {std.mean():+.4f}  sd {std.std():.4f}")
    print(f"  |z| > 2                    {100 * np.mean(np.abs(std) > 2):.1f}%  "
          f"(Gaussian expects 4.6%)")
    print(f"  |z| > 3                    {100 * np.mean(np.abs(std) > 3):.1f}%  "
          f"(Gaussian expects 0.3%)")

    # -- STEP 5 ----------------------------------------------------------
    apply_style()
    print("\nSTEP 5 -- figures")
    figure_scale(frame, out_dir)
    figure_distribution(frame, out_dir)
    flagged, _pacf, _band = figure_pacf(frame, out_dir)
    figure_seasonality(frame, out_dir)

    sd = float(std.std())
    print("\n" + "=" * 70)
    print("  READ-OUT")
    print("=" * 70)
    print(f"  Standardised residual sd is {sd:.2f}.")
    if sd > 1.0:
        print(f"  RANDUNC is too small by a factor of about {sd:.1f}. The")
        print("  likelihood over-weights every day, so the posterior is narrower")
        print("  than the data support.")
    else:
        print("  RANDUNC is not too small; the scale assumption holds.")
    if flagged:
        print(f"  Residuals are autocorrelated at lags {flagged}. Consecutive days")
        print("  repeat information the likelihood counts as independent, which")
        print("  narrows the posterior further.")
    else:
        print("  No significant partial autocorrelation; the independence")
        print("  assumption is not contradicted.")
    print("\n  Diagnostic only. No likelihood change is made or implied here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
