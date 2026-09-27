#!/usr/bin/env python
"""Figures and tables from an existing posterior. Samples nothing.

Usage, from the repository root:
    python showcase/make_results.py <posterior.nc> [<twin_posterior.nc>]

The daily bands come from re-running the forward model for a set of saved
posterior draws. The twin posterior defaults to paths.twin_posterior in
config.yaml, and the twin outputs are skipped if that file does not exist.
"""

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.config import load_config, resolve_path  # noqa: E402
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402
from dalec.parameters import PARAMETER_NAMES, DalecParameters  # noqa: E402
from dalec.plotting import OKABE_ITO, apply_style, save_figure  # noqa: E402
from dalec.priors import prior_sources  # noqa: E402
from model import load_data  # noqa: E402

matplotlib.use("Agg")
CONFIG = Path(__file__).with_name("config.yaml")
BLUE, VERMILLION, BLACK = OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[7]


def print_convergence(idata, label):
    """Worst R-hat, smallest bulk and tail ESS, divergences, BFMI per chain. Unrounded."""
    table = az.summary(idata, kind="diagnostics", round_to="none")
    diverging = idata["sample_stats"]["diverging"]
    bfmi = np.asarray(az.bfmi(idata)["energy"])
    print(f"\n=== {label} ===")
    print(f"worst R-hat         {table['r_hat'].max():.4f}  ({table['r_hat'].idxmax()})")
    print(f"smallest bulk ESS   {table['ess_bulk'].min():.1f}  ({table['ess_bulk'].idxmin()})")
    print(f"smallest tail ESS   {table['ess_tail'].min():.1f}  ({table['ess_tail'].idxmin()})")
    print(f"divergences         {int(diverging.sum())} of {diverging.size}")
    print("BFMI per chain      " + "  ".join(f"{value:.3f}" for value in bfmi))
    return float(table["r_hat"].max())


def forward_runs(idata, data, acm, n_runs):
    """Daily fluxes from the numpy DALEC2 in src/dalec, for evenly spaced posterior draws."""
    posterior = idata["posterior"]
    n_chains, n_draws = posterior["f_auto"].shape
    runs = {"nee": [], "gpp": [], "reco": [], "c_fol": []}
    for flat in np.linspace(0, n_chains * n_draws - 1, n_runs).astype(int):
        chain, draw = divmod(flat, n_draws)
        values = {name: float(posterior[name][chain, draw]) for name in PARAMETER_NAMES}
        output = run_dalec2(DalecParameters(**values), data,
                            gpp_fn=acm, phenology_fn=dalec2_phenology)
        runs["nee"].append(output.nee)
        runs["gpp"].append(output.gpp)
        runs["reco"].append(output.reco)
        runs["c_fol"].append(output.pool("c_fol")[1:])
    return {name: np.array(series) for name, series in runs.items()}


def band(series, prob):
    """Lower edge, median and upper edge of the central `prob` interval, per day."""
    tail = 50.0 * (1.0 - prob)
    return np.percentile(series, [tail, 50.0, 100.0 - tail], axis=0)


def figure_nee(data, runs, config, note, out_dir):
    prob = config["results"]["interval_prob"]
    low, median, high = band(runs["nee"], prob)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for key, colour, label in (("calibration", OKABE_ITO[5], "calibration"),
                               ("prediction", OKABE_ITO[4], "held-out prediction")):
        first, last = config["years"][key]
        ax.axvspan(np.datetime64(f"{first}-01-01"), np.datetime64(f"{last}-12-31"),
                   color=colour, alpha=0.15, lw=0, label=f"{label} {first}-{last}")
    ax.fill_between(data.time, low, high, color=BLUE, alpha=0.35, lw=0,
                    label=f"model, {prob:.0%} band")
    ax.plot(data.time, median, color=BLUE, lw=0.5, label="model, median")
    mask = data.nee_mask
    ax.plot(data.time[mask], data.nee_obs[mask], ".", ms=1.2, color=BLACK,
            label="observed (nee_mask days)")
    ax.set_ylabel("NEE (g C m-2 d-1)")
    ax.set_title(f"Daily NEE, observed and modelled   ({note})")
    ax.legend(ncol=3, loc="upper left")
    save_figure(fig, out_dir, "fig1_nee_timeseries")


def figure_prior_posterior(idata, bounds, names, out_dir, stem, truth=None):
    ncols = 4
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.0 * nrows))
    for ax, name in zip(axes.flat, names, strict=False):
        low, high = bounds[name]
        ax.hist(np.ravel(idata["posterior"][name]), bins=40, range=(low, high), density=True,
                color=VERMILLION, alpha=0.7, label="posterior")
        ax.hlines(1.0 / (high - low), low, high, color=BLUE, lw=2, label="prior")
        if truth is not None:
            ax.axvline(truth[name], color=BLACK, ls="--", label="true value")
        ax.set_title(name)
        ax.set_yticks([])
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    save_figure(fig, out_dir, stem)


def figure_fluxes(data, runs, prob, note, out_dir):
    panels = (("gpp", "GPP (g C m-2 d-1)", "gpp_nt"),
              ("reco", "Reco = Ra + Rh (g C m-2 d-1)", "reco_nt"),
              ("c_fol", "foliar carbon (g C m-2)", None))
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax, (key, label, product) in zip(axes, panels, strict=True):
        low, median, high = band(runs[key], prob)
        ax.fill_between(data.time, low, high, color=BLUE, alpha=0.35, lw=0,
                        label=f"model, {prob:.0%} band")
        ax.plot(data.time, median, color=BLUE, lw=0.5, label="model, median")
        if product in data.partitioned:
            ax.plot(data.time, data.partitioned[product], color=BLACK, lw=0.3, alpha=0.6,
                    label=f"FLUXNET {product.upper()} (not assimilated)")
        ax.set_ylabel(label)
        ax.legend(ncol=3, loc="upper left")
    axes[0].set_title(f"Posterior GPP, ecosystem respiration and foliar carbon   ({note})")
    save_figure(fig, out_dir, "fig3_gpp_reco")


def figure_rank(idata, names, out_dir):
    plot = az.plot_rank(idata, var_names=names, backend="matplotlib")
    for suffix in (".pdf", ".png"):
        plot.savefig(out_dir / f"fig4_rank{suffix}")


def interval_table(idata, names, prob, truth=None):
    """Posterior mean and central interval; with a truth, whether it was recovered."""
    summary = az.summary(idata, var_names=names, kind="stats", ci_prob=prob, ci_kind="eti",
                         round_to="none").loc[names]
    lower, upper = f"{prob:.0%} lower", f"{prob:.0%} upper"
    table = pd.DataFrame({
        "posterior mean": summary["mean"].to_numpy(),
        lower: summary.filter(like="_lb").iloc[:, 0].to_numpy(),
        upper: summary.filter(like="_ub").iloc[:, 0].to_numpy(),
    }, index=pd.Index(names, name="parameter"))
    if truth is not None:
        table.insert(0, "true value", [truth[name] for name in names])
        inside = table["true value"].between(table[lower], table[upper])
        table["recovered"] = np.where(inside, "yes", "no")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("posterior", type=Path, help="posterior .nc from a calibration run")
    parser.add_argument("twin", type=Path, nargs="?", help="twin posterior .nc")
    args = parser.parse_args()

    config = load_config(CONFIG)
    results = config["results"]
    prob, names = results["interval_prob"], results["main_parameters"]
    out_dir = resolve_path(config["paths"]["outputs_dir"])
    figures = resolve_path(config["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    apply_style()

    # ---- The real-data posterior -------------------------------------------
    idata = az.from_netcdf(args.posterior)
    bounds = prior_sources(idata.attrs.get("convention", config["lai_convention"]))
    worst_rhat = print_convergence(idata, f"posterior {args.posterior}")
    note = f"worst R-hat {worst_rhat:.2f}"

    first, last = config["years"]["calibration"][0], config["years"]["prediction"][1]
    data = load_data(config, first, last)
    runs = forward_runs(idata, data, acm_from_config(config), results["n_forward_draws"])
    figure_nee(data, runs, config, note, figures)
    figure_prior_posterior(idata, bounds, names, figures, "fig2_prior_posterior")
    figure_fluxes(data, runs, prob, note, figures)
    figure_rank(idata, results["rank_plot_parameters"], figures)
    table = interval_table(idata, names, prob)
    table.to_csv(out_dir / "posterior_summary.csv")
    print(f"\n{table.to_string()}")

    # ---- The synthetic twin ------------------------------------------------
    twin_path = args.twin or resolve_path(config["paths"]["twin_posterior"])
    if not twin_path.exists():
        print(f"\nno twin posterior at {twin_path}; twin outputs skipped")
        return 0
    twin = az.from_netcdf(twin_path)
    truth = json.loads(twin_path.with_suffix(".truth.json").read_text(encoding="utf-8"))
    print_convergence(twin, f"twin posterior {twin_path}")
    twin_bounds = prior_sources(twin.attrs.get("convention", config["lai_convention"]))
    figure_prior_posterior(twin, twin_bounds, names, figures, "fig5_twin_recovery", truth)
    table = interval_table(twin, names, prob, truth)
    table.to_csv(out_dir / "twin_recovery.csv")
    print(f"\n{table.to_string()}")
    print(f"\nfigures in {figures}, tables in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
