#!/usr/bin/env python
"""Compare DALEC2 and the evergreen DALEC fitted to the same data. Samples nothing.

Reads the two posterior traces written by ``scripts/04_calibrate.py`` and
reports, for each model:

1. **Convergence.** Worst r-hat, smallest bulk and tail ESS, divergences, BFMI
   per chain, from ``az.summary(..., round_to="none")``. Reported first because
   everything below assumes it: if either model has r-hat above 1.01, the
   summary opens with a warning that the comparison is not yet reliable.
   Whether the evergreen model converges where DALEC2 did not is itself an RQ3
   result.
2. **Posterior contraction**, 1 - posterior variance / prior variance, for every
   sampled quantity, with counts below 0.1 and at a prior bound. Reported for
   all parameters and separately for the shared ones, because removing the
   phenology removes four parameters that could never contract (``d_fall`` is
   inert), which flatters the evergreen totals by construction. Allocation is
   compared as shares of NPP, so the evergreen foliar share faces DALEC2's
   labile plus foliar share: the two have identical priors.
3. **Skill** on the assimilable (``nee_mask``) days of the calibration block and
   of the held-out prediction block, all days and spring (March-May) alone:
   RMSE and mean bias of the posterior mean, coverage of the posterior
   *predictive* interval (model plus Gaussian noise with RANDUNC as sd, which is
   what an observation is compared against), and the log predictive density
   per day, the one score that rewards fit and honest uncertainty together.

The daily NEE comes from forward runs of each model's own compiled PyTensor
graph for a thinned set of posterior draws, over the calibration and
prediction blocks joined into one record.

Usage
-----
    python scripts/compare_structures.py \\
        results/calibration_dalec2_hemisurface.nc \\
        results/calibration_evergreen_hemisurface.nc
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.compute import compile_function  # noqa: E402
from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    require_year_block,
    resolve_path,
)
from dalec.data_io import SiteData  # noqa: E402
from dalec.model import build_forward_graph  # noqa: E402
from dalec.model_evergreen import (  # noqa: E402
    EVERGREEN_ALLOCATION_ORDER,
    EVERGREEN_PARAMETER_NAMES,
    build_evergreen_graph,
    evergreen_allocation_concentration,
    evergreen_prior_sources,
)
from dalec.parameters import (  # noqa: E402
    ALLOCATION_WEIGHT_ORDER,
    DEFAULT_LAI_CONVENTION,
    PARAMETER_NAMES,
    allocation_concentration,
)
from dalec.plotting import OKABE_ITO, apply_style, save_figure  # noqa: E402
from dalec.priors import prior_sources  # noqa: E402

OUT_DIR = Path("results/comparison")
VARIANTS = ("dalec2", "evergreen")
COLOURS = {"dalec2": OKABE_ITO[0], "evergreen": OKABE_ITO[1]}

#: Posterior draws re-run forward over the whole record.
DEFAULT_FORWARD_DRAWS = 200
#: Noise realisations per posterior draw for the predictive interval.
NOISE_REPLICATES = 10
#: Thresholds named in the task.
RHAT_LIMIT = 1.01
LOW_CONTRACTION = 0.1
#: A posterior mean within this fraction of its prior range from either end is
#: "at a bound" -- the rule scripts/04_calibrate.py reports with.
BOUND_TOLERANCE = 0.05
SPRING_MONTHS = (3, 4, 5)

GRAPHS = {
    "dalec2": (PARAMETER_NAMES, build_forward_graph),
    "evergreen": (EVERGREEN_PARAMETER_NAMES, build_evergreen_graph),
}

#: Allocation compared as shares of NPP. Groups name the simplex components
#: summed into each share; DALEC2's labile and direct foliar weights are also
#: listed on their own, as DALEC2-only quantities.
SHARE_GROUPS = {
    "dalec2": {
        "foliage_share": ("f_lab", "f_fol"),
        "root_share": ("f_roo",),
        "wood_share": ("f_woo",),
        "labile_weight": ("f_lab",),
        "direct_foliar_weight": ("f_fol",),
    },
    "evergreen": {
        "foliage_share": ("f_fol",),
        "root_share": ("f_roo",),
        "wood_share": ("f_woo",),
    },
}
SHARED_SHARES = ("foliage_share", "root_share", "wood_share")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dalec2", type=Path, help="DALEC2 posterior .nc")
    parser.add_argument("evergreen", type=Path, help="evergreen posterior .nc")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--draws", type=int, default=DEFAULT_FORWARD_DRAWS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_record(config) -> tuple[SiteData, int]:
    """The calibration and prediction blocks joined into one daily record."""
    processed = resolve_path(config["paths"]["processed_dir"])
    slug = str(config["site"]["code"]).lower().replace("-", "_")
    blocks = []
    for key in ("calibration", "prediction"):
        first, last = require_year_block(config, key)
        blocks.append(SiteData.load(processed / f"{slug}_{key}_{first}_{last}.nc"))
    step = (blocks[1].time[0] - blocks[0].time[-1]).astype("timedelta64[D]").astype(int)
    if step != 1:
        raise SystemExit("the calibration and prediction blocks do not abut")
    fields = ("time", "doy", "t_air", "t_day", "t_night", "t_max", "t_min",
              "sw_in", "co2", "nee_obs", "nee_unc", "nee_qc", "nee_mask")
    joined = {name: np.concatenate([getattr(b, name) for b in blocks]) for name in fields}
    return SiteData(**joined), blocks[0].n_days


def forward_nee(idata, variant, record, acm, n_draws) -> np.ndarray:
    """Daily NEE, shape (n_draws, n_days), for evenly spaced posterior draws."""
    import pytensor.tensor as pt

    names, builder = GRAPHS[variant]
    theta = pt.dvector("theta")
    graph = builder(
        parameters={name: theta[i] for i, name in enumerate(names)},
        doy=record.doy.astype(float),
        t_air=record.t_air,
        t_max=record.t_max,
        t_min=record.t_min,
        sw_in=record.sw_in,
        co2=record.co2,
        latitude_deg=acm.latitude_deg,
        coefficients=acm.coefficients,
        frost_threshold_degc=acm.frost_threshold_degc,
    )
    run = compile_function([theta], graph.nee, on_unused_input="ignore")
    posterior = idata["posterior"]
    values = np.column_stack([posterior[name].values.reshape(-1) for name in names])
    picks = np.linspace(0, len(values) - 1, n_draws).astype(int)
    return np.array([run(values[i]) for i in picks])


# ---------------------------------------------------------------------------
# Convergence and contraction
# ---------------------------------------------------------------------------
def convergence(idata) -> dict[str, object]:
    import arviz as az

    table = az.summary(idata, kind="diagnostics", round_to="none")
    diverging = idata["sample_stats"]["diverging"]
    return {
        "worst_rhat": float(table["r_hat"].max()),
        "worst_rhat_var": str(table["r_hat"].idxmax()),
        "min_ess_bulk": float(table["ess_bulk"].min()),
        "min_ess_tail": float(table["ess_tail"].min()),
        "divergences": f"{int(diverging.sum())} of {diverging.size}",
        "bfmi": np.asarray(az.bfmi(idata)["energy"]).round(3).tolist(),
    }


def share_samples(idata, variant) -> dict[str, np.ndarray]:
    """Posterior allocation shares, flattened over chains and draws."""
    order = ALLOCATION_WEIGHT_ORDER if variant == "dalec2" else EVERGREEN_ALLOCATION_ORDER
    weights = idata["posterior"]["allocation_weights"].values
    weights = weights.reshape(-1, weights.shape[-1])
    return {
        share: weights[:, [order.index(c) for c in components]].sum(axis=1)
        for share, components in SHARE_GROUPS[variant].items()
    }


def share_priors(variant) -> dict[str, tuple[float, float]]:
    """Beta(a, b) marginal of each share under the model's Dirichlet."""
    if variant == "dalec2":
        order, alpha = ALLOCATION_WEIGHT_ORDER, allocation_concentration()
    else:
        order, alpha = EVERGREEN_ALLOCATION_ORDER, evergreen_allocation_concentration()
    total = float(alpha.sum())
    priors = {}
    for share, components in SHARE_GROUPS[variant].items():
        a = float(sum(alpha[order.index(c)] for c in components))
        priors[share] = (a, total - a)
    return priors


def contraction_table(idata, variant, convention) -> pd.DataFrame:
    """1 - posterior variance / prior variance for every sampled quantity."""
    bounds = prior_sources(convention) if variant == "dalec2" else \
        evergreen_prior_sources(convention)
    shared = set(evergreen_prior_sources(convention)) | set(SHARED_SHARES)
    rows = []
    for name, (low, high) in bounds.items():
        values = idata["posterior"][name].values.reshape(-1)
        position = (values.mean() - low) / (high - low)
        rows.append({
            "parameter": name,
            "prior_var": (high - low) ** 2 / 12.0,
            "posterior_mean": values.mean(),
            "posterior_var": values.var(),
            "position": position,
            "at_bound": bool(position <= BOUND_TOLERANCE or position >= 1 - BOUND_TOLERANCE),
        })
    for share, values in share_samples(idata, variant).items():
        a, b = share_priors(variant)[share]
        rows.append({
            "parameter": share,
            "prior_var": a * b / ((a + b) ** 2 * (a + b + 1)),
            "posterior_mean": values.mean(),
            "posterior_var": values.var(),
            "position": np.nan,
            "at_bound": False,
        })
    table = pd.DataFrame(rows).set_index("parameter")
    table["contraction"] = 1.0 - table["posterior_var"] / table["prior_var"]
    table["shared"] = [name in shared for name in table.index]
    return table


def contraction_counts(table) -> dict[str, int]:
    shared = table[table["shared"]]
    return {
        "n_all": len(table),
        "low_all": int((table["contraction"] < LOW_CONTRACTION).sum()),
        "bound_all": int(table["at_bound"].sum()),
        "n_shared": len(shared),
        "low_shared": int((shared["contraction"] < LOW_CONTRACTION).sum()),
        "bound_shared": int(shared["at_bound"].sum()),
    }


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------
def skill(nee, record, days, prob, rng) -> dict[str, float]:
    """RMSE, bias, predictive coverage and log predictive density on `days`."""
    from scipy.special import logsumexp

    model = nee[:, days]
    observed = record.nee_obs[days]
    sigma = record.nee_unc[days]
    mean = model.mean(axis=0)

    replicated = np.repeat(model, NOISE_REPLICATES, axis=0)
    predictive = replicated + rng.normal(0.0, sigma, size=replicated.shape)
    tail = 50.0 * (1.0 - prob)
    low, high = np.percentile(predictive, [tail, 100.0 - tail], axis=0)

    log_density = (-0.5 * ((observed - model) / sigma) ** 2
                   - np.log(sigma) - 0.5 * math.log(2.0 * math.pi))
    pointwise = logsumexp(log_density, axis=0) - math.log(model.shape[0])
    return {
        "n_days": int(days.size),
        "rmse": float(np.sqrt(np.mean((mean - observed) ** 2))),
        "bias": float(np.mean(mean - observed)),
        "coverage": float(np.mean((observed >= low) & (observed <= high))),
        "lpd_per_day": float(pointwise.mean()),
    }


def skill_table(runs, record, n_calibration_days, prob, seed) -> pd.DataFrame:
    months = pd.DatetimeIndex(record.time).month
    in_calibration = np.arange(record.n_days) < n_calibration_days
    windows = {
        "calibration": record.nee_mask & in_calibration,
        "held-out": record.nee_mask & ~in_calibration,
    }
    rows = []
    for variant, nee in runs.items():
        rng = np.random.default_rng(seed)
        for window, mask in windows.items():
            for subset, keep in (("all", mask), ("spring", mask & np.isin(months, SPRING_MONTHS))):
                days = np.flatnonzero(keep)
                rows.append({"model": variant, "window": window, "days": subset,
                             **skill(nee, record, days, prob, rng)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_nee(runs, record, n_calibration_days, prob, out_dir) -> None:
    import matplotlib.pyplot as plt

    tail = 50.0 * (1.0 - prob)
    held_out = record.time[n_calibration_days:]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, zoom in zip(axes, (False, True), strict=True):
        keep = np.arange(record.n_days) >= (n_calibration_days if zoom else 0)
        ax.axvspan(held_out[0], held_out[-1], color=OKABE_ITO[4], alpha=0.12, lw=0,
                   label="held-out prediction window")
        for variant, nee in runs.items():
            low, high = np.percentile(nee[:, keep], [tail, 100.0 - tail], axis=0)
            ax.fill_between(record.time[keep], low, high, color=COLOURS[variant],
                            alpha=0.35, lw=0, label=f"{variant}, {prob:.0%} band")
        mask = record.nee_mask & keep
        ax.plot(record.time[mask], record.nee_obs[mask], ".", ms=1.2, color=OKABE_ITO[7],
                label="observed (nee_mask days)")
        ax.set_ylabel("NEE (g C m-2 d-1)")
    axes[0].set_title("Observed NEE against both models' posterior bands")
    axes[1].set_title("Held-out prediction window")
    axes[0].legend(ncol=4, loc="upper left")
    save_figure(fig, out_dir, "fig1_nee_both_models")


def figure_shared_posteriors(idatas, convention, out_dir) -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    bounds = evergreen_prior_sources(convention)
    names = [*bounds, *SHARED_SHARES]
    ncols = 4
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.2 * nrows))
    for ax, name in zip(axes.flat, names, strict=False):
        if name in bounds:
            low, high = bounds[name]
            ax.hlines(1.0 / (high - low), low, high, color=OKABE_ITO[7], lw=1.5, label="prior")
        else:
            low, high = 0.0, 1.0
            grid = np.linspace(low, high, 200)
            ax.plot(grid, beta.pdf(grid, *share_priors("evergreen")[name]),
                    color=OKABE_ITO[7], lw=1.5, label="prior")
        for variant, idata in idatas.items():
            values = (idata["posterior"][name].values.reshape(-1) if name in bounds
                      else share_samples(idata, variant)[name])
            ax.hist(values, bins=40, range=(low, high), density=True, alpha=0.5,
                    color=COLOURS[variant], label=variant)
        ax.set_title(name)
        ax.set_yticks([])
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    save_figure(fig, out_dir, "fig2_shared_posteriors")


def figure_contraction(tables, out_dir) -> None:
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys([*tables["evergreen"].index, *tables["dalec2"].index]))
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1.5))
    for offset, (variant, table) in zip((-0.2, 0.2), tables.items(), strict=True):
        values = [table["contraction"].get(name, np.nan) for name in names]
        ax.barh(positions + offset, values, height=0.4, color=COLOURS[variant], label=variant)
    ax.axvline(LOW_CONTRACTION, color=OKABE_ITO[7], ls="--", lw=1,
               label=f"contraction {LOW_CONTRACTION}")
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("posterior contraction, 1 - var(posterior) / var(prior)")
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, "fig3_contraction")


# ---------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import arviz as az

    args = parse_args()
    config = load_config(args.config)
    prob = float(config["diagnostics"]["hdi_prob"])
    seed = int(config["seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    idatas = {"dalec2": az.from_netcdf(args.dalec2), "evergreen": az.from_netcdf(args.evergreen)}
    for variant, idata in idatas.items():
        stored = idata.attrs.get("model_variant")
        if stored is not None and stored != variant:
            raise SystemExit(f"{variant} argument is a {stored} trace")
    conventions = {idata.attrs.get("convention", DEFAULT_LAI_CONVENTION)
                   for idata in idatas.values()}
    if len(conventions) != 1:
        raise SystemExit(f"the traces use different LAI conventions: {conventions}")
    convention = conventions.pop()

    record, n_calibration_days = load_record(config)
    acm = acm_from_config(config)
    runs = {variant: forward_nee(idata, variant, record, acm, args.draws)
            for variant, idata in idatas.items()}

    health = {variant: convergence(idata) for variant, idata in idatas.items()}
    tables = {variant: contraction_table(idata, variant, convention)
              for variant, idata in idatas.items()}
    scores = skill_table(runs, record, n_calibration_days, prob, seed)

    lines = []
    unconverged = [v for v, h in health.items() if h["worst_rhat"] > RHAT_LIMIT]
    if unconverged:
        lines += [
            "!" * 74,
            f"  WARNING: worst r-hat above {RHAT_LIMIT} for {', '.join(unconverged)}.",
            "  The comparison is NOT yet reliable: contraction, skill and every",
            "  posterior number below pool chains that have not mixed.",
            "!" * 74, "",
        ]
    lines += [f"DALEC2 vs evergreen DALEC -- convention {convention}, "
              f"{args.draws} forward draws per model", "", "CONVERGENCE"]
    for variant, h in health.items():
        lines.append(f"  {variant:<10} worst r-hat {h['worst_rhat']:.4f} ({h['worst_rhat_var']})"
                     f"   min ESS bulk {h['min_ess_bulk']:.1f}, tail {h['min_ess_tail']:.1f}"
                     f"   divergences {h['divergences']}   BFMI {h['bfmi']}")
    lines += ["", f"CONTRACTION (low = below {LOW_CONTRACTION}; at bound = posterior mean "
                  f"within {BOUND_TOLERANCE:.0%} of a prior bound)"]
    for variant, table in tables.items():
        c = contraction_counts(table)
        lines.append(f"  {variant:<10} all {c['n_all']:>2}: {c['low_all']:>2} low, "
                     f"{c['bound_all']:>2} at bound   |   shared {c['n_shared']:>2}: "
                     f"{c['low_shared']:>2} low, {c['bound_shared']:>2} at bound")
        table.to_csv(args.out / f"contraction_{variant}.csv")
    lines += ["", "  shared parameters, contraction (dalec2 | evergreen)"]
    for name in tables["evergreen"].index[tables["evergreen"]["shared"].to_numpy()]:
        lines.append(f"    {name:<22} {tables['dalec2'].loc[name, 'contraction']:>7.3f}"
                     f" | {tables['evergreen'].loc[name, 'contraction']:>7.3f}")
    lines += ["", f"SKILL on nee_mask days ({prob:.0%} predictive interval includes RANDUNC "
                  "noise; lpd = mean log predictive density per day)",
              scores.to_string(index=False, float_format=lambda x: f"{x:.4f}")]
    scores.to_csv(args.out / "skill.csv", index=False)

    summary = "\n".join(lines) + "\n"
    (args.out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    figure_nee(runs, record, n_calibration_days, prob, args.out)
    figure_shared_posteriors(idatas, convention, args.out)
    figure_contraction(tables, args.out)
    print(f"figures, tables and summary in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
