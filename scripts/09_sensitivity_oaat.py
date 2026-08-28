#!/usr/bin/env python
"""Task 2, reduced: one-at-a-time sensitivity over the calibration block.

Two figures only:

    fig05_oaat_loglik_panels   log-likelihood against each parameter, small
                               multiples on a shared y-axis
    fig06_oaat_ranking         parameters ordered by d_loglik

The sweep anchor is the component-wise median of the 163 draws that survived
Task 1's screening -- **not** the prior median. The prior median is not merely a
worse anchor, it is a point Task 1's own screen rejects: it produces daily fluxes
above 30 g C m-2 d-1 and an RMSE of 35 against a record whose total net flux
never exceeds about 7. Sweeping around a point the model classifies as
non-physical measures sensitivity in a region the posterior will never visit.

No fig07. No prior is changed. Finite-difference gradients (amendment A3) are
dropped from this reduced scope, so grad_scaled is not computed.

Two deliberate departures from the letter of section 2.3, both serving its intent
rather than defeating it:

* fig05's shared y-axis is **symlog**. The specification asks for a shared axis
  "so flatness is comparable by eye". On a shared *linear* axis c_som_0 and
  theta_som span 9e7 units and every other panel renders as a flat line, hiding
  variation of 1e5 to 1e6. Symlog keeps one axis across all panels while leaving
  that variation visible.
* fig06 draws section 2.4's reading aids at 3 and 10 units but labels them
  inapplicable rather than treating them as thresholds. At a prior-derived anchor
  the absolute scale is meaningless; only the ranking carries information.

Usage
-----
    python scripts/09_sensitivity_oaat.py
    python scripts/09_sensitivity_oaat.py --figures-only
"""

from __future__ import annotations

import argparse
import sys
import textwrap
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
    SWEEP_POINTS,
    classify_prior_draw,
    gaussian_loglik,
    prior_sweep_values,
    sample_prior_parameters,
    simplex_edge_weights,
)
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402
from dalec.parameters import (  # noqa: E402
    ALLOCATION_WEIGHT_ORDER,
    PARAMETER_REGISTRY,
    DalecParameters,
    prior_bounds,
)
from dalec.plotting import OKABE_ITO, apply_style, save_figure  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")

#: Sweep trajectories, cached so the figures can be restyled without repeating
#: 360 forward integrations. results/ is gitignored per amendment A9, and
#: section 3 reuses these sweeps rather than recomputing them.
CACHE_PATH = Path("results/prior_diagnostics/cache/oaat_sweeps.csv")

#: Task 1 drew this many, from the master seed, and wrote its failures out.
TASK1_DRAWS = 1000

#: Reading aids from section 2.4. Drawn for reference, not applied as thresholds.
FLAT_BELOW = 3.0
CONTRACTION_ABOVE = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--points", type=int, default=SWEEP_POINTS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="rebuild the figures from the cached sweeps, running no model",
    )
    return parser.parse_args()


def usable_draw_anchor(config, out_dir: Path) -> tuple[DalecParameters, pd.DataFrame]:
    """Rebuild Task 1's usable draws and take their component-wise median.

    Task 1 is deterministic from the master seed and wrote out the indices of the
    draws that failed, so the surviving set is recoverable exactly, without
    repeating a thousand forward integrations.
    """
    failed_path = out_dir / "failed_draws.csv"
    if not failed_path.exists():
        raise SystemExit(
            f"no {failed_path}. Run scripts/08_prior_predictive.py first -- the "
            "anchor for these sweeps is defined by which draws survived it."
        )
    failed = set(pd.read_csv(failed_path)["draw"])
    _, frame = sample_prior_parameters(
        TASK1_DRAWS, rng=np.random.default_rng(int(config["seed"]))
    )
    usable = frame[~frame["draw"].isin(failed)]

    simplex = list(ALLOCATION_WEIGHT_ORDER)
    scalars = [n for n, p in PARAMETER_REGISTRY.items() if not p.simplex]

    # The four fractions are a split of 1 - f_auto, so their medians are taken as
    # weights and renormalised; component-wise medians of the fractions
    # themselves would not close to 1 - f_auto.
    weights = usable[simplex].to_numpy() / (1.0 - usable["f_auto"].to_numpy()[:, None])
    anchor_weights = np.median(weights, axis=0)
    anchor_weights = anchor_weights / anchor_weights.sum()

    anchor = DalecParameters.from_allocation_simplex(
        f_auto=float(usable["f_auto"].median()),
        allocation_weights=anchor_weights,
        **{n: float(usable[n].median()) for n in scalars if n != "f_auto"},
    )
    return anchor, usable


def compute_metrics(
    sweeps: dict[str, pd.DataFrame], simplex: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Metric table, the ranked subset, and any identically flat parameters."""
    rows = []
    for name, table in sweeps.items():
        finite = table["loglik"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "parameter": name,
                "simplex": name in simplex,
                "d_loglik": float(finite.max() - finite.min())
                if len(finite) > 1
                else float("nan"),
                "d_nee_annual": float(table["annual"].max() - table["annual"].min()),
                "d_rmse": float(table["rmse"].max() - table["rmse"].min()),
                "n_nonfinite": int(len(table) - len(finite)),
                "n_failed": int(table["reason"].notna().sum()),
            }
        )
    metrics = pd.DataFrame(rows).set_index("parameter")

    # A6: a zero profile is a transcription artefact, not a statement about what
    # the data can constrain. Applied generically rather than by naming d_fall.
    flat = metrics.index[metrics["d_loglik"].fillna(0.0) <= 0.0].tolist()
    ranked = metrics.drop(index=flat).sort_values("d_loglik", ascending=False)
    return metrics, ranked, flat


def build_figures(
    sweeps: dict[str, pd.DataFrame],
    ranked: pd.DataFrame,
    flat: list[str],
    simplex: list[str],
    out_dir: Path,
    n_usable: int,
) -> list[Path]:
    apply_style()
    import matplotlib.pyplot as plt

    written: list[Path] = []
    order = list(ranked.index) + flat

    # -- fig05 ---------------------------------------------------------------
    columns = 5
    n_rows = int(np.ceil(len(order) / columns))
    figure, axes = plt.subplots(
        n_rows, columns, figsize=(13.0, 2.3 * n_rows), sharey=True
    )
    axes = np.atleast_2d(axes)
    for position, name in enumerate(order):
        ax = axes[position // columns, position % columns]
        table = sweeps[name]
        ax.plot(
            table["value"],
            table["loglik"] - table["loglik"].max(),
            color=OKABE_ITO[0], lw=1.5, marker="o", ms=2.5,
        )
        is_flat = name in flat
        ax.set_title(
            name + ("  [inert]" if is_flat else ""),
            fontsize=8, color=OKABE_ITO[1] if is_flat else "black",
        )
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(
            "fraction of GPP" if name in simplex else PARAMETER_REGISTRY[name].unit,
            fontsize=6.5,
        )
    axes[0, 0].set_yscale("symlog", linthresh=1e3)
    for position in range(len(order), n_rows * columns):
        axes[position // columns, position % columns].axis("off")
    for row in range(n_rows):
        axes[row, 0].set_ylabel("log-lik $-$ max", fontsize=7)
    figure.suptitle(
        "fig05 -- log-likelihood across each parameter's prior range, shared "
        f"symlog y-axis (anchor: median of {n_usable} usable Task 1 draws)",
        fontsize=10.5, fontweight="semibold",
    )
    written += save_figure(figure, out_dir, "fig05_oaat_loglik_panels")
    plt.close(figure)

    # -- fig06 ---------------------------------------------------------------
    figure, ax = plt.subplots(figsize=(9.5, 0.32 * len(ranked) + 3.0))
    positions = np.arange(len(ranked))[::-1]
    ax.barh(positions, ranked["d_loglik"], color=OKABE_ITO[0], height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels(ranked.index, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("$\\Delta$ log-likelihood across the prior range (log scale)")
    for value, colour in ((FLAT_BELOW, OKABE_ITO[1]), (CONTRACTION_ABOVE, OKABE_ITO[2])):
        ax.axvline(value, color=colour, lw=1.0, ls=":")

    # The caveat belongs on the figure, not only in the findings: read without
    # it, these bars say every parameter is strongly constrained, which is the
    # opposite of the truth. Wrapped at runtime so the figure stays a sane width.
    ax.set_title(
        textwrap.fill(
            "Dotted lines are section 2.4's reading aids, 3 and 10 units. They do "
            "not apply at a prior-derived anchor: it sits about 7e6 units from the "
            "posterior mode, so every parameter clears them by orders of "
            "magnitude. The ranking is the result; the absolute scale is not.",
            width=94,
        ),
        fontsize=8, color="#5E6E6C", loc="left",
    )
    excluded = f"; {', '.join(flat)} excluded as identically flat" if flat else ""
    figure.suptitle(
        f"fig06 -- parameters ranked by what the likelihood can see{excluded}",
        fontsize=11, fontweight="semibold",
    )
    written += save_figure(figure, out_dir, "fig06_oaat_ranking")
    plt.close(figure)
    return written


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    config = load_config(args.config)
    out_dir = args.out if args.out is not None else Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    simplex = list(ALLOCATION_WEIGHT_ORDER)
    scalars = [n for n, p in PARAMETER_REGISTRY.items() if not p.simplex]

    print("=" * 78)
    print("  Task 2 -- one-at-a-time sensitivity")
    print("=" * 78)
    anchor, usable = usable_draw_anchor(config, out_dir)

    if args.figures_only:
        if not CACHE_PATH.exists():
            raise SystemExit(
                f"no cached sweeps at {CACHE_PATH}; run without --figures-only"
            )
        cached = pd.read_csv(CACHE_PATH)
        sweeps = {
            name: group.drop(columns="parameter").reset_index(drop=True)
            for name, group in cached.groupby("parameter", sort=False)
        }
        print(f"  loaded {len(sweeps)} cached sweeps from {CACHE_PATH}")
        _, ranked, flat = compute_metrics(sweeps, simplex)
        for path in build_figures(sweeps, ranked, flat, simplex, out_dir, len(usable)):
            print(f"  -> {path}")
        return 0

    calibration = require_year_block(config, "calibration")
    slug = str(config.get("site", {}).get("code", "")).lower().replace("-", "_")
    block = SiteData.load(
        resolve_path(config["paths"]["processed_dir"])
        / f"{slug}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )
    acm = acm_from_config(config)

    def evaluate(parameters: DalecParameters) -> tuple[float, float, float, str | None]:
        output = run_dalec2(parameters, block, gpp_fn=acm, phenology_fn=dalec2_phenology)
        reason = classify_prior_draw(output)
        _, _, mask = block.likelihood_arrays()
        return (
            gaussian_loglik(output.nee, block),
            float(np.mean(output.nee)) * 365.25,
            float(np.sqrt(np.mean((output.nee[mask] - block.nee_obs[mask]) ** 2))),
            reason,
        )

    print(f"  calibration block   {calibration[0]}-{calibration[1]}, "
          f"{block.n_days} days, {block.n_assimilated} assimilable")
    print(f"  anchor              median of {len(usable)} usable Task 1 draws")
    print(f"  sweep               {args.points} points, 1st-99th prior percentile")

    soil = anchor.theta_som * anchor.c_som_0
    litter = anchor.theta_lit * anchor.c_lit_0
    print(f"\n  anchor heterotrophic respiration at T=0: soil {soil:.2f} + "
          f"litter {litter:.2f} = {soil + litter:.2f} g C m-2 d-1")

    base = evaluate(anchor)
    prior_median = DalecParameters.from_allocation_simplex(
        f_auto=float(np.mean(prior_bounds("f_auto"))),
        allocation_weights=np.full(len(simplex), 1.0 / len(simplex)),
        **{n: float(np.mean(prior_bounds(n))) for n in scalars if n != "f_auto"},
    )
    rejected = evaluate(prior_median)

    print(f"\n  {'anchor':<26}{'log-likelihood':>18}{'annual NEE':>14}"
          f"{'RMSE':>9}   screen")
    for label, result in (
        ("usable-draw median", base), ("prior median (rejected)", rejected)
    ):
        loglik, annual, rmse, reason = result
        print(f"  {label:<26}{loglik:>18,.0f}{annual:>+14.1f}{rmse:>9.3f}   "
              f"{reason or 'passes'}")
    print(f"  the correction is worth {rejected[0] - base[0]:,.0f} log-likelihood "
          f"units and {rejected[2] - base[2]:.2f} g C m-2 d-1 of RMSE")

    anchor_weights = np.array([getattr(anchor, n) for n in simplex]) / (
        1.0 - anchor.f_auto
    )
    sweeps: dict[str, pd.DataFrame] = {}
    print(f"\n  sweeping {len(scalars) + len(simplex)} parameters "
          f"x {args.points} points...")

    for name in scalars:
        rows = []
        for value in prior_sweep_values(name, args.points):
            trial = DalecParameters.from_allocation_simplex(
                f_auto=value if name == "f_auto" else anchor.f_auto,
                allocation_weights=anchor_weights,
                **{
                    other: (value if other == name else getattr(anchor, other))
                    for other in scalars
                    if other != "f_auto"
                },
            )
            loglik, annual, rmse, reason = evaluate(trial)
            rows.append({"value": value, "loglik": loglik, "annual": annual,
                         "rmse": rmse, "reason": reason})
        sweeps[name] = pd.DataFrame(rows)
        print(f"    {name:<22} done")

    for index, name in enumerate(simplex):
        rows = []
        for weights in simplex_edge_weights(anchor_weights, index, args.points):
            trial = DalecParameters.from_allocation_simplex(
                f_auto=anchor.f_auto,
                allocation_weights=weights,
                **{n: getattr(anchor, n) for n in scalars if n != "f_auto"},
            )
            loglik, annual, rmse, reason = evaluate(trial)
            rows.append({"value": weights[index] * (1.0 - anchor.f_auto),
                         "loglik": loglik, "annual": annual, "rmse": rmse,
                         "reason": reason})
        sweeps[name] = pd.DataFrame(rows)
        print(f"    {name:<22} done (simplex edge)")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [table.assign(parameter=name) for name, table in sweeps.items()],
        ignore_index=True,
    ).to_csv(CACHE_PATH, index=False)
    print(f"\n  sweep cache -> {CACHE_PATH}")

    metrics, ranked, flat = compute_metrics(sweeps, simplex)
    metrics.to_csv(out_dir / "oaat_metrics.csv")
    print(f"  metric table -> {out_dir / 'oaat_metrics.csv'}")

    print("\n" + "=" * 78)
    print("  Ranking by d_loglik")
    print("=" * 78)
    print(f"  {'parameter':<22}{'d_loglik':>16}{'d_nee_annual':>15}{'d_rmse':>10}")
    for name, row in ranked.iterrows():
        print(f"  {name:<22}{row.d_loglik:>16,.0f}{row.d_nee_annual:>15,.1f}"
              f"{row.d_rmse:>10.3f}")
    if flat:
        print(f"\n  excluded, identically flat: {', '.join(flat)}")
        print("  d_fall is inert as transcribed (DECISIONS.md section 2,")
        print("  correction 2). Not a statement about what the data can see.")

    for path in build_figures(sweeps, ranked, flat, simplex, out_dir, len(usable)):
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
