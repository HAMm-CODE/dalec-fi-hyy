#!/usr/bin/env python
"""Sampling-stage diagnostic figures for the plumbing run.

``scripts/20_plumbing_run.py`` wrote text and CSV only, so nothing about the
sampler's behaviour was visible except as numbers in a table. These five figures
are the visual counterpart, and they read the stored trace rather than sampling
again.

**This is the plumbing run, not a result.** Two chains, 200 tune, 200 draws over
1997-1998. Far too short for any posterior statement. What the figures are for is
the machinery: whether the chains mixed, which parameters the prior is
truncating, and which pairs the data cannot separate.

One number here is worth stating plainly because it was previously wrong.
``az.summary(idata, round_to=None)`` does **not** disable rounding -- only the
string ``"none"`` does -- so Python ``None`` fell through to the default
2-significant-figure display and reported a worst r-hat of **1.0199** as
**"1.000"**. Two variables exceed 1.01, and the old table showed none.

Style comes from :mod:`dalec.plotting`, the same helper
``scripts/06``-``09`` use for the prior-diagnostics figures.

Figures
-------
``fig20_convergence``     r-hat and ESS per variable, against their thresholds.
``fig21_prior_position``  posterior 89% interval rescaled to each prior's range.
``fig22_phenology``       ``d_onset`` against ``d_fall`` on a shared prior.
``fig23_traces``          trace and rank plots for the four worst by r-hat.
``fig24_pairs``           the eight most-correlated parameter pairs.

Usage
-----
    python scripts/22_sampling_figures.py
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.diagnostics import reparameterised_bounds  # noqa: E402
from dalec.parameters import (  # noqa: E402
    ALLOCATION_WEIGHT_ORDER,
    PARAMETER_REGISTRY,
    canopy_bounds,
)
from dalec.plotting import OKABE_ITO, apply_style, save_figure  # noqa: E402
from dalec.priors import prior_sources  # noqa: E402

TRACE = Path("reports/sampling/plumbing_hemisurface.nc")
OUT_DIR = Path("reports/sampling")
CONVENTION = "hemisurface"

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400.0

#: Named in the request; asserted against the trace rather than trusted.
WORST_MIXING = ("d_onset", "c_lit_0", "c_fol_0", "c_lf")

COL_OK = OKABE_ITO[0]      # blue
COL_BAD = OKABE_ITO[1]     # vermillion
COL_ALT = OKABE_ITO[2]     # bluish green
COL_GREY = "0.55"

PLUMBING_NOTE = ("Plumbing run: 2 chains, 200 tune, 200 draws, 1997-1998. "
                 "Machinery check, NOT a posterior.")


def sampled_bounds() -> dict[str, tuple[float, float]]:
    """Prior bounds for every sampled scalar, from their recorded sources.

    Composed from the three the model actually reads -- ``PARAMETER_REGISTRY``
    for the published uniforms, ``canopy_bounds`` for ``lma``/``c_lf``/``ceff``,
    and ``reparameterised_bounds`` for the four respiration parameters, which is
    where ``F_SOM_BOUNDS`` enters. No bound is typed here.

    Checked against :func:`dalec.priors.prior_sources`, which is what the model
    itself builds the priors from, so this cannot drift away from the prior the
    trace was actually drawn under.
    """
    canopy = canopy_bounds(CONVENTION)
    respiration = reparameterised_bounds()
    bounds: dict[str, tuple[float, float]] = {}
    for name, entry in PARAMETER_REGISTRY.items():
        if entry.simplex or name in respiration or name.endswith("_0"):
            continue
        bounds[name] = canopy.get(name, (entry.lower, entry.upper))
    bounds.update(respiration)

    reference = prior_sources(CONVENTION)
    if set(bounds) != set(reference):
        raise AssertionError(
            f"bounds disagree with dalec.priors.prior_sources on which "
            f"parameters are sampled: {set(bounds) ^ set(reference)}"
        )
    for name, (lo, hi) in bounds.items():
        if not (np.isclose(lo, reference[name][0]) and np.isclose(hi, reference[name][1])):
            raise AssertionError(
                f"bounds for {name!r} disagree with prior_sources: "
                f"{(lo, hi)} vs {reference[name]}"
            )
    return bounds


def figure_convergence(summary, meta) -> None:
    """r-hat and ESS for every variable, against their thresholds."""
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("r_hat")
    names = [str(n) for n in ordered.index]
    y = np.arange(len(names))
    rhat = ordered["r_hat"].to_numpy()
    bulk = ordered["ess_bulk"].to_numpy()
    tail = ordered["ess_tail"].to_numpy()
    over = rhat > RHAT_THRESHOLD

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.30 * len(names) + 2.6),
                             sharey=True)

    ax = axes[0]
    ax.barh(y, rhat - 1.0, left=1.0, height=0.72,
            color=[COL_BAD if o else COL_OK for o in over], zorder=3)
    ax.axvline(RHAT_THRESHOLD, color=COL_BAD, lw=1.4, ls="--", zorder=4,
               label=f"threshold {RHAT_THRESHOLD}")
    ax.axvline(1.0, color="0.3", lw=0.9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("$\\hat{R}$")
    ax.set_xlim(0.995, max(1.0215, float(rhat.max()) + 0.002))
    ax.set_title("Convergence: $\\hat{R}$ per variable", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for i in np.flatnonzero(over):
        ax.annotate(f"  {names[i]}  {rhat[i]:.4f}", (rhat[i], y[i]),
                    va="center", fontsize=8.5, fontweight="bold", color=COL_BAD)

    ax = axes[1]
    ax.scatter(bulk, y, s=26, color=COL_OK, zorder=3, label="ESS bulk")
    ax.scatter(tail, y, s=26, facecolors="none", edgecolors=COL_ALT, zorder=3,
               label="ESS tail")
    ax.axvline(ESS_THRESHOLD, color=COL_BAD, lw=1.4, ls="--", zorder=4,
               label=f"threshold {ESS_THRESHOLD:.0f}")
    ax.set_xlabel("effective sample size (of 400 draws)")
    ax.set_title("Effective sample size", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, max(700, float(max(bulk.max(), tail.max())) * 1.08))

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.set_ylim(-0.8, len(names) - 0.2)

    caption = (
        f"{meta['divergences']}/{meta['chains'] * meta['draws']} divergences, "
        f"{meta['max_treedepth_hits']}/{meta['chains'] * meta['draws']} "
        f"max-treedepth hits. "
        f"Worst $\\hat{{R}}$ {meta['worst_r_hat']:.4f}, lowest bulk ESS "
        f"{meta['lowest_ess_bulk']:.0f}. {PLUMBING_NOTE}"
    )
    fig.text(0.005, 0.004, caption, fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    print("  ", save_figure(fig, OUT_DIR, "fig20_convergence"))
    plt.close(fig)


def figure_prior_position(post, bounds) -> None:
    """Posterior 89% interval rescaled so every prior range maps to [0, 1]."""
    import matplotlib.pyplot as plt

    rows = []
    for name, (lo, hi) in bounds.items():
        draws = post[name].values.ravel()
        span = hi - lo
        low, high = np.percentile(draws, [5.5, 94.5])   # 89% interval
        med = float(np.median(draws))
        p_lo, p_hi, p_med = ((low - lo) / span, (high - lo) / span,
                             (med - lo) / span)
        # distance from the interval to whichever bound it is nearer
        distance = min(p_lo, 1.0 - p_hi)
        rows.append((name, p_lo, p_hi, p_med, distance))

    rows.sort(key=lambda r: r[4])
    names = [r[0] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 2.4))
    ax.axvspan(0, 1, color="0.90", zorder=0, label="prior range")
    for i, (_name, p_lo, p_hi, p_med, distance) in enumerate(rows):
        pinned = distance < 0.02
        colour = COL_BAD if pinned else COL_OK
        ax.plot([p_lo, p_hi], [i, i], lw=5.0, color=colour, solid_capstyle="butt",
                zorder=3, alpha=0.9)
        # A tightly determined parameter has an interval narrower than a pixel,
        # so the bar alone renders as a blank row -- and those are exactly the
        # rows that matter. The median marker guarantees every row is visible.
        ax.plot([p_med], [i], marker="o", ms=5.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.9, zorder=5)
        ax.annotate(f"{distance:.3f}", (1.045, i), va="center",
                    fontsize=7.5, color=colour if pinned else "0.4")

    ax.axvline(0.0, color="0.25", lw=1.2, ls="--", zorder=5)
    ax.axvline(1.0, color="0.25", lw=1.2, ls="--", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("position within the prior range  (0 = lower bound, 1 = upper)")
    ax.set_xlim(-0.10, 1.16)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.invert_yaxis()
    ax.set_title(
        "Which parameters are pinned against their prior\n"
        "89% posterior interval, each rescaled to its own prior range; "
        "sorted by distance to the nearest bound",
        loc="left", pad=28,
    )
    handles = [
        plt.Line2D([], [], lw=5, color=COL_BAD, label="interval within 0.02 of a bound"),
        plt.Line2D([], [], lw=5, color=COL_OK, label="clear of both bounds"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))
    fig.text(0.005, 0.004,
             "Bounds read from PARAMETER_REGISTRY, canopy_bounds('hemisurface') "
             f"and reparameterised_bounds(). {PLUMBING_NOTE}",
             fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    print("  ", save_figure(fig, OUT_DIR, "fig21_prior_position"))
    plt.close(fig)


def figure_phenology(post, bounds) -> None:
    """``d_onset`` and ``d_fall`` on one day-of-year axis, prior drawn flat."""
    import matplotlib.pyplot as plt

    onset = post["d_onset"].values.ravel()
    fall = post["d_fall"].values.ravel()
    lo_o, hi_o = bounds["d_onset"]
    lo_f, hi_f = bounds["d_fall"]
    if (lo_o, hi_o) != (lo_f, hi_f):
        raise AssertionError("d_onset and d_fall no longer share a prior")
    density = 1.0 / (hi_o - lo_o)

    # One shared x-axis, two y-scales. Drawn together the posteriors differ by
    # a factor of ~400 in height: d_onset is a needle and d_fall is
    # indistinguishable from the flat prior, so d_fall reads as missing rather
    # than as lying exactly on its prior, which is the finding.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.hist(onset, bins=80, density=True, color=COL_OK, zorder=3,
            label=f"d_onset   median {np.median(onset):.1f}, sd {onset.std():.2f}")
    ax.axhline(density, color=COL_GREY, lw=2.0, ls="--", zorder=2,
               label=f"shared prior U({lo_o:.0f}, {hi_o:.0f}), density "
                     f"{density:.2e}")
    ax.set_ylabel("posterior density")
    ax.set_title(
        "Phenology: one parameter the data informs, one it cannot\n"
        "d_onset and d_fall share the prior U(1, 365)",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.annotate("d_onset: the whole posterior is\nnarrower than one day",
                xy=(float(np.median(onset)), ax.get_ylim()[1] * 0.55),
                xytext=(float(np.median(onset)) + 28, ax.get_ylim()[1] * 0.62),
                fontsize=9, color=COL_OK,
                arrowprops={"arrowstyle": "->", "color": COL_OK, "lw": 1.2})

    ax = axes[1]
    ax.hist(fall, bins=40, density=True, color=COL_BAD, alpha=0.75, zorder=3,
            label=f"d_fall   median {np.median(fall):.0f}, sd {fall.std():.1f}")
    ax.axhline(density, color=COL_GREY, lw=2.0, ls="--", zorder=4,
               label="shared prior U(1, 365)")
    ax.set_ylim(0, density * 2.4)
    ax.set_xlim(lo_o, hi_o)
    ax.set_xlabel("day of year")
    ax.set_ylabel("posterior density")
    ax.set_title("Same axis, rescaled: d_fall lies on its prior", loc="left",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(0.005, 0.004,
             "d_fall is inert as transcribed (DECISIONS section 2, correction 2): "
             "it cannot affect the likelihood, so recovering its prior is the "
             f"correct behaviour, not a failure to converge. {PLUMBING_NOTE}",
             fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    print("  ", save_figure(fig, OUT_DIR, "fig22_phenology"))
    plt.close(fig)


def figure_traces(post, summary) -> None:
    """Trace and rank plots for the four worst-mixing variables."""
    import matplotlib.pyplot as plt

    ranked = summary.sort_values("r_hat", ascending=False)
    worst = [str(n) for n in ranked.index[:4]]
    if set(worst) != set(WORST_MIXING):
        print(f"   note: worst four by r_hat are {worst}, "
              f"not {list(WORST_MIXING)}; plotting the measured four")

    n_chains = post.sizes["chain"]
    colours = [OKABE_ITO[i] for i in (0, 1, 2, 4)]
    fig, axes = plt.subplots(len(worst), 2, figsize=(12, 2.5 * len(worst)))

    for row, name in enumerate(worst):
        values = post[name].values          # (chain, draw)
        rhat = float(ranked.loc[name, "r_hat"])
        ess = float(ranked.loc[name, "ess_bulk"])

        ax = axes[row, 0]
        for c in range(n_chains):
            ax.plot(values[c], lw=0.7, color=colours[c % len(colours)],
                    alpha=0.85, label=f"chain {c}")
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(f"{name}   $\\hat{{R}}$ {rhat:.4f}   ESS {ess:.0f}",
                     loc="left", fontsize=10)
        if row == 0:
            ax.legend(frameon=False, fontsize=8, ncol=n_chains)
        if row == len(worst) - 1:
            ax.set_xlabel("draw")

        # rank plot: ranks pooled across chains, then one histogram per chain.
        # Uniform bars mean the chains are interleaved; a chain occupying one
        # end of the rank range is the signature r-hat is reacting to.
        ax = axes[row, 1]
        flat = values.ravel()
        ranks = flat.argsort().argsort().reshape(values.shape)
        bins = np.linspace(0, flat.size, 21)
        for c in range(n_chains):
            ax.hist(ranks[c], bins=bins, histtype="step", lw=1.6,
                    color=colours[c % len(colours)])
        ax.axhline(flat.size / (20 * n_chains), color="0.3", lw=1.0, ls="--")
        ax.set_yticks([])
        ax.set_title("rank plot (flat = well mixed)", loc="left", fontsize=9)
        if row == len(worst) - 1:
            ax.set_xlabel("rank across pooled draws")

    fig.suptitle("The four worst-mixing variables by $\\hat{R}$",
                 x=0.005, ha="left", y=0.999)
    fig.text(0.005, 0.004, PLUMBING_NOTE, fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.028, 1, 0.975))
    print("  ", save_figure(fig, OUT_DIR, "fig23_traces"))
    plt.close(fig)


def figure_pairs(post, bounds) -> None:
    """The eight most-correlated parameter pairs: the equifinality evidence."""
    import matplotlib.pyplot as plt

    columns: dict[str, np.ndarray] = {
        name: post[name].values.ravel() for name in bounds
    }
    simplex = set()
    for i, label in enumerate(ALLOCATION_WEIGHT_ORDER):
        key = f"w[{label}]"
        columns[key] = post["allocation_weights"].values[:, :, i].ravel()
        simplex.add(key)

    scored = []
    for a, b in combinations(sorted(columns), 2):
        r = float(np.corrcoef(columns[a], columns[b])[0, 1])
        if np.isfinite(r):
            scored.append((abs(r), r, a, b))
    scored.sort(reverse=True)
    top = scored[:8]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    for ax, (_absr, r, a, b) in zip(axes.ravel(), top, strict=True):
        structural = a in simplex and b in simplex
        colour = COL_GREY if structural else (COL_BAD if abs(r) > 0.8 else COL_OK)
        ax.scatter(columns[a], columns[b], s=7, alpha=0.35, color=colour,
                   edgecolors="none")
        ax.set_xlabel(a, fontsize=9)
        ax.set_ylabel(b, fontsize=9)
        tag = "  (simplex, structural)" if structural else ""
        ax.set_title(f"r = {r:+.3f}{tag}", loc="left", fontsize=10,
                     color=colour if not structural else "0.35")
        # Rate constants run to 1e-4 and their default tick labels collide.
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both",
                            useMathText=True)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Equifinality: the eight most-correlated parameter pairs\n"
        "Pairs inside the allocation simplex are correlated by construction "
        "and are marked as such",
        x=0.005, ha="left", y=0.999,
    )
    fig.text(0.005, 0.004, PLUMBING_NOTE, fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    print("  ", save_figure(fig, OUT_DIR, "fig24_pairs"))
    plt.close(fig)
    return top


def main() -> int:
    for category in (RuntimeWarning, UserWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)
    import json

    import arviz as az

    apply_style()
    idata = az.from_netcdf(str(TRACE))
    post = idata["posterior"]
    summary = az.summary(idata, round_to="none")
    meta = json.loads(
        (OUT_DIR / f"plumbing_{CONVENTION}_meta.json").read_text(encoding="utf-8")
    )
    bounds = sampled_bounds()

    print("figures ->")
    figure_convergence(summary, meta)
    figure_prior_position(post, bounds)
    figure_phenology(post, bounds)
    figure_traces(post, summary)
    top = figure_pairs(post, bounds)

    print("\ntop correlated pairs:")
    for _absr, r, a, b in top:
        print(f"  {a:<24} {b:<24} r = {r:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
