#!/usr/bin/env python
"""Three figures from the hemisurface calibration trace.

**The posterior these describe has not converged.** Worst r-hat is 1.50 and the
lowest bulk ESS is 7, because the chains found two widely separated modes and
never crossed between them: chains 0, 1 and 3 sit at a log-posterior near
-67,019 and chain 2 near -63,064, a gap of about 3,955 log units that no draw
ever bridges. Each chain is individually well mixed -- within-chain ESS is
400-1,100 of 1,000 draws and every chain is stationary across its own halves --
so this is genuine multimodality rather than a stalled sampler.

Every figure therefore separates the two modes rather than pooling them, and
every figure says on its face that it is not a converged posterior. Pooling
would report the majority mode as though it were the answer, and the majority is
an artefact of where three of four chains happened to initialise -- the minority
mode fits better by 3,955 log units.

Figures
-------
1. ``fig20_seasonal_fit``   observed NEE by day of year against the posterior
   predictive band, per mode.
2. ``fig21_prior_posterior`` prior against posterior for every sampled
   parameter, with the prior bounds drawn.
3. ``fig22_flux_densities`` posterior GPP and Reco densities against
   Ilvesniemi et al. (2009) Fig. 6.

Figure 1 needs daily NEE, which the trace does not carry: ``track_fluxes`` was
False, deliberately, so the trace is 1.7 MB rather than gigabytes. The daily
series is recomputed by ``scratchpad/postpred.py`` and cached to
``results/posterior_predictive_hemisurface.npz``.

Usage
-----
    python scripts/22_calibration_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.parameters import (  # noqa: E402
    ALLOCATION_WEIGHT_ORDER,
    allocation_concentration,
)
from dalec.plotting import OKABE_ITO, apply_style, save_figure  # noqa: E402
from dalec.priors import prior_sources  # noqa: E402

TRACE = Path("results/calibration_hemisurface.nc")
POSTPRED = Path("results/posterior_predictive_hemisurface.npz")
OUT_DIR = Path("reports/sampling")

#: Ilvesniemi et al. (2009) Fig. 6, g C m-2 yr-1.
MEASURED_GPP = (952.0, 1104.0)
MEASURED_RECO = (761.0, 898.0)
MEASURED_NEE = -215.8

#: Chains 0, 1, 3 share one mode; chain 2 is alone in the better-fitting one.
MODE_A_CHAINS = (0, 1, 3)
MODE_B_CHAINS = (2,)
COLOUR_A = OKABE_ITO[0]   # blue
COLOUR_B = OKABE_ITO[1]   # vermillion
COLOUR_OBS = OKABE_ITO[7]  # black

BANNER = ("NOT A CONVERGED POSTERIOR — r-hat 1.50, min ESS 7. "
          "Two modes, no mixing between them.")


def mode_label(chains, lp_mean) -> str:
    return f"chains {','.join(str(c) for c in chains)}  (lp ≈ {lp_mean:,.0f})"


def figure_seasonal(npz, lp_means) -> None:
    """Observed NEE by day of year against the posterior predictive band."""
    import matplotlib.pyplot as plt

    nee = npz["nee"]              # (n_draws, n_days)
    chain = npz["chain"]
    doy = npz["doy"].astype(int)
    obs = npz["nee_obs"]
    mask = npz["nee_mask"].astype(bool)

    days = np.arange(1, 367)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]

    # observed: median by doy over assimilable days only
    obs_med = np.full(days.size, np.nan)
    obs_lo = np.full(days.size, np.nan)
    obs_hi = np.full(days.size, np.nan)
    for i, d in enumerate(days):
        sel = (doy == d) & mask
        if sel.sum() >= 3:
            v = obs[sel]
            obs_med[i] = np.median(v)
            obs_lo[i], obs_hi[i] = np.percentile(v, [25, 75])

    ax.fill_between(days, obs_lo, obs_hi, color=COLOUR_OBS, alpha=0.15,
                    label="observed IQR across years", zorder=2)
    ax.plot(days, obs_med, color=COLOUR_OBS, lw=1.8,
            label="observed NEE, median by day of year", zorder=4)

    for chains, colour, name in ((MODE_A_CHAINS, COLOUR_A, "mode A"),
                                 (MODE_B_CHAINS, COLOUR_B, "mode B")):
        sel_draws = np.isin(chain, chains)
        block = nee[sel_draws]
        med = np.full(days.size, np.nan)
        lo = np.full(days.size, np.nan)
        hi = np.full(days.size, np.nan)
        for i, d in enumerate(days):
            cols = doy == d
            if cols.sum() == 0:
                continue
            v = block[:, cols].ravel()
            med[i] = np.median(v)
            lo[i], hi[i] = np.percentile(v, [3, 97])
        lp = lp_means[chains[0]]
        ax.fill_between(days, lo, hi, color=colour, alpha=0.25, zorder=3)
        ax.plot(days, med, color=colour, lw=1.6, zorder=5,
                label=f"{name}, {mode_label(chains, lp)}")

    ax.axhline(0.0, color="0.4", lw=0.8, ls=":", zorder=1)
    ax.set_ylabel("NEE, g C m$^{-2}$ d$^{-1}$")
    ax.set_title("Seasonal cycle of NEE: observed against posterior predictive\n"
                 "FI-Hyy 1997-2010, hemisurface convention", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.0, 1.0, "", transform=ax.transAxes)

    # residual panel
    axr = axes[1]
    for chains, colour, name in ((MODE_A_CHAINS, COLOUR_A, "mode A"),
                                 (MODE_B_CHAINS, COLOUR_B, "mode B")):
        sel_draws = np.isin(chain, chains)
        block = nee[sel_draws]
        resid = np.full(days.size, np.nan)
        for i, d in enumerate(days):
            cols = (doy == d) & mask
            if cols.sum() < 3:
                continue
            resid[i] = np.median(block[:, cols].ravel()) - np.median(obs[cols])
        axr.plot(days, resid, color=colour, lw=1.5, label=f"{name} - observed")
    axr.axhline(0.0, color=COLOUR_OBS, lw=1.0)
    axr.set_xlabel("day of year")
    axr.set_ylabel("model - observed\ng C m$^{-2}$ d$^{-1}$")
    axr.set_title("Residual. Positive = model releases more carbon than observed.",
                  loc="left", fontsize=9)
    axr.legend(frameon=False, fontsize=8)
    axr.set_xlim(1, 366)

    fig.text(0.5, 0.005, BANNER, ha="center", fontsize=8, style="italic",
             color=OKABE_ITO[1])
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    print("  ", save_figure(fig, OUT_DIR, "fig20_seasonal_fit"))
    plt.close(fig)


def figure_prior_posterior(post, lp_means) -> None:
    """Prior against posterior for every sampled parameter."""
    import matplotlib.pyplot as plt

    bounds = prior_sources("hemisurface")
    scalars = sorted(bounds)
    n_alloc = len(ALLOCATION_WEIGHT_ORDER)
    panels = scalars + [f"allocation_weights[{i}]" for i in range(n_alloc)]

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    concentration = allocation_concentration()
    rng = np.random.default_rng(0)
    dirichlet_prior = rng.dirichlet(concentration, size=20000)

    for ax, name in zip(axes, panels, strict=False):
        is_alloc = name.startswith("allocation_weights")
        if is_alloc:
            i = int(name.split("[")[1].rstrip("]"))
            draws_all = post["allocation_weights"].values[:, :, i]
            prior_draws = dirichlet_prior[:, i]
            ax.hist(prior_draws, bins=60, density=True, color="0.8",
                    label="prior", zorder=1)
            lo, hi = 0.0, max(float(prior_draws.max()), float(draws_all.max()))
            title = f"{ALLOCATION_WEIGHT_ORDER[i]}  (weight {i})"
        else:
            draws_all = post[name].values
            lo, hi = bounds[name]
            ax.axvspan(lo, hi, color="0.87", label="prior support", zorder=1)
            title = name

        # The posteriors are often orders of magnitude narrower than the prior
        # range -- which is the finding, not a nuisance. A histogram alone
        # renders them as an invisible hairline at the axis edge, so each mode
        # also gets an explicit median rule that cannot be missed.
        for chains, colour, label in ((MODE_A_CHAINS, COLOUR_A, "mode A"),
                                      (MODE_B_CHAINS, COLOUR_B, "mode B")):
            v = draws_all[list(chains)].ravel()
            ax.hist(v, bins=40, density=True, color=colour, alpha=0.75,
                    label=label, zorder=3)
            ax.axvline(float(np.median(v)), color=colour, lw=1.6, zorder=5)

        if not is_alloc:
            for edge in (lo, hi):
                ax.axvline(edge, color="0.25", lw=1.1, ls="--", zorder=4)
            pad = 0.06 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            # position of the pooled posterior median within the prior range
            pos = (float(np.median(draws_all)) - lo) / (hi - lo)
            pinned = pos <= 0.05 or pos >= 0.95
            ax.text(0.5, 0.86, f"pos {pos:.2f}" + ("  AT BOUND" if pinned else ""),
                    transform=ax.transAxes, ha="center", fontsize=7.5,
                    color=OKABE_ITO[1] if pinned else "0.35",
                    fontweight="bold" if pinned else "normal")
            if name == "d_fall":
                ax.text(0.5, 0.66, "inert by construction", transform=ax.transAxes,
                        ha="center", fontsize=7, style="italic", color="0.35")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

    for ax in axes[len(panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=9,
               ncol=3, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(
        "Prior against posterior, every sampled parameter\n"
        "FI-Hyy 1997-2010, hemisurface. Dashed lines are the prior bounds; "
        "'pos' is the posterior median's position in the prior range.",
        y=0.998, ha="left", x=0.01,
    )
    fig.text(0.01, 0.001, BANNER, fontsize=8, style="italic", color=OKABE_ITO[1])
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    print("  ", save_figure(fig, OUT_DIR, "fig21_prior_posterior"))
    plt.close(fig)


def figure_flux_densities(post, lp_means) -> None:
    """Posterior GPP and Reco against the measured ranges."""
    import matplotlib.pyplot as plt

    gpp_all = post["gpp_annual"].values
    nee_all = post["nee_annual"].values
    reco_all = gpp_all + nee_all      # exact: NEE = Reco - GPP (DECISIONS §4)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = (
        ("GPP", gpp_all, MEASURED_GPP, "Ilvesniemi Fig. 6, 952-1,104"),
        ("Reco", reco_all, MEASURED_RECO, "761-898"),
        ("NEE", nee_all, None, f"observed {MEASURED_NEE:.1f}"),
    )
    for ax, (label, values, measured, note) in zip(axes, panels, strict=True):
        for chains, colour, name in ((MODE_A_CHAINS, COLOUR_A, "mode A"),
                                     (MODE_B_CHAINS, COLOUR_B, "mode B")):
            v = values[list(chains)].ravel()
            ax.hist(v, bins=50, density=True, color=colour, alpha=0.75,
                    label=f"{name}  median {np.median(v):,.0f}", zorder=3)
        if measured is not None:
            ax.axvspan(measured[0], measured[1], color=OKABE_ITO[2], alpha=0.35,
                       zorder=1, label="measured range")
        else:
            ax.axvline(MEASURED_NEE, color=OKABE_ITO[2], lw=2.0, zorder=2,
                       label="observed")
        ax.set_title(f"{label}\n{note}", fontsize=10, loc="left")
        ax.set_xlabel("g C m$^{-2}$ yr$^{-1}$")
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("posterior density")
    fig.suptitle(
        "Posterior annual fluxes against Ilvesniemi et al. (2009) Fig. 6\n"
        "FI-Hyy 1997-2010, hemisurface convention",
        y=0.999, ha="left", x=0.01,
    )
    fig.text(0.01, 0.005, BANNER, fontsize=8, style="italic", color=OKABE_ITO[1])
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    print("  ", save_figure(fig, OUT_DIR, "fig22_flux_densities"))
    plt.close(fig)


def main() -> int:
    for category in (RuntimeWarning, UserWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)
    import arviz as az

    apply_style()
    idata = az.from_netcdf(str(TRACE))
    post = idata["posterior"]
    lp_means = idata["sample_stats"]["lp"].values.mean(axis=1)

    print("figures ->")
    if POSTPRED.exists():
        figure_seasonal(np.load(POSTPRED), lp_means)
    else:
        print(f"   SKIPPED fig20: {POSTPRED} not found. Daily fluxes were not")
        print("   stored (track_fluxes=False); recompute them first.")
    figure_prior_posterior(post, lp_means)
    figure_flux_densities(post, lp_means)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
