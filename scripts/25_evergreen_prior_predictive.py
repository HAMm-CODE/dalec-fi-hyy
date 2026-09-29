#!/usr/bin/env python
"""Prior predictive check for the evergreen DALEC, with DALEC2's three checks.

The checks and their definitions are those of
``scripts/10_reparameterised_prior.py``, so the two models are judged the same
way:

    1. does the derived c_som_0 land in the 5,000-10,000 g C m-2 inventory range?
    2. the share of physically impossible draws (dalec.diagnostics.classify_prior_draw:
       non-finite values, a negative pool, a pool growing past its limit, or
       |NEE| past its limit) and the coverage of the 90% observation-level band
    3. prior predictive median annual NEE, against the observed value

Draws come from the model's own PyMC prior block and forward graph, not from
the numpy sampler script 10 uses. ``--model-variant dalec2`` runs DALEC2
through exactly the same path, which is the like-for-like baseline: compare
the two reports this script writes, not this one against script 10's.

This script only reports; it changes nothing and fits nothing.

Usage
-----
    python scripts/25_evergreen_prior_predictive.py
    python scripts/25_evergreen_prior_predictive.py --model-variant dalec2
    python scripts/25_evergreen_prior_predictive.py --draws 200
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.compute import ensure_configured  # noqa: E402
from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    require_year_block,
    resolve_path,
)
from dalec.data_io import SiteData  # noqa: E402
from dalec.diagnostics import classify_prior_draw  # noqa: E402
from dalec.model import build_forward_graph  # noqa: E402
from dalec.model_evergreen import (  # noqa: E402
    EVERGREEN_POOL_NAMES,
    build_evergreen_graph,
    build_evergreen_priors,
)
from dalec.model_numpy import POOL_NAMES  # noqa: E402
from dalec.parameters import DAYS_PER_YEAR, DEFAULT_LAI_CONVENTION, LAI_CONVENTIONS  # noqa: E402
from dalec.priors import build_priors  # noqa: E402
from dalec.sampler import MODEL_VARIANTS  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")

#: As scripts/10_reparameterised_prior.py.
DEFAULT_DRAWS = 1000
COVERAGE_INTERVAL = (5.0, 95.0)
SOIL_INVENTORY_RANGE = (5000.0, 10000.0)
ANALYTIC_DRAWS = 60_000

BUILDERS = {
    "dalec2": (build_priors, build_forward_graph, POOL_NAMES),
    "evergreen": (build_evergreen_priors, build_evergreen_graph, EVERGREEN_POOL_NAMES),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-variant", default="evergreen", choices=MODEL_VARIANTS)
    parser.add_argument("--convention", default=DEFAULT_LAI_CONVENTION,
                        choices=sorted(LAI_CONVENTIONS))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def coverage_of(latent, block, rng):
    """Coverage of the 90% observation-level band, exactly as script 10 computes it."""
    sigma = np.where(block.nee_mask, block.nee_unc, np.nan)
    noise = np.nan_to_num(sigma, nan=float(np.nanmedian(sigma)))
    observed_level = latent + rng.normal(0.0, noise, size=latent.shape)
    low, high = np.percentile(observed_level, COVERAGE_INTERVAL, axis=0)
    inside = (block.nee_obs >= low) & (block.nee_obs <= high)
    return float(inside[block.nee_mask].mean())


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import pymc as pm
    import pytensor.tensor as pt

    args = parse_args()
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    calibration = require_year_block(config, "calibration")
    slug = str(config.get("site", {}).get("code", "")).lower().replace("-", "_")
    block = SiteData.load(
        resolve_path(config["paths"]["processed_dir"])
        / f"{slug}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )
    acm = acm_from_config(config)
    prior_builder, graph_builder, pool_names = BUILDERS[args.model_variant]

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    ensure_configured()
    with pm.Model():
        priors = prior_builder(t_air=block.t_air, convention=args.convention)
        graph = graph_builder(
            parameters=priors.parameters,
            doy=block.doy.astype(float),
            t_air=block.t_air,
            t_max=block.t_max,
            t_min=block.t_min,
            sw_in=block.sw_in,
            co2=block.co2,
            latitude_deg=acm.latitude_deg,
            coefficients=acm.coefficients,
            frost_threshold_degc=acm.frost_threshold_degc,
        )
        initial = pt.stack([priors.parameters[f"{name}_0"] for name in pool_names])

    bar = "=" * 74
    out(bar)
    out(f"  Prior predictive, {args.model_variant} -- the three checks of script 10")
    out(bar)
    out(f"  calibration block  {calibration[0]}-{calibration[1]}, "
        f"{block.n_days} days, {block.n_assimilated} assimilable")
    out(f"  convention         {args.convention}")
    out(f"  draws              {args.draws}")
    out(f"  master seed        {seed}")

    # -- CHECK 1 -------------------------------------------------------------
    c_som_0, c_lit_0 = pm.draw(
        [priors.derived["c_som_0"], priors.derived["c_lit_0"]],
        draws=ANALYTIC_DRAWS, random_seed=seed,
    )
    lo_t, hi_t = SOIL_INVENTORY_RANGE
    out("\n" + bar)
    out(f"  CHECK 1 -- derived c_som_0 against the {lo_t:,.0f}-{hi_t:,.0f} g C m-2 inventory")
    out(bar)
    for name, values in (("c_som_0", c_som_0), ("c_lit_0", c_lit_0)):
        q = np.percentile(values, [2.5, 50, 97.5])
        out(f"  {name}   2.5% {q[0]:8,.0f}   median {q[1]:8,.0f}   97.5% {q[2]:8,.0f}")
    inside = float(((c_som_0 >= lo_t) & (c_som_0 <= hi_t)).mean())
    out(f"  prior mass inside the inventory range   {100 * inside:.1f}%")

    # -- CHECK 2 -------------------------------------------------------------
    out("\n" + bar)
    out("  CHECK 2 -- share of physically impossible draws, and coverage")
    out(bar)
    nee, pools, starts = pm.draw(
        [graph.nee, graph.pools, initial], draws=args.draws, random_seed=seed
    )
    reasons = [
        classify_prior_draw(SimpleNamespace(nee=nee[i], pools=np.vstack([starts[i], pools[i]])))
        for i in range(args.draws)
    ]
    ok = np.array([reason is None for reason in reasons])
    for reason in sorted({str(r) for r in reasons}):
        count = sum(str(r) == reason for r in reasons)
        label = "usable" if reason == "None" else reason
        out(f"    {label:<24} {count:>5}   {100 * count / args.draws:5.1f}%")
    out(f"\n  impossible draws   {int((~ok).sum())} / {args.draws} = "
        f"{100 * (~ok).mean():.1f}%")
    if not ok.any():
        raise SystemExit("every draw failed; no coverage to report")
    latent = nee[ok]
    coverage = coverage_of(latent, block, np.random.default_rng(seed))
    out(f"  coverage of the 90% band   {coverage:.3f}   (target ~0.90)")

    # -- CHECK 3 -------------------------------------------------------------
    out("\n" + bar)
    out("  CHECK 3 -- prior predictive median annual NEE")
    out(bar)
    annual = latent.mean(axis=1) * DAYS_PER_YEAR
    q = np.percentile(annual, [5, 25, 50, 75, 95])
    observed = float(block.nee_obs[block.nee_mask].mean() * DAYS_PER_YEAR)
    out(f"  prior predictive median   {q[2]:+9.1f} g C m-2 yr-1")
    out(f"  IQR                       {q[1]:+9.1f} to {q[3]:+.1f}")
    out(f"  5-95%                     {q[0]:+9.1f} to {q[4]:+.1f}")
    out(f"  observed, assimilable days   {observed:+9.1f}")
    out(f"  observed inside the prior 5-95%?   "
        f"{'yes' if q[0] <= observed <= q[4] else 'NO'}")

    report = args.out / f"prior_predictive_{args.model_variant}_{args.convention}.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
