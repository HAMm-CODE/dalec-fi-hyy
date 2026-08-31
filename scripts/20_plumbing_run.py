#!/usr/bin/env python
"""Plumbing run: does the machinery work end to end?

**This is not a result.** Two chains, 200 tune, 200 draws, two calibration years.
Far too short for any posterior statement, and nothing here should be read as
one. The question is only whether priors, forward model, likelihood and NUTS
connect and produce well-formed output.

It also checks the two things the diagnostic phase predicted, because a plumbing
run is the first opportunity to look:

1. **Do divergences cluster at particular ``c_lf`` values?** The foliage
   separatrix sits at 30-98 g C m-2 while the derived ``c_fol_0`` spans 462-770
   (`separatrix.txt`), so at the prior there should be no crossing and therefore
   no clustering. If they cluster anyway, the separatrix moved.
2. **Do chains agree on NEE while disagreeing on ``c_fol``?** That is the
   signature of a canopy the data cannot see.

Usage
-----
    python scripts/20_plumbing_run.py
    python scripts/20_plumbing_run.py --convention projected
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.compute import resolved_linker_class  # noqa: E402
from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_path,
)
from dalec.data_io import build_site_data, load_daily_extremes, load_fluxnet_dd  # noqa: E402
from dalec.parameters import DEFAULT_LAI_CONVENTION, LAI_CONVENTIONS  # noqa: E402
from dalec.sampler import initial_point_is_finite, model_from_config, sample  # noqa: E402

REPORT_DIR = Path("reports/sampling")

DEFAULT_YEARS = (1997, 1998)
DEFAULT_CHAINS = 2
DEFAULT_TUNE = 200
DEFAULT_DRAWS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--convention", default=DEFAULT_LAI_CONVENTION,
                        choices=sorted(LAI_CONVENTIONS))
    parser.add_argument("--start-year", type=int, default=DEFAULT_YEARS[0])
    parser.add_argument("--end-year", type=int, default=DEFAULT_YEARS[1])
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    return parser.parse_args()


def main() -> int:
    for category in (RuntimeWarning, DeprecationWarning, UserWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)
    import arviz as az

    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    bar = "=" * 74

    print(bar)
    print("  PLUMBING RUN -- machinery check, NOT a result")
    print(bar)

    site_code = str(config.get("site", {}).get("code", ""))
    slug = site_code.lower().replace("-", "_")
    processed = resolve_path(config["paths"]["processed_dir"])
    frame = load_fluxnet_dd(resolve_path(config["paths"]["fluxnet_file"]))
    extremes = load_daily_extremes(processed / f"{slug}_tminmax.csv")
    block = build_site_data(
        frame,
        start_year=args.start_year,
        end_year=args.end_year,
        qc_threshold=float(config["data"]["qc_threshold"]),
        site_code=site_code,
        daily_extremes=extremes,
    )
    print(f"  block        {args.start_year}-{args.end_year}, {block.n_days} days, "
          f"{block.n_assimilated} assimilable")
    print(f"  convention   {args.convention}")
    print(f"  sampling     {args.chains} chains, {args.tune} tune, {args.draws} draws")
    print(f"  machine      {platform.machine()}, {resolved_linker_class()}")

    build_start = time.perf_counter()
    dalec = model_from_config(config, block, convention=args.convention)
    build_s = time.perf_counter() - build_start
    finite, logp = initial_point_is_finite(dalec)
    print(f"  model built in {build_s:.1f} s; initial logp {logp:,.1f} "
          f"({'finite' if finite else 'NOT FINITE'})")
    if not finite:
        raise SystemExit("initial logp is not finite; sampling would fail")

    start = time.perf_counter()
    idata = sample(
        dalec,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        seed=int(config["seed"]),
        cores=args.chains,
    )
    wall_s = time.perf_counter() - start

    stats = idata.sample_stats
    divergences = int(stats["diverging"].sum())
    total_draws = args.chains * args.draws
    max_depth_hits = 0
    depth_limit = None
    if "reached_max_treedepth" in stats:
        max_depth_hits = int(stats["reached_max_treedepth"].sum())
    tree_depth = stats["tree_depth"].values if "tree_depth" in stats else None
    if tree_depth is not None:
        depth_limit = int(tree_depth.max())

    print("\n" + bar)
    print("  Sampling")
    print(bar)
    print(f"  wall clock            {wall_s:.1f} s  ({wall_s / 60:.1f} min)")
    per_draw_ms = 1000 * wall_s / (args.chains * (args.tune + args.draws))
    print(f"  per draw              {per_draw_ms:.0f} ms")
    print(f"  divergences           {divergences} / {total_draws} "
          f"({100 * divergences / total_draws:.1f}%)")
    print(f"  max treedepth hits    {max_depth_hits} / {total_draws}")
    if tree_depth is not None:
        print(f"  tree depth            median {np.median(tree_depth):.0f}   "
              f"max {depth_limit}")
    for chain in range(args.chains):
        step_size = float(stats["step_size"].values[chain, -1])
        chain_div = int(stats["diverging"].values[chain].sum())
        print(f"    chain {chain}: step size {step_size:.3e}, "
              f"divergences {chain_div}")

    # -- convergence ---------------------------------------------------------
    summary = az.summary(idata, round_to=None)
    print("\n" + bar)
    print("  Convergence, every sampled variable")
    print(bar)
    print(f"  {'variable':<24} {'mean':>12} {'r_hat':>8} {'ess_bulk':>10} {'ess_tail':>10}")
    for name, row in summary.iterrows():
        print(f"  {name!s:<24} {row['mean']:>12.4g} {row['r_hat']:>8.3f} "
              f"{row['ess_bulk']:>10.0f} {row['ess_tail']:>10.0f}")

    worst_rhat = summary["r_hat"].max()
    worst_ess = summary["ess_bulk"].min()
    print(f"\n  worst r_hat {worst_rhat:.3f}   lowest ess_bulk {worst_ess:.0f}")

    # -- prediction 1: divergences and c_lf ----------------------------------
    print("\n" + bar)
    print("  Prediction 1: do divergences cluster at particular c_lf values?")
    print(bar)
    c_lf = idata.posterior["c_lf"].values.reshape(-1)
    diverging = stats["diverging"].values.reshape(-1).astype(bool)
    if divergences == 0:
        print("  No divergences, so there is nothing to cluster.")
        print("  Consistent with the separatrix prediction: it sits at 30-98 g C m-2")
        print("  while the derived c_fol_0 spans 462-770, so the prior never")
        print("  approaches it (reports/prior_diagnostics/separatrix.txt).")
    else:
        print(f"  c_lf, diverging draws     n={diverging.sum():4d}  "
              f"median {np.median(c_lf[diverging]):.4f}  "
              f"range {c_lf[diverging].min():.4f}-{c_lf[diverging].max():.4f}")
        print(f"  c_lf, non-diverging       n={(~diverging).sum():4d}  "
              f"median {np.median(c_lf[~diverging]):.4f}  "
              f"range {c_lf[~diverging].min():.4f}-{c_lf[~diverging].max():.4f}")
        from scipy import stats as scipy_stats

        result = scipy_stats.mannwhitneyu(c_lf[diverging], c_lf[~diverging])
        print(f"  Mann-Whitney U p = {result.pvalue:.4f}  "
              f"({'clustered' if result.pvalue < 0.05 else 'no evidence of clustering'})")

    # -- prediction 2: agree on NEE, disagree on c_fol -----------------------
    print("\n" + bar)
    print("  Prediction 2: chains agree on NEE, disagree on c_fol?")
    print(bar)
    for name in ("nee_annual", "c_fol_mean", "gpp_annual"):
        if name not in idata.posterior:
            continue
        values = idata.posterior[name].values
        per_chain = [values[chain] for chain in range(values.shape[0])]
        rhat = float(summary.loc[name, "r_hat"]) if name in summary.index else np.nan
        means = "  ".join(f"{chain.mean():10.2f}" for chain in per_chain)
        spread = abs(per_chain[0].mean() - per_chain[-1].mean())
        pooled = float(np.std(values))
        print(f"  {name:<12} chain means {means}   |diff| {spread:9.2f}   "
              f"pooled sd {pooled:9.2f}   r_hat {rhat:.3f}")
    print("\n  Read: a low r_hat on nee_annual beside a high one on c_fol_mean is")
    print("  the predicted signature. At 200 draws neither is conclusive.")

    # -- persist -------------------------------------------------------------
    idata.to_netcdf(out_dir / f"plumbing_{args.convention}.nc")
    summary.to_csv(out_dir / f"plumbing_{args.convention}_summary.csv")
    meta = {
        "years": [args.start_year, args.end_year],
        "convention": args.convention,
        "chains": args.chains, "tune": args.tune, "draws": args.draws,
        "wall_clock_s": wall_s,
        "divergences": divergences,
        "max_treedepth_hits": max_depth_hits,
        "worst_r_hat": float(worst_rhat),
        "lowest_ess_bulk": float(worst_ess),
        "n_days": int(block.n_days),
        "n_assimilated": int(block.n_assimilated),
    }
    (out_dir / f"plumbing_{args.convention}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\n  -> {out_dir / f'plumbing_{args.convention}.nc'}")
    print("\n  REMINDER: plumbing only. Not a posterior, not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
