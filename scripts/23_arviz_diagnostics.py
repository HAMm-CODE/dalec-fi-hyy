#!/usr/bin/env python
"""
ArviZ diagnostic pack for the FI-Hyy DALEC calibration.
Written for ArviZ 1.x (the DataTree rewrite). Verified against arviz 1.2.0.

Usage:
    python scripts/23_arviz_diagnostics.py --trace results/calibration_hemisurface.nc
"""

import argparse
import contextlib
import io
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import arviz as az
import numpy as np
import pandas as pd

az.rcParams["plot.max_subplots"] = 400


def as_dataset(node):
    return node.to_dataset() if hasattr(node, "to_dataset") else node.dataset


def flatten_posterior(ds):
    """Return (flat names, array of shape [n_draws_total, n_params])."""
    names, cols = [], []
    for var in ds.data_vars:
        da = ds[var]
        extra = [d for d in da.dims if d not in ("chain", "draw")]
        if not extra:
            names.append(var)
            cols.append(da.values.reshape(-1))
        else:
            stacked = da.stack(_flat=extra)
            for i in range(stacked.sizes["_flat"]):
                names.append(f"{var}[{i}]")
                cols.append(stacked.isel(_flat=i).values.reshape(-1))
    return names, np.column_stack(cols)


def label_modes(dt, gap=100.0):
    """Group chains by mean log-posterior. Returns (labels dict, mean lp array)."""
    if "/sample_stats" not in dt.groups:
        return None, None
    ss = as_dataset(dt["sample_stats"])
    if "lp" not in ss:
        return None, None
    means = ss["lp"].values.mean(axis=1)
    order = np.argsort(-means)
    labels, cur = {int(order[0]): 0}, 0
    for prev, this in zip(order[:-1], order[1:]):
        if abs(means[prev] - means[this]) > gap:
            cur += 1
        labels[int(this)] = cur
    return labels, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--outdir", default="reports/arviz")
    ap.add_argument("--n-worst", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    dt = az.from_netcdf(args.trace)
    post = as_dataset(dt["posterior"])
    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    say(f"arviz {az.__version__}   |   {args.trace}")
    say(f"{n_chain} chains x {n_draw} draws")
    say(f"groups: {dt.groups}")
    say(f"log_likelihood stored (needed for WAIC/LOO): {'/log_likelihood' in dt.groups}")
    say()

    # ---- 1. pooled summary -------------------------------------------------
    summ = az.summary(dt)
    summ.to_csv(out / "summary_pooled.csv")
    say("=== pooled summary, worst 15 by r-hat ===")
    say(summ.sort_values("r_hat", ascending=False).head(15).to_string())
    say()

    # ---- 2. per-chain summary (is each chain healthy on its own?) ----------
    frames = []
    for c in range(n_chain):
        s = az.summary(dt.sel(chain=[c]))
        s["chain"] = c
        frames.append(s.reset_index().rename(columns={"index": "variable"}))
    per_chain = pd.concat(frames, ignore_index=True)
    per_chain.to_csv(out / "summary_per_chain.csv", index=False)
    say("=== lowest bulk ESS within each chain considered alone ===")
    say(per_chain.groupby("chain")["ess_bulk"].min().to_string())
    say()

    # ---- 3. mode structure -------------------------------------------------
    labels, lp_means = label_modes(dt)
    if labels is not None:
        say("=== chains grouped by mean log-posterior ===")
        for c in range(n_chain):
            say(f"  chain {c}: mode {labels[c]}   mean lp = {lp_means[c]:.1f}")
        say()

    # ---- 4. built-in diagnostic readout ------------------------------------
    say("=== az.diagnose() ===")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        az.diagnose(dt)
    say(buf.getvalue())

    try:
        b = az.bfmi(dt)
        vals = b["energy"].values if hasattr(b, "__getitem__") else np.asarray(b)
        say(f"BFMI per chain: {np.round(np.asarray(vals), 3)}  (below 0.3 is a warning)")
    except Exception as e:
        say(f"BFMI unavailable: {e}")
    say()

    # ---- 5. figures --------------------------------------------------------
    def save(pc, name):
        for ext in ("pdf", "png"):
            pc.savefig(out / f"{name}.{ext}")
        say(f"wrote {name}.pdf / .png")

    worst = summ.sort_values("r_hat", ascending=False).index[: args.n_worst].tolist()
    worst_vars = sorted({str(w).split("[")[0] for w in worst})
    say(f"worst-mixing variables plotted: {worst_vars}")

    for name, fn, kw in [
        ("fig23_forest", az.plot_forest, dict(combined=False)),
        ("fig24_convergence", az.plot_convergence_dist, {}),
        ("fig25_rank", az.plot_rank, dict(var_names=worst_vars)),
        ("fig26_trace", az.plot_trace, dict(var_names=worst_vars)),
        ("fig27_energy", az.plot_energy, {}),
    ]:
        try:
            save(fn(dt, backend="matplotlib", **kw), name)
        except Exception as e:
            say(f"{name} skipped: {type(e).__name__}: {e}")

    # ---- 6. correlations -> pair plot (the equifinality picture for RQ3) ----
    names, X = flatten_posterior(post)
    keep = [i for i in range(len(names)) if X[:, i].std() > 0]
    C = np.corrcoef(X[:, keep], rowvar=False)
    pairs = sorted(
        (
            (abs(C[i, j]), names[keep[i]], names[keep[j]])
            for i, j in itertools.combinations(range(len(keep)), 2)
        ),
        reverse=True,
    )[:10]
    say()
    say("=== 10 strongest posterior correlations (candidate equifinality ridges) ===")
    for r, a, b in pairs:
        say(f"  |r| = {r:.3f}   {a} <-> {b}")

    # prefer scalar parameters: vector entries blow the panel count up fast
    scalar_pairs = [p for p in pairs if "[" not in p[1] and "[" not in p[2]]
    pair_vars = []
    for _, a, b in (scalar_pairs or pairs)[:3]:
        for v in (a.split("[")[0], b.split("[")[0]):
            if v not in pair_vars:
                pair_vars.append(v)
    try:
        save(az.plot_pair(dt, var_names=pair_vars[:5], backend="matplotlib"), "fig28_pair")
    except Exception as e:
        say(f"fig28_pair skipped: {type(e).__name__}: {e}")

    (out / "diagnostics.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nText readout: {out / 'diagnostics.txt'}")


if __name__ == "__main__":
    main()