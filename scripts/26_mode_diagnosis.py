#!/usr/bin/env python
"""Take the two comparison traces apart by chain. Samples nothing.

``scripts/compare_structures.py`` found a worst r-hat near 1.53 and a minimum
ESS below 11 for both models. Chains sitting in different modes produce those
numbers, and so do chains creeping along one ridge; pooled statistics cannot
tell which. This script reads the same two traces and reports, for each model
and each chain:

1. **Chain means and groups.** The per-chain mean of every variable in the
   posterior group: the sampled parameters and every recorded deterministic
   (allocation fractions, derived initial pools, ``rh_ref``, mean foliar carbon,
   annual fluxes). The gap between two chains is the largest, over those
   quantities, of the difference in means in units of the pooled within-chain
   sd. Chains closer than ``--threshold`` are put in one group (single linkage).
   The same measure between the two halves of one chain shows whether that
   chain had settled.
2. **Log posterior.** Per-chain mean of ``lp``, and of the calibration
   log-likelihood from the forward runs. ``lp`` is on the sampler's
   unconstrained scale and so carries the Jacobian of every bound transform,
   which lowers it for a chain near a prior bound; the log-likelihood does not.
3. **Held-out skill**, all days and spring, by
   ``compare_structures.skill_table`` itself with one chain's draws in place of
   one model's. The forward draws are the ones ``compare_structures.py`` ran, so
   the pooled rows should reproduce its ``skill.csv``; the report says whether
   they do.
4. **Annual GPP, Reco and NEE** over the calibration block, from the trace's own
   ``gpp_annual`` and ``nee_annual``, against the measured ranges
   ``scripts/04_calibrate.py`` reports against.

Writes ``results/comparison/modes.txt``.

Usage
-----
    python scripts/26_mode_diagnosis.py \\
        results/calibration_dalec2_hemisurface.nc \\
        results/calibration_evergreen_hemisurface.nc
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
for _path in (_SCRIPTS.parent / "src", _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from compare_structures import (  # noqa: E402
    DEFAULT_FORWARD_DRAWS,
    GRAPHS,
    OUT_DIR,
    load_record,
    skill_table,
)

from dalec.acm import acm_from_config  # noqa: E402
from dalec.compute import compile_function  # noqa: E402
from dalec.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from dalec.model_evergreen import EVERGREEN_ALLOCATION_ORDER  # noqa: E402
from dalec.parameters import ALLOCATION_WEIGHT_ORDER  # noqa: E402

#: Two chains share a group when no recorded quantity separates their means by
#: more than this many within-chain standard deviations.
DEFAULT_THRESHOLD = 1.0
#: Largest difference at which the pooled skill counts as reproducing skill.csv.
REPRODUCTION_TOLERANCE = 1e-6
SCORES = ("rmse", "bias", "coverage", "lpd_per_day")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dalec2", type=Path, help="DALEC2 posterior .nc")
    parser.add_argument("evergreen", type=Path, help="evergreen posterior .nc")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_FORWARD_DRAWS,
        help="Forward draws per model, shared between its chains. The default is "
             "compare_structures.py's, which is what lets the pooled skill reproduce it.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Largest gap, in within-chain sds, between two chains of one group.",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    return parser.parse_args()


def load_calibrate_script():
    """``scripts/04_calibrate.py`` as a module, for the measured ranges it reports against.

    Its file name starts with a digit, so ``import`` cannot reach it.
    """
    spec = importlib.util.spec_from_file_location("calibrate", _SCRIPTS / "04_calibrate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Chain means and groups
# ---------------------------------------------------------------------------
def recorded_quantities(idata, variant) -> tuple[list[str], np.ndarray]:
    """Every variable in the posterior group, shape (chains, draws, quantities).

    Vector variables are split by component; the allocation simplex is labelled
    with the pool each weight feeds.
    """
    posterior = idata["posterior"]
    order = ALLOCATION_WEIGHT_ORDER if variant == "dalec2" else EVERGREEN_ALLOCATION_ORDER
    labels, columns = [], []
    for name in posterior.data_vars:
        values = posterior[name].transpose("chain", "draw", ...).values
        values = values.reshape(*values.shape[:2], -1)
        size = values.shape[2]
        parts = order if name == "allocation_weights" else range(size)
        labels += [str(name)] if size == 1 else [f"{name}[{part}]" for part in parts]
        columns += [values[:, :, k] for k in range(size)]
    return labels, np.stack(columns, axis=2)


def separation(a, b) -> np.ndarray:
    """Gap between the means of two sets of draws, in pooled sds, per quantity."""
    gap = np.abs(a.mean(axis=0) - b.mean(axis=0))
    scale = np.sqrt(0.5 * (a.var(axis=0) + b.var(axis=0)))
    # No spread at all: apart if the means differ, together if they do not.
    return np.divide(gap, scale, out=np.where(gap > 0.0, np.inf, 0.0), where=scale > 0.0)


def group_chains(gap, threshold) -> list[str]:
    """A group letter per chain: two chains closer than `threshold` share one.

    Single linkage, so a chain joins a group by being close to any one member.
    """
    group = list(range(len(gap)))
    for a, b in combinations(range(len(gap)), 2):
        if gap[a, b] < threshold:
            group = [group[a] if g == group[b] else g for g in group]
    first_seen = list(dict.fromkeys(group))
    return [chr(ord("A") + first_seen.index(g)) for g in group]


def section_groups(labels, values, threshold) -> tuple[list[str], list[str]]:
    """The grouping and the per-chain mean of every quantity; also each chain's group."""
    n_chains = values.shape[0]
    gaps = np.zeros((n_chains, n_chains, len(labels)))
    for a, b in combinations(range(n_chains), 2):
        gaps[a, b] = gaps[b, a] = separation(values[a], values[b])
    names = group_chains(gaps.max(axis=2), threshold)

    members: dict[str, list[str]] = {}
    for chain, name in enumerate(names):
        members.setdefault(name, []).append(str(chain))
    lines = [
        f"CHAIN GROUPS (chains closer than {threshold:g} within-chain sd on every quantity)",
        "  " + "    ".join(f"group {name} = chain {', '.join(chains)}"
                           for name, chains in members.items()),
        "",
        "  gap between two chains: largest difference in means over all quantities,",
        "  in pooled within-chain sds",
        f"  {'pair':<8}{'gap':>8}   largest on",
    ]
    for a, b in combinations(range(n_chains), 2):
        worst = int(gaps[a, b].argmax())
        pair = f"{a} - {b}"
        lines.append(f"  {pair:<8}{gaps[a, b, worst]:>8.2f}   {labels[worst]}")

    half = values.shape[1] // 2
    lines += ["", "  first half of each chain against its second half, same measure",
              f"  {'chain':<8}{'gap':>8}   largest on"]
    for chain in range(n_chains):
        drift = separation(values[chain, :half], values[chain, half:])
        worst = int(drift.argmax())
        lines.append(f"  {chain:<8}{drift[worst]:>8.2f}   {labels[worst]}")
    lines += [
        "  Read: a chain whose own halves are as far apart as the groups are had not",
        "  settled, which is slow mixing along one ridge rather than a second mode.",
        "  A gap set by a quantity pinned at a prior bound can be large in sds and",
        "  negligible in value: check that quantity's row below before trusting a split.",
    ]

    means = values.mean(axis=1)
    largest = gaps.max(axis=(0, 1))
    lines += [
        "",
        "  per-chain mean of every recorded quantity, most separated first",
        f"  {'quantity':<26}" + "".join(f"{'chain ' + str(c):>12}" for c in range(n_chains))
        + f"{'gap':>8}",
        f"  {'group':<26}" + "".join(f"{name:>12}" for name in names),
    ]
    for q in np.argsort(-largest, kind="stable"):
        lines.append(f"  {labels[q]:<26}" + "".join(f"{mean:>12.4g}" for mean in means[:, q])
                     + f"{largest[q]:>8.2f}")
    return lines, names


# ---------------------------------------------------------------------------
# Forward runs, log posterior and skill
# ---------------------------------------------------------------------------
def forward_nee(idata, variant, record, acm, n_draws) -> tuple[np.ndarray, np.ndarray]:
    """Daily NEE, shape (n_draws, n_days), and the chain each draw came from.

    The graph and the draws are those of ``compare_structures.forward_nee``:
    evenly spaced through the chains laid end to end. Repeated here rather than
    called, because that function does not say which chain a draw belongs to.
    """
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
    chain_of = picks // int(posterior.sizes["draw"])
    return np.array([run(values[i]) for i in picks]), chain_of


def calibration_loglik(nee, record, n_calibration_days) -> np.ndarray:
    """NEE log-likelihood of each forward draw over the calibration nee_mask days.

    The likelihood the sampler used: Gaussian, RANDUNC as sd, masked days out.
    """
    in_calibration = np.arange(record.n_days) < n_calibration_days
    days = np.flatnonzero(record.nee_mask & in_calibration)
    sigma = record.nee_unc[days]
    residual = (record.nee_obs[days] - nee[:, days]) / sigma
    return (-0.5 * residual**2 - np.log(sigma) - 0.5 * math.log(2.0 * math.pi)).sum(axis=1)


def spread(per_chain) -> str:
    """Range of the chain means, also in units of the pooled within-chain sd."""
    gap = np.ptp([values.mean() for values in per_chain])
    within = np.sqrt(np.mean([values.var() for values in per_chain]))
    return f"{gap:.1f} = {gap / within:.1f} within-chain sd"


def section_log_posterior(idata, loglik, chain_of, names) -> list[str]:
    stats = idata["sample_stats"]
    recorded = "lp" in stats
    lp = stats["lp"].values if recorded else np.full((len(names), 1), np.nan)
    lp_chains = [lp[chain] for chain in range(len(names))]
    loglik_chains = [loglik[chain_of == chain] for chain in range(len(names))]

    lines = [
        "LOG POSTERIOR",
        "  lp          PyMC's model logp on the sampler's unconstrained scale, every draw",
        "  cal loglik  NEE log-likelihood over the calibration nee_mask days, forward draws",
        f"  {'chain':<6}{'group':<6}{'lp mean':>12}{'lp sd':>9}{'lp max':>12}"
        f"{'cal loglik':>13}{'sd':>9}{'draws':>7}",
    ]
    for chain, (lp_c, loglik_c) in enumerate(zip(lp_chains, loglik_chains, strict=True)):
        lines.append(f"  {chain:<6}{names[chain]:<6}{lp_c.mean():>12.1f}{lp_c.std():>9.1f}"
                     f"{lp_c.max():>12.1f}{loglik_c.mean():>13.1f}{loglik_c.std():>9.1f}"
                     f"{loglik_c.size:>7}")
    if not recorded:
        lines.append("  lp is not in sample_stats for this trace.")
    lines += [
        f"  range of the chain means: lp {spread(lp_chains)}; "
        f"cal loglik {spread(loglik_chains)}",
        "  Read: lp carries the Jacobian of every bound transform, which lowers it for a",
        "  chain near a prior bound; cal loglik does not. Neither says how much posterior",
        "  mass a group holds: chains that never cross cannot weigh one against another.",
    ]
    return lines


def reproduction_check(pooled, variant, out_dir) -> str:
    """Whether the pooled rows equal what compare_structures.py wrote to skill.csv."""
    path = out_dir / "skill.csv"
    if not path.exists():
        return f"the 'all' rows were not checked: no {path}"
    reference = pd.read_csv(path)
    merged = pooled.merge(reference[reference["model"] == variant], on=["window", "days"],
                          suffixes=("", "_reference"))
    if len(merged) != len(pooled):
        return f"the 'all' rows were not checked: {path} has no matching {variant} rows"
    worst = max(float((merged[score] - merged[f"{score}_reference"]).abs().max())
                for score in SCORES)
    if worst <= REPRODUCTION_TOLERANCE:
        return f"the 'all' rows reproduce {path} (largest difference {worst:.1e})"
    return (f"the 'all' rows DO NOT reproduce {path} (largest difference {worst:.1e}): "
            "other draws, seed, data or traces went into it")


def section_skill(nee, chain_of, record, n_calibration_days, prob, seed, names, variant,
                  out_dir) -> list[str]:
    """Held-out skill by chain, scored by ``compare_structures.skill_table``."""
    runs = {"all": nee, **{str(chain): nee[chain_of == chain] for chain in range(len(names))}}
    scores = skill_table(runs, record, n_calibration_days, prob, seed)
    scores = scores.rename(columns={"model": "chain"})
    group_of = {"all": "-", **{str(chain): name for chain, name in enumerate(names)}}
    scores.insert(1, "group", [group_of[chain] for chain in scores["chain"]])

    held_out = scores[scores["window"] == "held-out"].drop(columns="window")
    table = held_out.sort_values("days", kind="stable").to_string(
        index=False, float_format=lambda x: f"{x:.4f}")
    return [
        f"HELD-OUT SKILL on nee_mask days, method of compare_structures.py ({prob:.0%} "
        "predictive interval)",
        *("  " + line for line in table.splitlines()),
        "  " + reproduction_check(scores[scores["chain"] == "all"], variant, out_dir),
        "  Read: each chain is scored on its own forward draws only (the draws column of",
        "  the log posterior table), so small differences between chains are noise.",
    ]


# ---------------------------------------------------------------------------
# Annual fluxes
# ---------------------------------------------------------------------------
def section_fluxes(idata, calibrate, prob, names) -> list[str]:
    """Annual GPP, Reco and NEE by chain, against the measured ranges.

    From the trace's own annual deterministics, so every draw is used and the
    numbers are those of the calibration log split by chain. Reco is
    ``gpp_annual + nee_annual``, exact because NEE = Reco - GPP every timestep
    (DECISIONS §4).
    """
    posterior = idata["posterior"]
    gpp = posterior["gpp_annual"].values
    nee = posterior["nee_annual"].values
    tail = 50.0 * (1.0 - prob)
    measured_gpp, measured_reco = calibrate.MEASURED_GPP, calibrate.MEASURED_RECO
    measured_nee = calibrate.MEASURED_NEE

    def cell(values, note) -> str:
        median = float(np.median(values))
        low, high = np.percentile(values, [tail, 100.0 - tail])
        return f"{median:>9.1f} [{low:>7.1f}, {high:>7.1f}] {note(median):<7}"

    lines = [
        f"ANNUAL FLUXES over the calibration block, g C m-2 yr-1, every draw: "
        f"median [{prob:.0%} interval]",
        f"  measured: GPP {measured_gpp[0]:.0f}-{measured_gpp[1]:.0f}, "
        f"Reco {measured_reco[0]:.0f}-{measured_reco[1]:.0f}, NEE {measured_nee:.1f} "
        "(the NEE column ends with median minus this)",
        f"  {'chain':<6}{'group':<6}{'GPP':>9}{'Reco':>36}{'NEE':>36}",
    ]
    rows = [(str(chain), name, slice(chain, chain + 1)) for chain, name in enumerate(names)]
    for chain, name, pick in [*rows, ("all", "-", slice(None))]:
        lines.append(
            f"  {chain:<6}{name:<6}"
            + cell(gpp[pick], lambda median: calibrate.band(median, *measured_gpp))
            + cell((gpp + nee)[pick], lambda median: calibrate.band(median, *measured_reco))
            + cell(nee[pick], lambda median: f"{median - measured_nee:+.1f}")
        )
    return lines


# ---------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import arviz as az

    args = parse_args()
    config = load_config(args.config)
    prob = float(config["diagnostics"]["hdi_prob"])
    seed = int(config["seed"])
    calibrate = load_calibrate_script()
    record, n_calibration_days = load_record(config)
    acm = acm_from_config(config)
    args.out.mkdir(parents=True, exist_ok=True)

    idatas = {"dalec2": az.from_netcdf(args.dalec2), "evergreen": az.from_netcdf(args.evergreen)}
    for variant, idata in idatas.items():
        stored = idata.attrs.get("model_variant")
        if stored is not None and stored != variant:
            raise SystemExit(f"{variant} argument is a {stored} trace")
        missing = [name for name in ("gpp_annual", "nee_annual")
                   if name not in idata["posterior"]]
        if missing:
            raise SystemExit(f"the {variant} trace has no {', '.join(missing)}")
        if args.draws < 2 * int(idata["posterior"].sizes["chain"]):
            raise SystemExit("--draws leaves fewer than two forward draws per chain")

    # Everything that needs only the traces comes first: it takes seconds, so a
    # problem there shows before the forward graphs are compiled.
    from_trace = {}
    for variant, idata in idatas.items():
        labels, values = recorded_quantities(idata, variant)
        groups, names = section_groups(labels, values, args.threshold)
        from_trace[variant] = (names, groups, section_fluxes(idata, calibrate, prob, names))

    bar = "=" * 74
    lines = [
        f"PER-CHAIN DIAGNOSIS -- {args.draws} forward draws per model",
        "Chains are numbered from 0 in trace order. Group letters are assigned within",
        "one model: group A of DALEC2 has nothing to do with group A of the evergreen.",
    ]
    for variant, idata in idatas.items():
        names, groups, fluxes = from_trace[variant]
        nee, chain_of = forward_nee(idata, variant, record, acm, args.draws)
        loglik = calibration_loglik(nee, record, n_calibration_days)
        sizes = idata["posterior"].sizes
        lines += [
            "", bar,
            f"  {variant} -- {sizes['chain']} chains x {sizes['draw']} draws, "
            f"convention {idata.attrs.get('convention', 'not recorded')}",
            bar, "",
            *groups, "",
            *section_log_posterior(idata, loglik, chain_of, names), "",
            *section_skill(nee, chain_of, record, n_calibration_days, prob, seed, names,
                           variant, args.out), "",
            *fluxes,
        ]

    report = "\n".join(lines) + "\n"
    (args.out / "modes.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"report in {args.out / 'modes.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
