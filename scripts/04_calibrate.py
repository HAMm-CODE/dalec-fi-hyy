#!/usr/bin/env python
"""Real-data NUTS calibration over the calibration block.

Assimilates daily ``NEE_VUT_REF`` only, with a Gaussian likelihood whose per-day
sd is ``NEE_VUT_REF_RANDUNC``. Writes the trace as NetCDF (never pickle) with the
seed recorded, plus the convergence summary.

**Everything NUTS-specific is left at PyMC's defaults.** No ``target_accept``, no
``max_treedepth``, no ``init``, no step-size hint -- the config carries values for
some of these and they are deliberately not read. A first real run should measure
the problem rather than a tuning choice, and a number produced under tuning
cannot be compared against one produced without it. The single exception is
``cores``, which is a resource-allocation setting rather than a sampler
parameter: PyMC would otherwise infer it from ``os.cpu_count()``, which on a
shared SLURM node reports the whole machine rather than the allocation.

The convention is an argument, not a default
--------------------------------------------
Both LAI conventions stay live and neither is adopted (DECISIONS §10,
LIMITATIONS §15). The calibration is run under each and the difference reported
as a sensitivity, so the projected run is this same script with
``--convention projected`` and the convention appears in every output filename.
A GPP number from this script is not interpretable without it.

The re-timing comes out of this run, not a separate one
-------------------------------------------------------
Before sampling starts the job prints interpreter architecture and hardware
architecture on **separate lines**, the resolved linker, ``config.cxx``, and one
forward pass and one gradient at the real step count. That is the measurement
``scripts/21_cluster_timing.py`` exists for, taken here as a by-product so the
cluster figures arrive with the run rather than needing their own job.

The architecture lines are printed separately because ``platform.machine()``
reports the *hardware*: on the development laptop it prints ``ARM64`` while the
interpreter is an emulated x86-64 build, which is how every earlier timing report
came to carry a misleading header (DECISIONS §13).

Usage
-----
    python scripts/04_calibrate.py
    python scripts/04_calibrate.py --convention projected
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import sysconfig
import time
import warnings
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.compute import resolved_linker_class  # noqa: E402
from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    require_year_block,
    resolve_path,
)
from dalec.data_io import SiteData  # noqa: E402
from dalec.model import build_forward_graph  # noqa: E402
from dalec.parameters import (  # noqa: E402
    DEFAULT_LAI_CONVENTION,
    LAI_CONVENTIONS,
    PARAMETER_NAMES,
    DalecParameters,
    prior_bounds,
)
from dalec.sampler import initial_point_is_finite, model_from_config, sample  # noqa: E402

RESULTS_DIR = Path("results")

#: Ilvesniemi et al. (2009) Fig. 6, eddy-covariance GPP, g C m-2 yr-1.
#: Verified against the paper; DECISIONS §7.
MEASURED_GPP = (952.0, 1104.0)

#: Ecosystem respiration over the same block, g C m-2 yr-1.
MEASURED_RECO = (761.0, 898.0)

#: Observed annual NEE over the calibration block, g C m-2 yr-1 (DECISIONS §7).
MEASURED_NEE = -215.8

#: A sampled scalar is called "at a bound" when its posterior mean sits within
#: this fraction of the prior range from either end.
BOUND_TOLERANCE = 0.05

N_TIMING_WARMUP = 2
N_TIMING_RUNS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--convention",
        default=DEFAULT_LAI_CONVENTION,
        choices=sorted(LAI_CONVENTIONS),
        help="LAI convention. The projected run is this script with this flag.",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Defaults to --chains. A resource setting, not a sampler parameter.",
    )
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip the pre-sampling forward/gradient timing.",
    )
    return parser.parse_args()


def reference_parameters(seed: int = 5) -> DalecParameters:
    """A single prior-range point, only ever used to time the graph."""
    rng = np.random.default_rng(seed)
    scalars = {
        name: float(rng.uniform(*prior_bounds(name)))
        for name in (
            "f_auto", "theta_roo", "theta_woo", "theta_lit", "theta_som",
            "theta_min", "temperature_exponent", "d_onset", "cr_onset",
            "d_fall", "cr_fall", "lma", "ceff",
        )
    }
    scalars["c_lf"] = 0.25
    pools = {
        "c_lab_0": 80.0, "c_fol_0": 600.0, "c_roo_0": 240.0,
        "c_woo_0": 6000.0, "c_lit_0": 120.0, "c_som_0": 5800.0,
    }
    return DalecParameters.from_allocation_simplex(
        f_auto=scalars.pop("f_auto"),
        allocation_weights=rng.dirichlet(np.ones(4)),
        **scalars,
        **pools,
    )


def print_provenance(out, config, block) -> dict[str, object]:
    """Architecture, backend and compiler, before anything expensive happens."""
    import pytensor

    interpreter = sysconfig.get_platform()
    hardware = platform.machine()
    emulated = ("amd64" in interpreter.lower() or "x86_64" in interpreter.lower()) and (
        "arm" in hardware.lower() or "aarch64" in hardware.lower()
    )
    linker = resolved_linker_class()

    out(f"  hostname                {platform.node()}")
    out(f"  interpreter arch        {interpreter}")
    out(f"  hardware arch           {hardware}  ({platform.processor()})")
    if emulated:
        out("  ** EMULATED: x86-64 interpreter on ARM hardware. These timings")
        out("     are inflated and are not a native measurement.")
    out(f"  python                  {platform.python_version()}")
    out(f"  pytensor                {pytensor.__version__}")
    out(f"  resolved linker         {linker}")
    out(f"  pytensor.config.cxx     {pytensor.config.cxx!r}")
    out("  C backend               {}".format(
        "available" if pytensor.config.cxx else "NOT available"))
    return {
        "hostname": platform.node(),
        "interpreter_arch": interpreter,
        "hardware_arch": hardware,
        "processor": platform.processor(),
        "emulated": bool(emulated),
        "python": platform.python_version(),
        "pytensor": pytensor.__version__,
        "resolved_linker": linker,
        "config_cxx": pytensor.config.cxx,
    }


def time_graph(out, config, block, acm) -> dict[str, float]:
    """One forward pass and one gradient at the real step count.

    Compiles a standalone copy of the forward graph over a plain ``dvector`` of
    parameters. That is the same graph the sampler runs, but compiled separately
    so the timing is of the model and not of PyMC's logp wrapper.
    """
    import pytensor
    import pytensor.tensor as pt

    theta = pt.dvector("theta")
    named = dict(
        zip(PARAMETER_NAMES, [theta[i] for i in range(len(PARAMETER_NAMES))],
            strict=True)
    )
    graph = build_forward_graph(
        parameters=named,
        doy=block.doy.astype(float),
        t_air=block.t_air,
        t_max=block.t_max,
        t_min=block.t_min,
        sw_in=block.sw_in,
        co2=block.co2,
        latitude_deg=float(config["site"]["latitude_deg"]),
        coefficients=acm.coefficients,
        frost_threshold_degc=acm.frost_threshold_degc,
    )
    observed = np.where(block.nee_mask, block.nee_obs, 0.0)
    sigma = np.where(block.nee_mask, block.nee_unc, 1.0)
    weight = block.nee_mask.astype(float)
    loss = pt.sum(weight * pt.sqr((graph.nee - observed) / sigma))

    compile_start = time.perf_counter()
    forward = pytensor.function([theta], graph.nee, on_unused_input="ignore")
    gradient = pytensor.function(
        [theta], pt.grad(loss, theta), on_unused_input="ignore"
    )
    compile_s = time.perf_counter() - compile_start

    theta0 = np.array(
        [reference_parameters().to_dict()[name] for name in PARAMETER_NAMES]
    )

    def measure(function) -> np.ndarray:
        for _ in range(N_TIMING_WARMUP):
            function(theta0)
        times = []
        for _ in range(N_TIMING_RUNS):
            start = time.perf_counter()
            function(theta0)
            times.append(time.perf_counter() - start)
        return np.array(times)

    forward_ms = 1000 * float(np.median(measure(forward)))
    gradient_ms = 1000 * float(np.median(measure(gradient)))
    ratio = gradient_ms / forward_ms

    out(f"  compile                 {compile_s:.1f} s")
    out(f"  forward pass            {forward_ms:.2f} ms   at {block.n_days} steps")
    out(f"  gradient                {gradient_ms:.2f} ms")
    out(f"  ratio                   {ratio:.2f}x")
    out("  (healthy is 2-5x; the regression this guards against is 37.6x)")
    return {
        "compile_s": compile_s,
        "forward_ms": forward_ms,
        "gradient_ms": gradient_ms,
        "ratio": ratio,
        "n_steps": int(block.n_days),
    }


def report_bounds(out, idata, bounds) -> list[dict[str, object]]:
    """Which sampled scalars sit at a prior bound, and in which direction.

    Reported because a posterior pressed against a bound means the prior is
    truncating the answer: the data wanted to go further and the range would not
    let it. It is a statement about the prior, not about the parameter.
    """
    posterior = idata.posterior
    at_bounds: list[dict[str, object]] = []
    out("  {:<24} {:>12} {:>10} {:>10} {:>9}".format(
        "parameter", "mean", "lower", "upper", "position"))
    for name in sorted(bounds):
        if name not in posterior:
            continue
        lower, upper = bounds[name]
        mean = float(posterior[name].values.mean())
        span = upper - lower
        position = (mean - lower) / span if span else float("nan")
        flag = ""
        if position <= BOUND_TOLERANCE:
            flag = "  <- AT LOWER BOUND"
            at_bounds.append({"parameter": name, "direction": "lower",
                              "mean": mean, "position": position})
        elif position >= 1.0 - BOUND_TOLERANCE:
            flag = "  <- AT UPPER BOUND"
            at_bounds.append({"parameter": name, "direction": "upper",
                              "mean": mean, "position": position})
        out(f"  {name:<24} {mean:>12.4g} {lower:>10.4g} {upper:>10.4g} {position:>8.2f}{flag}")

    out("")
    if at_bounds:
        upper = [b["parameter"] for b in at_bounds if b["direction"] == "upper"]
        lower = [b["parameter"] for b in at_bounds if b["direction"] == "lower"]
        out(f"  {len(at_bounds)} of {len(bounds)} sampled scalars sit within "
            f"{BOUND_TOLERANCE:.0%} of a bound.")
        if upper:
            out("    upper: {}".format(", ".join(upper)))
        if lower:
            out("    lower: {}".format(", ".join(lower)))
        out("  A posterior pressed against a bound is the prior truncating the")
        out("  answer. Report it as a property of the prior range, not as an")
        out("  estimate of the parameter.")
    else:
        out(f"  No sampled scalar sits within {BOUND_TOLERANCE:.0%} of a bound.")
    return at_bounds


def band(value: float, low: float, high: float) -> str:
    if value < low:
        return "BELOW"
    if value > high:
        return "ABOVE"
    return "inside"


def report_fluxes(out, idata, block) -> dict[str, object]:
    """Posterior GPP, Reco and NEE against the measured ranges.

    ``Reco`` is not a recorded deterministic. It does not need to be: carbon
    conservation makes ``NEE = Reco - GPP`` an exact identity every timestep
    (DECISIONS §4), so ``reco_annual = gpp_annual + nee_annual`` exactly, with no
    discretisation error to worry about.
    """
    posterior = idata.posterior
    gpp = posterior["gpp_annual"].values.reshape(-1)
    nee = posterior["nee_annual"].values.reshape(-1)
    reco = gpp + nee

    def describe(name, values, measured):
        median = float(np.median(values))
        lo, hi = np.percentile(values, [3, 97])
        verdict = band(median, *measured)
        out(f"  {name:<10} median {median:>9.1f}   94% HDI {lo:>8.1f} - {hi:<8.1f}  "
            f"measured {measured[0]:>6.0f}-{measured[1]:<6.0f}  {verdict}")
        return {"median": median, "hdi_low": float(lo), "hdi_high": float(hi),
                "verdict": verdict}

    gpp_stats = describe("GPP", gpp, MEASURED_GPP)
    reco_stats = describe("Reco", reco, MEASURED_RECO)

    nee_median = float(np.median(nee))
    nee_lo, nee_hi = np.percentile(nee, [3, 97])
    observed_daily = block.nee_obs[block.nee_mask]
    observed_annual = float(observed_daily.mean() * 365.25)
    out("  {:<10} median {:>9.1f}   94% HDI {:>8.1f} - {:<8.1f}  "
        "observed {:>7.1f}".format("NEE", nee_median, nee_lo, nee_hi, MEASURED_NEE))
    out("")
    out(f"  Observed annual NEE from this block's assimilable days: {observed_annual:.1f}")
    out(f"  (recorded value {MEASURED_NEE:.1f}; the modelled scalar averages ALL "
        f"{block.n_days} days")
    out(f"   while the observed average is over the {block.n_assimilated} assimilable ones, so the")
    out("   two are close but not a like-for-like comparison.)")
    return {
        "gpp": gpp_stats,
        "reco": reco_stats,
        "nee": {"median": nee_median, "hdi_low": float(nee_lo),
                "hdi_high": float(nee_hi)},
        "observed_nee_assimilable": observed_annual,
    }


def report_rq3(out, fluxes) -> dict[str, object]:
    """Test the prediction recorded before sampling.

    From ``reports/prior_diagnostics/FINDINGS_gpp.md``:

      **Falsified by** a posterior with GPP near 1,030 and Reco near 850 and no
      seasonal structure in the residuals. **Confirmed by** both gross fluxes
      biased high by the same absolute amount, leaving NEE right for the wrong
      reasons.

    Only the magnitude half is testable from the scalars this run records. The
    seasonal half needs monthly posterior-predictive residuals, which is a
    separate pass over the saved trace.
    """
    gpp = fluxes["gpp"]["median"]
    reco = fluxes["reco"]["median"]
    nee = fluxes["nee"]["median"]

    gpp_mid = sum(MEASURED_GPP) / 2
    reco_mid = sum(MEASURED_RECO) / 2
    gpp_excess = gpp - gpp_mid
    reco_excess = reco - reco_mid
    nee_error = nee - MEASURED_NEE

    out("  Recorded before sampling (FINDINGS_gpp.md):")
    out("    CONFIRMED by both gross fluxes biased high by the same absolute")
    out("      amount, leaving NEE right for the wrong reasons.")
    out("    FALSIFIED by GPP near 1,030 and Reco near 850.")
    out("")
    out(f"  GPP  excess over measured midpoint  {gpp_excess:>+9.1f} g C m-2 yr-1")
    out(f"  Reco excess over measured midpoint  {reco_excess:>+9.1f}")
    out(f"  NEE  error against observed         {nee_error:>+9.1f}")
    out("")

    # "Biased high" means above the measured RANGE, not merely above its
    # midpoint. Testing against the midpoint would call a posterior sitting
    # comfortably inside the measured range "biased high" whenever it landed in
    # the upper half of it, which is the opposite of what the prediction says.
    both_high = (
        fluxes["gpp"]["verdict"] == "ABOVE" and fluxes["reco"]["verdict"] == "ABOVE"
    )
    nee_ok = abs(nee_error) < 50.0
    similar = (
        abs(gpp_excess - reco_excess) < 0.25 * max(abs(gpp_excess), abs(reco_excess))
        if max(abs(gpp_excess), abs(reco_excess)) > 0
        else False
    )
    both_inside = (
        fluxes["gpp"]["verdict"] == "inside" and fluxes["reco"]["verdict"] == "inside"
    )

    if both_high and nee_ok and similar:
        verdict = "CONFIRMED"
        out("  -> CONFIRMED. Both gross fluxes are biased high by comparable")
        out("     amounts and NEE is close to observed: the net flux is right")
        out("     for the wrong reasons, which is the compensating-error result.")
    elif both_inside and nee_ok:
        verdict = "FALSIFIED"
        out("  -> FALSIFIED on the magnitude half. Both gross fluxes sit inside")
        out("     their measured ranges with NEE close to observed.")
    else:
        verdict = "NEITHER"
        out("  -> NEITHER cleanly. The numbers above do not match either arm of")
        out("     the prediction as stated; describe what they do show rather")
        out("     than forcing them into one.")
    out("")
    out("  The seasonal half of the prediction -- a systematic autumn deficit and")
    out("  spring surplus in the monthly residuals -- is NOT tested here. It needs")
    out("  monthly posterior-predictive residuals from the saved trace, which is")
    out("  a separate pass. Do not report this verdict as covering it.")
    return {
        "verdict": verdict,
        "gpp_excess": gpp_excess,
        "reco_excess": reco_excess,
        "nee_error": nee_error,
        "both_gross_fluxes_high": bool(both_high),
    }


def main() -> int:
    for category in (RuntimeWarning, DeprecationWarning, UserWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)
    import arviz as az

    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cores = args.cores if args.cores is not None else args.chains

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    calibration = require_year_block(config, "calibration")
    site_code = str(config.get("site", {}).get("code", ""))
    slug = site_code.lower().replace("-", "_")
    block = SiteData.load(
        resolve_path(config["paths"]["processed_dir"])
        / f"{slug}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )
    acm = acm_from_config(config)
    seed = int(config["seed"])

    bar = "=" * 74
    out(bar)
    out("  DALEC2 CALIBRATION -- FI-Hyy, NEE_VUT_REF only")
    out(bar)
    out(f"  block                   {calibration[0]}-{calibration[1]}, "
        f"{block.n_days} days, {block.n_assimilated} assimilable")
    out(f"  convention              {args.convention}")
    out(f"  sampling                {args.chains} chains, {args.tune} tune, "
        f"{args.draws} draws, {cores} cores")
    out(f"  seed                    {seed}")
    out("  NUTS settings           PyMC defaults throughout -- target_accept,")
    out("                          max_treedepth, init and step size are NOT set")
    out("")

    out(bar)
    out("  Environment")
    out(bar)
    environment = print_provenance(out, config, block)

    timing: dict[str, float] = {}
    if not args.skip_timing:
        out("")
        out(bar)
        out("  Graph timing at the real step count")
        out(bar)
        timing = time_graph(out, config, block, acm)

    out("")
    out(bar)
    out("  Model")
    out(bar)
    build_start = time.perf_counter()
    dalec = model_from_config(config, block, convention=args.convention)
    build_s = time.perf_counter() - build_start
    finite, logp = initial_point_is_finite(dalec)
    out(f"  built in                {build_s:.1f} s")
    out("  initial logp            {:,.1f}  ({})".format(
        logp, "finite" if finite else "NOT FINITE"))
    if not finite:
        raise SystemExit("initial logp is not finite; sampling would fail")

    out("")
    out("  Sampling now. This is the long part.")
    start = time.perf_counter()
    idata = sample(
        dalec,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        seed=seed,
        cores=cores,
    )
    wall_s = time.perf_counter() - start

    # -- trace out first, before any analysis can fail -----------------------
    trace_path = out_dir / f"calibration_{args.convention}.nc"
    idata.attrs["seed"] = seed
    idata.attrs["convention"] = args.convention
    idata.attrs["calibration_years"] = list(calibration)
    idata.attrs["n_days"] = int(block.n_days)
    idata.attrs["n_assimilated"] = int(block.n_assimilated)
    idata.to_netcdf(str(trace_path))
    out(f"  trace written to        {trace_path}")

    stats = idata.sample_stats
    total_draws = args.chains * args.draws
    divergences = int(stats["diverging"].sum())
    max_depth_hits = (
        int(stats["reached_max_treedepth"].sum())
        if "reached_max_treedepth" in stats
        else 0
    )
    tree_depth = stats["tree_depth"].values if "tree_depth" in stats else None

    out("")
    out(bar)
    out("  Sampling")
    out(bar)
    out(f"  wall clock              {wall_s:.1f} s  ({wall_s / 3600:.2f} h)")
    per_draw_ms = 1000 * wall_s / (args.chains * (args.tune + args.draws))
    out(f"  per draw                {per_draw_ms:.0f} ms")
    out(f"  divergences             {divergences} / {total_draws}  "
        f"({100 * divergences / total_draws:.2f}%)")
    out(f"  max treedepth hits      {max_depth_hits} / {total_draws}")
    if tree_depth is not None:
        out(f"  tree depth              median {np.median(tree_depth):.0f}   "
            f"max {int(tree_depth.max())}")
    for chain in range(args.chains):
        out("    chain {}: step size {:.3e}, divergences {}".format(
            chain,
            float(stats["step_size"].values[chain, -1]),
            int(stats["diverging"].values[chain].sum()),
        ))

    summary = az.summary(idata, round_to=None)
    out("")
    out(bar)
    out("  Convergence, every variable")
    out(bar)
    out("  {:<24} {:>12} {:>8} {:>10} {:>10}".format(
        "variable", "mean", "r_hat", "ess_bulk", "ess_tail"))
    for name, row in summary.iterrows():
        out("  {!s:<24} {:>12.4g} {:>8.3f} {:>10.0f} {:>10.0f}".format(
            name, row["mean"], row["r_hat"], row["ess_bulk"], row["ess_tail"]))
    worst_rhat = float(summary["r_hat"].max())
    lowest_ess = float(summary["ess_bulk"].min())
    out("")
    out(f"  worst r_hat {worst_rhat:.3f}   lowest ess_bulk {lowest_ess:.0f}")
    if worst_rhat > 1.01:
        out("  ** r_hat above 1.01: the chains have not mixed. Treat every number")
        out("     below as provisional.")

    out("")
    out(bar)
    out("  Parameters at prior bounds")
    out(bar)
    at_bounds = report_bounds(out, idata, dalec.priors.bounds)

    out("")
    out(bar)
    out("  Posterior fluxes against measurement")
    out(bar)
    out(f"  convention {args.convention} -- a GPP number is not interpretable without it")
    out("")
    fluxes = report_fluxes(out, idata, block)

    out("")
    out(bar)
    out("  RQ3: the pre-sampling prediction")
    out(bar)
    rq3 = report_rq3(out, fluxes)

    summary_path = out_dir / f"calibration_{args.convention}_summary.csv"
    summary.to_csv(summary_path)
    meta = {
        "convention": args.convention,
        "block": list(calibration),
        "n_days": int(block.n_days),
        "n_assimilated": int(block.n_assimilated),
        "seed": seed,
        "chains": args.chains,
        "tune": args.tune,
        "draws": args.draws,
        "cores": cores,
        "nuts_settings": "pymc defaults",
        "wall_clock_s": wall_s,
        "divergences": divergences,
        "max_treedepth_hits": max_depth_hits,
        "tree_depth_median": float(np.median(tree_depth)) if tree_depth is not None
        else None,
        "tree_depth_max": int(tree_depth.max()) if tree_depth is not None else None,
        "worst_r_hat": worst_rhat,
        "lowest_ess_bulk": lowest_ess,
        "environment": environment,
        "timing": timing,
        "at_bounds": at_bounds,
        "fluxes": fluxes,
        "rq3": rq3,
    }
    (out_dir / f"calibration_{args.convention}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    out("")
    out(bar)
    out("  Outputs")
    out(bar)
    out(f"  trace     {trace_path}")
    out(f"  summary   {summary_path}")
    out("  meta      {}".format(out_dir / f"calibration_{args.convention}_meta.json"))

    log_path = out_dir / f"calibration_{args.convention}.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  log       {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
