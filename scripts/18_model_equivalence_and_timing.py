#!/usr/bin/env python
"""Graph against numpy: max absolute difference, and the cost of one gradient.

Reports the equivalence residual on the real 1997-2010 driver record, then times
one forward pass and one gradient at 5113 steps. The timing is the number that
decides whether NUTS is affordable here.

Usage
-----
    python scripts/18_model_equivalence_and_timing.py
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.compute import compile_function, resolved_linker_class  # noqa: E402
from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    require_year_block,
    resolve_path,
)
from dalec.data_io import SiteData  # noqa: E402
from dalec.model import build_forward_graph  # noqa: E402
from dalec.model_numpy import dalec2_phenology, run_dalec2  # noqa: E402
from dalec.parameters import PARAMETER_NAMES, DalecParameters, prior_bounds  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
N_WARMUP = 3
N_TIMED = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def reference_parameters(seed: int = 5) -> DalecParameters:
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


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import pytensor
    import pytensor.tensor as pt

    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = require_year_block(config, "calibration")
    site_code = str(config.get("site", {}).get("code", "")).lower().replace("-", "_")
    block = SiteData.load(
        resolve_path(config["paths"]["processed_dir"])
        / f"{site_code}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )
    acm = acm_from_config(config)
    parameters = reference_parameters()
    drivers = {
        "doy": block.doy.astype(float), "t_air": block.t_air,
        "t_max": block.t_max, "t_min": block.t_min,
        "sw_in": block.sw_in, "co2": block.co2,
    }
    bar = "=" * 74

    print(bar)
    print("  Graph against numpy, on the real driver record")
    print(bar)
    print(f"  block {calibration[0]}-{calibration[1]}, {block.n_days} steps")
    print(f"  {platform.machine()}, python {platform.python_version()}, "
          f"pytensor linker {pytensor.config.linker} -> {resolved_linker_class()}")

    theta = pt.dvector("theta")
    named = dict(zip(PARAMETER_NAMES, [theta[i] for i in range(len(PARAMETER_NAMES))],
                     strict=True))
    graph = build_forward_graph(
        parameters=named,
        latitude_deg=float(config["site"]["latitude_deg"]),
        coefficients=acm.coefficients,
        frost_threshold_degc=acm.frost_threshold_degc,
        **drivers,
    )
    theta0 = np.array([parameters.to_dict()[name] for name in PARAMETER_NAMES])

    # compile_function pins the linker. Timing on the default 'auto' linker
    # measures whichever backend the machine happened to resolve to, which is
    # the exact failure dalec.compute exists to prevent.
    forward = compile_function([theta], graph.nee, on_unused_input="ignore")
    modelled = forward(theta0)
    reference = run_dalec2(
        parameters, block, gpp_fn=acm, phenology_fn=dalec2_phenology
    )

    difference = np.abs(modelled - reference.nee)
    print("\n  NEE, graph minus numpy:")
    print(f"    max absolute difference   {difference.max():.6e} g C m-2 d-1")
    print(f"    mean absolute difference  {difference.mean():.6e}")
    print(f"    max relative difference   "
          f"{difference.max() / np.abs(reference.nee).max():.6e}")
    print(f"    NEE scale (max |numpy|)   {np.abs(reference.nee).max():.4f}")
    print(f"    double-precision epsilon  {np.finfo(float).eps:.3e}")

    # -- timing ---------------------------------------------------------------
    observed = np.where(block.nee_mask, block.nee_obs, 0.0)
    sigma = np.where(block.nee_mask, block.nee_unc, 1.0)
    weight = block.nee_mask.astype(float)
    loss = pt.sum(weight * pt.sqr((graph.nee - observed) / sigma))
    gradient = compile_function(
        [theta], pt.grad(loss, theta), on_unused_input="ignore"
    )

    print("\n" + bar)
    print(f"  Timing at {block.n_days} steps, {N_TIMED} runs after {N_WARMUP} warmups")
    print(bar)
    results = {}
    for label, function in (("forward pass", forward), ("gradient", gradient)):
        for _ in range(N_WARMUP):
            function(theta0)
        times = []
        for _ in range(N_TIMED):
            start = time.perf_counter()
            function(theta0)
            times.append(time.perf_counter() - start)
        times = np.array(times)
        results[label] = times
        print(f"  {label:<14} median {1000 * np.median(times):8.2f} ms   "
              f"min {1000 * times.min():8.2f}   max {1000 * times.max():8.2f}")

    ratio = np.median(results["gradient"]) / np.median(results["forward pass"])
    print(f"\n  gradient / forward = {ratio:.2f}x")
    per_gradient = np.median(results["gradient"])
    print("\n  Projection, 4 chains x (1000 tune + 1000 draws), one gradient per")
    print("  leapfrog step and a nominal 2^5 = 32 steps per iteration:")
    total = 4 * 2000 * 32 * per_gradient
    print(f"    {total / 3600:.1f} hours of gradient evaluation")
    print("    Treat as an order of magnitude: the real step count adapts, and")
    print("    this machine's timings have been unstable across runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
