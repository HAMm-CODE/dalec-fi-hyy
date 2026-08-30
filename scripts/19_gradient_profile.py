#!/usr/bin/env python
"""Why did the gradient cost 44x a forward pass, and does hoisting fix it?

Reverse-mode AD normally costs 2-5x a forward pass. 44x is an order of magnitude
off, and it is a graph problem rather than a hardware one.

The hypothesis: ``psi`` solves A9 from ``c_lf`` alone, so it is time-invariant,
but it was being computed **inside** the scan. That means 24 unrolled Newton
steps evaluated forward and taped for reverse at every one of 5113 timesteps.

Three variants are built and timed, so the attribution is measured rather than
argued:

``newton_inside``
    The original. Newton unrolled per timestep.
``constant_inside``
    ``psi`` frozen at its start value, still inside the scan. **Numerically
    wrong** -- it isolates the cost of the Newton solve from everything else.
``hoisted``
    The fix. ``psi`` and the phenology amplitudes solved once outside the scan.

Correctness is checked, not assumed: ``hoisted`` must reproduce
``newton_inside`` to machine precision, and ``constant_inside`` must not.

Usage
-----
    python scripts/19_gradient_profile.py
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
from dalec.parameters import PARAMETER_NAMES, DalecParameters, prior_bounds  # noqa: E402

REPORT_DIR = Path("reports/prior_diagnostics")
N_WARMUP = 3
N_TIMED = 15
MODES = ("newton_inside", "constant_inside", "hoisted")

#: Chains x (tune + draws) x nominal leapfrog steps, for the projection.
NUTS_GRADIENTS = 4 * 2000 * 32


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


def timed(function, theta0):
    for _ in range(N_WARMUP):
        function(theta0)
    times = []
    for _ in range(N_TIMED):
        start = time.perf_counter()
        function(theta0)
        times.append(time.perf_counter() - start)
    return np.array(times)


def main() -> int:
    for category in (RuntimeWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=category)
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
    theta0 = np.array([parameters.to_dict()[name] for name in PARAMETER_NAMES])
    drivers = {
        "doy": block.doy.astype(float), "t_air": block.t_air,
        "t_max": block.t_max, "t_min": block.t_min,
        "sw_in": block.sw_in, "co2": block.co2,
    }
    observed = np.where(block.nee_mask, block.nee_obs, 0.0)
    sigma = np.where(block.nee_mask, block.nee_unc, 1.0)
    weight = block.nee_mask.astype(float)

    bar = "=" * 74
    print(bar)
    print("  Gradient profile: where the 44x came from")
    print(bar)
    print(f"  block {calibration[0]}-{calibration[1]}, {block.n_days} steps, "
          f"{platform.machine()}, {resolved_linker_class()}")
    print(f"  {N_TIMED} runs after {N_WARMUP} warmups")

    results = {}
    values = {}
    for mode in MODES:
        theta = pt.dvector("theta")
        named = dict(
            zip(PARAMETER_NAMES, [theta[i] for i in range(len(PARAMETER_NAMES))],
                strict=True)
        )
        graph = build_forward_graph(
            parameters=named,
            latitude_deg=float(config["site"]["latitude_deg"]),
            coefficients=acm.coefficients,
            frost_threshold_degc=acm.frost_threshold_degc,
            psi_mode=mode,
            **drivers,
        )
        loss = pt.sum(weight * pt.sqr((graph.nee - observed) / sigma))
        compile_start = time.perf_counter()
        forward = compile_function([theta], graph.nee, on_unused_input="ignore")
        gradient = compile_function(
            [theta], pt.grad(loss, theta), on_unused_input="ignore"
        )
        compile_s = time.perf_counter() - compile_start

        values[mode] = (forward(theta0), gradient(theta0))
        forward_times = timed(forward, theta0)
        gradient_times = timed(gradient, theta0)
        ratio = np.median(gradient_times) / np.median(forward_times)
        results[mode] = (forward_times, gradient_times, ratio, compile_s)

        print(f"\n  {mode}")
        print(f"    compile        {compile_s:8.1f} s")
        print(f"    forward pass   {1000 * np.median(forward_times):8.2f} ms   "
              f"({1000 * forward_times.min():.2f} - {1000 * forward_times.max():.2f})")
        print(f"    gradient       {1000 * np.median(gradient_times):8.2f} ms   "
              f"({1000 * gradient_times.min():.2f} - {1000 * gradient_times.max():.2f})")
        print(f"    ratio          {ratio:8.2f}x")

    # -- correctness, checked not assumed ------------------------------------
    print("\n" + bar)
    print("  Correctness of the fix")
    print(bar)
    base_nee, base_grad = values["newton_inside"]
    hoisted_nee, hoisted_grad = values["hoisted"]
    nee_difference = np.max(np.abs(hoisted_nee - base_nee))
    grad_difference = np.max(np.abs(hoisted_grad - base_grad))
    grad_scale = np.max(np.abs(base_grad))
    print(f"  hoisted vs newton_inside, NEE       max abs diff {nee_difference:.3e}")
    print(f"  hoisted vs newton_inside, gradient  max abs diff {grad_difference:.3e}")
    print(f"                                      relative     "
          f"{grad_difference / grad_scale:.3e}")
    print(f"  gradient scale (max |newton_inside|)             {grad_scale:.4e}")

    frozen_nee, _frozen_grad = values["constant_inside"]
    frozen_difference = np.max(np.abs(frozen_nee - base_nee))
    print(f"\n  constant_inside vs newton_inside, NEE  max abs diff "
          f"{frozen_difference:.3e}")
    print("  (this one SHOULD differ -- it freezes psi at its start value and is")
    print("   a timing probe, not a model)")

    # -- projection ----------------------------------------------------------
    print("\n" + bar)
    print("  Revised projection")
    print(bar)
    print("  4 chains x (1000 tune + 1000 draws) x nominal 32 leapfrog steps")
    print(f"  = {NUTS_GRADIENTS:,} gradient evaluations")
    print("  mode              gradient    ratio    one run    two conventions")
    for mode in MODES:
        _f, gradient_times, ratio, _c = results[mode]
        hours = NUTS_GRADIENTS * float(np.median(gradient_times)) / 3600.0
        print(f"  {mode:<16} {1000 * np.median(gradient_times):8.1f} ms  "
              f"{ratio:6.2f}x  {hours:8.1f} h        {2 * hours:8.1f} h")

    baseline = float(np.median(results["newton_inside"][1]))
    fixed = float(np.median(results["hoisted"][1]))
    print(f"\n  speed-up from hoisting: {baseline / fixed:.2f}x")
    print("  Order of magnitude only: the leapfrog count adapts, and this")
    print("  machine's timings vary 20-30% between runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
