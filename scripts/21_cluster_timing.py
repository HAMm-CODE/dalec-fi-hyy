#!/usr/bin/env python
"""Re-time the FIXED graph at the full calibration block, on both backends.

Supersedes the projection in ``reports/prior_diagnostics/gradient_profile.txt``,
which was measured on the laptop. Two things make a re-measurement necessary
rather than a rescaling:

**The laptop numbers were measured under emulation.** The interpreter there is an
x86-64 (AMD64) build running on ARM64 Qualcomm hardware. ``platform.machine()``
reports the *hardware* and so prints ``ARM64``, which is why every earlier report
header reads "ARM64, NumbaLinker" and reads as a native run. It was not one. This
script prints the interpreter architecture and the hardware architecture
separately and says plainly when they disagree, so the same mistake cannot be
made twice.

**``timing_spike.py`` is not the model.** Numbers taken from it are the *unfixed*
graph: it still closes over its parameters instead of passing them as
``non_sequences``, which is the exact pattern DECISIONS §12 measured at 12.2x.
This script times :func:`dalec.model.build_forward_graph`, the graph the sampler
actually runs.

Both backends, because the pin now means something different
------------------------------------------------------------
``dalec.compute`` pins Numba, and its docstring justifies that partly on there
being no C++ compiler available to pin instead. That is true on the laptop
(``config.cxx`` is empty) and **false on Roihu**, where ``g++`` resolves to the
Tykky container's own compiler and ``config.cxx`` is populated. The pin is
therefore now a choice between two available backends rather than a statement
about what exists, and a choice should be made on a measurement.

**This script does not change the pin.** It reports what each backend resolves to
and what each costs. Note that PyTensor accepts a linker it cannot provide and
silently substitutes a slower one, so the resolved class is read back and a
backend that did not resolve as asked is reported and skipped rather than timed
-- timing a substituted ``VMLinker`` would produce a number roughly 130x off and
present it as a comparison.

Usage
-----
    python scripts/21_cluster_timing.py
    python scripts/21_cluster_timing.py --backends numba
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
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
from dalec.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from dalec.data_io import SiteData  # noqa: E402
from dalec.model import build_forward_graph  # noqa: E402
from dalec.parameters import (  # noqa: E402
    PARAMETER_NAMES,
    DalecParameters,
    prior_bounds,
)

REPORT_DIR = Path("reports/sampling")

N_WARMUP = 3
N_TIMED = 15

#: Linker name -> the linker classes that count as "resolved as asked".
BACKENDS: dict[str, tuple[str, ...]] = {
    "numba": ("NumbaLinker",),
    "cvm": ("CVMLinker", "CLinker"),
}

#: The sampling design the projection is for. Not configurable here on purpose:
#: it is the run being planned, and it comes from config/default.yaml.
CHAINS = 4
TUNE = 1000
DRAWS = 1000

#: Tree depths to project at. The plumbing run measured median 5, max 6 over two
#: years; a fourteen-year record is more informative and should go deeper, so 6
#: is the optimistic case and 8 the pessimistic one.
TREE_DEPTHS = (6, 7, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKENDS),
        choices=list(BACKENDS),
        help="Backends to time. Default: both.",
    )
    return parser.parse_args()


def reference_parameters(seed: int = 5) -> DalecParameters:
    """The same reference point script 19 used, so the numbers are comparable."""
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


def timed(function, theta0) -> np.ndarray:
    for _ in range(N_WARMUP):
        function(theta0)
    times = []
    for _ in range(N_TIMED):
        start = time.perf_counter()
        function(theta0)
        times.append(time.perf_counter() - start)
    return np.array(times)


def interpreter_architecture() -> str:
    """What the *interpreter* was built for, as opposed to what it runs on."""
    return sysconfig.get_platform()


def gxx_report() -> dict[str, object]:
    """Which g++ is on PATH, what it is, and what PyTensor made of it.

    Recorded because the answer differs between environments in a way that
    changes which backends exist: on Roihu the first ``g++`` on PATH is the
    Tykky container's own compiler rather than the system GCC, and that is what
    populates ``config.cxx``.
    """
    import pytensor

    found = shutil.which("g++")
    version = None
    if found:
        try:
            completed = subprocess.run(
                [found, "--version"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            version = completed.stdout.splitlines()[0] if completed.stdout else None
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
            version = None
    return {
        "which_gxx": found,
        "gxx_version": version,
        "pytensor_config_cxx": pytensor.config.cxx,
    }


def print_environment(out) -> dict[str, object]:
    """Provenance block. Prints, and returns the same facts for the meta file."""
    import pytensor

    interp = interpreter_architecture()
    machine = platform.machine()
    emulated = ("amd64" in interp.lower() or "x86_64" in interp.lower()) and (
        "arm" in machine.lower() or "aarch64" in machine.lower()
    )
    compiler = gxx_report()

    out(f"  hostname            {platform.node()}")
    out(f"  hardware            {machine}  ({platform.processor()})")
    out(f"  interpreter built   {interp}   (python {platform.python_version()})")
    if emulated:
        out("  ** EMULATED: an x86-64 interpreter on ARM hardware. Timings from")
        out("     this machine are inflated and are not a native measurement.")
        out("     platform.machine() reports the HARDWARE, so a report header")
        out("     reading 'ARM64' here does NOT mean a native ARM64 run.")
    else:
        out("  emulation           none detected (interpreter matches hardware)")
    out(f"  which g++           {compiler['which_gxx'] or '<not found>'}")
    out(f"  g++ --version       {compiler['gxx_version'] or '<none>'}")
    out(f"  pytensor.config.cxx {compiler['pytensor_config_cxx']!r}")
    out(
        "  C backend           "
        + ("available" if compiler["pytensor_config_cxx"] else "NOT available")
    )
    out(f"  pytensor            {pytensor.__version__}")
    return {
        "hostname": platform.node(),
        "hardware": machine,
        "processor": platform.processor(),
        "interpreter_platform": interp,
        "python": platform.python_version(),
        "emulated": bool(emulated),
        "pytensor": pytensor.__version__,
        **compiler,
    }


def build_timed_functions(config, block, acm, backend: str):
    """Compile forward and gradient on ``backend``; return them and what resolved.

    Sets ``config.linker`` directly rather than going through
    ``dalec.compute.configure_pytensor``, which asserts the pinned Numba backend
    and would refuse to let the alternative be measured at all. The resolved
    class is read back here instead, so the substitution failure mode is still
    caught -- it is the reason this is not simply trusted.
    """
    import pytensor
    import pytensor.tensor as pt

    pytensor.config.linker = backend
    resolved = type(pytensor.compile.mode.get_default_mode().linker).__name__
    if resolved not in BACKENDS[backend]:
        return None, resolved

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

    start = time.perf_counter()
    forward = pytensor.function([theta], graph.nee, on_unused_input="ignore")
    gradient = pytensor.function(
        [theta], pt.grad(loss, theta), on_unused_input="ignore"
    )
    compile_s = time.perf_counter() - start
    return (forward, gradient, compile_s), resolved


def main() -> int:
    for category in (RuntimeWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=category)

    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    calibration = config["years"]["calibration"]
    site_code = str(config.get("site", {}).get("code", "")).lower().replace("-", "_")
    block = SiteData.load(
        resolve_path(config["paths"]["processed_dir"])
        / f"{site_code}_calibration_{calibration[0]}_{calibration[1]}.nc"
    )
    acm = acm_from_config(config)
    parameters = reference_parameters()
    theta0 = np.array([parameters.to_dict()[name] for name in PARAMETER_NAMES])

    bar = "=" * 74
    out(bar)
    out("  Cluster timing: the FIXED graph, both backends")
    out(bar)
    out(f"  block {calibration[0]}-{calibration[1]}, {block.n_days} steps")
    out(f"  {N_TIMED} runs after {N_WARMUP} warmups")
    out("")
    environment = print_environment(out)

    results: dict[str, dict[str, float]] = {}
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: dict[str, str] = {}

    for backend in args.backends:
        out("")
        out(bar)
        out(f"  backend: {backend}")
        out(bar)
        built, resolved = build_timed_functions(config, block, acm, backend)
        if built is None:
            out(f"  requested {backend!r} -> resolved {resolved!r}")
            out("  NOT the requested backend. Skipped rather than timed: PyTensor")
            out("  substitutes a slower linker silently, and timing the")
            out("  substitute would report a number ~130x off as a comparison.")
            skipped[backend] = resolved
            continue

        forward, gradient, compile_s = built
        values[backend] = (forward(theta0), gradient(theta0))
        forward_times = timed(forward, theta0)
        gradient_times = timed(gradient, theta0)
        f_med = float(np.median(forward_times))
        g_med = float(np.median(gradient_times))
        ratio = g_med / f_med

        out(f"  resolved linker    {resolved}")
        out(f"  compile            {compile_s:8.1f} s")
        out(f"  forward pass       {1000 * f_med:8.2f} ms   "
            f"({1000 * forward_times.min():.2f} - {1000 * forward_times.max():.2f})")
        out(f"  gradient           {1000 * g_med:8.2f} ms   "
            f"({1000 * gradient_times.min():.2f} - "
            f"{1000 * gradient_times.max():.2f})")
        out(f"  ratio              {ratio:8.2f}x")
        results[backend] = {
            "resolved_linker": resolved,
            "compile_s": compile_s,
            "forward_ms": 1000 * f_med,
            "gradient_ms": 1000 * g_med,
            "ratio": ratio,
        }

    # -- the backends must agree, or a speed comparison is meaningless --------
    if len(values) > 1:
        out("")
        out(bar)
        out("  Do the backends agree?")
        out(bar)
        names = list(values)
        base = names[0]
        for other in names[1:]:
            nee_diff = float(np.max(np.abs(values[other][0] - values[base][0])))
            grad_diff = float(np.max(np.abs(values[other][1] - values[base][1])))
            scale = float(np.max(np.abs(values[base][1])))
            out(f"  {other} vs {base}, NEE       max abs diff {nee_diff:.3e}")
            out(f"  {other} vs {base}, gradient  max abs diff {grad_diff:.3e}")
            out(f"                                relative     "
                f"{grad_diff / scale if scale else float('nan'):.3e}")
            if nee_diff > 1e-10 or (scale and grad_diff / scale > 1e-10):
                out("  ** The backends DISAGREE. A speed comparison between them")
                out("     is not a like-for-like comparison. Investigate before")
                out("     using either number.")

    # -- projection ----------------------------------------------------------
    out("")
    out(bar)
    out("  Wall-clock projection for the calibration run")
    out(bar)
    out(f"  {CHAINS} chains x ({TUNE} tune + {DRAWS} draws) = "
        f"{TUNE + DRAWS} iterations per chain")
    out("  One gradient per leapfrog step; treedepth d gives 2^d steps per")
    out("  iteration. Wall clock assumes the chains run concurrently on at")
    out(f"  least {CHAINS} physical cores -- with fewer, multiply accordingly.")
    out("")
    iterations = TUNE + DRAWS
    projection: dict[str, dict[str, float]] = {}
    for backend, measured in results.items():
        grad_s = measured["gradient_ms"] / 1000.0
        out(f"  {backend} ({measured['gradient_ms']:.1f} ms per gradient)")
        out("    treedepth   leapfrogs/iter   wall clock   total gradient-hours")
        for depth in TREE_DEPTHS:
            leapfrogs = 2 ** depth
            wall_h = iterations * leapfrogs * grad_s / 3600.0
            total_h = wall_h * CHAINS
            out(f"    {depth:>9}   {leapfrogs:>14}   {wall_h:>8.2f} h   "
                f"{total_h:>18.2f} h")
            projection[f"{backend}_treedepth_{depth}"] = {
                "leapfrogs_per_iteration": leapfrogs,
                "wall_clock_h": wall_h,
                "total_gradient_h": total_h,
            }
        out("")
    out("  Both LAI conventions doubles the wall clock unless they are run")
    out("  concurrently, which needs twice the cores.")

    if skipped:
        out("")
        out(bar)
        out("  Backends not timed")
        out(bar)
        for backend, resolved in skipped.items():
            out(f"  {backend}: resolved to {resolved}, not a {backend} linker.")

    out("")
    out("  NOTE: this supersedes the projection in")
    out("  reports/prior_diagnostics/gradient_profile.txt, which was measured")
    out("  on an emulated x86-64 interpreter on ARM hardware.")

    report = out_dir / "cluster_timing.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "cluster_timing_meta.json").write_text(
        json.dumps(
            {
                "block": list(calibration),
                "n_days": int(block.n_days),
                "environment": environment,
                "backends": results,
                "skipped": skipped,
                "design": {
                    "chains": CHAINS, "tune": TUNE, "draws": DRAWS,
                },
                "projection": projection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
