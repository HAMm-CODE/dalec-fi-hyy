#!/usr/bin/env python
"""Build the DALEC2 model, sample it with NUTS, and save the posterior as NetCDF.

Usage, from the repository root:
    python showcase/run_calibration.py                          # hemisurface LAI
    python showcase/run_calibration.py --convention projected   # the sensitivity run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.config import load_config, resolve_path  # noqa: E402
from model import build_model, load_data, sample  # noqa: E402

CONFIG = Path(__file__).with_name("config.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--convention", choices=["hemisurface", "projected"],
                        help="LAI convention; default from config.yaml")
    args = parser.parse_args()

    config = load_config(CONFIG)
    convention = args.convention or config["lai_convention"]
    first, last = config["years"]["calibration"]

    data = load_data(config, first, last)
    print(f"calibration {first}-{last}: {data.n_days} days, "
          f"{data.n_assimilated} in the likelihood, LAI convention {convention}")

    model, _ = build_model(data, acm_from_config(config), convention)
    idata = sample(model, config)

    idata.attrs.update(
        seed=config["seed"],
        convention=convention,
        calibration_years=[first, last],
        n_days=data.n_days,
        n_assimilated=data.n_assimilated,
    )
    out = resolve_path(config["paths"]["outputs_dir"]) / f"calibration_{convention}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(out))
    print(f"posterior written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
