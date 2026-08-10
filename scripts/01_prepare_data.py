#!/usr/bin/env python
"""Raw FLUXNET csv -> processed, model-ready driver and observation arrays.

Writes one NetCDF file per block (calibration and evaluation) into
``data/processed``. No rows are dropped: QC screening produces a boolean
likelihood mask, not a filtered time series.

Requires ``years.calibration`` (and, for the evaluation block,
``years.evaluation``) to be set in the config. Fill them in from the output of
``scripts/00_qc_coverage.py``.

Usage
-----
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --block calibration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    require_year_block,
    resolve_path,
)
from dalec.data_io import build_site_data, load_fluxnet_dd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--block",
        choices=["calibration", "evaluation", "both"],
        default="both",
        help="which year block(s) to prepare",
    )
    parser.add_argument("--fluxnet-file", type=Path, default=None)
    parser.add_argument("--qc-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    fluxnet_file = (
        args.fluxnet_file
        if args.fluxnet_file is not None
        else resolve_path(config["paths"]["fluxnet_file"])
    )
    qc_threshold = (
        args.qc_threshold
        if args.qc_threshold is not None
        else float(config["data"]["qc_threshold"])
    )
    site_code = str(config.get("site", {}).get("code", ""))
    processed_dir = resolve_path(config["paths"]["processed_dir"])

    frame = load_fluxnet_dd(fluxnet_file)

    blocks = ["calibration", "evaluation"] if args.block == "both" else [args.block]
    for block in blocks:
        start_year, end_year = require_year_block(config, block)
        site_data = build_site_data(
            frame,
            start_year=start_year,
            end_year=end_year,
            qc_threshold=qc_threshold,
            site_code=site_code,
        )
        site_data.attrs["block"] = block
        slug = site_code.lower().replace("-", "_")
        out_path = processed_dir / f"{slug}_{block}_{start_year}_{end_year}.nc"
        site_data.save(out_path)
        print(
            f"{block:>12}  {start_year}-{end_year}  "
            f"{site_data.n_days} days, {site_data.n_assimilated} assimilated "
            f"({100.0 * site_data.n_assimilated / site_data.n_days:.1f}%)  -> {out_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
