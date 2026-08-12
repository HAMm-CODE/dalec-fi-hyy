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
from dalec.data_io import (  # noqa: E402
    TEMPERATURE_SOURCES,
    build_site_data,
    load_daily_extremes,
    load_fluxnet_dd,
)


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
    parser.add_argument(
        "--extremes-file",
        type=Path,
        default=None,
        help="daily min/max csv from scripts/01b_derive_tminmax.py",
    )
    parser.add_argument(
        "--temperature-source",
        choices=list(TEMPERATURE_SOURCES),
        default="extremes",
        help=(
            "where the daily maximum and minimum come from. The day/night proxy "
            "understates the daily range by about 4.8 degC at this site and is "
            "the comparison baseline only"
        ),
    )
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

    extremes = None
    if args.temperature_source == "extremes":
        slug = site_code.lower().replace("-", "_")
        extremes_file = (
            args.extremes_file
            if args.extremes_file is not None
            else processed_dir / f"{slug}_tminmax.csv"
        )
        if not extremes_file.exists():
            raise SystemExit(
                f"no derived daily extremes at {extremes_file}. Run "
                "scripts/01b_derive_tminmax.py first, or pass "
                "--temperature-source day_night_proxy to fall back to "
                "TA_F_DAY/TA_F_NIGHT -- which understates the daily temperature "
                "range by about 4.8 degC at this site."
            )
        extremes = load_daily_extremes(extremes_file)
        print(f"daily extremes: {extremes_file.name} ({len(extremes)} days)")
    else:
        print("daily extremes: NOT USED -- falling back to the TA_F_DAY/TA_F_NIGHT proxy")

    blocks = ["calibration", "evaluation"] if args.block == "both" else [args.block]
    for block in blocks:
        start_year, end_year = require_year_block(config, block)
        site_data = build_site_data(
            frame,
            start_year=start_year,
            end_year=end_year,
            qc_threshold=qc_threshold,
            site_code=site_code,
            daily_extremes=extremes,
            temperature_source=args.temperature_source,
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
