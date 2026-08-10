#!/usr/bin/env python
"""Per-year data coverage table for the FI-Hyy FLUXNET2015 daily file.

This is the table that decides which years go into the calibration block and
which are held out for evaluation, so every screening criterion is reported
separately rather than only the intersection.

Usage
-----
    python scripts/00_qc_coverage.py
    python scripts/00_qc_coverage.py --qc-threshold 0.5
    python scripts/00_qc_coverage.py --csv results/qc_coverage.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from dalec.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from dalec.data_io import DRIVER_COLUMNS, coverage_table, load_fluxnet_dd  # noqa: E402

COLUMN_LABELS = {
    "days": "rows",
    "days_in_year": "cal.days",
    "nee_present": "NEE ok",
    "nee_qc_pass": "QC pass",
    "nee_qc_mean": "mean QC",
    "randunc_present": "RANDUNC",
    "assimilable": "usable",
    "assimilable_pct": "usable %",
    "driver_gaps": "drv.gaps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to the YAML config"
    )
    parser.add_argument(
        "--fluxnet-file", type=Path, default=None, help="override paths.fluxnet_file"
    )
    parser.add_argument(
        "--qc-threshold", type=float, default=None, help="override data.qc_threshold"
    )
    parser.add_argument("--csv", type=Path, default=None, help="also write the table as csv")
    parser.add_argument(
        "--min-usable-pct",
        type=float,
        default=70.0,
        help="threshold used only for the contiguous-block hint at the end",
    )
    return parser.parse_args()


def contiguous_blocks(years: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted list of years into inclusive contiguous ranges."""
    blocks: list[tuple[int, int]] = []
    for year in years:
        if blocks and year == blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], year)
        else:
            blocks.append((year, year))
    return blocks


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

    frame = load_fluxnet_dd(fluxnet_file)
    table = coverage_table(frame, qc_threshold=qc_threshold)

    site = config.get("site", {})
    print()
    print("=" * 92)
    print(f"  FLUXNET2015 daily coverage -- {site.get('code', '?')} ({site.get('name', '?')})")
    print("=" * 92)
    print(f"  file            {fluxnet_file}")
    print(f"  record          {frame.index[0].date()} to {frame.index[-1].date()} "
          f"({len(frame)} days)")
    print("  target          NEE_VUT_REF, g C m-2 d-1")
    print(f"  QC threshold    NEE_VUT_REF_QC >= {qc_threshold:g}")
    print()
    print("  A day is 'usable' when NEE is present, QC meets the threshold, and")
    print("  RANDUNC is present and positive (it is the likelihood sd).")
    print("  'drv.gaps' counts days missing any of: "
          + ", ".join(DRIVER_COLUMNS.values()) + ".")
    print("  A year with any driver gap cannot be used at all -- the forward model")
    print("  cannot integrate through a missing driver.")
    print()

    display = table.rename(columns=COLUMN_LABELS)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(
            display.to_string(
                formatters={
                    "mean QC": lambda v: "     -" if pd.isna(v) else f"{v:6.3f}",
                    "usable %": lambda v: f"{v:7.1f}",
                }
            )
        )

    total_usable = int(table["assimilable"].sum())
    total_days = int(table["days"].sum())
    print("-" * 92)
    print(
        f"  total   rows={total_days}   usable={total_usable} "
        f"({100.0 * total_usable / max(total_days, 1):.1f}% of record)"
    )
    print()

    # --- Guidance only. The calibration/evaluation split is your call. --------
    good_years = [
        int(year)
        for year, row in table.iterrows()
        if row["driver_gaps"] == 0
        and row["days"] == row["days_in_year"]
        and row["assimilable_pct"] >= args.min_usable_pct
    ]
    blocks = contiguous_blocks(good_years)
    print(
        f"  Contiguous whole years with no driver gaps and >= {args.min_usable_pct:g}% usable days:"
    )
    if blocks:
        for start, end in blocks:
            print(f"    {start}-{end}  ({end - start + 1} year(s))")
    else:
        print("    none -- lower --min-usable-pct or --qc-threshold and re-run")
    print()
    print("  Set years.calibration and years.evaluation in config/default.yaml to two")
    print("  contiguous, non-overlapping blocks (calibration first, evaluation held out).")
    print()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv)
        print(f"  wrote {args.csv}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
