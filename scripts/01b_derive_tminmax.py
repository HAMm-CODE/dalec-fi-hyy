#!/usr/bin/env python
"""Half-hourly FLUXNET csv -> true daily maximum and minimum air temperature.

Why this exists
---------------
ACM's ``Tr`` is the daily temperature *range*, ``T_max - T_min``, and a range
cannot be negative. The FULLSET **daily** product carries no ``TA_F_MAX`` or
``TA_F_MIN``, only daytime and nighttime *means*, and those invert -- the daytime
mean falling below the nighttime mean, as it does under a warm front overnight.
At FI-Hyy that happens on 14.8% of calibration days and drives the canopy
conductance denominator negative on 5.0% of them.

The half-hourly product does carry ``TA_F``, so the true daily extremes are a
groupby. Derived here once, ``Tr`` is non-negative by construction and the
interim floor in ``dalec.acm`` becomes unreachable.

Coverage is reported rather than assumed: a day assembled from a handful of
surviving half-hours has a max and a min, but they are not the day's extremes.
Days below ``--min-halfhours`` are flagged, not dropped -- the same policy the
daily QC screening follows.

Usage
-----
    python scripts/01b_derive_tminmax.py
    python scripts/01b_derive_tminmax.py --min-halfhours 36
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from a checkout without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from dalec.data_io import MISSING_VALUE  # noqa: E402

#: Half-hours in a complete day.
HALFHOURS_PER_DAY = 48

#: Below this many valid half-hours, the day's extremes are not trustworthy.
DEFAULT_MIN_HALFHOURS = 24

#: Only these are read. The half-hourly file is ~500 MB across 243 columns, and
#: reading three of them keeps this to seconds and a modest footprint.
NEEDED_COLUMNS = ["TIMESTAMP_START", "TA_F", "TA_F_QC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--hh-file",
        type=Path,
        default=None,
        help="half-hourly FULLSET csv; defaults to the DD path with _DD_ -> _HH_",
    )
    parser.add_argument(
        "--min-halfhours",
        type=int,
        default=DEFAULT_MIN_HALFHOURS,
        help=f"valid half-hours below which a day is flagged (default {DEFAULT_MIN_HALFHOURS})",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def default_hh_path(config: dict) -> Path:
    """The HH file sits beside the DD file, differing only in the product code."""
    daily = resolve_path(config["paths"]["fluxnet_file"])
    candidate = daily.with_name(daily.name.replace("_DD_", "_HH_"))
    if not candidate.exists():
        raise FileNotFoundError(
            f"expected the half-hourly file at {candidate}. Pass --hh-file if it "
            "is named differently; the FLUXNET2015 pattern is "
            "FLX_<SITE>_FLUXNET2015_FULLSET_HH_<YEARS>_<VERSION>.csv"
        )
    return candidate


def derive_daily_extremes(hh_file: Path, *, min_halfhours: int) -> pd.DataFrame:
    """Group half-hourly ``TA_F`` into true daily maximum, minimum and range."""
    frame = pd.read_csv(hh_file, usecols=NEEDED_COLUMNS)
    frame["TA_F"] = frame["TA_F"].replace(MISSING_VALUE, pd.NA)
    frame["TA_F"] = pd.to_numeric(frame["TA_F"], errors="coerce")

    # TIMESTAMP_START is YYYYMMDDHHMM; the date is its first eight digits.
    frame["date"] = pd.to_datetime(
        frame["TIMESTAMP_START"].astype("int64").astype(str).str[:8], format="%Y%m%d"
    )

    grouped = frame.groupby("date")["TA_F"]
    daily = pd.DataFrame(
        {
            "t_max": grouped.max(),
            "t_min": grouped.min(),
            "n_halfhours": grouped.count().astype(int),
        }
    )
    daily["t_range"] = daily["t_max"] - daily["t_min"]
    daily["reliable"] = daily["n_halfhours"] >= min_halfhours
    return daily


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    hh_file = args.hh_file if args.hh_file is not None else default_hh_path(config)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    site_code = str(config.get("site", {}).get("code", ""))
    slug = site_code.lower().replace("-", "_")
    out_path = args.out if args.out is not None else processed_dir / f"{slug}_tminmax.csv"

    print(f"reading {hh_file.name} ({hh_file.stat().st_size / 1e6:.0f} MB)...")
    daily = derive_daily_extremes(hh_file, min_halfhours=args.min_halfhours)

    negative = int((daily["t_range"] < 0).sum())
    unreliable = int((~daily["reliable"]).sum())
    print(f"  {len(daily)} days, {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"  daily range: min {daily['t_range'].min():.2f}, "
          f"mean {daily['t_range'].mean():.2f}, max {daily['t_range'].max():.2f} degC")
    print(f"  days with a negative range:            {negative}  (must be zero)")
    print(f"  days below {args.min_halfhours} valid half-hours:      {unreliable} "
          f"({100.0 * unreliable / len(daily):.1f}%)")
    print(f"  days with no valid half-hours at all:  {int((daily['n_halfhours'] == 0).sum())}")

    if negative:
        raise SystemExit(
            f"{negative} day(s) have a negative temperature range, which is "
            "impossible from a max and a min over the same day. Check the parse."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index_label="date")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
