"""FLUXNET2015 loading, QC screening and unit conversion for the FI-Hyy assimilation.

Unit conventions used throughout this project
---------------------------------------------
carbon pools        g C m-2
carbon fluxes       g C m-2 d-1
temperature         degrees Celsius
radiation           MJ m-2 d-1  (daily *total*, converted on load)
CO2                 umol mol-1  (ppm)

FLUXNET2015 FULLSET **daily** (DD) files already report NEE, GPP and RECO in
g C m-2 d-1 -- the umol m-2 s-1 convention applies to the half-hourly files
only -- so no flux unit conversion happens here. The single conversion applied
is incoming shortwave radiation; see ``SW_W_M2_TO_MJ_M2_DAY``.

Design points that are easy to get wrong, and are therefore enforced here
------------------------------------------------------------------------
1. **Rows are never dropped.** The forward model integrates carbon pools day by
   day and has to run continuously through bad-quality days. QC screening
   produces a boolean *likelihood mask*, not a filtered time series.
2. **Drivers must be gap-free.** Observations may be missing; drivers may not.
   A NaN driver propagates into every pool for the rest of the run. The loader
   raises instead of silently interpolating.
3. **RANDUNC gates the likelihood too.** The Gaussian likelihood takes its
   per-day standard deviation from ``NEE_VUT_REF_RANDUNC``. A day with a valid
   NEE but a missing or non-positive RANDUNC has no usable sd and is masked
   out, regardless of its QC flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "DRIVER_COLUMNS",
    "EXTREME_COLUMNS",
    "MISSING_VALUE",
    "OBSERVATION_COLUMNS",
    "PARTITIONED_COLUMNS",
    "SW_W_M2_TO_MJ_M2_DAY",
    "TEMPERATURE_SOURCES",
    "SiteData",
    "build_site_data",
    "coverage_table",
    "load_daily_extremes",
    "load_fluxnet_dd",
    "load_site_data",
    "sw_in_to_mj_per_day",
    "sw_in_to_w_per_m2",
]

# FLUXNET2015 missing-data sentinel, used in every numeric column.
MISSING_VALUE: Final[float] = -9999.0

# Incoming shortwave conversion.
#
#   SW_IN_F in the DD file is a *daily mean* flux density in W m-2 = J s-1 m-2.
#   The ACM photosynthesis routine wants a *daily total* in MJ m-2 d-1.
#
#   1 W m-2 sustained over one day = 1 J s-1 m-2 * 86400 s = 86400 J m-2
#                                  = 8.64e4 J m-2 = 0.0864 MJ m-2 d-1.
SW_W_M2_TO_MJ_M2_DAY: Final[float] = 0.0864

TIMESTAMP_COLUMN: Final[str] = "TIMESTAMP"

#: Driver columns. Day of year is derived from TIMESTAMP rather than read.
#:
#: TA_F_DAY and TA_F_NIGHT are the mean daytime and mean nighttime air
#: temperature. They were used as stand-ins for the daily maximum and minimum
#: that ACM expects; they are **no longer** the default source for those --
#: see :func:`load_daily_extremes` -- but they are still loaded, because the
#: comparison between proxy and truth is a recorded result.
#:
#: VPD_F is deliberately absent and must stay absent.
DRIVER_COLUMNS: Final[dict[str, str]] = {
    "t_air": "TA_F",
    "t_day": "TA_F_DAY",
    "t_night": "TA_F_NIGHT",
    "sw_in": "SW_IN_F",
    "co2": "CO2_F_MDS",
}

#: Where the daily temperature extremes come from.
#:
#: ``"extremes"``
#:     True daily maximum and minimum, derived from the half-hourly product by
#:     ``scripts/01b_derive_tminmax.py``. The default, and the only source that
#:     should reach a published result.
#: ``"day_night_proxy"``
#:     ``TA_F_DAY`` and ``TA_F_NIGHT``. Retained deliberately as the comparison
#:     baseline for ``dalec.diagnostics.temperature_proxy_comparison``, and
#:     reachable only by asking for it explicitly. Measured over 1997-2005 the
#:     proxy range correlates 0.641 with the truth and understates it by
#:     4.79 degC -- the daytime mean is 2.51 degC below the true maximum and the
#:     nighttime mean 2.28 degC above the true minimum, and those biases compound
#:     in the difference.
TEMPERATURE_SOURCES: Final[tuple[str, str]] = ("extremes", "day_night_proxy")

#: Columns expected in the derived daily-extremes file.
EXTREME_COLUMNS: Final[tuple[str, ...]] = ("t_max", "t_min")

#: Assimilation target and its metadata. NEE_VUT_REF only.
OBSERVATION_COLUMNS: Final[dict[str, str]] = {
    "nee_obs": "NEE_VUT_REF",
    "nee_qc": "NEE_VUT_REF_QC",
    "nee_unc": "NEE_VUT_REF_RANDUNC",
}

#: Night-time- and daytime-partitioned GPP and RECO products.
#:
#: Loaded for the Phase 8 posterior *consistency* plots only. These are model
#: products, not observations, and they never enter the likelihood.
PARTITIONED_COLUMNS: Final[dict[str, str]] = {
    "gpp_nt": "GPP_NT_VUT_REF",
    "gpp_dt": "GPP_DT_VUT_REF",
    "reco_nt": "RECO_NT_VUT_REF",
    "reco_dt": "RECO_DT_VUT_REF",
}


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

#: Constrained so that the conversions preserve their argument type: an array in
#: gives an array out, a scalar gives a scalar.
Numeric = TypeVar("Numeric", float, np.ndarray)


def sw_in_to_mj_per_day(sw_w_m2: Numeric) -> Numeric:
    """Convert daily-mean shortwave irradiance to a daily total.

    Parameters
    ----------
    sw_w_m2
        Daily mean incoming shortwave radiation, W m-2.

    Returns
    -------
    Daily total incoming shortwave radiation, MJ m-2 d-1.
    """
    return sw_w_m2 * SW_W_M2_TO_MJ_M2_DAY


def sw_in_to_w_per_m2(sw_mj_m2_day: Numeric) -> Numeric:
    """Inverse of :func:`sw_in_to_mj_per_day`.

    Parameters
    ----------
    sw_mj_m2_day
        Daily total incoming shortwave radiation, MJ m-2 d-1.

    Returns
    -------
    Daily mean incoming shortwave radiation, W m-2.
    """
    return sw_mj_m2_day / SW_W_M2_TO_MJ_M2_DAY


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteData:
    """Model-ready drivers and observations for a contiguous block of days.

    All arrays share the leading time dimension and are ordered by date with no
    gaps in the calendar. Driver arrays are guaranteed finite. Observation
    arrays may contain NaN; ``nee_mask`` says which days the likelihood uses.

    Attributes
    ----------
    time
        Calendar dates, ``datetime64[ns]``, strictly increasing, one per day.
    doy
        Day of year, 1-366, derived from ``time``.
    t_air
        Daily mean air temperature, degrees C (``TA_F``).
    t_day
        Daily mean daytime air temperature, degrees C (``TA_F_DAY``). Retained
        as the comparison baseline only -- photosynthesis reads ``t_max``.
    t_night
        Daily mean nighttime air temperature, degrees C (``TA_F_NIGHT``). As
        ``t_day``: baseline only.
    t_max, t_min
        Daily maximum and minimum air temperature, degrees C. What the ACM
        photosynthesis routine reads, for both its ``exp(a8 * Tmax)`` term and
        the daily range. True extremes by default, derived from the half-hourly
        product; ``t_max - t_min`` is then non-negative by construction.

        **Three distinct temperatures live in this record and confusing them is
        silent.** Decomposition and respiration read ``t_air``, the daily mean.
        Photosynthesis reads ``t_max`` and ``t_min``. ``t_day`` and ``t_night``
        reach neither.
    sw_in
        Daily total incoming shortwave radiation, MJ m-2 d-1 (``SW_IN_F``,
        converted from W m-2 on load).
    co2
        Atmospheric CO2 mole fraction, umol mol-1 (``CO2_F_MDS``).
    nee_obs
        Observed net ecosystem exchange, g C m-2 d-1 (``NEE_VUT_REF``). NaN
        where missing. Sign convention is FLUXNET's: negative is net uptake.
    nee_unc
        Random uncertainty on NEE, g C m-2 d-1 (``NEE_VUT_REF_RANDUNC``), used
        as the Gaussian likelihood standard deviation. NaN where missing.
    nee_qc
        NEE quality fraction, 0-1 (``NEE_VUT_REF_QC``). NaN where missing.
    nee_mask
        True on days that enter the likelihood.
    partitioned
        Partitioned GPP and RECO products, g C m-2 d-1, for Phase 8 consistency
        plots only. Never used in the likelihood.
    attrs
        Provenance: source file, site code, QC threshold, year range.
    """

    time: np.ndarray
    doy: np.ndarray
    t_air: np.ndarray
    t_day: np.ndarray
    t_night: np.ndarray
    t_max: np.ndarray
    t_min: np.ndarray
    sw_in: np.ndarray
    co2: np.ndarray
    nee_obs: np.ndarray
    nee_unc: np.ndarray
    nee_qc: np.ndarray
    nee_mask: np.ndarray
    partitioned: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    # -- convenience --------------------------------------------------------

    @property
    def n_days(self) -> int:
        """Number of daily timesteps in the block."""
        return int(self.time.shape[0])

    @property
    def n_assimilated(self) -> int:
        """Number of days entering the likelihood."""
        return int(np.count_nonzero(self.nee_mask))

    @property
    def years(self) -> np.ndarray:
        """Calendar year of each timestep."""
        return pd.DatetimeIndex(self.time).year.to_numpy()

    def drivers(self) -> dict[str, np.ndarray]:
        """Return the driver arrays keyed by their canonical short names."""
        return {name: getattr(self, name) for name in DRIVER_COLUMNS}

    def likelihood_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return finite-safe ``(observations, sigma, mask)`` for the likelihood.

        PyMC/PyTensor will happily propagate a NaN through a masked-out term and
        poison the whole logp, so the masked-out entries are replaced with
        harmless finite placeholders (0 for the observation, 1 for the standard
        deviation). Only ``mask`` decides what actually contributes.

        Returns
        -------
        observations
            NEE, g C m-2 d-1, finite everywhere.
        sigma
            Likelihood standard deviation, g C m-2 d-1, finite and strictly
            positive everywhere.
        mask
            Boolean, True where the day contributes to the likelihood.
        """
        observations = np.where(self.nee_mask, self.nee_obs, 0.0)
        sigma = np.where(self.nee_mask, self.nee_unc, 1.0)
        return (
            np.nan_to_num(observations, nan=0.0).astype(float),
            np.nan_to_num(sigma, nan=1.0).astype(float),
            self.nee_mask.astype(bool),
        )

    # -- serialisation ------------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        """Convert to an :class:`xarray.Dataset` with units recorded per variable."""
        units = {
            "doy": "1",
            "t_air": "degC",
            "t_day": "degC",
            "t_night": "degC",
            "t_max": "degC",
            "t_min": "degC",
            "sw_in": "MJ m-2 d-1",
            "co2": "umol mol-1",
            "nee_obs": "g C m-2 d-1",
            "nee_unc": "g C m-2 d-1",
            "nee_qc": "1",
            "nee_mask": "1",
        }
        data_vars: dict[str, Any] = {}
        for name, unit in units.items():
            values = getattr(self, name)
            # NetCDF has no boolean type; store as int8 and restore on load.
            if values.dtype == bool:
                values = values.astype(np.int8)
            data_vars[name] = ("time", values, {"units": unit})
        for name, values in self.partitioned.items():
            data_vars[name] = (
                "time",
                values,
                {"units": "g C m-2 d-1", "note": "consistency check only, not assimilated"},
            )
        return xr.Dataset(data_vars, coords={"time": self.time}, attrs=self.attrs)

    def save(self, path: str | Path) -> Path:
        """Write to NetCDF. Returns the path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataset().to_netcdf(out)
        return out

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> SiteData:
        """Rebuild from an :class:`xarray.Dataset` produced by :meth:`to_dataset`."""
        partitioned = {
            name: dataset[name].to_numpy().astype(float)
            for name in PARTITIONED_COLUMNS
            if name in dataset
        }
        return cls(
            time=dataset["time"].to_numpy(),
            doy=dataset["doy"].to_numpy().astype(int),
            t_air=dataset["t_air"].to_numpy().astype(float),
            t_day=dataset["t_day"].to_numpy().astype(float),
            t_night=dataset["t_night"].to_numpy().astype(float),
            t_max=dataset["t_max"].to_numpy().astype(float),
            t_min=dataset["t_min"].to_numpy().astype(float),
            sw_in=dataset["sw_in"].to_numpy().astype(float),
            co2=dataset["co2"].to_numpy().astype(float),
            nee_obs=dataset["nee_obs"].to_numpy().astype(float),
            nee_unc=dataset["nee_unc"].to_numpy().astype(float),
            nee_qc=dataset["nee_qc"].to_numpy().astype(float),
            nee_mask=dataset["nee_mask"].to_numpy().astype(bool),
            partitioned=partitioned,
            attrs=dict(dataset.attrs),
        )

    @classmethod
    def load(cls, path: str | Path) -> SiteData:
        """Read back a NetCDF file written by :meth:`save`."""
        with xr.open_dataset(path) as dataset:
            return cls.from_dataset(dataset.load())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_fluxnet_dd(path: str | Path, *, require_partitioned: bool = False) -> pd.DataFrame:
    """Load a FLUXNET2015 FULLSET daily (DD) csv.

    Handles the -9999 missing sentinel and parses ``TIMESTAMP`` (``YYYYMMDD``)
    into a real datetime column. No unit conversion and no screening happen
    here -- this is the raw table with missing values made explicit.

    Parameters
    ----------
    path
        Path to the ``FLX_<site>_FLUXNET2015_FULLSET_DD_<years>_<ver>.csv`` file.
    require_partitioned
        If True, also require the partitioned GPP/RECO columns to be present.

    Returns
    -------
    pandas.DataFrame
        Indexed by a ``DatetimeIndex`` named ``time``, sorted, with -9999
        replaced by NaN in every numeric column.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If any required driver or observation column is missing. A SUBSET file
        will trip this on ``NEE_VUT_REF_RANDUNC``.
    ValueError
        If the timestamps are not unique or not daily-contiguous.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"FLUXNET file not found: {csv_path}\n"
            "Expected the FULLSET daily (DD) csv. Note that SUBSET files omit "
            "NEE_VUT_REF_RANDUNC and cannot be used."
        )

    frame = pd.read_csv(csv_path)

    required = [TIMESTAMP_COLUMN, *DRIVER_COLUMNS.values(), *OBSERVATION_COLUMNS.values()]
    if require_partitioned:
        required += list(PARTITIONED_COLUMNS.values())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{csv_path.name} is missing required columns: {missing}. "
            "Check that this is a FLUXNET2015 FULLSET (not SUBSET) daily file."
        )

    timestamps = pd.to_datetime(
        frame[TIMESTAMP_COLUMN].astype("int64").astype(str), format="%Y%m%d"
    )

    numeric = frame.drop(columns=[TIMESTAMP_COLUMN]).apply(pd.to_numeric, errors="coerce")
    # Replace the sentinel with NaN. Compared with a tolerance rather than by
    # equality because the csv may serialise it as -9999.0, -9999.000, etc.
    numeric = numeric.mask(
        np.isclose(numeric.to_numpy(dtype=float), MISSING_VALUE, rtol=0.0, atol=1e-6)
    )

    numeric.index = pd.DatetimeIndex(timestamps, name="time")
    numeric = numeric.sort_index()

    if numeric.index.has_duplicates:
        duplicates = numeric.index[numeric.index.duplicated()].unique()[:5]
        raise ValueError(f"duplicate timestamps in {csv_path.name}: {list(duplicates)}")

    gaps = numeric.index.to_series().diff().dropna()
    if not gaps.empty and not (gaps == pd.Timedelta(days=1)).all():
        first_gap = gaps[gaps != pd.Timedelta(days=1)].index[0]
        raise ValueError(
            f"{csv_path.name} is not daily-contiguous; first break before {first_gap.date()}. "
            "The forward model requires an unbroken daily calendar."
        )

    numeric.attrs["source_file"] = str(csv_path)
    return numeric


# ---------------------------------------------------------------------------
# QC screening
# ---------------------------------------------------------------------------


def _likelihood_mask(frame: pd.DataFrame, qc_threshold: float) -> np.ndarray:
    """Boolean mask of days that enter the Gaussian likelihood.

    A day qualifies when the NEE observation is present, its QC fraction meets
    the threshold, and its random uncertainty is present and strictly positive
    (a zero or missing sd has no valid Gaussian).
    """
    nee = frame[OBSERVATION_COLUMNS["nee_obs"]].to_numpy(dtype=float)
    qc = frame[OBSERVATION_COLUMNS["nee_qc"]].to_numpy(dtype=float)
    unc = frame[OBSERVATION_COLUMNS["nee_unc"]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(nee)
            & np.isfinite(qc)
            & (qc >= qc_threshold)
            & np.isfinite(unc)
            & (unc > 0.0)
        )


def coverage_table(frame: pd.DataFrame, qc_threshold: float = 0.75) -> pd.DataFrame:
    """Per-year data coverage summary.

    This is the table that decides which years are usable for calibration and
    which are held out for evaluation, so it reports each screening criterion
    separately rather than only the intersection.

    Parameters
    ----------
    frame
        Output of :func:`load_fluxnet_dd`.
    qc_threshold
        Minimum ``NEE_VUT_REF_QC`` fraction for a day to enter the likelihood.

    Returns
    -------
    pandas.DataFrame
        One row per calendar year, indexed by year, with columns:

        ``days``
            Rows present in the file for that year.
        ``days_in_year``
            Calendar length, 365 or 366; a shortfall means a partial year.
        ``nee_present``
            Days with a non-missing ``NEE_VUT_REF``.
        ``nee_qc_pass``
            Days with non-missing NEE whose QC fraction meets the threshold.
        ``nee_qc_mean``
            Mean QC fraction over days with non-missing NEE.
        ``randunc_present``
            Days with a non-missing ``NEE_VUT_REF_RANDUNC``.
        ``assimilable``
            Days satisfying every likelihood criterion at once.
        ``assimilable_pct``
            ``assimilable`` as a percentage of ``days_in_year``.
        ``driver_gaps``
            Days with at least one missing driver. Must be 0 for a year to be
            usable at all -- the forward model cannot step through a NaN driver.
    """
    nee_column = OBSERVATION_COLUMNS["nee_obs"]
    qc_column = OBSERVATION_COLUMNS["nee_qc"]
    unc_column = OBSERVATION_COLUMNS["nee_unc"]

    working = frame.copy()
    working["_year"] = working.index.year
    working["_assimilable"] = _likelihood_mask(frame, qc_threshold)
    working["_driver_gap"] = (
        working[list(DRIVER_COLUMNS.values())].isna().any(axis=1).to_numpy()
    )

    rows = []
    for year, group in working.groupby("_year", sort=True):
        nee = group[nee_column]
        qc = group[qc_column]
        nee_present = nee.notna()
        days_in_year = 366 if pd.Timestamp(int(year), 12, 31).dayofyear == 366 else 365
        assimilable = int(group["_assimilable"].sum())
        rows.append(
            {
                "year": int(year),
                "days": len(group),
                "days_in_year": days_in_year,
                "nee_present": int(nee_present.sum()),
                "nee_qc_pass": int((nee_present & (qc >= qc_threshold)).sum()),
                "nee_qc_mean": float(qc[nee_present].mean()) if nee_present.any() else np.nan,
                "randunc_present": int(group[unc_column].notna().sum()),
                "assimilable": assimilable,
                "assimilable_pct": 100.0 * assimilable / days_in_year,
                "driver_gaps": int(group["_driver_gap"].sum()),
            }
        )

    table = pd.DataFrame(rows).set_index("year")
    table.attrs["qc_threshold"] = qc_threshold
    table.attrs["source_file"] = frame.attrs.get("source_file", "")
    return table


def _check_drivers_complete(frame: pd.DataFrame) -> None:
    """Raise if any driver has a gap over the selected block.

    Observations can be masked out of the likelihood; drivers cannot be masked
    out of the forward model. One NaN driver day contaminates every pool from
    that day onward, and it does so silently.
    """
    problems: list[str] = []
    for short_name, column in DRIVER_COLUMNS.items():
        missing = frame[column].isna()
        count = int(missing.sum())
        if count:
            examples = ", ".join(str(d.date()) for d in frame.index[missing][:5])
            suffix = ", ..." if count > 5 else ""
            problems.append(
                f"  {short_name} ({column}): {count} missing day(s) [{examples}{suffix}]"
            )
    if problems:
        raise ValueError(
            "driver gaps in the selected year range -- the forward model cannot "
            "integrate through missing drivers:\n"
            + "\n".join(problems)
            + "\nPick a different year block (see scripts/00_qc_coverage.py)."
        )


def load_daily_extremes(path: str | Path) -> pd.DataFrame:
    """Read the derived daily temperature extremes written by ``01b``.

    Parameters
    ----------
    path
        CSV written by ``scripts/01b_derive_tminmax.py``, indexed by date with
        ``t_max`` and ``t_min`` columns.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``DatetimeIndex``.

    Raises
    ------
    ValueError
        If a required column is missing, or any range is negative -- which
        cannot happen from a genuine maximum and minimum over the same day, so
        it means the file was not produced the way it claims.
    """
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    missing = [name for name in EXTREME_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}; expected a file written by "
            "scripts/01b_derive_tminmax.py"
        )
    negative = int((frame["t_max"] < frame["t_min"]).sum())
    if negative:
        raise ValueError(
            f"{path} has {negative} day(s) where t_max is below t_min. A range "
            "taken from one day's maximum and minimum cannot be negative, so "
            "this file was not produced as claimed."
        )
    return frame


def _extremes_for_block(
    extremes: pd.DataFrame, index: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """Align the derived extremes onto the block, refusing to guess at gaps."""
    aligned = extremes.reindex(index.normalize())
    absent = aligned["t_max"].isna() | aligned["t_min"].isna()
    count = int(absent.sum())
    if count:
        examples = ", ".join(str(d.date()) for d in index[absent.to_numpy()][:5])
        suffix = ", ..." if count > 5 else ""
        raise ValueError(
            f"the derived daily extremes cover only part of this block: "
            f"{count} day(s) have no matching row [{examples}{suffix}]. "
            "Re-run scripts/01b_derive_tminmax.py over a period that spans the "
            "block, or pass temperature_source='day_night_proxy' to fall back "
            "to TA_F_DAY/TA_F_NIGHT -- knowing that the proxy understates the "
            "daily range by about 4.8 degC at this site."
        )
    return (
        aligned["t_max"].to_numpy(dtype=float),
        aligned["t_min"].to_numpy(dtype=float),
    )


def build_site_data(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    qc_threshold: float = 0.75,
    site_code: str = "",
    daily_extremes: pd.DataFrame | None = None,
    temperature_source: str = "extremes",
) -> SiteData:
    """Slice a contiguous inclusive year range and assemble model-ready arrays.

    Applies the shortwave unit conversion and computes the likelihood mask. No
    rows are dropped: every day in ``[start_year, end_year]`` appears in the
    output, whether or not it is assimilated.

    Parameters
    ----------
    frame
        Output of :func:`load_fluxnet_dd`.
    start_year, end_year
        Inclusive calendar year bounds of the block.
    qc_threshold
        Minimum ``NEE_VUT_REF_QC`` fraction for a day to enter the likelihood.
    site_code
        Recorded in the output provenance attributes.
    daily_extremes
        Output of :func:`load_daily_extremes`. Required unless
        ``temperature_source`` is ``"day_night_proxy"``.
    temperature_source
        One of :data:`TEMPERATURE_SOURCES`. Defaults to the true extremes; the
        proxy has to be asked for by name, because it understates the daily
        range fourfold at this site and no published result should rest on it.

    Returns
    -------
    SiteData

    Raises
    ------
    ValueError
        If the range is empty, is not fully covered by the file, contains a
        driver gap, or the requested temperature source is unavailable.
    """
    if temperature_source not in TEMPERATURE_SOURCES:
        raise ValueError(
            f"temperature_source must be one of {TEMPERATURE_SOURCES}, "
            f"got {temperature_source!r}"
        )
    if temperature_source == "extremes" and daily_extremes is None:
        raise ValueError(
            "temperature_source='extremes' needs daily_extremes, which comes "
            "from scripts/01b_derive_tminmax.py via load_daily_extremes(). Pass "
            "temperature_source='day_night_proxy' to use TA_F_DAY/TA_F_NIGHT "
            "instead -- that is the comparison baseline, not a substitute: it "
            "understates the daily temperature range by about 4.8 degC here."
        )
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} is before start_year {start_year}")

    selection = frame.loc[str(start_year) : str(end_year)]
    if selection.empty:
        available = f"{frame.index.year.min()}-{frame.index.year.max()}"
        raise ValueError(
            f"no rows for {start_year}-{end_year}; file covers {available}"
        )

    expected_start = pd.Timestamp(start_year, 1, 1)
    expected_end = pd.Timestamp(end_year, 12, 31)
    if selection.index[0] != expected_start or selection.index[-1] != expected_end:
        raise ValueError(
            f"{start_year}-{end_year} is only partially covered by the file "
            f"({selection.index[0].date()} to {selection.index[-1].date()}). "
            "Calibration and evaluation blocks must be whole years."
        )

    _check_drivers_complete(selection)

    index = pd.DatetimeIndex(selection.index)
    mask = _likelihood_mask(selection, qc_threshold)

    t_day = selection[DRIVER_COLUMNS["t_day"]].to_numpy(dtype=float)
    t_night = selection[DRIVER_COLUMNS["t_night"]].to_numpy(dtype=float)
    if temperature_source == "extremes":
        assert daily_extremes is not None  # guarded above
        t_max, t_min = _extremes_for_block(daily_extremes, index)
    else:
        t_max, t_min = t_day, t_night

    partitioned = {
        short_name: selection[column].to_numpy(dtype=float)
        for short_name, column in PARTITIONED_COLUMNS.items()
        if column in selection.columns
    }

    attrs: dict[str, Any] = {
        "site_code": site_code,
        "source_file": frame.attrs.get("source_file", ""),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "qc_threshold": float(qc_threshold),
        "n_days": len(selection),
        "n_assimilated": int(mask.sum()),
        "sw_conversion_factor_w_m2_to_mj_m2_day": SW_W_M2_TO_MJ_M2_DAY,
        "temperature_source": temperature_source,
        "assimilation_target": OBSERVATION_COLUMNS["nee_obs"],
        "note": (
            "Partitioned GPP/RECO products are stored for posterior consistency "
            "checks only and are not assimilated."
        ),
    }

    return SiteData(
        time=index.to_numpy(),
        doy=index.dayofyear.to_numpy().astype(int),
        t_air=selection[DRIVER_COLUMNS["t_air"]].to_numpy(dtype=float),
        t_day=t_day,
        t_night=t_night,
        t_max=t_max,
        t_min=t_min,
        # W m-2 daily mean -> MJ m-2 d-1 daily total.
        sw_in=sw_in_to_mj_per_day(selection[DRIVER_COLUMNS["sw_in"]].to_numpy(dtype=float)),
        co2=selection[DRIVER_COLUMNS["co2"]].to_numpy(dtype=float),
        nee_obs=selection[OBSERVATION_COLUMNS["nee_obs"]].to_numpy(dtype=float),
        nee_unc=selection[OBSERVATION_COLUMNS["nee_unc"]].to_numpy(dtype=float),
        nee_qc=selection[OBSERVATION_COLUMNS["nee_qc"]].to_numpy(dtype=float),
        nee_mask=mask,
        partitioned=partitioned,
        attrs=attrs,
    )


def load_site_data(
    path: str | Path,
    *,
    start_year: int,
    end_year: int,
    qc_threshold: float = 0.75,
    site_code: str = "",
    extremes_file: str | Path | None = None,
    temperature_source: str = "extremes",
) -> SiteData:
    """Load a FLUXNET DD csv and return model-ready arrays for a year range.

    Convenience wrapper over :func:`load_fluxnet_dd` and :func:`build_site_data`.
    ``extremes_file`` is the csv from ``scripts/01b_derive_tminmax.py``, required
    unless ``temperature_source`` is ``"day_night_proxy"``.
    """
    frame = load_fluxnet_dd(path)
    extremes = load_daily_extremes(extremes_file) if extremes_file is not None else None
    return build_site_data(
        frame,
        start_year=start_year,
        end_year=end_year,
        qc_threshold=qc_threshold,
        site_code=site_code,
        daily_extremes=extremes,
        temperature_source=temperature_source,
    )
