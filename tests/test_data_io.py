"""Tests for FLUXNET loading, QC screening and unit conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import (
    DRIVER_GAP_DATES,
    GOOD_QC_VALUE,
    LOW_QC_DATES,
    LOW_QC_VALUE,
    NEE_MISSING_DATES,
    RANDUNC_MISSING_DATES,
    SyntheticFluxnet,
    synthetic_extremes,
)
from dalec.data_io import (
    DRIVER_COLUMNS,
    MISSING_VALUE,
    SW_W_M2_TO_MJ_M2_DAY,
    SiteData,
    build_site_data,
    coverage_table,
    load_fluxnet_dd,
    sw_in_to_mj_per_day,
    sw_in_to_w_per_m2,
)

QC_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_missing_sentinel_becomes_nan(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    assert not (frame.to_numpy(dtype=float) == MISSING_VALUE).any()
    assert frame.loc[NEE_MISSING_DATES, "NEE_VUT_REF"].isna().all()
    assert frame.loc[RANDUNC_MISSING_DATES, "NEE_VUT_REF_RANDUNC"].isna().all()
    assert frame.loc[DRIVER_GAP_DATES, "SW_IN_F"].isna().all()


def test_timestamp_parsed_to_contiguous_daily_index(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.name == "time"
    assert frame.index[0] == pd.Timestamp("2000-01-01")
    assert frame.index[-1] == pd.Timestamp("2002-12-31")
    assert frame.index.is_monotonic_increasing
    assert (frame.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    # 2000 is a leap year.
    assert len(frame) == 366 + 365 + 365


def test_subset_file_missing_randunc_raises(
    synthetic_fluxnet: SyntheticFluxnet, tmp_path
) -> None:
    """A SUBSET file has no NEE_VUT_REF_RANDUNC, and must fail loudly."""
    raw = pd.read_csv(synthetic_fluxnet.path).drop(columns=["NEE_VUT_REF_RANDUNC"])
    subset_path = tmp_path / "subset.csv"
    raw.to_csv(subset_path, index=False)

    with pytest.raises(KeyError, match="NEE_VUT_REF_RANDUNC"):
        load_fluxnet_dd(subset_path)


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fluxnet_dd(tmp_path / "does_not_exist.csv")


def test_non_contiguous_calendar_raises(
    synthetic_fluxnet: SyntheticFluxnet, tmp_path
) -> None:
    raw = pd.read_csv(synthetic_fluxnet.path)
    raw = raw.drop(index=100)  # punch a hole in the calendar
    path = tmp_path / "gappy.csv"
    raw.to_csv(path, index=False)

    with pytest.raises(ValueError, match="not daily-contiguous"):
        load_fluxnet_dd(path)


def test_vpd_is_never_a_driver() -> None:
    """VPD is out of scope by design; the fixture file contains VPD_F anyway."""
    assert "VPD_F" not in DRIVER_COLUMNS.values()


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def test_sw_conversion_factor_is_seconds_per_day_in_mj() -> None:
    # 1 W m-2 for 86400 s = 86400 J m-2 = 0.0864 MJ m-2.
    assert SW_W_M2_TO_MJ_M2_DAY == pytest.approx(86400.0 / 1.0e6)
    assert sw_in_to_mj_per_day(1.0) == pytest.approx(0.0864)
    assert sw_in_to_mj_per_day(100.0) == pytest.approx(8.64)


def test_sw_conversion_round_trips() -> None:
    values = np.array([0.0, 1.0, 37.5, 100.0, 412.7])
    assert sw_in_to_w_per_m2(sw_in_to_mj_per_day(values)) == pytest.approx(values)
    assert sw_in_to_mj_per_day(sw_in_to_w_per_m2(values)) == pytest.approx(values)


def test_site_data_radiation_is_converted(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    site_data = build_site_data(
        frame, start_year=2000, end_year=2001, qc_threshold=QC_THRESHOLD,
        daily_extremes=synthetic_extremes(pd.DatetimeIndex(frame.index)),
    )
    expected = frame.loc["2000":"2001", "SW_IN_F"].to_numpy() * SW_W_M2_TO_MJ_M2_DAY
    assert site_data.sw_in == pytest.approx(expected)
    # Sanity: a boreal daily total is single-digit MJ m-2 d-1, not hundreds.
    assert site_data.sw_in.max() < 40.0


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------


def test_coverage_table_counts_each_criterion_separately(
    synthetic_fluxnet: SyntheticFluxnet,
) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    table = coverage_table(frame, qc_threshold=QC_THRESHOLD)

    assert list(table.index) == [2000, 2001, 2002]

    # 2000: complete except for five missing RANDUNC days.
    assert table.loc[2000, "days"] == 366
    assert table.loc[2000, "days_in_year"] == 366
    assert table.loc[2000, "nee_present"] == 366
    assert table.loc[2000, "nee_qc_pass"] == 366
    assert table.loc[2000, "randunc_present"] == 366 - len(RANDUNC_MISSING_DATES)
    assert table.loc[2000, "assimilable"] == 366 - len(RANDUNC_MISSING_DATES)
    assert table.loc[2000, "driver_gaps"] == 0

    # 2001: ten missing NEE days and twenty low-QC days, non-overlapping.
    n_present = 365 - len(NEE_MISSING_DATES)
    n_pass = n_present - len(LOW_QC_DATES)
    assert table.loc[2001, "nee_present"] == n_present
    assert table.loc[2001, "nee_qc_pass"] == n_pass
    assert table.loc[2001, "randunc_present"] == 365
    assert table.loc[2001, "assimilable"] == n_pass
    expected_mean_qc = (
        n_pass * GOOD_QC_VALUE + len(LOW_QC_DATES) * LOW_QC_VALUE
    ) / n_present
    assert table.loc[2001, "nee_qc_mean"] == pytest.approx(expected_mean_qc)

    # 2002: NEE is fine, but the driver gap makes the year unusable.
    assert table.loc[2002, "assimilable"] == 365
    assert table.loc[2002, "driver_gaps"] == len(DRIVER_GAP_DATES)

    assert table.loc[2000, "assimilable_pct"] == pytest.approx(
        100.0 * (366 - len(RANDUNC_MISSING_DATES)) / 366
    )


def test_coverage_table_threshold_is_honoured(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    strict = coverage_table(frame, qc_threshold=QC_THRESHOLD)
    lenient = coverage_table(frame, qc_threshold=0.05)
    # The low-QC days come back when the threshold drops below 0.10.
    assert lenient.loc[2001, "assimilable"] - strict.loc[2001, "assimilable"] == len(LOW_QC_DATES)


# ---------------------------------------------------------------------------
# SiteData construction and masking
# ---------------------------------------------------------------------------


@pytest.fixture
def site_data(synthetic_fluxnet: SyntheticFluxnet) -> SiteData:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    return build_site_data(
        frame, start_year=2000, end_year=2001, qc_threshold=QC_THRESHOLD,
        site_code="XX-Syn",
        daily_extremes=synthetic_extremes(pd.DatetimeIndex(frame.index)),
    )


def test_no_rows_are_dropped(site_data: SiteData) -> None:
    """QC screening masks the likelihood; it must not shorten the time series."""
    assert site_data.n_days == 366 + 365
    assert site_data.time[0] == np.datetime64("2000-01-01")
    assert site_data.time[-1] == np.datetime64("2001-12-31")
    for name, values in site_data.drivers().items():
        assert values.shape == (site_data.n_days,), name
    assert site_data.nee_obs.shape == (site_data.n_days,)
    assert site_data.nee_mask.shape == (site_data.n_days,)


def test_doy_is_derived_and_handles_leap_years(site_data: SiteData) -> None:
    assert site_data.doy[0] == 1
    assert site_data.doy[365] == 366  # 2000-12-31, leap year
    assert site_data.doy[366] == 1  # 2001-01-01
    assert site_data.doy.max() == 366
    assert site_data.doy.min() == 1


def test_mask_excludes_missing_nee_low_qc_and_missing_randunc(site_data: SiteData) -> None:
    time = pd.DatetimeIndex(site_data.time)
    mask = pd.Series(site_data.nee_mask, index=time)

    assert not mask.loc[NEE_MISSING_DATES].any()
    assert not mask.loc[LOW_QC_DATES].any()
    assert not mask.loc[RANDUNC_MISSING_DATES].any()

    # ...and the days themselves are still present in the series.
    assert len(mask.loc[NEE_MISSING_DATES]) == len(NEE_MISSING_DATES)
    # RANDUNC-missing days have a perfectly good NEE value; they are excluded
    # purely because the Gaussian has no standard deviation.
    assert np.isfinite(
        pd.Series(site_data.nee_obs, index=time).loc[RANDUNC_MISSING_DATES]
    ).all()

    expected_masked = len(NEE_MISSING_DATES) + len(LOW_QC_DATES) + len(RANDUNC_MISSING_DATES)
    assert site_data.n_assimilated == site_data.n_days - expected_masked


def test_likelihood_arrays_are_finite_and_positive(site_data: SiteData) -> None:
    observations, sigma, mask = site_data.likelihood_arrays()
    assert np.isfinite(observations).all()
    assert np.isfinite(sigma).all()
    assert (sigma > 0.0).all()
    assert mask.dtype == bool
    # Unmasked entries are untouched.
    assert observations[mask] == pytest.approx(site_data.nee_obs[mask])
    assert sigma[mask] == pytest.approx(site_data.nee_unc[mask])


def test_driver_gap_raises_with_a_useful_message(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    with pytest.raises(ValueError, match="driver gaps") as excinfo:
        build_site_data(frame, start_year=2002, end_year=2002, qc_threshold=QC_THRESHOLD,
                        temperature_source="day_night_proxy")
    message = str(excinfo.value)
    assert "SW_IN_F" in message
    assert "2002-06-01" in message


def test_partial_year_range_raises(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    truncated = frame.loc[:"2001-06-30"]
    truncated.attrs = frame.attrs
    with pytest.raises(ValueError, match="partially covered"):
        build_site_data(truncated, start_year=2000, end_year=2001,
                        temperature_source="day_night_proxy")


def test_year_range_outside_record_raises(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    with pytest.raises(ValueError, match="no rows"):
        build_site_data(frame, start_year=1990, end_year=1991,
                        temperature_source="day_night_proxy")


def test_reversed_year_range_raises(synthetic_fluxnet: SyntheticFluxnet) -> None:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    with pytest.raises(ValueError, match="before start_year"):
        build_site_data(frame, start_year=2001, end_year=2000,
                        temperature_source="day_night_proxy")


def test_partitioned_products_are_carried_but_flagged(site_data: SiteData) -> None:
    assert set(site_data.partitioned) == {"gpp_nt", "gpp_dt", "reco_nt", "reco_dt"}
    assert "not assimilated" in site_data.attrs["note"]
    assert site_data.attrs["assimilation_target"] == "NEE_VUT_REF"


def test_provenance_is_recorded(site_data: SiteData) -> None:
    attrs = site_data.attrs
    assert attrs["site_code"] == "XX-Syn"
    assert attrs["start_year"] == 2000
    assert attrs["end_year"] == 2001
    assert attrs["qc_threshold"] == QC_THRESHOLD
    assert attrs["sw_conversion_factor_w_m2_to_mj_m2_day"] == SW_W_M2_TO_MJ_M2_DAY
    assert attrs["source_file"].endswith(".csv")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_netcdf_round_trip(site_data: SiteData, tmp_path) -> None:
    path = site_data.save(tmp_path / "processed" / "block.nc")
    assert path.is_file()

    restored = SiteData.load(path)
    assert restored.n_days == site_data.n_days
    assert restored.nee_mask.dtype == bool
    np.testing.assert_array_equal(restored.nee_mask, site_data.nee_mask)
    np.testing.assert_array_equal(restored.doy, site_data.doy)
    np.testing.assert_array_equal(restored.time, site_data.time)
    for name in ("t_air", "t_day", "t_night", "sw_in", "co2", "nee_obs", "nee_unc", "nee_qc"):
        np.testing.assert_allclose(
            getattr(restored, name), getattr(site_data, name), rtol=0, atol=0, equal_nan=True
        )
    assert restored.attrs["qc_threshold"] == site_data.attrs["qc_threshold"]
    assert set(restored.partitioned) == set(site_data.partitioned)


def test_dataset_records_units(site_data: SiteData) -> None:
    dataset = site_data.to_dataset()
    assert dataset["sw_in"].attrs["units"] == "MJ m-2 d-1"
    assert dataset["nee_obs"].attrs["units"] == "g C m-2 d-1"
    assert dataset["nee_unc"].attrs["units"] == "g C m-2 d-1"
    assert dataset["t_air"].attrs["units"] == "degC"
    assert dataset["co2"].attrs["units"] == "umol mol-1"
