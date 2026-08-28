"""Tests for the pre-calibration GPP magnitude gate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import SyntheticFluxnet, make_parameters, synthetic_extremes
from dalec.data_io import MISSING_VALUE, SiteData, build_site_data, load_fluxnet_dd
from dalec.diagnostics import (
    HALFHOURS_PER_DAY,
    SUGGESTED_PEAK_OFFSET_DAYS,
    SWEEP_POINTS,
    annual_gpp_comparison,
    binned_median_iqr,
    canopy_efficiency_sweep,
    classify_prior_draw,
    format_gpp_magnitude_report,
    format_seasonal_timing_report,
    gaussian_loglik,
    gpp_magnitude_gate,
    prior_decade_mass,
    prior_sweep_values,
    randunc_audit,
    randunc_relationships,
    sample_prior_parameters,
    seasonal_timing,
    simplex_edge_weights,
    temperature_proxy_comparison,
)
from dalec.model_numpy import DalecOutput, run_dalec2
from dalec.parameters import PARAMETER_REGISTRY, prior_bounds

#: Roughly calibrates the fake photosynthesis routine below so that its annual
#: total lands near the synthetic partitioned products at mid-range ceff.
WELL_SCALED = 0.474


@pytest.fixture
def block(synthetic_fluxnet: SyntheticFluxnet) -> SiteData:
    """Two clean years from the synthetic FLUXNET file, with GPP products."""
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    return build_site_data(
        frame, start_year=2000, end_year=2001, qc_threshold=0.75,
        daily_extremes=synthetic_extremes(pd.DatetimeIndex(frame.index)),
    )


def saturating_gpp(scale: float) -> Callable[..., float]:
    """A stand-in photosynthesis routine that saturates in ceff, as ACM does."""

    def _gpp(*, sw_in: float, ceff: float, **_: object) -> float:
        return scale * (ceff / (ceff + 20.0)) * sw_in

    return _gpp


# ---------------------------------------------------------------------------
# Annual comparison
# ---------------------------------------------------------------------------


def test_annual_comparison_reports_one_row_per_year(block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    table = annual_gpp_comparison(output, block)

    assert list(table.index) == [2000, 2001]
    assert list(table["days"]) == [366, 365]
    for column in ("gpp_model", "gpp_nt", "gpp_dt", "ratio_nt", "ratio_dt", "ratio_ref"):
        assert np.isfinite(table[column]).all(), column


def test_modelled_total_is_the_plain_annual_sum(block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    table = annual_gpp_comparison(output, block)

    in_2000 = block.years == 2000
    assert table.loc[2000, "gpp_model"] == pytest.approx(output.gpp[in_2000].sum())


def test_ratio_is_computed_on_matched_days_only(block: SiteData) -> None:
    """A gappy product must not inflate the ratio in proportion to its gaps."""
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    complete = annual_gpp_comparison(output, block)

    # Punch out half of one year's NT product. The matched-day ratio must not
    # move; a naive full-year-over-available-days ratio would roughly double.
    gappy_nt = block.partitioned["gpp_nt"].copy()
    holes = (block.years == 2000) & (np.arange(block.n_days) % 2 == 0)
    gappy_nt[holes] = np.nan
    gappy = SiteData(
        **{
            **{
                name: getattr(block, name)
                for name in (
                    "time", "doy", "t_air", "t_day", "t_night", "t_max", "t_min", "sw_in", "co2",
                    "nee_obs", "nee_unc", "nee_qc", "nee_mask", "attrs",
                )
            },
            "partitioned": {**block.partitioned, "gpp_nt": gappy_nt},
        }
    )
    holed = annual_gpp_comparison(output, gappy)

    assert holed.loc[2000, "coverage_nt"] == pytest.approx(0.5)
    assert holed.loc[2000, "ratio_nt"] == pytest.approx(
        complete.loc[2000, "ratio_nt"], rel=0.05
    )
    # ...whereas the raw product total does fall by about half.
    assert holed.loc[2000, "gpp_nt"] < 0.6 * complete.loc[2000, "gpp_nt"]


def test_run_length_mismatch_raises(block: SiteData) -> None:
    frame = load_fluxnet_dd(block.attrs["source_file"])
    shorter = build_site_data(
        frame, start_year=2000, end_year=2000,
        daily_extremes=synthetic_extremes(pd.DatetimeIndex(frame.index)),
    )
    output = run_dalec2(make_parameters(), shorter, gpp_fn=saturating_gpp(WELL_SCALED))

    with pytest.raises(ValueError, match="same period"):
        annual_gpp_comparison(output, block)


def test_missing_partitioned_products_raise(block: SiteData) -> None:
    stripped = SiteData(
        **{**{k: v for k, v in vars(block).items() if k != "partitioned"}, "partitioned": {}}
    )
    output = run_dalec2(make_parameters(), stripped, gpp_fn=saturating_gpp(WELL_SCALED))

    with pytest.raises(ValueError, match="partitioned product"):
        annual_gpp_comparison(output, stripped)


# ---------------------------------------------------------------------------
# Canopy efficiency sweep
# ---------------------------------------------------------------------------


def test_sweep_spans_the_ceff_prior_range(block: SiteData) -> None:
    sweep = canopy_efficiency_sweep(
        make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED)
    )

    assert sweep.index.min() == pytest.approx(10.0)
    assert sweep.index.max() == pytest.approx(100.0)
    assert sweep["gpp_model"].is_monotonic_increasing


def test_sweep_honours_explicit_ceff_values(block: SiteData) -> None:
    sweep = canopy_efficiency_sweep(
        make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED),
        ceff_values=np.array([10.0, 40.0, 100.0]),
    )
    assert list(sweep.index) == [10.0, 40.0, 100.0]


def test_sweep_overrides_ceff_rather_than_mutating_params(block: SiteData) -> None:
    params = make_parameters(ceff=15.0)
    canopy_efficiency_sweep(params, block, gpp_fn=saturating_gpp(WELL_SCALED))
    assert params.ceff == 15.0


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_gate_passes_for_a_well_scaled_forward_model(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))

    assert gate.passed
    assert 1.0 / gate.tolerance <= gate.best_ratio <= gate.tolerance
    assert 10.0 <= gate.best_ceff <= 100.0


def test_gate_fails_when_gpp_is_five_times_too_high(block: SiteData) -> None:
    """The reported failure mode: 5x on annual GPP must block calibration."""
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(5 * WELL_SCALED))

    assert not gate.passed
    assert gate.best_ratio > 1.5
    assert "DO NOT PROCEED" in format_gpp_magnitude_report(gate)


def test_gate_fails_when_gpp_is_far_too_low(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED / 10))

    assert not gate.passed
    assert gate.best_ratio < 1.0 / 1.5


def test_gate_picks_the_ceff_closest_to_a_ratio_of_one(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))

    distances = np.abs(np.log(gate.sweep["ratio_ref"].to_numpy(dtype=float)))
    assert gate.best_ceff == pytest.approx(float(gate.sweep.index[int(distances.argmin())]))


def test_gate_treats_over_and_under_shoot_symmetrically(block: SiteData) -> None:
    """Closest in log space, so 2x high and 2x low are equally far off."""
    high = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(3 * WELL_SCALED))
    low = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED / 3))

    assert not high.passed
    assert not low.passed
    assert high.best_ratio > 1.0
    assert low.best_ratio < 1.0


def test_gate_requires_a_photosynthesis_routine(block: SiteData) -> None:
    """Left at the default stub, the gate must raise rather than report nonsense."""
    with pytest.raises(NotImplementedError, match="Aggregated Canopy Model"):
        gpp_magnitude_gate(make_parameters(), block)


def test_gate_rejects_a_meaningless_tolerance(block: SiteData) -> None:
    with pytest.raises(ValueError, match="tolerance must be greater than 1"):
        gpp_magnitude_gate(
            make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED), tolerance=1.0
        )


def test_report_is_readable_and_states_what_the_products_are(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    report = format_gpp_magnitude_report(gate)

    assert "GPP magnitude gate: PASS" in report
    assert "never assimilated" in report
    assert "matched days" in report
    assert "2000" in report and "2001" in report


# ---------------------------------------------------------------------------
# Canopy density check
#
# The GPP ratio is dominated by LAI rather than by ceff, so a gate reporting
# only the ratio could be made to pass or fail by the parameter set handed to
# it. These pin the second, independent check.
# ---------------------------------------------------------------------------


def test_gate_reports_the_lai_trajectory(block: SiteData) -> None:
    params = make_parameters()
    gate = gpp_magnitude_gate(params, block, gpp_fn=saturating_gpp(WELL_SCALED))

    expected = (
        run_dalec2(
            replace(params, ceff=gate.best_ceff), block, gpp_fn=saturating_gpp(WELL_SCALED)
        ).pool("c_fol")[:-1]
        / params.lma
    )
    assert gate.lai_min == pytest.approx(expected.min())
    assert gate.lai_mean == pytest.approx(expected.mean())
    assert gate.lai_max == pytest.approx(expected.max())
    assert gate.lai_end == pytest.approx(expected[-1])


def test_lai_appears_in_the_report(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    report = format_gpp_magnitude_report(gate)

    assert "LAI at best ceff" in report
    assert "plausible band" in report
    assert "lai_mean" in report and "lai_max" in report  # the sweep columns


def test_implausible_lai_fails_the_gate_even_when_the_ratio_is_fine(
    block: SiteData,
) -> None:
    """The two checks are independent, and the ratio one still passes here."""
    gate = gpp_magnitude_gate(
        make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED), lai_band=(50.0, 60.0)
    )

    assert gate.ratio_ok, "the GPP ratio itself is still within tolerance"
    assert not gate.lai_ok
    assert not gate.passed


def test_the_two_failures_read_differently(block: SiteData) -> None:
    # Each failure is isolated deliberately. Five times too much GPP also grows
    # the canopy out of any plausible band -- which is the coupling this check
    # exists to expose -- so the band is widened here to leave only the ratio
    # failing, and the LAI case below is scaled correctly and fails on the band
    # alone.
    ratio_failure = format_gpp_magnitude_report(
        gpp_magnitude_gate(
            make_parameters(), block, gpp_fn=saturating_gpp(5 * WELL_SCALED),
            lai_band=(0.1, 100.0),
        )
    )
    lai_failure = format_gpp_magnitude_report(
        gpp_magnitude_gate(
            make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED),
            lai_band=(50.0, 60.0),
        )
    )

    assert "GPP ratio failed." in ratio_failure
    assert "Canopy density failed" not in ratio_failure

    assert "Canopy density failed" in lai_failure
    assert "says nothing about ACM" in lai_failure
    assert "GPP ratio failed." not in lai_failure
    assert "DO NOT PROCEED" in lai_failure


def test_gate_rejects_a_meaningless_lai_band(block: SiteData) -> None:
    with pytest.raises(ValueError, match="positive and increasing"):
        gpp_magnitude_gate(
            make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED), lai_band=(8.0, 1.0)
        )


# ---------------------------------------------------------------------------
# Seasonal timing
# ---------------------------------------------------------------------------


def test_seasonal_timing_reports_every_series(block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    table = seasonal_timing(output, block)

    assert list(table.index) == ["model", "gpp_nt", "gpp_dt"]
    for column in ("peak_doy", "onset_doy", "cessation_doy", "season_length_days"):
        assert (table[column] > 0).all(), column
    assert (table["onset_doy"] < table["cessation_doy"]).all()


def test_seasonal_timing_detects_a_deliberate_phase_shift(block: SiteData) -> None:
    """The check the magnitude gate cannot do: roll the model and see it move."""
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    aligned = seasonal_timing(output, block)

    shifted = replace(output, gpp=np.roll(output.gpp, 30))
    moved = seasonal_timing(shifted, block)

    assert abs(int(moved.loc["model", "peak_offset_days"])) > abs(
        int(aligned.loc["model", "peak_offset_days"])
    )
    assert int(moved.loc["model", "peak_doy"]) != int(aligned.loc["model", "peak_doy"])


def test_a_phase_shift_barely_moves_the_annual_ratio(block: SiteData) -> None:
    """Why the timing check has to exist separately from the magnitude gate."""
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    shifted = replace(output, gpp=np.roll(output.gpp, 30))

    before = annual_gpp_comparison(output, block)["gpp_model"].sum()
    after = annual_gpp_comparison(shifted, block)["gpp_model"].sum()
    assert after == pytest.approx(before, rel=0.02), (
        "an annual integral is nearly blind to phase, which is the whole point"
    )


def test_the_products_define_the_reference_peak(block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    table = seasonal_timing(output, block)

    product_offsets = table.loc[["gpp_nt", "gpp_dt"], "peak_offset_days"]
    assert product_offsets.sum() == 0, "the two products straddle their own mean"


def test_timing_report_states_that_it_is_not_enforced(block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    report = format_seasonal_timing_report(seasonal_timing(output, block))

    assert "NOT ENFORCED" in report
    assert "nearly blind to" in report
    assert str(SUGGESTED_PEAK_OFFSET_DAYS) in report


def test_timing_rejects_a_mismatched_run(block: SiteData) -> None:
    frame = load_fluxnet_dd(block.attrs["source_file"])
    shorter = build_site_data(
        frame, start_year=2000, end_year=2000,
        daily_extremes=synthetic_extremes(pd.DatetimeIndex(frame.index)),
    )
    output = run_dalec2(make_parameters(), shorter, gpp_fn=saturating_gpp(WELL_SCALED))

    with pytest.raises(ValueError, match="same period"):
        seasonal_timing(output, block)


# ---------------------------------------------------------------------------
# Temperature proxy comparison
# ---------------------------------------------------------------------------


def _extremes_for(
    block: SiteData, *, spread: float = 4.0, vary: bool = True
) -> pd.DataFrame:
    """True daily extremes, as 01b_derive_tminmax.py would write them.

    The spread varies through the year by default: a constant range is
    degenerate, and correlation against it is undefined.
    """
    mean = 0.5 * (block.t_day + block.t_night)
    half = spread + (np.sin(2.0 * np.pi * block.doy / 365.25) if vary else 0.0)
    return pd.DataFrame(
        {
            "t_max": mean + half,
            "t_min": mean - half,
            "t_range": 2.0 * half,
            "n_halfhours": np.full(block.n_days, 48),
            "reliable": np.full(block.n_days, True),
        },
        index=pd.DatetimeIndex(block.time).normalize(),
    )


def test_proxy_comparison_reports_every_quantity(block: SiteData) -> None:
    table = temperature_proxy_comparison(block, _extremes_for(block))

    assert list(table.index) == ["t_max vs TA_F_DAY", "t_min vs TA_F_NIGHT", "t_range"]
    assert (table["n_days"] == block.n_days).all()
    for column in ("correlation", "mean_bias", "rmse"):
        assert np.isfinite(table[column]).all(), column


def test_proxy_comparison_measures_the_range_understatement(block: SiteData) -> None:
    """The whole point: the proxy range is not the true range."""
    extremes = _extremes_for(block, spread=4.0, vary=False)  # true range 8.0 flat
    table = temperature_proxy_comparison(block, extremes)

    proxy_range = float((block.t_day - block.t_night).mean())
    assert table.loc["t_range", "mean_bias"] == pytest.approx(proxy_range - 8.0)
    assert table.loc["t_range", "truth_min"] == pytest.approx(8.0)


def test_correlation_against_a_constant_series_is_nan_not_a_warning(
    block: SiteData,
) -> None:
    """A flat truth series makes correlation undefined; say so, quietly."""
    table = temperature_proxy_comparison(block, _extremes_for(block, vary=False))
    assert np.isnan(table.loc["t_range", "correlation"])
    assert np.isfinite(table.loc["t_range", "mean_bias"]), "the bias is still defined"


def test_proxy_comparison_skips_unreliable_days(block: SiteData) -> None:
    extremes = _extremes_for(block)
    extremes.iloc[: block.n_days // 2, extremes.columns.get_loc("reliable")] = False
    table = temperature_proxy_comparison(block, extremes)

    assert table["n_days"].iloc[0] == block.n_days - block.n_days // 2


def test_proxy_comparison_needs_overlapping_dates(block: SiteData) -> None:
    extremes = _extremes_for(block)
    extremes.index = extremes.index + pd.Timedelta(days=10_000)

    with pytest.raises(ValueError, match="share no usable dates"):
        temperature_proxy_comparison(block, extremes)


def test_sweep_carries_lai_alongside_gpp(block: SiteData) -> None:
    sweep = canopy_efficiency_sweep(
        make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED)
    )

    assert (sweep["lai_max"] >= sweep["lai_mean"]).all()
    assert (sweep["lai_mean"] > 0.0).all()


# ---------------------------------------------------------------------------
# Task 4 -- RANDUNC characterisation
# ---------------------------------------------------------------------------


def _randunc_frame(
    n_days: int = 400, *, missing: slice | None = None, sentinel: bool = False
) -> pd.DataFrame:
    """A minimal FLUXNET-shaped frame with known answers."""
    index = pd.date_range("2000-01-01", periods=n_days, freq="D", name="time")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "NEE_VUT_REF": rng.normal(0.0, 2.0, n_days),
            "NEE_VUT_REF_QC": np.linspace(0.0, 1.0, n_days).round(6),
            "NEE_VUT_REF_RANDUNC": rng.uniform(0.05, 0.4, n_days),
        },
        index=index,
    )
    if missing is not None:
        frame.iloc[missing, frame.columns.get_loc("NEE_VUT_REF_RANDUNC")] = (
            MISSING_VALUE if sentinel else np.nan
        )
    return frame


def test_randunc_audit_counts_are_exact() -> None:
    frame = _randunc_frame(n_days=400, missing=slice(10, 40))
    audit = randunc_audit(frame, qc_threshold=0.75, calibration_years=(2000, 2000))

    assert audit.total_days == 400
    assert audit.valid_nee == 400
    assert audit.valid_nee_missing_unc == 30
    assert audit.valid_nee_zero_unc == 0
    # QC rises linearly from 0 to 1, so exactly the top quarter clears 0.75.
    assert audit.qc_pass == int((frame["NEE_VUT_REF_QC"] >= 0.75).sum())


def test_randunc_audit_separates_missing_from_zero() -> None:
    """A missing sigma and a zero sigma assert different things."""
    frame = _randunc_frame(n_days=100)
    frame.iloc[0, frame.columns.get_loc("NEE_VUT_REF_RANDUNC")] = 0.0
    frame.iloc[1, frame.columns.get_loc("NEE_VUT_REF_RANDUNC")] = np.nan
    audit = randunc_audit(frame, qc_threshold=0.0)

    assert audit.valid_nee_zero_unc == 1
    assert audit.valid_nee_missing_unc == 1
    # Both are unusable, so both are counted against the likelihood.
    assert audit.qc_pass_no_sigma == 2


def test_randunc_sentinel_conversion(tmp_path) -> None:
    """-9999 must become NaN on load and be excluded from every statistic."""
    frame = _randunc_frame(n_days=120, missing=slice(0, 20), sentinel=True)
    # Straight from the raw frame the sentinel is a huge negative number...
    assert frame["NEE_VUT_REF_RANDUNC"].min() == MISSING_VALUE
    # ...and once converted, as load_fluxnet_dd does, it is simply absent.
    converted = frame.replace(MISSING_VALUE, np.nan)
    audit = randunc_audit(converted, qc_threshold=0.0)
    assert audit.valid_nee_missing_unc == 20
    assert audit.valid_nee_zero_unc == 0
    fits = randunc_relationships(converted)
    assert fits.on_abs_nee.n == 100, "sentinel days must not enter the fit"


def test_audit_reports_contiguous_blocks_not_scattered_days() -> None:
    frame = _randunc_frame(n_days=200, missing=slice(50, 60))
    frame.iloc[120, frame.columns.get_loc("NEE_VUT_REF_RANDUNC")] = np.nan
    audit = randunc_audit(frame, qc_threshold=0.0)

    blocks = audit.missing_blocks
    assert len(blocks) == 2
    assert list(blocks["days"]) == [10, 1]
    assert int(blocks["days"].sum()) == audit.qc_pass_no_sigma


def test_audit_restricts_the_block_count_to_the_calibration_years() -> None:
    frame = _randunc_frame(n_days=800, missing=slice(0, 30))
    inside = randunc_audit(frame, qc_threshold=0.0, calibration_years=(2000, 2000))
    outside = randunc_audit(frame, qc_threshold=0.0, calibration_years=(2001, 2001))

    assert inside.qc_pass_no_sigma_in_block == 30
    assert outside.qc_pass_no_sigma_in_block == 0
    assert inside.qc_pass_no_sigma == outside.qc_pass_no_sigma == 30


def test_qc_multiple_of_halfhour_detection() -> None:
    """QC * 48 is an exact half-hour count only if every value is a multiple."""
    exact = _randunc_frame(n_days=49)
    exact["NEE_VUT_REF_QC"] = np.arange(49) / 48.0
    assert randunc_audit(exact, qc_threshold=0.0).qc_is_multiple_of_halfhour

    ragged = _randunc_frame(n_days=49)
    ragged["NEE_VUT_REF_QC"] = np.linspace(0.0, 1.0, 49) + 0.001
    assert not randunc_audit(ragged, qc_threshold=0.0).qc_is_multiple_of_halfhour


def test_relationships_recover_a_planted_sqrt_n_law() -> None:
    """If sigma really is the SE of a mean, the log-log slope must come back -0.5.

    This is the positive control for the headline result: without it, a slope
    near zero could just mean the metric is broken.
    """
    n_days = 480
    index = pd.date_range("2000-01-01", periods=n_days, freq="D", name="time")
    qc = np.tile(np.arange(1, 49) / 48.0, n_days // 48)
    # |NEE| varies but sigma does not depend on it: the planted law is purely
    # 1/sqrt(n). A constant NEE would make the |NEE| regression degenerate and
    # exercise a numerically ill-conditioned path that says nothing useful.
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "NEE_VUT_REF": rng.uniform(0.5, 5.0, n_days),
            "NEE_VUT_REF_QC": qc,
            "NEE_VUT_REF_RANDUNC": 1.0 / np.sqrt(qc * HALFHOURS_PER_DAY),
        },
        index=index,
    )
    fits = randunc_relationships(frame)

    assert fits.log_log_slope == pytest.approx(-0.5, abs=1e-6)
    assert fits.on_inv_sqrt_n.r_squared == pytest.approx(1.0, abs=1e-6)


def test_relationships_reject_a_magnitude_proportional_law() -> None:
    """The negative control: sigma tied to |NEE| gives no 1/sqrt(n) signal."""
    n_days = 480
    index = pd.date_range("2000-01-01", periods=n_days, freq="D", name="time")
    rng = np.random.default_rng(7)
    nee = rng.uniform(0.5, 5.0, n_days)
    frame = pd.DataFrame(
        {
            "NEE_VUT_REF": nee,
            "NEE_VUT_REF_QC": np.tile(np.arange(1, 49) / 48.0, n_days // 48),
            "NEE_VUT_REF_RANDUNC": 0.08 + 0.05 * nee,
        },
        index=index,
    )
    fits = randunc_relationships(frame)

    assert abs(fits.log_log_slope) < 0.1
    assert fits.on_abs_nee.r_squared == pytest.approx(1.0, abs=1e-6)
    assert abs(fits.partial_r_given_abs_nee) < 0.05


def test_binned_median_iqr_drops_empty_bins_and_orders_quartiles() -> None:
    x = np.array([0.1, 0.2, 0.9, 0.95])
    y = np.array([1.0, 3.0, 10.0, 20.0])
    table = binned_median_iqr(x, y, np.array([0.0, 0.5, 0.8, 1.0]))

    assert len(table) == 2, "the 0.5-0.8 bin is empty and must be dropped"
    assert (table["q25"] <= table["median"]).all()
    assert (table["median"] <= table["q75"]).all()
    assert list(table["n"]) == [2, 2]


# ---------------------------------------------------------------------------
# Task 1 -- prior sampling, decade mass, failed-draw classification
# ---------------------------------------------------------------------------


def test_scalar_prior_draws_come_from_the_registry_not_from_typed_bounds() -> None:
    """The binding constraint of amendment A2.

    Every scalar draw must sit inside the registry's own bounds. If a bound were
    retyped anywhere, the guarantee that these are the priors that will later be
    sampled is gone, and this is what would catch it.

    The four simplex fractions are excluded deliberately -- see the test below.
    """
    _, frame = sample_prior_parameters(300, rng=np.random.default_rng(0))

    for name, entry in PARAMETER_REGISTRY.items():
        if entry.simplex:
            continue
        lower, upper = prior_bounds(name)
        values = frame[name].to_numpy()
        assert (values >= lower).all(), f"{name} below its registry lower bound"
        assert (values <= upper).all(), f"{name} above its registry upper bound"


def test_simplex_fractions_are_not_confined_to_their_tabulated_marginals() -> None:
    """They are a Dirichlet split of 1 - f_auto, and that support is wider.

    DECISIONS.md section 4: the Dirichlet support is a *superset* of the
    published marginal ranges of 0.01-0.5, and an individual fraction may reach
    1 - f_auto, up to 0.7. The registry keeps the tabulated bounds only because
    Morris screening needs a range to perturb over, which is why those four
    carry simplex=True so priors.py cannot build a Uniform from them.

    Enforcing the marginals would need a truncated Dirichlet, reintroducing
    exactly the rejection wall the reparameterisation exists to remove. So this
    pins the deviation rather than treating it as a bug to be fixed later.
    """
    _, frame = sample_prior_parameters(600, rng=np.random.default_rng(5))
    simplex = [name for name, p in PARAMETER_REGISTRY.items() if p.simplex]

    below = sum(int((frame[name].to_numpy() < prior_bounds(name)[0]).any())
                for name in simplex)
    assert below > 0, (
        "no simplex fraction fell below its tabulated marginal; the Dirichlet "
        "support is supposed to be wider than the published range"
    )
    for name in simplex:
        values = frame[name].to_numpy()
        assert (values >= 0.0).all(), f"{name} went negative"
        assert (values <= 0.7).all(), f"{name} exceeded 1 - min(f_auto)"


def test_allocation_fractions_close_exactly_on_every_draw() -> None:
    """f_auto + f_lab + f_fol + f_roo + f_woo = 1, the corrected five-term form."""
    params, _ = sample_prior_parameters(200, rng=np.random.default_rng(1))
    for parameters in params:
        total = (
            parameters.f_auto + parameters.f_lab + parameters.f_fol
            + parameters.f_roo + parameters.f_woo
        )
        assert total == pytest.approx(1.0)


def test_the_simplex_fractions_are_not_drawn_independently() -> None:
    """Independent uniforms could not close; a Dirichlet split of 1 - f_auto does.

    Also guards the registry's own claim: the four are flagged simplex=True
    precisely so nothing builds a Uniform from their tabulated bounds.
    """
    _, frame = sample_prior_parameters(400, rng=np.random.default_rng(2))
    simplex = [name for name, p in PARAMETER_REGISTRY.items() if p.simplex]
    assert simplex == ["f_lab", "f_fol", "f_roo", "f_woo"]

    allocated = frame[simplex].sum(axis=1).to_numpy()
    assert allocated == pytest.approx(1.0 - frame["f_auto"].to_numpy())


def test_prior_draws_are_reproducible_from_the_seed() -> None:
    first, _ = sample_prior_parameters(20, rng=np.random.default_rng(20260809))
    second, _ = sample_prior_parameters(20, rng=np.random.default_rng(20260809))
    assert [p.to_dict() for p in first] == [p.to_dict() for p in second]


def test_decade_mass_sums_to_one_per_parameter() -> None:
    table = prior_decade_mass()
    for name, group in table.groupby("parameter"):
        assert group["mass_fraction"].sum() == pytest.approx(1.0), name


def test_decade_mass_shows_the_uniform_prior_is_not_ignorance() -> None:
    """theta_som on [1e-7, 1e-3] puts 90% of its mass in the top decade alone."""
    table = prior_decade_mass()
    som = table[table["parameter"] == "theta_som"].sort_values("decade_low")

    assert len(som) == 4
    assert som["mass_fraction"].iloc[-1] == pytest.approx(0.9, abs=0.01)
    assert som["mass_fraction"].iloc[:2].sum() < 0.011, "bottom two decades are ~1%"


def test_decade_mass_only_reports_wide_priors() -> None:
    reported = set(prior_decade_mass()["parameter"])
    # ceff spans exactly one order and must not appear.
    assert "ceff" not in reported
    # theta_lit spans exactly two, which is not *more* than two.
    assert "theta_lit" not in reported
    assert {"theta_som", "theta_min", "c_woo_0", "c_som_0"} <= reported


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda p, n: p.__setitem__((0, 0), np.nan), "non-finite"),
        (lambda p, n: p.__setitem__((5, 2), -1.0), "negative pool"),
        (lambda p, n: p.__setitem__((5, 3), p[0, 3] * 500.0), "pool growth"),
        (lambda p, n: n.__setitem__(7, 45.0), "|NEE|"),
    ],
)
def test_failed_draw_classifier_catches_each_criterion(mutate, expected) -> None:
    """Hand-built trajectories, one defect each."""
    pools = np.tile(np.array([100.0, 300.0, 200.0, 8000.0, 150.0, 10000.0]), (11, 1))
    nee = np.full(10, 0.5)
    mutate(pools, nee)
    output = _output_from(pools, nee)

    reason = classify_prior_draw(output)
    assert reason is not None and expected in reason


def test_failed_draw_classifier_passes_a_clean_trajectory() -> None:
    pools = np.tile(np.array([100.0, 300.0, 200.0, 8000.0, 150.0, 10000.0]), (11, 1))
    assert classify_prior_draw(_output_from(pools, np.full(10, 0.5))) is None


def _output_from(pools: np.ndarray, nee: np.ndarray) -> DalecOutput:
    zeros = np.zeros_like(nee)
    return DalecOutput(
        pools=pools, gpp=zeros, ra=zeros, rh=zeros, reco=zeros, nee=nee,
        phi_onset=zeros, phi_fall=zeros,
    )


# ---------------------------------------------------------------------------
# Task 2 -- sweep design and the likelihood
# ---------------------------------------------------------------------------


def test_sweep_endpoints_lie_inside_the_prior_support() -> None:
    """1st to 99th percentile, so no sweep point sits on a bound."""
    for name, entry in PARAMETER_REGISTRY.items():
        if entry.simplex:
            continue
        values = prior_sweep_values(name)
        assert len(values) == SWEEP_POINTS
        assert values[0] > entry.lower, f"{name} starts on its lower bound"
        assert values[-1] < entry.upper, f"{name} ends on its upper bound"
        assert (values >= entry.lower).all() and (values <= entry.upper).all()


def test_sweep_is_linear_because_the_priors_are_uniform() -> None:
    """Amendment A1: linear spacing, not geometric.

    The original specification called for geometric spacing on the grounds that
    the priors were log-uniform. They are not, and a geometric centre would put
    theta_som two orders of magnitude off.
    """
    values = prior_sweep_values("theta_som")
    steps = np.diff(values)
    assert np.allclose(steps, steps[0]), "spacing is not linear"

    lower, upper = prior_bounds("theta_som")
    geometric_centre = np.sqrt(lower * upper)
    linear_centre = values[len(values) // 2]
    assert linear_centre / geometric_centre > 10.0, (
        "the two conventions should differ by orders here, which is the point"
    )


def test_simplex_edge_rows_stay_on_the_simplex() -> None:
    anchor = np.array([0.30, 0.29, 0.28, 0.13])
    for target in range(4):
        rows = simplex_edge_weights(anchor, target)
        assert rows.shape == (SWEEP_POINTS, 4)
        assert np.allclose(rows.sum(axis=1), 1.0), "rows must remain a simplex"
        assert (rows >= 0.0).all()
        assert rows[0, target] < rows[-1, target], "target must increase"


def test_simplex_edge_holds_the_other_fractions_in_proportion() -> None:
    """Raising one fraction lowers the others proportionally, not arbitrarily."""
    anchor = np.array([0.40, 0.30, 0.20, 0.10])
    rows = simplex_edge_weights(anchor, 0)
    others = rows[:, 1:]
    first = others[0] / others[0].sum()
    last = others[-1] / others[-1].sum()
    assert np.allclose(first, last), "the remainder changed its internal shares"
    assert np.allclose(first, anchor[1:] / anchor[1:].sum())


def test_loglik_is_computed_only_on_assimilable_days(block: SiteData) -> None:
    """Masked days must contribute nothing, however wrong the prediction is."""
    _, _, mask = block.likelihood_arrays()
    predicted = np.where(mask, block.nee_obs, 0.0)
    baseline = gaussian_loglik(predicted, block)

    wrecked = predicted.copy()
    wrecked[~mask] = 1e6
    assert gaussian_loglik(wrecked, block) == pytest.approx(baseline)


def test_loglik_is_maximised_by_predicting_the_observations(block: SiteData) -> None:
    _, _, mask = block.likelihood_arrays()
    perfect = np.where(mask, block.nee_obs, 0.0)
    worse = perfect + np.where(mask, 0.5, 0.0)
    assert gaussian_loglik(perfect, block) > gaussian_loglik(worse, block)


def test_loglik_weights_days_by_their_own_sigma(block: SiteData) -> None:
    """The headline metric exists because a big NEE move can be invisible.

    Displacing a low-sigma day costs far more likelihood than displacing a
    high-sigma one by the same amount, which is exactly why d_loglik is the
    ranking statistic rather than delta-NEE.
    """
    # The synthetic fixture's RANDUNC spans only 0.30-0.40, too narrow to make
    # the point, so the spread is constructed here rather than borrowed.
    wide = block.nee_unc.copy()
    wide[::2] = 0.05
    wide[1::2] = 1.00
    block = replace(block, nee_unc=wide)

    observations, sigma, mask = block.likelihood_arrays()
    indices = np.flatnonzero(mask)
    order = np.argsort(sigma[indices])
    tightest, loosest = indices[order[0]], indices[order[-1]]
    assert sigma[loosest] > sigma[tightest] * 2, "need a real spread to test"

    base = np.where(mask, observations, 0.0)
    shift_tight = base.copy()
    shift_tight[tightest] += 1.0
    shift_loose = base.copy()
    shift_loose[loosest] += 1.0

    cost_tight = gaussian_loglik(base, block) - gaussian_loglik(shift_tight, block)
    cost_loose = gaussian_loglik(base, block) - gaussian_loglik(shift_loose, block)
    assert cost_tight > cost_loose
