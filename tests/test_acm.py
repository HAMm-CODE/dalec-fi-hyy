"""Tests for the Aggregated Canopy Model (Williams et al. 1997).

The hand-computed reference below is worked independently from the printed
equations, in solve order 2-5-6-4-8-7-9, rather than by calling the
implementation -- otherwise it would only assert that the code equals itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import SyntheticFluxnet, make_parameters
from dalec.acm import (
    ACM_CALIBRATION_BOUNDS,
    ACM_COEFFICIENTS,
    DEFAULT_FROST_THRESHOLD_DEGC,
    MIDSUMMER_DOY,
    AcmModel,
    acm_from_config,
    acm_terms,
    average_daily_temperature,
    daily_temperature_half_range,
    frost_mask,
    leaf_area_index,
    make_acm,
)
from dalec.data_io import SiteData, build_site_data, load_fluxnet_dd
from dalec.diagnostics import calibration_bound_coverage, shoulder_season_gpp
from dalec.model_numpy import run_dalec2
from dalec.parameters import prior_bounds

# Fixed site constants used throughout. These are the values the project author
# used when measuring the reference tables, not FI-Hyy values.
HEIGHT_M = 18.0
PSI_D_MPA = -1.5

#: A driver set entirely inside the Table 1 calibration bounds (T = 15 degC).
REFERENCE_CASE = {
    "doy": 180,
    "t_day": 20.0,
    "t_night": 10.0,
    "sw_in": 15.0,
    "co2": 400.0,
    "c_fol": 180.0,
    "lma": 60.0,
    "ceff": 40.0,
}


@pytest.fixture
def acm() -> AcmModel:
    return make_acm(canopy_height_m=HEIGHT_M, psi_d_mpa=PSI_D_MPA)


@pytest.fixture
def block(synthetic_fluxnet: SyntheticFluxnet) -> SiteData:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    return build_site_data(frame, start_year=2000, end_year=2001, qc_threshold=0.75)


# ---------------------------------------------------------------------------
# Coefficients and constants
# ---------------------------------------------------------------------------


def test_coefficients_match_table_2() -> None:
    assert ACM_COEFFICIENTS == {
        "a2": 0.018,
        "theta": 32.6,
        "k": 576.7,
        "b1": -0.029,
        "b2": 0.315,
        "c1": 0.989,
        "c2": 0.873,
        "d1": -0.0018,
        "d2": 1.81,
    }


def test_the_nue_parameter_is_not_among_the_coefficients_in_use() -> None:
    """DALEC2 replaces the whole a1 * N product with the sampled ceff."""
    assert "a1" not in ACM_COEFFICIENTS
    assert "n" not in ACM_COEFFICIENTS


def test_midsummer_reference_day() -> None:
    assert MIDSUMMER_DOY == 173.0


# ---------------------------------------------------------------------------
# Driver transforms
# ---------------------------------------------------------------------------


def test_d_t_is_half_the_daily_range_not_the_full_range() -> None:
    """A factor-of-two error here is silent; Table 1 defines D_T = (Tmax-Tmin)/2."""
    t_day, t_night = 21.0, 5.0
    assert daily_temperature_half_range(t_day, t_night) == pytest.approx(8.0)
    assert daily_temperature_half_range(t_day, t_night) == pytest.approx(
        0.5 * (t_day - t_night)
    )
    # Explicitly *not* the full range.
    assert daily_temperature_half_range(t_day, t_night) != pytest.approx(t_day - t_night)


def test_average_daily_temperature_is_the_day_night_mean() -> None:
    assert average_daily_temperature(20.0, 10.0) == pytest.approx(15.0)


def test_photosynthesis_temperature_is_not_the_daily_mean_driver(block: SiteData) -> None:
    """A5/A6 use TA_F; ACM uses (TA_F_DAY + TA_F_NIGHT)/2. They must not be conflated."""
    t_mean = average_daily_temperature(block.t_day, block.t_night)
    assert t_mean.shape == block.t_air.shape
    # In the synthetic file the two happen to coincide; the point is that ACM
    # reads the day/night pair, never t_air.
    assert not np.shares_memory(t_mean, block.t_air)


def test_leaf_area_index_is_foliar_carbon_over_lma() -> None:
    assert leaf_area_index(180.0, 60.0) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="leaf mass per area must be positive"):
        leaf_area_index(180.0, 0.0)


# ---------------------------------------------------------------------------
# The hand-computed reference
# ---------------------------------------------------------------------------


def test_reproduces_a_hand_computed_gpp(acm: AcmModel) -> None:
    """Worked independently from the printed equations, in solve order."""
    a2, theta, k = 0.018, 32.6, 576.7
    b1, b2 = -0.029, 0.315
    c1, c2 = 0.989, 0.873
    d1, d2 = -0.0018, 1.81

    t = 0.5 * (20.0 + 10.0)  # 15.0 degC, inside the 7-30 calibration bound
    d_t = 0.5 * (20.0 - 10.0)  # 5.0 degC, half range
    lai = 180.0 / 60.0  # 3.0
    c_a, irradiance, doy, ceff = 400.0, 15.0, 180, 40.0

    p_n = ceff * lai * math.exp(a2 * t)  # Eq 2
    g_c = (-PSI_D_MPA * math.exp(b1 * t)) / (b2 * HEIGHT_M + d_t)  # Eq 5
    q = theta - k  # Eq 6
    p = p_n / g_c
    c_i = 0.5 * (
        c_a + q - p + math.sqrt((c_a + q - p) ** 2 - 4.0 * (c_a * q - p * theta))
    )
    p_d = g_c * (c_a - c_i)  # Eq 4
    e_0 = c1 * lai**2 / (c2 + lai**2)  # Eq 8
    p_i = (e_0 * irradiance * p_d) / (e_0 * irradiance + p_d)  # Eq 7
    expected = p_i * (d1 * abs(173.0 - doy) + d2)  # Eq 9

    assert acm(**REFERENCE_CASE) == pytest.approx(expected, rel=1e-12)
    # And the independently reported value for these conditions.
    assert acm(**REFERENCE_CASE) == pytest.approx(15.543, abs=5e-4)


def test_intermediates_match_the_hand_computation(acm: AcmModel) -> None:
    terms = acm.terms(**REFERENCE_CASE)
    assert float(terms["t_mean"]) == pytest.approx(15.0)
    assert float(terms["d_t"]) == pytest.approx(5.0)
    assert float(terms["lai"]) == pytest.approx(3.0)
    assert float(terms["e_0"]) == pytest.approx(0.989 * 9.0 / (0.873 + 9.0))
    assert float(terms["d_ms"]) == pytest.approx(7.0)
    assert float(terms["day_length_factor"]) == pytest.approx(-0.0018 * 7.0 + 1.81)


def test_canopy_quantum_yield_matches_the_reference_column(acm: AcmModel) -> None:
    """Eq. 8 against independently measured E_0 values."""
    for lai, expected in [(0.5, 0.220), (1.0, 0.528), (2.0, 0.812),
                          (3.0, 0.902), (6.0, 0.966)]:
        terms = acm.terms(**{**REFERENCE_CASE, "c_fol": lai * 60.0})
        assert float(terms["e_0"]) == pytest.approx(expected, abs=5e-4)


# ---------------------------------------------------------------------------
# Frost cutoff
# ---------------------------------------------------------------------------


def test_gpp_is_exactly_zero_below_the_frost_threshold(acm: AcmModel) -> None:
    frozen = {**REFERENCE_CASE, "t_day": -2.0, "t_night": -8.0}  # T = -5 degC
    assert average_daily_temperature(-2.0, -8.0) < DEFAULT_FROST_THRESHOLD_DEGC
    assert acm(**frozen) == 0.0


def test_gpp_is_finite_and_positive_above_the_threshold(acm: AcmModel) -> None:
    thawed = {**REFERENCE_CASE, "t_day": 2.0, "t_night": -2.0}  # T = 0 degC
    value = acm(**thawed)
    assert math.isfinite(value)
    assert value > 0.0


def test_frost_mask_depends_only_on_temperature_drivers(block: SiteData) -> None:
    """Parameter-independent, so it is a fixed mask NUTS never has to traverse."""
    baseline = frost_mask(block.t_day, block.t_night)
    assert baseline.dtype == bool
    assert baseline.shape == (block.n_days,)
    # Recomputing it cannot depend on ceff, lma or foliar carbon: those are not
    # even arguments.
    np.testing.assert_array_equal(baseline, frost_mask(block.t_day, block.t_night))


def test_frost_mask_zeroes_gpp_over_a_whole_record(acm: AcmModel, block: SiteData) -> None:
    terms = acm.terms(
        doy=block.doy, t_day=block.t_day, t_night=block.t_night,
        sw_in=block.sw_in, co2=block.co2, c_fol=np.full(block.n_days, 180.0),
        lma=60.0, ceff=40.0,
    )
    frozen = terms["frost_masked"]
    assert frozen.any(), "the synthetic record should contain some frost days"
    assert np.all(terms["gpp"][frozen] == 0.0)
    assert np.all(np.isfinite(terms["gpp"]))
    assert np.all(terms["gpp"][~frozen] > 0.0)


def test_the_masked_days_would_otherwise_emit_the_floor(acm: AcmModel, block: SiteData) -> None:
    """The cutoff is load-bearing: unmasked, frost days still produce carbon."""
    terms = acm.terms(
        doy=block.doy, t_day=block.t_day, t_night=block.t_night,
        sw_in=block.sw_in, co2=block.co2, c_fol=np.full(block.n_days, 180.0),
        lma=60.0, ceff=40.0,
    )
    frozen = terms["frost_masked"]
    suppressed = float(terms["gpp_unmasked"][frozen].sum())
    assert suppressed > 0.0


def test_a_higher_threshold_suppresses_strictly_more(block: SiteData) -> None:
    strict = make_acm(canopy_height_m=HEIGHT_M, psi_d_mpa=PSI_D_MPA, frost_threshold_degc=5.0)
    loose = make_acm(canopy_height_m=HEIGHT_M, psi_d_mpa=PSI_D_MPA, frost_threshold_degc=-20.0)
    shared = {
        "doy": block.doy, "t_day": block.t_day, "t_night": block.t_night,
        "sw_in": block.sw_in, "co2": block.co2,
        "c_fol": np.full(block.n_days, 180.0), "lma": 60.0, "ceff": 40.0,
    }
    assert strict.terms(**shared)["gpp"].sum() < loose.terms(**shared)["gpp"].sum()


# ---------------------------------------------------------------------------
# Structural behaviour
# ---------------------------------------------------------------------------


def test_gpp_is_zero_without_foliage(acm: AcmModel) -> None:
    """No leaf area means no photosynthesis, and no 0/0 in Eq. 7."""
    value = acm(**{**REFERENCE_CASE, "c_fol": 0.0})
    assert value == 0.0
    assert math.isfinite(value)


def test_ceff_response_magnitude_not_merely_its_direction(acm: AcmModel) -> None:
    """Monotonicity passes while ceff is nearly inert, so assert the size.

    Across the full 10-100 prior range the measured response is about 1.53x.
    A correct ``a1 * N`` substitution would move GPP far more, so this fails if
    the reading of ceff is wrong -- which a monotonicity check would not.
    """
    lower, upper = prior_bounds("ceff")
    low = acm(**{**REFERENCE_CASE, "ceff": lower})
    high = acm(**{**REFERENCE_CASE, "ceff": upper})

    assert high > low
    ratio = high / low
    assert ratio == pytest.approx(1.53, abs=0.05), (
        f"a tenfold ceff change moved GPP by {ratio:.2f}x; if this has shifted, "
        "the ceff substitution has changed"
    )


def test_ceff_response_saturates_hard(acm: AcmModel) -> None:
    """Sensitivity per unit ceff collapses above 20, by roughly an order of magnitude.

    Stated as a slope, not a total increment: the 20-100 interval is eight times
    wider than 10-20, so it accumulates a comparable total while being far less
    responsive. The slope is what makes the parameter nearly inert up there.
    """
    at_10 = acm(**{**REFERENCE_CASE, "ceff": 10.0})
    at_20 = acm(**{**REFERENCE_CASE, "ceff": 20.0})
    at_100 = acm(**{**REFERENCE_CASE, "ceff": 100.0})

    slope_below = (at_20 - at_10) / 10.0
    slope_above = (at_100 - at_20) / 80.0
    assert slope_below > 5.0 * slope_above


def test_gpp_rises_with_irradiance_and_leaf_area(acm: AcmModel) -> None:
    assert acm(**{**REFERENCE_CASE, "sw_in": 20.0}) > acm(**REFERENCE_CASE)
    assert acm(**{**REFERENCE_CASE, "c_fol": 300.0}) > acm(**REFERENCE_CASE)


def test_day_length_correction_peaks_at_midsummer(acm: AcmModel) -> None:
    midsummer = acm(**{**REFERENCE_CASE, "doy": int(MIDSUMMER_DOY)})
    for doy in (1, 60, 300, 365):
        assert acm(**{**REFERENCE_CASE, "doy": doy}) < midsummer


def test_low_irradiance_gpp_is_nearly_temperature_independent(acm: AcmModel) -> None:
    """The structural insensitivity the frost cutoff exists to mask.

    Recorded as a test so the limitation is measured, not merely asserted in
    prose. The frost mask is disabled here to expose the bare behaviour.
    """
    unmasked = make_acm(
        canopy_height_m=HEIGHT_M, psi_d_mpa=PSI_D_MPA, frost_threshold_degc=-999.0
    )
    dim = {**REFERENCE_CASE, "sw_in": 2.0, "doy": 30}
    values = [
        unmasked(**{**dim, "t_day": t + 3.0, "t_night": t - 3.0})
        for t in (20.0, 7.0, 0.0, -10.0, -20.0)
    ]
    spread = (max(values) - min(values)) / max(values)
    assert spread < 0.02, f"expected near-total insensitivity, got {spread:.3%}"


# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------


def test_discriminant_stays_positive_across_the_prior_range(block: SiteData) -> None:
    """Sweep the parameter box and the whole driver record.

    A negative Eq. 6 discriminant yields NaN and poisons every gradient
    downstream. It cannot happen: with ``q = theta - k = -544.1 < 0``,
    ``C_a > 0`` and ``p >= 0``, the term ``C_a*q - p*theta`` is strictly
    negative, so ``-4(C_a*q - p*theta)`` is strictly positive and the
    discriminant is a square plus a positive number. The sweep confirms it
    empirically over the real ranges.
    """
    ceff_lo, ceff_hi = prior_bounds("ceff")
    lma_lo, lma_hi = prior_bounds("lma")
    fol_lo, fol_hi = prior_bounds("c_fol_0")

    worst = np.inf
    for ceff in (ceff_lo, ceff_hi):
        for lma in (lma_lo, lma_hi):
            for c_fol in (0.0, fol_lo, fol_hi):
                terms = acm_terms(
                    doy=block.doy, t_day=block.t_day, t_night=block.t_night,
                    sw_in=block.sw_in, co2=block.co2,
                    c_fol=np.full(block.n_days, c_fol), lma=lma, ceff=ceff,
                    canopy_height_m=HEIGHT_M, psi_d_mpa=PSI_D_MPA,
                )
                assert not terms["discriminant_clamped"].any()
                worst = min(worst, float(terms["discriminant"].min()))
                assert np.isfinite(terms["gpp"]).all()

    assert worst > 0.0


def test_discriminant_is_non_negative_for_every_reachable_p() -> None:
    """The guard is provably unreachable, not merely untriggered in practice.

    Written as a quadratic in ``p = p_N / g_c``::

        D(p) = p^2 - 2p(C_a - theta - k) + (C_a - theta + k)^2

    whose own discriminant is negative -- so ``D > 0`` for every real ``p`` --
    exactly when ``(C_a - theta) * k > 0``, i.e. ``C_a > theta = 32.6``. Below
    the compensation point the negative window exists but lies entirely at
    ``p < 0``, and ``p`` cannot be negative because ``p_N >= 0`` and
    ``g_c > 0``.
    """
    theta, k = ACM_COEFFICIENTS["theta"], ACM_COEFFICIENTS["k"]
    q = theta - k

    c_a_grid = np.linspace(0.1, 2000.0, 400)[:, None]
    p_grid = np.linspace(0.0, 500_000.0, 400)[None, :]
    discriminant = (c_a_grid + q - p_grid) ** 2 - 4.0 * (c_a_grid * q - p_grid * theta)

    assert discriminant.min() > 0.0


def test_a_negative_discriminant_would_warn_rather_than_clamp_silently() -> None:
    """Guard the guard: unreachable is not the same as unwired.

    Forcing it needs a negative ``ceff``, which is not a valid parameter -- that
    is the point. If the guard ever does fire on real input, it must be loud.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        terms = acm_terms(
            doy=180, t_day=20.0, t_night=10.0, sw_in=15.0, co2=10.0,
            c_fol=180.0, lma=60.0, ceff=-14.0,
            canopy_height_m=HEIGHT_M, psi_d_mpa=PSI_D_MPA,
        )

    assert terms["discriminant_clamped"].any()
    assert any("discriminant" in str(warning.message) for warning in caught)
    assert np.isfinite(terms["gpp"]).all()


def test_conductance_denominator_is_guarded() -> None:
    """b2*H + D_T must stay away from zero."""
    with pytest.raises(ValueError, match=r"Eq. 5 denominator"):
        acm_terms(
            **{**REFERENCE_CASE, "t_day": -50.0, "t_night": 50.0},
            canopy_height_m=1.0,
            psi_d_mpa=PSI_D_MPA,
        )


def test_clamp_count_starts_at_zero_and_is_inspectable(acm: AcmModel, block: SiteData) -> None:
    assert acm.clamp_count == 0
    for index in range(20):
        acm(
            doy=int(block.doy[index]), t_day=float(block.t_day[index]),
            t_night=float(block.t_night[index]), sw_in=float(block.sw_in[index]),
            co2=float(block.co2[index]), c_fol=180.0, lma=60.0, ceff=40.0,
        )
    assert acm.clamp_count == 0


# ---------------------------------------------------------------------------
# Site constants and config wiring
# ---------------------------------------------------------------------------


def test_canopy_height_must_be_positive() -> None:
    with pytest.raises(ValueError, match="canopy height must be positive"):
        make_acm(canopy_height_m=0.0, psi_d_mpa=PSI_D_MPA)


def test_psi_d_must_be_negative() -> None:
    with pytest.raises(ValueError, match="must be negative"):
        make_acm(canopy_height_m=HEIGHT_M, psi_d_mpa=1.5)


@pytest.mark.parametrize("missing", ["canopy_height_m", "psi_d_mpa"])
def test_config_without_a_site_constant_raises(missing: str) -> None:
    site = {"canopy_height_m": HEIGHT_M, "psi_d_mpa": PSI_D_MPA}
    site[missing] = None
    with pytest.raises(ValueError, match=f"site.{missing}"):
        acm_from_config({"site": site})


def test_shipped_config_leaves_the_site_constants_unset() -> None:
    """They come from site literature; the config must not guess them."""
    from dalec.config import load_config

    config = load_config()
    assert config["site"]["canopy_height_m"] is None
    assert config["site"]["psi_d_mpa"] is None
    assert config["acm"]["frost_threshold_degc"] == DEFAULT_FROST_THRESHOLD_DEGC
    with pytest.raises(ValueError, match=r"site\."):
        acm_from_config(config)


def test_config_builds_a_working_model() -> None:
    model = acm_from_config(
        {
            "site": {"canopy_height_m": HEIGHT_M, "psi_d_mpa": PSI_D_MPA},
            "acm": {"frost_threshold_degc": -3.5},
        }
    )
    assert model.canopy_height_m == HEIGHT_M
    assert model.psi_d_mpa == PSI_D_MPA
    assert model.frost_threshold_degc == -3.5
    assert model(**REFERENCE_CASE) > 0.0


# ---------------------------------------------------------------------------
# Wired into the forward model
# ---------------------------------------------------------------------------


def test_forward_run_with_acm_conserves_carbon(acm: AcmModel, block: SiteData) -> None:
    """The last blocker closed: A1-A9 running end to end on real drivers."""
    params = make_parameters(lma=60.0, ceff=40.0)
    output = run_dalec2(params, block, gpp_fn=acm)

    assert np.abs(output.carbon_imbalance).max() < 1e-8
    np.testing.assert_allclose(np.diff(output.total_carbon), -output.nee, atol=1e-8)
    assert np.isfinite(output.gpp).all()
    assert (output.gpp >= 0.0).all()
    assert output.gpp.max() > 0.0


def test_forward_run_gpp_is_zero_on_frost_days(acm: AcmModel, block: SiteData) -> None:
    params = make_parameters(lma=60.0, ceff=40.0)
    output = run_dalec2(params, block, gpp_fn=acm)
    frozen = frost_mask(block.t_day, block.t_night)

    assert np.all(output.gpp[frozen] == 0.0)


def test_run_without_a_photosynthesis_routine_points_at_make_acm(block: SiteData) -> None:
    with pytest.raises(NotImplementedError, match="make_acm"):
        run_dalec2(make_parameters(), block)


# ---------------------------------------------------------------------------
# Structural-bias diagnostics
# ---------------------------------------------------------------------------


def test_shoulder_season_diagnostic_quantifies_the_residual_floor(
    acm: AcmModel, block: SiteData
) -> None:
    params = make_parameters(lma=60.0, ceff=40.0)
    output = run_dalec2(params, block, gpp_fn=acm)
    table = shoulder_season_gpp(params, block, output, acm=acm)

    assert list(table.index) == [2000, 2001]
    for year in table.index:
        row = table.loc[year]
        assert row["frost_days"] >= 0
        assert row["light_limited_days"] >= 0
        # Frost days and light-limited-unmasked days are disjoint by definition.
        assert row["frost_days"] + row["light_limited_days"] <= row["days"]
        assert 0.0 <= row["fraction_light_limited"] <= 1.0 + 1e-9
        assert row["gpp_light_limited"] <= row["gpp_total"] + 1e-9


def test_shoulder_season_diagnostic_rejects_a_mismatched_run(
    acm: AcmModel, block: SiteData
) -> None:
    params = make_parameters(lma=60.0, ceff=40.0)
    frame = load_fluxnet_dd(block.attrs["source_file"])
    shorter = build_site_data(frame, start_year=2000, end_year=2000)
    output = run_dalec2(params, shorter, gpp_fn=acm)

    with pytest.raises(ValueError, match="same period"):
        shoulder_season_gpp(params, block, output, acm=acm)


def test_calibration_bound_coverage_reports_extrapolation(block: SiteData) -> None:
    table = calibration_bound_coverage(block)

    assert "t_mean" in table.index
    row = table.loc["t_mean"]
    assert (row["lower"], row["upper"]) == ACM_CALIBRATION_BOUNDS["t_mean"]
    assert row["n_days"] == block.n_days
    assert row["n_below"] + row["n_above"] <= row["n_days"]
    assert 0.0 <= row["fraction_outside"] <= 1.0
    # A boreal record spends most of the year below the 7 degC calibration floor.
    assert row["n_below"] > 0
