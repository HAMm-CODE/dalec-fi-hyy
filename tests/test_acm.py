"""Tests for the Aggregated Canopy Model, in the form DALEC implements.

Source: Chuter et al. (2015), Eqs. 12 and B1-B6, coefficients Appendices A and D.

The hand-computed reference below is transcribed independently from the published
equations using plain arithmetic, rather than by calling the implementation --
otherwise it would only assert that the code equals itself.

The overstatement table is a regression test with a purpose: it pins the measured
difference between the DALEC day-length term and the Williams et al. (1997) one,
so that reverting to the 1997 form cannot pass silently.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from conftest import SyntheticFluxnet, make_parameters
from dalec.acm import (
    ACM_CALIBRATION_BOUNDS,
    DEFAULT_FROST_THRESHOLD_DEGC,
    LOOBOS_EVERGREEN,
    OREGON_PONDEROSA,
    WILLIAMS_1997_COEFFICIENTS,
    AcmModel,
    acm_from_config,
    average_daily_temperature,
    chuter_day_length_factor,
    daily_temperature_half_range,
    daily_temperature_range,
    day_length_hours,
    frost_mask,
    leaf_area_index,
    make_acm,
    solar_declination_rad,
    williams1997_day_length_factor,
    williams1997_terms,
)
from dalec.config import load_config
from dalec.data_io import SiteData, build_site_data, load_fluxnet_dd
from dalec.diagnostics import calibration_bound_coverage, shoulder_season_gpp
from dalec.model_numpy import run_dalec2
from dalec.parameters import prior_bounds

#: FI-Hyy. The one site constant ACM needs.
LATITUDE = 61.8475

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
    return make_acm(latitude_deg=LATITUDE)


@pytest.fixture
def block(synthetic_fluxnet: SyntheticFluxnet) -> SiteData:
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    return build_site_data(frame, start_year=2000, end_year=2001, qc_threshold=0.75)


# ---------------------------------------------------------------------------
# Coefficient sets
# ---------------------------------------------------------------------------


def test_loobos_coefficients_match_appendix_a() -> None:
    c = LOOBOS_EVERGREEN
    assert (c.a2, c.a3, c.a4, c.a5, c.a6) == (0.0156, 4.22273, 208.868, 0.0453, 0.3783)
    assert (c.a7, c.a8, c.a9, c.a10) == (7.1929, 0.0111, 2.1001, 0.7897)
    assert (c.psi_mpa, c.r_tot) == (2.0, 1.0)


def test_oregon_coefficients_match_appendix_d() -> None:
    c = OREGON_PONDEROSA
    assert (c.a2, c.a3, c.a4, c.a5, c.a6) == (0.0142, 0.980, 217.9, 0.155, 2.653)
    assert (c.a7, c.a8, c.a9, c.a10) == (4.309, 0.060, 1.062, 0.0006)
    assert (c.psi_mpa, c.r_tot) == (0.8502, 1.0)


def test_the_two_published_sets_really_do_differ() -> None:
    """They are site-calibrated, not universal. Neither is boreal."""
    assert LOOBOS_EVERGREEN.a7 != OREGON_PONDEROSA.a7
    assert LOOBOS_EVERGREEN.a10 != OREGON_PONDEROSA.a10


def test_ceff_equivalents_bracket_the_published_prior() -> None:
    """This is what resolves the ceff scale question: the prior is vindicated."""
    lower, upper = prior_bounds("ceff")
    assert LOOBOS_EVERGREEN.ceff_equivalent == pytest.approx(29.6)
    assert OREGON_PONDEROSA.ceff_equivalent == pytest.approx(5.8185)
    # The evergreen set -- the one used here -- sits comfortably inside 10-100.
    assert lower <= LOOBOS_EVERGREEN.ceff_equivalent <= upper


def test_foliar_nitrogen_is_not_used_in_the_arithmetic(acm: AcmModel) -> None:
    """ceff replaces the p11 * N product, exactly as it replaced a1 * N."""
    baseline = acm.terms(**REFERENCE_CASE)["gpp"]
    other = make_acm(
        latitude_deg=LATITUDE,
        coefficients=type(LOOBOS_EVERGREEN)(
            **{**vars(LOOBOS_EVERGREEN), "p11": 999.0, "n_foliar": 999.0}
        ),
    )
    assert other.terms(**REFERENCE_CASE)["gpp"] == pytest.approx(float(baseline))


# ---------------------------------------------------------------------------
# Day length -- the difference that matters
# ---------------------------------------------------------------------------


def test_day_length_is_twelve_hours_at_the_equator() -> None:
    """tan(0) = 0, so the arccos argument vanishes for every day of the year."""
    for doy in range(1, 367):
        assert day_length_hours(doy, 0.0) == pytest.approx(12.0)


def test_day_length_peaks_in_summer_and_troughs_in_winter() -> None:
    """A latitude sign error is silent and severe, so this is asserted directly.

    The peak lands at doy 182, not at the true June solstice near 172 -- see
    ``test_the_published_declination_formula_lags_the_true_solstice``.
    """
    lengths = np.array([float(day_length_hours(doy, LATITUDE)) for doy in range(1, 366)])
    peak_doy = int(lengths.argmax()) + 1
    trough_doy = int(lengths.argmin()) + 1

    assert 178 <= peak_doy <= 187, f"peak at doy {peak_doy}"
    assert trough_doy >= 360 or trough_doy <= 5, f"trough at doy {trough_doy}"
    assert lengths.max() == pytest.approx(19.18, abs=0.05)
    assert lengths.min() == pytest.approx(4.82, abs=0.05)
    assert lengths.max() / lengths.min() > 3.0


def test_the_published_declination_formula_lags_the_true_solstice() -> None:
    """Chuter B5 carries no phase offset, so it runs about ten days late.

    ``delta = -0.408 * cos(2*pi*doy/365)`` peaks where the cosine is -1, at
    doy 182.5, whereas the June solstice falls near doy 172. Implemented as
    published rather than silently corrected, and pinned here so the offset is a
    recorded property of the source rather than a surprise. The consequence is a
    seasonal cycle shifted roughly ten days late; it does not affect the
    amplitude, which is what the 1997 form got wrong.
    """
    declination = solar_declination_rad(np.arange(1, 366))
    assert int(declination.argmax()) + 1 == 182
    assert int(declination.argmin()) + 1 == 365


def test_southern_hemisphere_seasons_are_inverted() -> None:
    """The sign of latitude has to reach the answer, not be swallowed."""
    assert day_length_hours(173, -LATITUDE) < day_length_hours(355, -LATITUDE)
    assert day_length_hours(173, LATITUDE) > day_length_hours(355, LATITUDE)


def test_polar_day_and_night_are_clamped_rather_than_nan() -> None:
    """Inside the polar circles the arccos argument genuinely leaves [-1, 1]."""
    assert day_length_hours(173, 80.0) == pytest.approx(24.0)
    assert day_length_hours(355, 80.0) == pytest.approx(0.0)
    assert np.isfinite(day_length_hours(np.arange(1, 367), 89.0)).all()


def test_declination_peaks_near_the_june_solstice() -> None:
    declination = solar_declination_rad(np.arange(1, 366))
    assert float(declination.max()) == pytest.approx(0.408, abs=1e-3)
    assert int(declination.argmax()) + 1 == pytest.approx(182, abs=3)


def test_at_this_latitude_the_arccos_argument_never_needs_clamping() -> None:
    """So the clamp is a guard here, not a silent modification of the answer."""
    doy = np.arange(1, 367)
    argument = -np.tan(np.radians(LATITUDE)) * np.tan(solar_declination_rad(doy))
    assert np.abs(argument).max() < 1.0


# ---------------------------------------------------------------------------
# The regression test that stops anyone reverting to the 1997 form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("doy", "williams", "dalec", "overstatement"),
    [
        (15, 1.526, 0.126, 12.1),
        (60, 1.607, 0.184, 8.7),
        (173, 1.810, 0.342, 5.3),
        (300, 1.581, 0.192, 8.3),
        (350, 1.491, 0.126, 11.8),
    ],
)
def test_the_1997_day_length_term_overstates_this_site(
    doy: int, williams: float, dalec: float, overstatement: float
) -> None:
    """Measured at FI-Hyy. The 1997 term contains no latitude at all.

    This is the dominant error and the origin of the winter GPP floor: a factor
    with no latitude in it cannot know that Hyytiala gets five hours of daylight
    in December.
    """
    measured_williams = float(williams1997_day_length_factor(doy))
    measured_dalec = float(chuter_day_length_factor(doy, LATITUDE))

    assert measured_williams == pytest.approx(williams, abs=5e-4)
    assert measured_dalec == pytest.approx(dalec, abs=5e-4)
    assert measured_williams / measured_dalec == pytest.approx(overstatement, abs=0.05)


def test_the_1997_day_length_term_carries_no_latitude() -> None:
    """The structural statement behind the table above."""
    for latitude in (0.0, 30.0, 61.8475, 80.0):
        assert float(williams1997_day_length_factor(173)) == pytest.approx(1.810, abs=5e-4)
        assert float(chuter_day_length_factor(173, latitude)) != pytest.approx(1.810)


def test_the_1997_variant_is_not_wired_into_the_forward_model(block: SiteData) -> None:
    """It is retained as thesis material, not as a code path."""
    acm = make_acm(latitude_deg=LATITUDE)
    assert not hasattr(acm, "canopy_height_m")
    assert "day_length" in acm.terms(**REFERENCE_CASE)


# ---------------------------------------------------------------------------
# The hand-computed reference
# ---------------------------------------------------------------------------


def test_reproduces_a_hand_computed_gpp(acm: AcmModel) -> None:
    """Transcribed independently from Chuter Eqs. 12 and B1-B6."""
    c = LOOBOS_EVERGREEN
    doy, t_day, t_night = 180, 20.0, 10.0
    sw_in, co2, c_fol, lma, ceff = 15.0, 400.0, 180.0, 60.0, 40.0

    # B1 -- conductance. No canopy height.
    t_range = t_day - t_night
    g_c = abs(c.psi_mpa) ** c.a10 / (0.5 * t_range + c.a6 * c.r_tot)

    # B3 -- photosynthate, on *maximum* daily temperature.
    lai = c_fol / lma
    p = ceff * lai * math.exp(c.a8 * t_day) / g_c

    # B2 -- internal CO2.
    q = c.a3 - c.a4
    discriminant = (co2 + q - p) ** 2 - 4.0 * (co2 * q - c.a3 * p)
    c_i = 0.5 * (co2 + q - p + math.sqrt(discriminant))

    # Diffusion limit, B4 quantum yield, light limitation.
    p_d = g_c * (co2 - c_i)
    e_0 = c.a7 * c_fol**2 / (c_fol**2 + c.a9 * lma**2)
    p_i = (e_0 * sw_in * p_d) / (e_0 * sw_in + p_d)

    # Eq. 12 -- true day length.
    declination = -0.408 * math.cos(2.0 * math.pi * doy / 365.0)
    s = 24.0 * math.acos(-math.tan(math.radians(LATITUDE)) * math.tan(declination)) / math.pi
    expected = p_i * (c.a2 * s + c.a5)

    assert acm(**REFERENCE_CASE) == pytest.approx(expected, rel=1e-12)


def test_intermediates_match_the_hand_computation(acm: AcmModel) -> None:
    c = LOOBOS_EVERGREEN
    terms = acm.terms(**REFERENCE_CASE)

    assert float(terms["t_max"]) == pytest.approx(20.0), "B3 takes Tmax, not the mean"
    assert float(terms["t_range"]) == pytest.approx(10.0)
    assert float(terms["lai"]) == pytest.approx(3.0)
    assert float(terms["g_c"]) == pytest.approx(
        abs(c.psi_mpa) ** c.a10 / (5.0 + c.a6 * c.r_tot)
    )
    assert float(terms["e_0"]) == pytest.approx(
        c.a7 * 180.0**2 / (180.0**2 + c.a9 * 60.0**2)
    )
    assert float(terms["day_length"]) == pytest.approx(
        float(day_length_hours(180, LATITUDE))
    )


def test_quantum_yield_is_the_lai_form_in_disguise(acm: AcmModel) -> None:
    """B4 as published is a7*Cf^2/(Cf^2 + a9*lma^2); in L it is a7*L^2/(L^2 + a9)."""
    c = LOOBOS_EVERGREEN
    terms = acm.terms(**REFERENCE_CASE)
    lai = float(terms["lai"])
    assert float(terms["e_0"]) == pytest.approx(c.a7 * lai**2 / (lai**2 + c.a9))


# ---------------------------------------------------------------------------
# Driver transforms
# ---------------------------------------------------------------------------


def test_tr_is_the_full_range_and_the_conductance_takes_half_of_it() -> None:
    """B1 divides by 0.5*Tr. Passing a half-range as Tr would halve it twice."""
    t_day, t_night = 21.0, 5.0
    assert daily_temperature_range(t_day, t_night) == pytest.approx(16.0)
    assert daily_temperature_half_range(t_day, t_night) == pytest.approx(8.0)


def test_average_daily_temperature_is_the_day_night_mean() -> None:
    assert average_daily_temperature(20.0, 10.0) == pytest.approx(15.0)


def test_photosynthesis_temperature_is_not_the_daily_mean_driver(block: SiteData) -> None:
    """A5/A6 use TA_F; ACM uses the day/night pair. They must not be conflated."""
    t_mean = average_daily_temperature(block.t_day, block.t_night)
    assert t_mean.shape == block.t_air.shape
    assert not np.shares_memory(t_mean, block.t_air)


def test_leaf_area_index_is_foliar_carbon_over_lma() -> None:
    assert leaf_area_index(180.0, 60.0) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="leaf mass per area must be positive"):
        leaf_area_index(180.0, 0.0)


# ---------------------------------------------------------------------------
# Frost cutoff
# ---------------------------------------------------------------------------


def test_gpp_is_exactly_zero_below_the_frost_threshold(acm: AcmModel) -> None:
    frozen = {**REFERENCE_CASE, "t_day": -2.0, "t_night": -8.0}
    assert acm(**frozen) == 0.0


def test_gpp_is_finite_and_positive_above_the_threshold(acm: AcmModel) -> None:
    mild = {**REFERENCE_CASE, "t_day": 3.0, "t_night": 1.0}
    value = acm(**mild)
    assert np.isfinite(value) and value > 0.0


def test_frost_mask_depends_only_on_temperature_drivers(block: SiteData) -> None:
    """Parameter-independent, so it can be precomputed and is safe under NUTS."""
    mask = frost_mask(block.t_day, block.t_night)
    assert mask.dtype == bool
    assert mask.shape == block.t_day.shape
    expected = average_daily_temperature(block.t_day, block.t_night) < DEFAULT_FROST_THRESHOLD_DEGC
    assert np.array_equal(mask, expected)


def test_a_higher_threshold_suppresses_strictly_more(block: SiteData) -> None:
    lenient = frost_mask(block.t_day, block.t_night, -10.0)
    strict = frost_mask(block.t_day, block.t_night, 5.0)
    assert np.count_nonzero(strict) >= np.count_nonzero(lenient)
    assert np.all(strict[lenient])


def test_frost_mask_zeroes_gpp_over_a_whole_record(acm: AcmModel, block: SiteData) -> None:
    params = make_parameters()
    output = run_dalec2(params, block, gpp_fn=acm)
    masked = frost_mask(block.t_day, block.t_night)
    assert np.all(output.gpp[masked] == 0.0)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_gpp_is_zero_without_foliage(acm: AcmModel) -> None:
    assert acm(**{**REFERENCE_CASE, "c_fol": 0.0}) == pytest.approx(0.0)


def test_gpp_increases_with_ceff(acm: AcmModel) -> None:
    lower, upper = prior_bounds("ceff")
    values = [acm(**{**REFERENCE_CASE, "ceff": c}) for c in np.linspace(lower, upper, 12)]
    assert all(b > a for a, b in itertools.pairwise(values))


def test_gpp_rises_with_irradiance_and_leaf_area(acm: AcmModel) -> None:
    assert acm(**{**REFERENCE_CASE, "sw_in": 20.0}) > acm(**{**REFERENCE_CASE, "sw_in": 5.0})
    assert acm(**{**REFERENCE_CASE, "c_fol": 300.0}) > acm(**{**REFERENCE_CASE, "c_fol": 60.0})


def test_direct_temperature_response_is_still_weak(acm: AcmModel) -> None:
    """Measured, and it is not what the rewrite fixed.

    Above the frost threshold, GPP moves only about 2% across 20 degrees:

        T (degC)   +20     +7      0
        GPP       3.455   3.411   3.384     (I = 2 MJ m-2 d-1)

    Temperature reaches GPP only through ``exp(a8 * Tmax)`` with a8 = 0.0111,
    and through the daily range in the B1 denominator; the light limitation
    damps both. The 1997 form gave 0.5% across 40 degrees, so this is better in
    kind but not in magnitude. **What the rewrite fixed is the seasonal
    amplitude, through day length -- not the temperature sensitivity.** Pinned
    here so it is not later reported as resolved.
    """
    low_light = {**REFERENCE_CASE, "sw_in": 2.0}
    warm = acm(**{**low_light, "t_day": 25.0, "t_night": 15.0})
    mid = acm(**{**low_light, "t_day": 12.0, "t_night": 2.0})
    cool = acm(**{**low_light, "t_day": 5.0, "t_night": -5.0})

    assert warm > mid > cool, "the response has the right sign, at least"
    assert (warm - cool) / warm < 0.05, "and it is under 5% across 20 degrees"


def test_the_seasonal_cycle_comes_from_day_length_and_light(acm: AcmModel) -> None:
    """The 1997 form capped the achievable summer/winter ratio near 8.2.

    Decomposed at this reference canopy: day length alone accounts for a 2.7x
    summer-to-winter drop and irradiance alone for 3.9x, giving 10.5x together
    with realistic winter temperatures -- before the frost mask contributes
    anything at all.
    """
    summer = acm(**{**REFERENCE_CASE, "doy": 180, "sw_in": 15.0})
    day_length_only = acm(**{**REFERENCE_CASE, "doy": 15, "sw_in": 15.0})
    light_only = acm(**{**REFERENCE_CASE, "doy": 180, "sw_in": 2.0})
    winter = acm(
        **{**REFERENCE_CASE, "doy": 15, "sw_in": 2.0, "t_day": 1.0, "t_night": -1.0}
    )

    assert summer / day_length_only == pytest.approx(2.7, abs=0.2)
    assert summer / light_only == pytest.approx(3.9, abs=0.2)
    assert summer / winter > 8.2, "the ceiling the 1997 form could not get past"


# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------


def test_discriminant_stays_positive_across_the_prior_range(block: SiteData) -> None:
    """A NaN here would poison every gradient downstream."""
    acm = make_acm(latitude_deg=LATITUDE)
    ceff_lo, ceff_hi = prior_bounds("ceff")
    lma_lo, lma_hi = prior_bounds("lma")

    for ceff in np.linspace(ceff_lo, ceff_hi, 6):
        for lma in np.linspace(lma_lo, lma_hi, 6):
            for c_fol in (20.0, 500.0, 2000.0):
                terms = acm.terms(
                    doy=block.doy, t_day=block.t_day, t_night=block.t_night,
                    sw_in=block.sw_in, co2=block.co2,
                    c_fol=np.full(block.n_days, c_fol), lma=float(lma), ceff=float(ceff),
                )
                assert (terms["discriminant"] >= 0.0).all()
                assert not terms["discriminant_clamped"].any()
                assert np.isfinite(terms["gpp"]).all()


def test_discriminant_is_non_negative_whenever_co2_exceeds_a3() -> None:
    """D > 0 for all real p exactly when C_a > a3, and a3 is 4.22 umol/mol."""
    c = LOOBOS_EVERGREEN
    q = c.a3 - c.a4
    for c_a in (300.0, 400.0, 500.0):
        for p in np.linspace(0.0, 5000.0, 400):
            assert (c_a + q - p) ** 2 - 4.0 * (c_a * q - c.a3 * p) > 0.0


def test_an_inverted_day_night_pair_is_floored_not_divided_by(acm: AcmModel) -> None:
    """Tr is a range and cannot be negative, but the day/night proxy can invert.

    At FI-Hyy this happens on 14.8% of calibration days and drives the B1
    denominator negative on 5.0% of them. The FULLSET daily product carries no
    true TA_F_MAX or TA_F_MIN, so the proxy is the only option; the range is
    floored at zero and counted rather than integrated as a physical state.
    """
    inverted = {**REFERENCE_CASE, "t_day": -50.0, "t_night": 50.0}
    terms = acm.terms(**inverted)

    assert float(terms["t_range_raw"]) == pytest.approx(-100.0)
    assert float(terms["t_range"]) == 0.0
    assert bool(terms["t_range_floored"])
    assert np.isfinite(float(terms["gpp_unmasked"]))


def test_the_floor_is_counted_and_reported_not_silent(acm: AcmModel) -> None:
    assert acm.range_floor_count == 0
    acm(**{**REFERENCE_CASE, "t_day": 5.0, "t_night": 10.0})
    assert acm.range_floor_count == 1
    acm(**REFERENCE_CASE)
    assert acm.range_floor_count == 1, "a normal day must not be counted"


def test_the_vectorised_path_warns_about_floored_ranges(acm: AcmModel) -> None:
    with pytest.warns(RuntimeWarning, match="daily temperature range was negative"):
        acm.terms(
            **{
                **REFERENCE_CASE,
                "doy": np.array([100, 101]),
                "t_day": np.array([5.0, 20.0]),
                "t_night": np.array([10.0, 10.0]),
            }
        )


def test_flooring_leaves_the_conductance_at_its_minimum(acm: AcmModel) -> None:
    """Tr = 0 gives the smallest denominator, hence the largest conductance."""
    c = LOOBOS_EVERGREEN
    terms = acm.terms(**{**REFERENCE_CASE, "t_day": 5.0, "t_night": 10.0})
    assert float(terms["g_c"]) == pytest.approx(
        abs(c.psi_mpa) ** c.a10 / (c.a6 * c.r_tot)
    )


def test_conductance_denominator_is_still_guarded(acm: AcmModel) -> None:
    """With Tr floored, only a non-positive a6*Rtot can break it."""
    broken = make_acm(
        latitude_deg=LATITUDE,
        coefficients=type(LOOBOS_EVERGREEN)(**{**vars(LOOBOS_EVERGREEN), "a6": 0.0}),
    )
    with pytest.raises(ValueError, match="B1 denominator"):
        broken.terms(**{**REFERENCE_CASE, "t_day": 5.0, "t_night": 10.0})


def test_clamp_count_starts_at_zero_and_is_inspectable(
    acm: AcmModel, block: SiteData
) -> None:
    assert acm.clamp_count == 0
    run_dalec2(make_parameters(), block, gpp_fn=acm)
    assert acm.clamp_count == 0


# ---------------------------------------------------------------------------
# Construction and config
# ---------------------------------------------------------------------------


def test_latitude_must_be_a_real_latitude() -> None:
    with pytest.raises(ValueError, match="latitude must lie"):
        make_acm(latitude_deg=120.0)


def test_config_without_a_latitude_raises() -> None:
    with pytest.raises(ValueError, match=r"site\.latitude_deg must be set"):
        acm_from_config({"site": {"latitude_deg": None}})


def test_shipped_config_builds_a_working_model() -> None:
    """Latitude is set in the shipped config, so this no longer raises."""
    acm = acm_from_config(load_config())
    assert acm.latitude_deg == pytest.approx(61.8474, abs=1e-3)
    assert acm.coefficients is LOOBOS_EVERGREEN
    assert np.isfinite(acm(**REFERENCE_CASE))


def test_canopy_height_is_no_longer_required() -> None:
    """It does not appear in the DALEC conductance equation at all."""
    config = load_config()
    assert "canopy_height_m" not in config["site"]
    assert np.isfinite(acm_from_config(config)(**REFERENCE_CASE))


def test_config_honours_the_frost_threshold() -> None:
    acm = acm_from_config(
        {"site": {"latitude_deg": LATITUDE}, "acm": {"frost_threshold_degc": 5.0}}
    )
    assert acm.frost_threshold_degc == 5.0
    assert acm(**{**REFERENCE_CASE, "t_day": 5.0, "t_night": 1.0}) == 0.0


# ---------------------------------------------------------------------------
# Wired into the forward model
# ---------------------------------------------------------------------------


def test_forward_run_with_acm_conserves_carbon(acm: AcmModel, block: SiteData) -> None:
    """Conservation is an identity and must hold for any GPP whatsoever."""
    output = run_dalec2(make_parameters(), block, gpp_fn=acm)
    assert np.abs(output.carbon_imbalance).max() < 1e-8


def test_forward_run_keeps_pools_non_negative(acm: AcmModel, block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=acm)
    assert (output.pools >= 0.0).all()


def test_run_without_a_photosynthesis_routine_points_at_make_acm(block: SiteData) -> None:
    with pytest.raises(NotImplementedError, match="make_acm"):
        run_dalec2(make_parameters(), block)


# ---------------------------------------------------------------------------
# Structural diagnostics
# ---------------------------------------------------------------------------


def test_shoulder_season_diagnostic_quantifies_the_residual_floor(
    acm: AcmModel, block: SiteData
) -> None:
    params = make_parameters()
    output = run_dalec2(params, block, gpp_fn=acm)
    table = shoulder_season_gpp(params, block, output, acm=acm)

    assert list(table.index) == [2000, 2001]
    for column in ("frost_days", "light_limited_days", "gpp_light_limited", "gpp_total"):
        assert column in table
    assert (table["gpp_light_limited"] <= table["gpp_total"] + 1e-9).all()


def test_shoulder_season_diagnostic_rejects_a_mismatched_run(
    acm: AcmModel, block: SiteData
) -> None:
    frame = load_fluxnet_dd(block.attrs["source_file"])
    shorter = build_site_data(frame, start_year=2000, end_year=2000)
    params = make_parameters()
    output = run_dalec2(params, shorter, gpp_fn=acm)

    with pytest.raises(ValueError, match="same period"):
        shoulder_season_gpp(params, block, output, acm=acm)


def test_calibration_bound_coverage_reports_extrapolation(block: SiteData) -> None:
    table = calibration_bound_coverage(block)
    assert "t_mean" in table.index
    assert ACM_CALIBRATION_BOUNDS["t_mean"] == (7.0, 30.0)
    assert 0.0 <= float(table.loc["t_mean", "fraction_outside"]) <= 1.0


# ---------------------------------------------------------------------------
# The retained 1997 variant still runs, so the comparison stays reproducible
# ---------------------------------------------------------------------------


def test_retained_1997_variant_still_evaluates() -> None:
    terms = williams1997_terms(
        **REFERENCE_CASE, canopy_height_m=18.0, psi_d_mpa=-1.5
    )
    assert np.isfinite(float(terms["gpp"]))
    assert WILLIAMS_1997_COEFFICIENTS["c1"] == 0.989


def test_the_two_forms_disagree_at_this_site(acm: AcmModel) -> None:
    """If these ever agree, something has been reverted."""
    dalec = acm(**REFERENCE_CASE)
    williams = float(
        williams1997_terms(**REFERENCE_CASE, canopy_height_m=18.0, psi_d_mpa=-1.5)["gpp"]
    )
    assert williams != pytest.approx(dalec, rel=0.1)
