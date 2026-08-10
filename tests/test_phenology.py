"""Tests for the DALEC2 phenology functions (published A7, A8, A9).

Source: Bloom & Williams (2015), Biogeosciences 12, 1299-1315 -- the published
paper, not the 2014 preprint.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import constant_gpp, make_drivers, make_parameters
from dalec.model_numpy import (
    ONSET_OFFSET_CONSTANT,
    PHENOLOGY_S,
    dalec2_phenology,
    phi_fall,
    phi_onset,
    run_dalec2,
)
from dalec.parameters import _psi_residual, phenology_psi_f, prior_bounds, solve_psi

DAYS_OF_YEAR = np.arange(1, 366)

#: Reference roots of A9, supplied with the equations.
PSI_REFERENCE = {
    0.125: -0.91483330,
    0.250: -0.64600950,
    0.333: -0.52742954,
    0.500: -0.35801684,
}


# ---------------------------------------------------------------------------
# The psi solver (published A9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("c_lf", "expected"), sorted(PSI_REFERENCE.items()))
def test_psi_matches_the_reference_values(c_lf: float, expected: float) -> None:
    assert solve_psi(c_lf) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("c_lf", sorted(PSI_REFERENCE))
def test_psi_residual_is_at_machine_precision(c_lf: float) -> None:
    """Stronger than matching the printed reference: the equation is satisfied."""
    assert abs(_psi_residual(solve_psi(c_lf), c_lf)) < 1e-14


def test_the_constant_s_is_a_half_period_per_year() -> None:
    assert PHENOLOGY_S == pytest.approx(365.25 / math.pi)
    assert 365.25 / PHENOLOGY_S == pytest.approx(math.pi)


@pytest.mark.parametrize("c_lf", [0.125, 0.2, 0.333, 0.5, 0.75, 0.9, 0.99, 0.999])
def test_psi_root_is_always_negative(c_lf: float) -> None:
    assert solve_psi(c_lf) < 0.0


def test_psi_is_monotone_increasing_in_c_lf_across_the_prior_range() -> None:
    """Monotone and unique, so a single fixed negative bracket always works."""
    lower, _ = prior_bounds("c_lf")
    grid = np.linspace(lower, 0.999, 300)
    roots = np.array([solve_psi(float(value)) for value in grid])
    assert np.all(np.diff(roots) > 0.0)
    assert np.isfinite(roots).all()


@pytest.mark.parametrize("c_lf", [-0.1, 0.0, 1.0, 1.5])
def test_psi_rejects_c_lf_outside_the_open_unit_interval(c_lf: float) -> None:
    """c_lf = 1 is the published upper bound but log(1 - c_lf) diverges there."""
    with pytest.raises(ValueError, match=r"c_lf must lie strictly inside \(0, 1\)"):
        solve_psi(c_lf)


def test_psi_f_scales_the_root_by_cr_fall_over_sqrt_two() -> None:
    for c_lf in (0.125, 0.5, 0.9):
        for cr_fall in (20.0, 40.0, 150.0):
            assert phenology_psi_f(c_lf, cr_fall) == pytest.approx(
                solve_psi(c_lf) * cr_fall / math.sqrt(2.0)
            )


def test_psi_f_is_negative_and_finite_across_the_prior_box() -> None:
    lower_c, _ = prior_bounds("c_lf")
    lower_cr, upper_cr = prior_bounds("cr_fall")
    for c_lf in np.linspace(lower_c, 0.999, 20):
        for cr_fall in np.linspace(lower_cr, upper_cr, 20):
            value = phenology_psi_f(float(c_lf), float(cr_fall))
            assert math.isfinite(value)
            assert value < 0.0


def test_psi_solve_is_cached() -> None:
    """It is a startup constant, not a per-timestep solve."""
    solve_psi.cache_clear()
    solve_psi(0.4321)
    before = solve_psi.cache_info()
    for _ in range(50):
        solve_psi(0.4321)
    after = solve_psi.cache_info()
    assert after.misses == before.misses
    assert after.hits == before.hits + 50


# ---------------------------------------------------------------------------
# phi_onset (published A7)
# ---------------------------------------------------------------------------


def test_phi_onset_peaks_once_a_year_near_the_onset_day() -> None:
    d_onset, cr_onset = 120.0, 30.0
    values = np.array([phi_onset(t, d_onset, cr_onset) for t in DAYS_OF_YEAR])

    peak_doy = int(DAYS_OF_YEAR[values.argmax()])
    expected = d_onset + ONSET_OFFSET_CONSTANT * cr_onset
    assert abs(peak_doy - expected) <= 1.0

    # A single pulse: the series rises then falls, with no second maximum.
    assert (np.diff(np.sign(np.diff(values))) < 0).sum() == 1


def test_phi_onset_tracks_the_onset_day() -> None:
    peaks = []
    for d_onset in (60.0, 120.0, 200.0, 300.0):
        values = np.array([phi_onset(t, d_onset, 30.0) for t in DAYS_OF_YEAR])
        peaks.append(int(DAYS_OF_YEAR[values.argmax()]))
    assert peaks == sorted(peaks)
    assert peaks[-1] - peaks[0] == pytest.approx(240, abs=2)


def test_phi_onset_peak_height_scales_inversely_with_the_release_period() -> None:
    """A shorter labile release period means a taller, narrower pulse.

    Evaluated on a sub-daily grid: sampling the narrow pulse only at integer
    days misses its true maximum by enough to blur the 1/cr_onset scaling.
    """
    fine_grid = np.linspace(1.0, 365.0, 40_000)
    narrow = max(phi_onset(t, 120.0, 10.0) for t in fine_grid)
    wide = max(phi_onset(t, 120.0, 100.0) for t in fine_grid)
    assert narrow > wide
    assert narrow / wide == pytest.approx(10.0, rel=1e-4)


def test_phi_onset_stays_within_zero_and_one_over_the_prior_box() -> None:
    lower_d, upper_d = prior_bounds("d_onset")
    lower_cr, upper_cr = prior_bounds("cr_onset")
    for d_onset in np.linspace(lower_d, upper_d, 25):
        for cr_onset in np.linspace(lower_cr, upper_cr, 25):
            values = [phi_onset(t, float(d_onset), float(cr_onset)) for t in DAYS_OF_YEAR]
            assert 0.0 <= min(values)
            assert max(values) <= 1.0


# ---------------------------------------------------------------------------
# phi_fall (published A8)
# ---------------------------------------------------------------------------


def test_d_fall_is_inert_as_transcribed() -> None:
    """The A8 sine argument is ``doy - cr_fall + psi_f``, so d_fall does nothing.

    Implemented as transcribed rather than silently corrected. This test pins
    the behaviour so the anomaly cannot be forgotten. **If it starts failing,
    the anchor was changed to d_fall -- that is the expected correction, and
    this test should be replaced by one asserting the pulse tracks d_fall.**
    """
    cr_fall, c_lf = 40.0, 0.5
    baseline = [phi_fall(t, 1.0, cr_fall, c_lf) for t in DAYS_OF_YEAR]

    for d_fall in (90.0, 200.0, 280.0, 365.0):
        assert [phi_fall(t, d_fall, cr_fall, c_lf) for t in DAYS_OF_YEAR] == baseline


def test_phi_fall_coefficient_uses_the_published_log_form() -> None:
    """Leading coefficient is -log(1 - c_lf) / cr_fall, not the preprint form."""
    cr_fall = 40.0
    for c_lf in (0.125, 0.25, 0.5, 0.9):
        peak = max(phi_fall(t, 280.0, cr_fall, c_lf) for t in DAYS_OF_YEAR)
        # The Gaussian envelope reaches ~1 at the pulse centre, so the peak is
        # the leading coefficient to within the daily sampling of the maximum.
        expected = math.sqrt(2.0 / math.pi) * (-math.log(1.0 - c_lf) / cr_fall)
        assert peak == pytest.approx(expected, rel=1e-3)


def test_phi_fall_increases_with_the_leaf_fall_fraction() -> None:
    peaks = [max(phi_fall(t, 280.0, 40.0, c) for t in DAYS_OF_YEAR)
             for c in (0.125, 0.25, 0.5, 0.9)]
    assert peaks == sorted(peaks)


def test_phi_fall_is_non_negative_and_bounded_over_the_prior_box() -> None:
    lower_cr, upper_cr = prior_bounds("cr_fall")
    lower_c, _ = prior_bounds("c_lf")
    for cr_fall in np.linspace(lower_cr, upper_cr, 20):
        for c_lf in np.linspace(lower_c, 0.99, 20):
            values = [phi_fall(t, 280.0, float(cr_fall), float(c_lf)) for t in DAYS_OF_YEAR]
            assert 0.0 <= min(values)
            assert max(values) <= 1.0


def test_phi_fall_accepts_a_precomputed_psi_f() -> None:
    cr_fall, c_lf = 40.0, 0.5
    psi_f = phenology_psi_f(c_lf, cr_fall)
    for t in (1, 100, 250, 365):
        assert phi_fall(t, 280.0, cr_fall, c_lf, psi_f=psi_f) == pytest.approx(
            phi_fall(t, 280.0, cr_fall, c_lf)
        )


# ---------------------------------------------------------------------------
# Wiring into the forward model
# ---------------------------------------------------------------------------


def test_dalec2_phenology_returns_both_fractions() -> None:
    params = make_parameters()
    onset, fall = dalec2_phenology(
        doy=140,
        d_onset=params.d_onset,
        cr_onset=params.cr_onset,
        d_fall=params.d_fall,
        cr_fall=params.cr_fall,
        c_lf=params.c_lf,
    )
    assert onset == pytest.approx(phi_onset(140, params.d_onset, params.cr_onset))
    assert fall == pytest.approx(
        phi_fall(140, params.d_fall, params.cr_fall, params.c_lf)
    )


def test_forward_run_with_real_phenology_conserves_carbon() -> None:
    """A7/A8 wired into A1, A2 and A5 must not break the carbon budget."""
    params = make_parameters()
    drivers = make_drivers(365 * 5, seed=21)

    output = run_dalec2(params, drivers, gpp_fn=constant_gpp(3.5))

    assert np.abs(output.carbon_imbalance).max() < 1e-8
    np.testing.assert_allclose(np.diff(output.total_carbon), -output.nee, atol=1e-8)
    assert output.pools.min() >= 0.0


def test_phenology_drives_labile_and_foliage_seasonally() -> None:
    """Leaf onset must actually move carbon out of labile and into foliage."""
    params = make_parameters()
    drivers = make_drivers(365, seed=22)

    output = run_dalec2(params, drivers, gpp_fn=constant_gpp(0.0))

    onset_peak = int(output.phi_onset.argmax())
    labile = output.pool("c_lab")
    foliage = output.pool("c_fol")

    # Across the onset pulse, labile falls and foliage rises by the same amount.
    window = slice(max(onset_peak - 30, 0), min(onset_peak + 30, output.n_steps))
    labile_drop = labile[window.start] - labile[window.stop]
    foliage_gain = foliage[window.stop] - foliage[window.start]
    assert labile_drop > 0.0
    # Foliage also loses carbon to litter over the window, so it gains less.
    assert 0.0 < foliage_gain < labile_drop


def test_phenology_fractions_recorded_in_the_output_match_the_functions() -> None:
    params = make_parameters()
    drivers = make_drivers(90, seed=23)

    output = run_dalec2(params, drivers, gpp_fn=constant_gpp(1.0))

    expected_onset = [phi_onset(int(d), params.d_onset, params.cr_onset) for d in drivers.doy]
    expected_fall = [
        phi_fall(int(d), params.d_fall, params.cr_fall, params.c_lf) for d in drivers.doy
    ]
    np.testing.assert_allclose(output.phi_onset, expected_onset)
    np.testing.assert_allclose(output.phi_fall, expected_fall)
