"""Tests for the NumPy DALEC2 forward model (Bloom & Williams 2015, A1-A6).

The sharpest test here is carbon conservation. Summing A1-A6 the internal
transfers cancel exactly, leaving ``d(total C) = GPP - Reco = -NEE`` with no
discretisation error, so any mistake that creates or destroys carbon -- a
dropped term, a transposed allocation, a sequential rather than simultaneous
update -- shows up as a non-zero residual.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from conftest import (
    DEFAULT_ALLOCATION_WEIGHTS,
    constant_gpp,
    constant_phenology,
    make_drivers,
    make_parameters,
    seasonal_phenology,
)
from dalec.model_numpy import (
    N_POOLS,
    POOL_NAMES,
    dalec2_step,
    gpp_not_implemented,
    run_dalec2,
)
from dalec.parameters import DalecParameters, allocation_fractions

# Total carbon is order 20 000 g C m-2, so a float64 difference of totals
# carries roughly 1e-11 of rounding. Anything above this is a real leak.
CONSERVATION_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# The equations themselves
# ---------------------------------------------------------------------------


def test_single_step_matches_the_appendix_by_hand() -> None:
    """One step against values worked out from A1-A6 with pen and paper."""
    params = make_parameters(
        f_auto=0.5,
        allocation_weights=(0.10, 0.30, 0.20, 0.40),  # -> 0.05, 0.15, 0.10, 0.20
        theta_roo=0.002,
        theta_woo=0.0001,
        theta_lit=0.01,
        theta_som=0.0001,
        theta_min=0.005,
        temperature_exponent=0.05,
    )
    state = np.array([10.0, 100.0, 50.0, 1000.0, 200.0, 5000.0])
    gpp, phi_onset, phi_fall, t_air = 4.0, 0.02, 0.003, 10.0

    next_state, (out_gpp, ra, rh, nee) = dalec2_step(
        state, gpp=gpp, phi_onset=phi_onset, phi_fall=phi_fall, t_air=t_air, params=params
    )

    rate = math.exp(0.05 * 10.0)  # exp(Theta * T) = exp(0.5)

    # A1  (1 - 0.02) * 10 + 0.05 * 4        = 9.8   + 0.2       = 10.0
    # A2  (1 - 0.003) * 100 + 0.02 * 10 + 0.15 * 4               = 100.5
    # A3  (1 - 0.002) * 50 + 0.10 * 4      = 49.9  + 0.4        = 50.3
    # A4  (1 - 0.0001) * 1000 + 0.20 * 4   = 999.9 + 0.8        = 1000.7
    assert next_state[0] == pytest.approx(10.0)
    assert next_state[1] == pytest.approx(100.5)
    assert next_state[2] == pytest.approx(50.3)
    assert next_state[3] == pytest.approx(1000.7)

    # A5 and A6 carry the temperature multiplier, written out from the appendix.
    assert next_state[4] == pytest.approx(
        (1.0 - (0.01 + 0.005) * rate) * 200.0 + 0.002 * 50.0 + 0.003 * 100.0
    )
    assert next_state[5] == pytest.approx(
        (1.0 - 0.0001 * rate) * 5000.0 + 0.0001 * 1000.0 + 0.005 * rate * 200.0
    )

    assert out_gpp == pytest.approx(4.0)
    assert ra == pytest.approx(0.5 * 4.0)
    assert rh == pytest.approx((0.01 * 200.0 + 0.0001 * 5000.0) * rate)
    assert nee == pytest.approx(ra + rh - gpp)

    # And the budget closes on the hand-worked numbers too.
    assert float(next_state.sum() - state.sum()) == pytest.approx(gpp - (ra + rh))


def test_update_is_simultaneous_not_sequential() -> None:
    """Every right-hand side must read the time-t state.

    With half the labile pool released and no GPP, foliage must receive the
    whole 50 g C. A sequential implementation that overwrote ``C_lab`` before
    evaluating A2 would pass on only 25 and quietly destroy the rest.
    """
    params = make_parameters(theta_roo=0.0, theta_woo=0.0, theta_lit=0.0, theta_som=0.0,
                             theta_min=0.0)
    state = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    next_state, _ = dalec2_step(
        state, gpp=0.0, phi_onset=0.5, phi_fall=0.5, t_air=10.0, params=params
    )

    assert next_state[0] == pytest.approx(50.0)
    assert next_state[1] == pytest.approx(50.0)
    assert float(next_state.sum()) == pytest.approx(100.0)


def test_step_does_not_mutate_the_input_state() -> None:
    params = make_parameters()
    state = np.array([10.0, 100.0, 150.0, 8000.0, 200.0, 12000.0])
    before = state.copy()

    dalec2_step(state, gpp=5.0, phi_onset=0.01, phi_fall=0.01, t_air=12.0, params=params)

    np.testing.assert_array_equal(state, before)


def test_theta_min_is_a_transfer_not_a_respiration_loss() -> None:
    """Mineralised litter moves to soil; it must not appear in Reco."""
    params = make_parameters(
        theta_roo=0.0,
        theta_woo=0.0,
        theta_lit=0.0,
        theta_som=0.0,
        theta_min=0.01,
    )
    state = np.array([0.0, 0.0, 0.0, 0.0, 500.0, 1000.0])

    next_state, (_, ra, rh, nee) = dalec2_step(
        state, gpp=0.0, phi_onset=0.0, phi_fall=0.0, t_air=15.0, params=params
    )

    rate = math.exp(params.temperature_exponent * 15.0)
    transferred = 0.01 * rate * 500.0

    assert rh == pytest.approx(0.0)
    assert ra == pytest.approx(0.0)
    assert nee == pytest.approx(0.0)
    assert next_state[4] == pytest.approx(500.0 - transferred)
    assert next_state[5] == pytest.approx(1000.0 + transferred)
    assert float(next_state.sum()) == pytest.approx(1500.0)


def test_zero_gpp_and_zero_rates_is_a_fixed_point() -> None:
    params = make_parameters(
        theta_roo=0.0, theta_woo=0.0, theta_lit=0.0, theta_som=0.0, theta_min=0.0
    )
    state = np.array([10.0, 100.0, 150.0, 8000.0, 200.0, 12000.0])

    next_state, (_, ra, rh, nee) = dalec2_step(
        state, gpp=0.0, phi_onset=0.0, phi_fall=0.0, t_air=20.0, params=params
    )

    np.testing.assert_allclose(next_state, state)
    assert (ra, rh, nee) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Carbon conservation
# ---------------------------------------------------------------------------


def test_carbon_is_conserved_every_timestep() -> None:
    """d(total C) - (GPP - Reco) must be zero at every step, not just on average."""
    params = make_parameters()
    drivers = make_drivers(365 * 3, seed=1)

    output = run_dalec2(
        params,
        drivers,
        gpp_fn=constant_gpp(4.0),
        phenology_fn=seasonal_phenology(),
    )

    assert np.abs(output.carbon_imbalance).max() < CONSERVATION_TOLERANCE


def test_change_in_total_carbon_equals_minus_nee() -> None:
    """The same identity stated the way the thesis will state it."""
    params = make_parameters()
    drivers = make_drivers(400, seed=2)

    output = run_dalec2(
        params, drivers, gpp_fn=constant_gpp(3.2), phenology_fn=seasonal_phenology()
    )

    np.testing.assert_allclose(
        np.diff(output.total_carbon), -output.nee, atol=CONSERVATION_TOLERANCE
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_carbon_is_conserved_for_random_parameters_and_fluxes(seed: int) -> None:
    """Conservation is structural: it must hold for arbitrary rates and doubles."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(4))
    params = make_parameters(
        f_auto=float(rng.uniform(0.3, 0.7)),
        allocation_weights=tuple(weights),
        theta_roo=float(rng.uniform(1e-4, 1e-2)),
        theta_woo=float(rng.uniform(1e-6, 1e-3)),
        theta_lit=float(rng.uniform(1e-4, 2e-2)),
        theta_som=float(rng.uniform(1e-7, 1e-3)),
        theta_min=float(rng.uniform(1e-4, 1e-2)),
        temperature_exponent=float(rng.uniform(0.01, 0.09)),
    )
    drivers = make_drivers(500, seed=seed)

    # Random GPP and random phenology: conservation does not care about either.
    # Each double keeps its own counter -- run_dalec2 calls phenology before GPP,
    # so a shared one would desynchronise.
    gpp_series = rng.uniform(0.0, 8.0, size=500)
    onset_series = rng.uniform(0.0, 0.2, size=500)
    fall_series = rng.uniform(0.0, 0.2, size=500)
    gpp_steps = itertools.count()
    phenology_steps = itertools.count()

    def gpp_fn(**_: object) -> float:
        return float(gpp_series[next(gpp_steps)])

    def phenology_fn(**_: object) -> tuple[float, float]:
        index = next(phenology_steps)
        return float(onset_series[index]), float(fall_series[index])

    output = run_dalec2(params, drivers, gpp_fn=gpp_fn, phenology_fn=phenology_fn)

    assert np.abs(output.carbon_imbalance).max() < CONSERVATION_TOLERANCE


def test_reco_decomposes_into_autotrophic_and_heterotrophic() -> None:
    params = make_parameters()
    drivers = make_drivers(200, seed=3)

    output = run_dalec2(
        params, drivers, gpp_fn=constant_gpp(5.0), phenology_fn=constant_phenology(0.01, 0.005)
    )

    np.testing.assert_allclose(output.reco, output.ra + output.rh)
    np.testing.assert_allclose(output.nee, output.reco - output.gpp)
    np.testing.assert_allclose(output.ra, params.f_auto * output.gpp)


# ---------------------------------------------------------------------------
# Long runs and non-negativity
# ---------------------------------------------------------------------------


def test_pools_stay_non_negative_over_a_long_run() -> None:
    """Twenty years of daily steps under a stable parameter set."""
    params = make_parameters()
    drivers = make_drivers(365 * 20, seed=4)

    output = run_dalec2(
        params, drivers, gpp_fn=constant_gpp(3.0), phenology_fn=seasonal_phenology()
    )

    assert output.pools.min() >= 0.0
    assert np.isfinite(output.pools).all()
    assert np.isfinite(output.nee).all()


def test_pools_decay_monotonically_without_input() -> None:
    """No GPP and no phenology: every pool that only loses carbon must shrink."""
    params = make_parameters()
    drivers = make_drivers(365 * 5, seed=5, constant_t_air=10.0)

    output = run_dalec2(
        params, drivers, gpp_fn=constant_gpp(0.0), phenology_fn=constant_phenology(0.0, 0.0)
    )

    for name in ("c_roo", "c_woo"):
        trajectory = output.pool(name)
        assert np.all(np.diff(trajectory) < 0.0), name
    assert output.total_carbon[-1] < output.total_carbon[0]
    # Losing carbon means positive NEE throughout.
    assert np.all(output.nee > 0.0)
    assert np.abs(output.carbon_imbalance).max() < CONSERVATION_TOLERANCE


def test_full_labile_release_empties_the_labile_pool() -> None:
    params = make_parameters(f_auto=1.0, allocation_weights=(0.25, 0.25, 0.25, 0.25))
    drivers = make_drivers(1, seed=6)

    output = run_dalec2(
        params, drivers, gpp_fn=constant_gpp(6.0), phenology_fn=constant_phenology(1.0, 0.0)
    )

    assert output.pool("c_lab")[1] == pytest.approx(0.0)
    assert output.pool("c_fol")[1] == pytest.approx(
        params.c_fol_0 + params.c_lab_0
    )
    # f_auto = 1 means every gram of GPP is respired straight back.
    assert output.ra[0] == pytest.approx(output.gpp[0])


# ---------------------------------------------------------------------------
# Wiring: drivers, output shapes, injected components
# ---------------------------------------------------------------------------


def test_output_shapes_and_initial_row() -> None:
    params = make_parameters()
    n_days = 120
    drivers = make_drivers(n_days, seed=7)

    output = run_dalec2(
        params, drivers, gpp_fn=constant_gpp(2.0), phenology_fn=constant_phenology(0.01, 0.01)
    )

    assert output.n_steps == n_days
    assert output.pools.shape == (n_days + 1, N_POOLS)
    np.testing.assert_allclose(output.pools[0], params.initial_pools)
    for series in (output.gpp, output.ra, output.rh, output.reco, output.nee,
                   output.phi_onset, output.phi_fall):
        assert series.shape == (n_days,)


def test_pool_accessor_matches_column_order() -> None:
    params = make_parameters()
    drivers = make_drivers(10, seed=8)
    output = run_dalec2(
        params, drivers, gpp_fn=constant_gpp(1.0), phenology_fn=constant_phenology(0.0, 0.0)
    )

    for index, name in enumerate(POOL_NAMES):
        np.testing.assert_array_equal(output.pool(name), output.pools[:, index])

    with pytest.raises(KeyError, match="unknown pool"):
        output.pool("c_nonsense")


def test_photosynthesis_receives_time_t_foliar_carbon() -> None:
    """GPP must see C_fol(t), consistent with the simultaneous update."""
    params = make_parameters()
    drivers = make_drivers(30, seed=9)
    seen: list[float] = []

    def recording_gpp(*, c_fol: float, **_: object) -> float:
        seen.append(c_fol)
        return 2.0

    output = run_dalec2(
        params, drivers, gpp_fn=recording_gpp, phenology_fn=constant_phenology(0.02, 0.01)
    )

    np.testing.assert_allclose(np.array(seen), output.pool("c_fol")[:-1])


def test_photosynthesis_receives_the_drivers_and_acm_parameters() -> None:
    params = make_parameters()
    drivers = make_drivers(3, seed=10)
    calls: list[dict[str, object]] = []

    def recording_gpp(**kwargs: object) -> float:
        calls.append(kwargs)
        return 1.0

    run_dalec2(params, drivers, gpp_fn=recording_gpp,
               phenology_fn=constant_phenology(0.0, 0.0))

    first = calls[0]
    assert first["doy"] == int(drivers.doy[0])
    assert first["t_max"] == pytest.approx(drivers.t_max[0])
    assert first["t_min"] == pytest.approx(drivers.t_min[0])
    assert first["sw_in"] == pytest.approx(drivers.sw_in[0])
    assert first["co2"] == pytest.approx(drivers.co2[0])
    assert first["lma"] == params.lma
    assert first["ceff"] == params.ceff
    # VPD is out of scope and must never be handed to photosynthesis.
    assert "vpd" not in first


def test_phenology_receives_its_five_parameters() -> None:
    params = make_parameters()
    drivers = make_drivers(2, seed=11)
    calls: list[dict[str, object]] = []

    def recording_phenology(**kwargs: object) -> tuple[float, float]:
        calls.append(kwargs)
        return 0.0, 0.0

    run_dalec2(params, drivers, gpp_fn=constant_gpp(0.0), phenology_fn=recording_phenology)

    first = calls[0]
    assert first["doy"] == int(drivers.doy[0])
    for name in ("d_onset", "cr_onset", "d_fall", "cr_fall", "c_lf"):
        assert first[name] == getattr(params, name)


def test_decomposition_uses_daily_mean_air_temperature() -> None:
    """A5/A6 scale with TA_F, not with the daytime/nighttime ACM inputs."""
    params = make_parameters()
    warm = make_drivers(50, seed=12, constant_t_air=20.0)
    cold = make_drivers(50, seed=12, constant_t_air=0.0)

    warm_out = run_dalec2(params, warm, gpp_fn=constant_gpp(0.0),
                          phenology_fn=constant_phenology(0.0, 0.0))
    cold_out = run_dalec2(params, cold, gpp_fn=constant_gpp(0.0),
                          phenology_fn=constant_phenology(0.0, 0.0))

    assert warm_out.rh[0] > cold_out.rh[0]
    assert warm_out.rh[0] == pytest.approx(
        cold_out.rh[0] * math.exp(params.temperature_exponent * 20.0)
    )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_unimplemented_photosynthesis_raises() -> None:
    params = make_parameters()
    drivers = make_drivers(5, seed=13)

    with pytest.raises(NotImplementedError, match="Aggregated Canopy Model"):
        run_dalec2(params, drivers, phenology_fn=constant_phenology(0.0, 0.0))

    with pytest.raises(NotImplementedError):
        gpp_not_implemented(
            doy=1, t_max=1.0, t_min=0.0, sw_in=1.0, co2=380.0,
            c_fol=100.0, lma=60.0, ceff=15.0,
        )


def test_phenology_defaults_to_the_published_functions() -> None:
    """Only photosynthesis is still stubbed; A7/A8 are wired in as the default."""
    params = make_parameters()
    drivers = make_drivers(5, seed=14)

    output = run_dalec2(params, drivers, gpp_fn=constant_gpp(1.0))

    assert np.isfinite(output.phi_onset).all()
    assert np.isfinite(output.phi_fall).all()
    assert (output.phi_onset > 0.0).all()
    assert (output.phi_fall > 0.0).all()


def test_mismatched_driver_lengths_raise() -> None:
    params = make_parameters()
    drivers = make_drivers(10, seed=15)
    broken = type(drivers)(
        doy=drivers.doy[:5],
        t_air=drivers.t_air,
        t_max=drivers.t_max,
        t_min=drivers.t_min,
        sw_in=drivers.sw_in,
        co2=drivers.co2,
    )

    with pytest.raises(ValueError, match="disagree in length"):
        run_dalec2(
            params, broken, gpp_fn=constant_gpp(1.0),
            phenology_fn=constant_phenology(0.0, 0.0)
        )


def test_non_finite_drivers_raise() -> None:
    params = make_parameters()
    drivers = make_drivers(10, seed=16)
    t_air = drivers.t_air.copy()
    t_air[4] = np.nan
    broken = type(drivers)(
        doy=drivers.doy,
        t_air=t_air,
        t_max=drivers.t_max,
        t_min=drivers.t_min,
        sw_in=drivers.sw_in,
        co2=drivers.co2,
    )

    with pytest.raises(ValueError, match="non-finite"):
        run_dalec2(params, broken, gpp_fn=constant_gpp(1.0),
                   phenology_fn=constant_phenology(0.0, 0.0))


def test_empty_driver_record_raises() -> None:
    params = make_parameters()
    empty = make_drivers(1, seed=17)
    empty = type(empty)(
        doy=np.array([], dtype=int),
        t_air=np.array([]),
        t_max=np.array([]),
        t_min=np.array([]),
        sw_in=np.array([]),
        co2=np.array([]),
    )

    with pytest.raises(ValueError, match="empty"):
        run_dalec2(params, empty, gpp_fn=constant_gpp(1.0),
                   phenology_fn=constant_phenology(0.0, 0.0))


# ---------------------------------------------------------------------------
# Allocation closure, as seen by the model
# ---------------------------------------------------------------------------


def test_allocated_carbon_equals_one_minus_f_auto_times_gpp() -> None:
    """The four allocation fractions must deliver exactly (1 - f_auto) * GPP."""
    params = make_parameters(f_auto=0.42, allocation_weights=DEFAULT_ALLOCATION_WEIGHTS)
    gpp = 7.0
    state = np.zeros(N_POOLS)

    next_state, (_, ra, rh, _) = dalec2_step(
        state, gpp=gpp, phi_onset=0.0, phi_fall=0.0, t_air=10.0, params=params
    )

    assert rh == pytest.approx(0.0)  # no carbon in litter or soil to respire
    assert ra == pytest.approx(0.42 * gpp)
    assert float(next_state.sum()) == pytest.approx((1.0 - 0.42) * gpp)


@pytest.mark.parametrize("seed", range(10))
def test_allocation_fractions_sum_to_one_for_prior_style_draws(seed: int) -> None:
    """Every draw from the f_auto + Dirichlet reparameterisation closes exactly."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        f_auto = float(rng.uniform(0.3, 0.7))
        weights = rng.dirichlet(np.ones(4))
        f_lab, f_fol, f_roo, f_woo = allocation_fractions(f_auto, weights)

        assert f_auto + f_lab + f_fol + f_roo + f_woo == pytest.approx(1.0, abs=1e-12)
        assert min(f_lab, f_fol, f_roo, f_woo) >= 0.0
        # Constructing DalecParameters would raise if the closure failed.
        params = make_parameters(f_auto=f_auto, allocation_weights=tuple(weights))
        assert isinstance(params, DalecParameters)
