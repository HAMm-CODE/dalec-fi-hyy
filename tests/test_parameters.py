"""Tests for the parameter container and the allocation simplex."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_parameters
from dalec.model_numpy import POOL_NAMES
from dalec.parameters import (
    ALLOCATION_PARAMETERS,
    ALLOCATION_WEIGHT_ORDER,
    PARAMETER_NAMES,
    PARAMETER_REGISTRY,
    PHENOLOGY_PARAMETERS,
    PHOTOSYNTHESIS_PARAMETERS,
    POOL_STATE_PARAMETERS,
    SIMPLEX_PARAMETERS,
    TURNOVER_PARAMETERS,
    DalecParameters,
    allocation_fractions,
    prior_bounds,
)

# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_parameter_set_has_twenty_four_fields_and_twenty_three_free() -> None:
    """One allocation fraction is determined by the closure, hence 24 -> 23."""
    assert len(PARAMETER_NAMES) == 24
    assert len(set(PARAMETER_NAMES)) == 24
    assert len(PARAMETER_NAMES) - 1 == 23


def test_parameter_groups_partition_the_registry() -> None:
    grouped = (
        ALLOCATION_PARAMETERS
        + TURNOVER_PARAMETERS
        + ("temperature_exponent",)
        + PHENOLOGY_PARAMETERS
        + PHOTOSYNTHESIS_PARAMETERS
        + POOL_STATE_PARAMETERS
    )
    assert sorted(grouped) == sorted(PARAMETER_NAMES)
    assert len(grouped) == len(set(grouped))


def test_group_sizes_match_the_appendix() -> None:
    assert len(ALLOCATION_PARAMETERS) == 5
    assert len(TURNOVER_PARAMETERS) == 5
    assert len(PHENOLOGY_PARAMETERS) == 5
    assert len(PHOTOSYNTHESIS_PARAMETERS) == 2
    assert len(POOL_STATE_PARAMETERS) == 6


def test_initial_pool_parameters_follow_the_model_pool_order() -> None:
    """c_lab_0 ... c_som_0 must line up with POOL_NAMES, or pools get swapped."""
    assert POOL_STATE_PARAMETERS == tuple(f"{name}_0" for name in POOL_NAMES)


def test_allocation_weight_order_is_the_documented_one() -> None:
    assert ALLOCATION_WEIGHT_ORDER == ("f_lab", "f_fol", "f_roo", "f_woo")
    assert ALLOCATION_PARAMETERS[0] == "f_auto"


def test_initial_pools_array_matches_the_fields() -> None:
    params = make_parameters()
    np.testing.assert_allclose(
        params.initial_pools,
        [params.c_lab_0, params.c_fol_0, params.c_roo_0,
         params.c_woo_0, params.c_lit_0, params.c_som_0],
    )


def test_to_dict_round_trips() -> None:
    params = make_parameters()
    assert DalecParameters(**params.to_dict()) == params
    assert set(params.to_dict()) == set(PARAMETER_NAMES)


# ---------------------------------------------------------------------------
# The allocation simplex
# ---------------------------------------------------------------------------


def test_allocation_fractions_split_the_remainder_in_order() -> None:
    f_lab, f_fol, f_roo, f_woo = allocation_fractions(0.4, [0.1, 0.2, 0.3, 0.4])
    remainder = 0.6
    assert f_lab == pytest.approx(0.1 * remainder)
    assert f_fol == pytest.approx(0.2 * remainder)
    assert f_roo == pytest.approx(0.3 * remainder)
    assert f_woo == pytest.approx(0.4 * remainder)
    assert f_lab + f_fol + f_roo + f_woo == pytest.approx(remainder)


def test_wood_fraction_never_goes_negative_at_the_extremes() -> None:
    """The failure mode the simplex exists to prevent.

    Subtracting three independent uniforms from (1 - f_auto) to obtain the
    wood fraction can produce a negative value, which is a rejection wall NUTS
    cannot differentiate through. Under the simplex, even a degenerate draw
    that puts all mass on one component leaves the rest at exactly zero.
    """
    for weights in ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]):
        fractions = allocation_fractions(0.7, weights)
        assert min(fractions) >= 0.0
        assert sum(fractions) == pytest.approx(0.3)


@pytest.mark.parametrize("seed", range(5))
def test_dirichlet_draws_always_close(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        f_auto = float(rng.uniform(0.0, 1.0))
        weights = rng.dirichlet(rng.uniform(0.2, 5.0, size=4))
        fractions = allocation_fractions(f_auto, weights)
        assert min(fractions) >= 0.0
        assert f_auto + sum(fractions) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("f_auto", [-0.01, 1.01, 2.0])
def test_f_auto_outside_the_unit_interval_raises(f_auto: float) -> None:
    with pytest.raises(ValueError, match=r"f_auto must lie in \[0, 1\]"):
        allocation_fractions(f_auto, [0.25, 0.25, 0.25, 0.25])


def test_wrong_number_of_weights_raises() -> None:
    with pytest.raises(ValueError, match="expected 4 allocation weights"):
        allocation_fractions(0.5, [0.5, 0.5])


def test_negative_weights_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        allocation_fractions(0.5, [-0.1, 0.4, 0.4, 0.3])


def test_weights_that_do_not_sum_to_one_raise() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        allocation_fractions(0.5, [0.3, 0.3, 0.3, 0.3])


def test_non_finite_weights_raise() -> None:
    with pytest.raises(ValueError, match="finite"):
        allocation_fractions(0.5, [np.nan, 0.4, 0.3, 0.3])


# ---------------------------------------------------------------------------
# Closure enforcement on the container
# ---------------------------------------------------------------------------


def test_hand_built_fractions_that_do_not_close_are_rejected() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        make_parameters().__class__(
            **{**make_parameters().to_dict(), "f_woo": 0.99}
        )


def test_from_allocation_simplex_produces_a_closing_set() -> None:
    params = make_parameters(f_auto=0.55, allocation_weights=(0.4, 0.3, 0.2, 0.1))
    total = params.f_auto + params.f_lab + params.f_fol + params.f_roo + params.f_woo
    assert total == pytest.approx(1.0, abs=1e-12)
    assert params.f_lab == pytest.approx(0.4 * 0.45)
    assert params.f_woo == pytest.approx(0.1 * 0.45)


def test_parameters_are_frozen() -> None:
    params = make_parameters()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
        params.f_auto = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Prior ranges (published Table 1)
# ---------------------------------------------------------------------------


def test_registry_covers_exactly_the_parameter_fields() -> None:
    assert tuple(PARAMETER_REGISTRY) == PARAMETER_NAMES


@pytest.mark.parametrize(
    ("name", "lower", "upper"),
    [
        ("f_auto", 0.3, 0.7),
        ("f_lab", 0.01, 0.5),
        ("f_fol", 0.01, 0.5),
        ("f_roo", 0.01, 0.5),
        ("f_woo", 0.01, 0.5),
        ("theta_woo", 2.5e-5, 1.0e-3),
        ("theta_roo", 1.0e-4, 1.0e-2),
        ("theta_lit", 1.0e-4, 1.0e-2),
        ("theta_som", 1.0e-7, 1.0e-3),
        ("theta_min", 1.0e-5, 1.0e-2),
        ("temperature_exponent", 0.018, 0.08),
        ("d_onset", 1.0, 365.0),
        ("d_fall", 1.0, 365.0),
        ("cr_onset", 10.0, 100.0),
        ("ceff", 10.0, 100.0),
        ("lma", 10.0, 400.0),
        ("c_lf", 0.125, 1.0),
        ("c_woo_0", 100.0, 1.0e5),
        ("c_som_0", 100.0, 2.0e5),
    ],
)
def test_prior_bounds_match_the_published_table(name: str, lower: float, upper: float) -> None:
    assert prior_bounds(name) == (pytest.approx(lower), pytest.approx(upper))


@pytest.mark.parametrize("name", ["c_lab_0", "c_fol_0", "c_roo_0", "c_lit_0"])
def test_corrected_initial_pool_ranges(name: str) -> None:
    """Published values, 20-2000 -- not the preprint's 10-1000."""
    assert prior_bounds(name) == (pytest.approx(20.0), pytest.approx(2000.0))


def test_corrected_cr_fall_range() -> None:
    """Published value, 20-150 day -- not the preprint's 10-100."""
    assert prior_bounds("cr_fall") == (pytest.approx(20.0), pytest.approx(150.0))


def test_every_bound_is_ordered_and_finite() -> None:
    for name in PARAMETER_NAMES:
        lower, upper = prior_bounds(name)
        assert np.isfinite([lower, upper]).all(), name
        assert lower < upper, name


def test_unknown_parameter_raises() -> None:
    with pytest.raises(KeyError, match="unknown parameter"):
        prior_bounds("f_nonsense")


def test_allocation_fractions_are_flagged_as_simplex_sampled() -> None:
    """priors.py must not build a Uniform from these four tabulated ranges."""
    assert SIMPLEX_PARAMETERS == ALLOCATION_WEIGHT_ORDER
    for name in PARAMETER_NAMES:
        assert PARAMETER_REGISTRY[name].simplex == (name in ALLOCATION_WEIGHT_ORDER), name


def test_every_parameter_has_a_description_and_unit() -> None:
    for name, entry in PARAMETER_REGISTRY.items():
        assert entry.description.strip(), name
        assert entry.unit.strip(), name


def test_baseline_parameters_lie_inside_their_prior_ranges() -> None:
    """The test fixture must not sit outside the priors it is meant to represent."""
    params = make_parameters()
    for name, value in params.to_dict().items():
        lower, upper = prior_bounds(name)
        assert lower <= value <= upper, f"{name}={value} outside [{lower}, {upper}]"
