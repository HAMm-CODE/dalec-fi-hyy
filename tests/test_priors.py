"""Tests for the PyMC prior block.

Three things are asserted, in the order they matter:

1. **Every sampled prior matches its recorded source.** Each bound is looked up
   independently here, from the registry or from the recorded constant, so these
   tests fail if ``priors.py`` ever hard-codes a number instead of reading it.
2. **Every derived pool satisfies its defining relation exactly.** Not
   approximately, and not only in the aggregate -- on every draw.
3. **The LAI convention switch changes only ``lma``.**
"""

from __future__ import annotations

import numpy as np
import pytest

from dalec.diagnostics import reparameterised_bounds
from dalec.parameters import (
    ALLOCATION_WEIGHT_ORDER,
    BELOWGROUND_LITTERFALL_G_C_M2,
    DAYS_PER_YEAR,
    LAI_CONVENTIONS,
    NEEDLE_LITTERFALL_G_C_M2,
    PARAMETER_NAMES,
    PARAMETER_REGISTRY,
    POOL_STATE_PARAMETERS,
    TREE_CARBON_STOCK_G_C_M2,
    canopy_bounds,
    derive_initial_pools,
    derive_litter_som_pools,
    prior_bounds,
)
from dalec.priors import (
    DERIVED_POOLS,
    REPARAMETERISED_RESPIRATION,
    DalecPriors,
    build_priors,
    prior_sources,
)

pytestmark = pytest.mark.pytensor

DRAWS = 400
SEED = 20260809


@pytest.fixture(scope="module")
def t_air() -> np.ndarray:
    """A synthetic driver series with FI-Hyy's mean and seasonal amplitude."""
    rng = np.random.default_rng(3)
    day = np.arange(2000)
    return 4.32 - 12.0 * np.cos(2 * np.pi * day / DAYS_PER_YEAR) + rng.normal(0, 2.0, day.size)


def _sample(t_air, names, convention="hemisurface", draws=DRAWS):
    """Draw from the prior block and return a name to array mapping."""
    import pymc as pm

    with pm.Model():
        priors = build_priors(t_air=t_air, convention=convention)
        wanted = {}
        for name in names:
            for source in (priors.parameters, priors.derived, priors.sampled):
                if name in source:
                    wanted[name] = source[name]
                    break
            else:  # pragma: no cover - a typo in a test name
                raise KeyError(name)
        values = pm.draw(list(wanted.values()), draws=draws, random_seed=SEED)
    return dict(zip(wanted, values, strict=True))


@pytest.fixture(scope="module")
def drawn(t_air) -> dict[str, np.ndarray]:
    """One prior draw set, shared by the derived-pool relation tests."""
    return _sample(
        t_air,
        [
            "rh_annual", "rh_ref", "f_som", "theta_lit", "theta_som",
            "theta_roo", "c_lf", "temperature_exponent",
            "decomposition_multiplier",
            *POOL_STATE_PARAMETERS, "f_lab", "f_fol",
        ],
    )


# ---------------------------------------------------------------------------
# 1. Every sampled prior matches its recorded source
# ---------------------------------------------------------------------------


class TestPriorsMatchTheirRecordedSources:
    def test_registry_scalars_take_the_published_bounds(self):
        """Anything not overridden must come straight from the registry."""
        bounds = prior_sources("hemisurface")
        replaced = set(REPARAMETERISED_RESPIRATION) | set(DERIVED_POOLS)
        canopy = canopy_bounds("hemisurface")

        for name, entry in PARAMETER_REGISTRY.items():
            if entry.simplex or name in replaced or name in canopy:
                continue
            assert bounds[name] == prior_bounds(name), name

    @pytest.mark.parametrize("convention", sorted(LAI_CONVENTIONS))
    def test_canopy_parameters_take_the_canopy_bounds(self, convention):
        """lma, c_lf and ceff override the registry; each must match its source."""
        bounds = prior_sources(convention)
        for name, expected in canopy_bounds(convention).items():
            assert bounds[name] == expected, name
            assert bounds[name] != prior_bounds(name), name

    def test_respiration_parameters_take_the_reparameterised_bounds(self):
        bounds = prior_sources("hemisurface")
        for name, expected in reparameterised_bounds().items():
            assert bounds[name] == expected, name

    def test_no_initial_pool_is_sampled(self):
        """All six pools are derived; none may appear as a sampled bound."""
        bounds = prior_sources("hemisurface")
        for name in POOL_STATE_PARAMETERS:
            assert name not in bounds, name
            assert name in DERIVED_POOLS, name

    def test_the_allocation_fractions_are_not_sampled_independently(self):
        """They come from the Dirichlet, so they carry no scalar bound."""
        bounds = prior_sources("hemisurface")
        for name in ALLOCATION_WEIGHT_ORDER:
            assert name not in bounds, name

    def test_every_model_parameter_is_present_exactly_once(self, t_air):
        import pymc as pm

        with pm.Model():
            priors = build_priors(t_air=t_air)
        assert set(priors.parameters) == set(PARAMETER_NAMES)
        assert not set(priors.sampled) & set(priors.derived)

    def test_draws_respect_every_recorded_bound(self, t_air):
        """The bounds are not merely recorded, they bind."""
        bounds = prior_sources("hemisurface")
        drawn = _sample(t_air, list(bounds))
        for name, (lower, upper) in bounds.items():
            assert drawn[name].min() >= lower - 1e-9, name
            assert drawn[name].max() <= upper + 1e-9, name

    def test_the_dirichlet_concentration_comes_from_the_measured_fluxes(self, t_air):
        """A flat simplex would send 24% of GPP to foliage against a measured 15%."""
        drawn = _sample(t_air, ["f_lab", "f_fol", "f_roo", "f_woo", "f_auto"])
        share = np.median(drawn["f_lab"] + drawn["f_fol"])
        assert 0.12 < share < 0.18
        assert np.median(drawn["f_woo"]) > np.median(drawn["f_roo"])

    def test_allocation_closes_on_every_draw(self, t_air):
        drawn = _sample(t_air, ["f_auto", *ALLOCATION_WEIGHT_ORDER])
        total = sum(drawn[name] for name in ("f_auto", *ALLOCATION_WEIGHT_ORDER))
        assert np.allclose(total, 1.0)


# ---------------------------------------------------------------------------
# 2. Every derived pool satisfies its defining relation exactly
# ---------------------------------------------------------------------------


class TestDerivedPoolsSatisfyTheirRelations:
    def test_litter_and_soil_respire_exactly_rh_ref(self, drawn):
        """theta_lit*c_lit_0 + theta_som*c_som_0 == rh_ref, on every draw."""
        respiration = (
            drawn["theta_lit"] * drawn["c_lit_0"]
            + drawn["theta_som"] * drawn["c_som_0"]
        )
        assert np.allclose(respiration, drawn["rh_ref"])

    def test_rh_ref_inverts_the_annual_total_at_each_draws_own_theta(self, drawn):
        """The correction that made the realised annual respiration exact."""
        realised = (
            drawn["rh_ref"] * drawn["decomposition_multiplier"] * DAYS_PER_YEAR
        )
        assert np.allclose(realised, drawn["rh_annual"])

    def test_foliage_reproduces_the_measured_litterfall(self, drawn):
        """c_lf * c_fol_0 == the measured needle litterfall."""
        assert np.allclose(
            drawn["c_lf"] * drawn["c_fol_0"], NEEDLE_LITTERFALL_G_C_M2[1]
        )

    def test_fine_roots_reproduce_the_measured_root_litterfall(self, drawn):
        assert np.allclose(
            drawn["theta_roo"] * drawn["c_roo_0"] * DAYS_PER_YEAR,
            BELOWGROUND_LITTERFALL_G_C_M2,
        )

    def test_the_tree_pools_sum_to_the_measured_tree_carbon(self, drawn):
        total = (
            drawn["c_lab_0"] + drawn["c_fol_0"]
            + drawn["c_roo_0"] + drawn["c_woo_0"]
        )
        assert np.allclose(total, TREE_CARBON_STOCK_G_C_M2)

    def test_the_labile_pool_takes_its_share_of_the_foliar_flux(self, drawn):
        share = drawn["f_lab"] / (drawn["f_lab"] + drawn["f_fol"])
        assert np.allclose(
            drawn["c_lab_0"], NEEDLE_LITTERFALL_G_C_M2[1] * share
        )

    def test_the_symbolic_derivation_agrees_with_the_numpy_one(self, drawn):
        """The graph and dalec.parameters must not drift apart.

        Two implementations of the same algebra is a real risk, so they are
        checked against each other on the same inputs rather than each against
        its own idea of the answer.
        """
        c_lit, c_som = derive_litter_som_pools(
            drawn["rh_ref"], drawn["f_som"],
            drawn["theta_lit"], drawn["theta_som"],
        )
        assert np.allclose(c_lit, drawn["c_lit_0"])
        assert np.allclose(c_som, drawn["c_som_0"])

        pools = derive_initial_pools(
            drawn["c_lf"], drawn["theta_roo"], drawn["f_lab"], drawn["f_fol"]
        )
        for name, expected in pools.items():
            assert np.allclose(expected, drawn[name]), name

    def test_every_derived_pool_is_positive(self, drawn):
        for name in POOL_STATE_PARAMETERS:
            assert drawn[name].min() > 0.0, name


# ---------------------------------------------------------------------------
# 3. The convention switch changes only lma
# ---------------------------------------------------------------------------


class TestTheConventionSwitchChangesOnlyLma:
    def test_only_lma_differs_between_the_two_conventions(self):
        first, second = sorted(LAI_CONVENTIONS)
        left, right = prior_sources(first), prior_sources(second)

        assert set(left) == set(right)
        differing = {name for name in left if left[name] != right[name]}
        assert differing == {"lma"}, differing

    def test_the_two_conventions_give_genuinely_different_lma(self):
        left = prior_sources("hemisurface")["lma"]
        right = prior_sources("projected")["lma"]
        # projected needs more carbon per unit leaf area, so its lma is larger
        assert right[0] > left[0]
        assert right[1] > left[1]
        # and by about the ratio of the two divisors
        expected = LAI_CONVENTIONS["projected"] / LAI_CONVENTIONS["hemisurface"]
        assert right[1] / left[1] == pytest.approx(expected, rel=0.02)

    def test_the_derived_pools_do_not_move_with_the_convention(self, t_air):
        """lma enters GPP, not the pool derivations. A pool that moved would
        mean the convention had leaked somewhere it does not belong."""
        names = [*POOL_STATE_PARAMETERS, "rh_ref"]
        left = _sample(t_air, names, convention="hemisurface", draws=64)
        right = _sample(t_air, names, convention="projected", draws=64)
        for name in names:
            assert np.allclose(left[name], right[name]), name

    def test_an_unknown_convention_raises_before_anything_is_built(self, t_air):
        import pymc as pm

        with pm.Model(), pytest.raises(KeyError, match="unknown LAI convention"):
            build_priors(t_air=t_air, convention="all-sided")

    def test_neither_convention_is_a_default_that_hides_the_choice(self):
        """Both are live; the switch must accept either explicitly."""
        for convention in LAI_CONVENTIONS:
            assert "lma" in prior_sources(convention)


# ---------------------------------------------------------------------------
# Guards on the block itself
# ---------------------------------------------------------------------------


class TestPriorBlockGuards:
    def test_it_refuses_an_unusable_driver_series(self, t_air):
        import pymc as pm

        for bad in (np.array([]), np.array([1.0, np.nan]), np.zeros((2, 2))):
            with pm.Model(), pytest.raises(ValueError):
                build_priors(t_air=bad)

    def test_it_refuses_a_block_missing_a_model_parameter(self):
        with pytest.raises(ValueError, match="missing model parameters"):
            DalecPriors(
                parameters={"f_auto": 0.5}, sampled={}, derived={},
                convention="hemisurface",
            )

    def test_it_refuses_a_block_inventing_a_parameter(self):
        parameters = dict.fromkeys(PARAMETER_NAMES, 0.0)
        parameters["not_a_parameter"] = 0.0
        with pytest.raises(ValueError, match="invented parameters"):
            DalecPriors(
                parameters=parameters, sampled={}, derived={},
                convention="hemisurface",
            )
