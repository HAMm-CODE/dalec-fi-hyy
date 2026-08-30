"""PyMC prior construction, read entirely from :mod:`dalec.parameters`.

**No bound is typed here.** Every range comes from the Parameter registry or
from a recorded constant, so a prior can only be changed by changing its
recorded source. That is the single rule this module exists to enforce:
``grep`` for a numeric literal below and you should find only structural
constants -- array shapes, tolerances, the number of allocation components.

What the priors are, and where each is recorded
-----------------------------------------------

============================  =========================================  =======
group                         source                                     record
============================  =========================================  =======
scalars                       ``PARAMETER_REGISTRY`` uniform bounds       §1
canopy: lma, c_lf, ceff       ``canopy_bounds(convention)``               §8, §9
respiration: rh_annual,       ``reparameterised_bounds()``                §7
  f_som, theta_lit, theta_som
allocation simplex            ``allocation_concentration()``              §8
derived pools                 ``derive_*`` relations, mirrored here       §7, §9
============================  =========================================  =======

Three things are deliberately not independent draws.

**Allocation** is a Dirichlet over the 4-simplex scaled by ``1 - f_auto``.
Sampling the fractions independently and taking one by subtraction lets it go
negative, which is a rejection wall NUTS cannot differentiate through.

**Every initial pool is derived** from a measured flux and a turnover rate, not
sampled. ``c_fol_0``'s published U(20, 2000) has median 1,010 against the
462-770 the canopy priors imply, and on that prior the modelled foliage never
reached steady state inside the calibration window, so every calibration year
was a relaxation transient from a wrong initial state.

**``rh_ref`` is derived per draw** from that draw's own ``temperature_exponent``.
A single multiplier at one fixed Theta drifts the realised annual respiration
off target for every draw whose Theta differs.

The LAI convention is a switch, not a decision
----------------------------------------------
ACM aggregates SPA, parametrised for a broadleaf stand where projected,
hemisurface and half-total coincide, so no needle geometry entered the
calibration and its own provenance cannot supply a convention for a conifer
canopy. **Both stay live and neither is adopted**: the calibration is run under
each and the difference reported as a sensitivity. See DECISIONS §10 and
LIMITATIONS §15. The switch changes ``lma`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from dalec.parameters import (
    ALLOCATION_WEIGHT_ORDER,
    DAYS_PER_YEAR,
    DEFAULT_LAI_CONVENTION,
    PARAMETER_NAMES,
    PARAMETER_REGISTRY,
    POOL_STATE_PARAMETERS,
    allocation_concentration,
    canopy_bounds,
    lai_divisor,
    prior_bounds,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pymc as pm

__all__ = [
    "DERIVED_POOLS",
    "REPARAMETERISED_RESPIRATION",
    "DalecPriors",
    "build_priors",
    "prior_sources",
]

#: Respiration parameters sampled in place of the published stock priors.
REPARAMETERISED_RESPIRATION: Final[tuple[str, ...]] = (
    "rh_annual",
    "f_som",
    "theta_lit",
    "theta_som",
)

#: Pools that are derived rather than sampled. All six of them.
DERIVED_POOLS: Final[tuple[str, ...]] = POOL_STATE_PARAMETERS


@dataclass(frozen=True)
class DalecPriors:
    """The prior block: sampled variables, derived quantities, and provenance.

    Attributes
    ----------
    parameters
        Name to tensor for every field of :class:`dalec.parameters.DalecParameters`,
        ready to feed the forward model.
    sampled
        Name to tensor for what is actually drawn. Strictly smaller than
        ``parameters``: the pools and the allocation fractions are derived.
    derived
        Name to tensor for everything computed from the sampled set.
    convention
        Which LAI convention this block was built on.
    bounds
        Name to ``(lower, upper)`` for every sampled scalar, so a test can check
        each against its recorded source without re-deriving it.
    """

    parameters: dict[str, Any]
    sampled: dict[str, Any]
    derived: dict[str, Any]
    convention: str
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(PARAMETER_NAMES) - set(self.parameters)
        if missing:
            raise ValueError(
                f"prior block is missing model parameters: {sorted(missing)}"
            )
        extra = set(self.parameters) - set(PARAMETER_NAMES)
        if extra:
            raise ValueError(
                f"prior block invented parameters not in the registry: {sorted(extra)}"
            )


def prior_sources(convention: str = DEFAULT_LAI_CONVENTION) -> dict[str, tuple[float, float]]:
    """Bounds for every sampled scalar, resolved from their recorded sources.

    Built the same way :func:`build_priors` builds them, so a test comparing the
    two is checking that the model reads its sources rather than that two copies
    of a number agree.
    """
    from dalec.diagnostics import reparameterised_bounds

    canopy = canopy_bounds(convention)
    respiration = reparameterised_bounds()
    replaced = set(REPARAMETERISED_RESPIRATION) | set(DERIVED_POOLS)

    bounds: dict[str, tuple[float, float]] = {}
    for name, entry in PARAMETER_REGISTRY.items():
        if entry.simplex or name in replaced:
            continue
        bounds[name] = canopy.get(name, prior_bounds(name))
    for name in REPARAMETERISED_RESPIRATION:
        bounds[name] = respiration[name]
    return bounds


def build_priors(
    *,
    t_air: np.ndarray,
    convention: str = DEFAULT_LAI_CONVENTION,
    model: pm.Model | None = None,
) -> DalecPriors:
    """Construct the DALEC prior block inside a PyMC model context.

    Parameters
    ----------
    t_air
        Daily mean air temperature over the calibration block, degrees C. Needed
        because ``rh_ref`` inverts an annual respiration total through each
        draw's own temperature multiplier.
    convention
        LAI convention, ``"hemisurface"`` or ``"projected"``. Changes ``lma``
        and nothing else.
    model
        Model to attach to; defaults to the enclosing ``pm.Model()`` context.

    Returns
    -------
    :class:`DalecPriors` with every model parameter as a tensor.
    """
    import pymc as pm
    import pytensor.tensor as pt

    lai_divisor(convention)  # raises on an unknown name before anything is built

    temperatures = np.asarray(t_air, dtype=float)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("t_air must be a non-empty one-dimensional series")
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("t_air contains non-finite values")

    bounds = prior_sources(convention)
    model = pm.modelcontext(model)
    sampled: dict[str, Any] = {}
    derived: dict[str, Any] = {}

    with model:
        # -- sampled scalars, every bound read from its recorded source ------
        for name, (lower, upper) in bounds.items():
            sampled[name] = pm.Uniform(name, lower=lower, upper=upper)

        # -- allocation: a Dirichlet simplex, not independent uniforms -------
        weights = pm.Dirichlet(
            "allocation_weights", a=allocation_concentration()
        )
        sampled["allocation_weights"] = weights
        available = 1.0 - sampled["f_auto"]
        fractions = {
            component: pm.Deterministic(component, available * weights[index])
            for index, component in enumerate(ALLOCATION_WEIGHT_ORDER)
        }
        derived.update(fractions)

        # -- rh_ref: invert the annual total at this draw's own Theta --------
        multiplier = pt.mean(
            pt.exp(sampled["temperature_exponent"] * temperatures)
        )
        rh_ref = pm.Deterministic(
            "rh_ref", sampled["rh_annual"] / (multiplier * DAYS_PER_YEAR)
        )
        derived["rh_ref"] = rh_ref
        derived["decomposition_multiplier"] = pm.Deterministic(
            "decomposition_multiplier", multiplier
        )

        # -- the six initial pools, all derived ------------------------------
        # Litter and soil: the stocks that respire rh_ref at the reference
        # temperature, split by f_som.
        derived["c_lit_0"] = pm.Deterministic(
            "c_lit_0",
            (1.0 - sampled["f_som"]) * rh_ref / sampled["theta_lit"],
        )
        derived["c_som_0"] = pm.Deterministic(
            "c_som_0", sampled["f_som"] * rh_ref / sampled["theta_som"]
        )

        # Foliage, labile, fine root and wood: measured fluxes over turnover.
        litterfall, root_litterfall, tree_carbon = _measured_canopy_fluxes()
        foliar_share = weights[ALLOCATION_WEIGHT_ORDER.index("f_fol")]
        labile_share = weights[ALLOCATION_WEIGHT_ORDER.index("f_lab")]
        derived["c_fol_0"] = pm.Deterministic(
            "c_fol_0", litterfall / sampled["c_lf"]
        )
        derived["c_lab_0"] = pm.Deterministic(
            "c_lab_0",
            litterfall * labile_share / (labile_share + foliar_share),
        )
        derived["c_roo_0"] = pm.Deterministic(
            "c_roo_0",
            root_litterfall / (sampled["theta_roo"] * DAYS_PER_YEAR),
        )
        derived["c_woo_0"] = pm.Deterministic(
            "c_woo_0",
            tree_carbon
            - derived["c_lab_0"]
            - derived["c_fol_0"]
            - derived["c_roo_0"],
        )

    parameters = {
        name: sampled[name] for name in PARAMETER_NAMES if name in sampled
    }
    parameters.update(
        {name: derived[name] for name in PARAMETER_NAMES if name in derived}
    )
    return DalecPriors(
        parameters=parameters,
        sampled=sampled,
        derived=derived,
        convention=convention,
        bounds=bounds,
    )


def _measured_canopy_fluxes() -> tuple[float, float, float]:
    """The three measured stocks and fluxes the pool derivations need.

    Read from their recorded constants rather than restated, so this function
    exists only to name what is used and keep the caller readable.
    """
    from dalec.parameters import (
        BELOWGROUND_LITTERFALL_G_C_M2,
        NEEDLE_LITTERFALL_G_C_M2,
        TREE_CARBON_STOCK_G_C_M2,
    )

    return (
        NEEDLE_LITTERFALL_G_C_M2[1],
        BELOWGROUND_LITTERFALL_G_C_M2,
        TREE_CARBON_STOCK_G_C_M2,
    )
