"""Evergreen DALEC as a differentiable PyTensor graph, with its prior block.

The evergreen DALEC of Williams et al. (2005), *Global Change Biology* 11,
89-105, Appendix "Carbon dynamic model equations", p. 104: five pools -- foliage,
wood, fine roots, fresh litter, and soil organic matter plus woody debris -- with
no labile pool and no leaf onset or leaf fall phenology. Foliage turns over at a
constant daily rate. Model assumptions 1-5 are on p. 95.

It exists for one structural question (RQ3): is a model without the labile pool
and the A7/A8 phenology better identified than DALEC2 at FI-Hyy? For a
comparison to answer that, everything that is not structure is DALEC2's:

- **GPP is the same ACM and frost mask** as :mod:`dalec.model`, through the same
  ``_acm_gpp``. Not the ACM printed in the Williams (2005) Appendix, whose
  parameter table lists a3 = 217.9 and a4 = 0.980, apparently swapped against
  Chuter et al. (2015) Appendix D.
- **Every shared parameter takes exactly its DALEC2 prior**, read through
  :func:`dalec.priors.prior_sources`. No bound is typed here.
- **The decomposition temperature response is DALEC2's** ``exp(Theta * T)`` with
  Theta sampled, not Williams' fixed Q10 = 2, ``T = 0.5 * exp(0.0693 * A)``.
- **The initial pools are derived as in** :mod:`dalec.priors`.

Three departures from the printed Appendix, each decided and recorded in
``docs/evergreen.md``:

1. **Wood litter enters the soil pool, not the litter pool.** The Appendix prints
   ``dClit = Lf + Lw + Lr - Rh1 - D``, but the same paper (p. 95) defines Clit as
   fresh leaf and fine root litter and the soil pool as SOM plus woody debris,
   and so does Fox et al. (2009) section 2.1. Implemented as
   ``dCsom = D + Lw - Rh2``, which is also where DALEC2 sends wood turnover.
2. **Foliage turnover is t5 = c_lf / 365.25**, with ``c_lf`` on its DALEC2 prior.
   At steady state annual litterfall is then ``c_lf * Cf``, the relation that
   ``c_fol_0 = litterfall / c_lf`` rests on (DECISIONS section 9).
3. **Allocation is a Dirichlet over (foliage, roots, wood)**: DALEC2's
   concentration with the labile and foliar weights merged. Merging two
   components of a Dirichlet gives a Dirichlet with the summed concentration, so
   every allocation marginal equals DALEC2's. Williams' t3 and t4, the fractions
   of NPP to foliage and to fine roots, are this simplex under another name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from dalec.acm import AcmCoefficients, day_length_hours, frost_mask
from dalec.model import _acm_gpp
from dalec.parameters import (
    ALLOCATION_WEIGHT_ORDER,
    BELOWGROUND_LITTERFALL_G_C_M2,
    DAYS_PER_YEAR,
    DEFAULT_LAI_CONVENTION,
    NEEDLE_LITTERFALL_G_C_M2,
    PARAMETER_NAMES,
    TREE_CARBON_STOCK_G_C_M2,
    allocation_concentration,
    lai_divisor,
)
from dalec.priors import prior_sources

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pymc as pm

__all__ = [
    "EVERGREEN_ALLOCATION_ORDER",
    "EVERGREEN_FOLIAGE_INDEX",
    "EVERGREEN_PARAMETER_NAMES",
    "EVERGREEN_POOL_NAMES",
    "REMOVED_PHENOLOGY",
    "EvergreenGraph",
    "EvergreenPriors",
    "build_evergreen_graph",
    "build_evergreen_priors",
    "evergreen_allocation_concentration",
    "evergreen_prior_sources",
]

#: The five pools of Williams et al. (2005), p. 95, in the order the scan carries
#: them. The soil pool is soil organic matter plus woody debris.
EVERGREEN_POOL_NAMES: Final[tuple[str, ...]] = ("c_fol", "c_roo", "c_woo", "c_lit", "c_som")

#: Position of foliar carbon in :data:`EVERGREEN_POOL_NAMES`.
EVERGREEN_FOLIAGE_INDEX: Final[int] = EVERGREEN_POOL_NAMES.index("c_fol")

#: Allocation simplex order. Load-bearing in the same way as DALEC2's
#: ``ALLOCATION_WEIGHT_ORDER``: a transposed simplex swaps carbon between pools
#: while every conservation check still passes.
EVERGREEN_ALLOCATION_ORDER: Final[tuple[str, ...]] = ("f_fol", "f_roo", "f_woo")

#: DALEC2 sampled parameters with no counterpart here: the A7/A8 phenology.
REMOVED_PHENOLOGY: Final[tuple[str, ...]] = ("d_onset", "cr_onset", "d_fall", "cr_fall")

#: Every evergreen model parameter: DALEC2's, less the phenology, the labile
#: allocation fraction and the labile initial pool.
EVERGREEN_PARAMETER_NAMES: Final[tuple[str, ...]] = tuple(
    name for name in PARAMETER_NAMES
    if name not in (*REMOVED_PHENOLOGY, "f_lab", "c_lab_0")
)


@dataclass(frozen=True)
class EvergreenGraph:
    """The forward model as tensors. Same fields as :class:`dalec.model.DalecGraph`.

    ``pools`` holds the carbon pools **after** each step, shape ``(n_days, 5)``,
    ordered as :data:`EVERGREEN_POOL_NAMES`.
    """

    nee: Any
    gpp: Any
    ra: Any
    rh: Any
    pools: Any
    n_days: int


@dataclass(frozen=True)
class EvergreenPriors:
    """The evergreen prior block. Same fields as :class:`dalec.priors.DalecPriors`."""

    parameters: dict[str, Any]
    sampled: dict[str, Any]
    derived: dict[str, Any]
    convention: str
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(EVERGREEN_PARAMETER_NAMES) - set(self.parameters)
        if missing:
            raise ValueError(f"evergreen prior block is missing: {sorted(missing)}")


def evergreen_prior_sources(
    convention: str = DEFAULT_LAI_CONVENTION,
) -> dict[str, tuple[float, float]]:
    """DALEC2's bounds for every scalar the evergreen model still samples."""
    return {
        name: bounds
        for name, bounds in prior_sources(convention).items()
        if name not in REMOVED_PHENOLOGY
    }


def evergreen_allocation_concentration() -> np.ndarray:
    """DALEC2's Dirichlet concentration with the labile and foliar weights merged.

    Ordered as :data:`EVERGREEN_ALLOCATION_ORDER`. The total is unchanged, so the
    foliar share has exactly the prior DALEC2 gives ``f_lab + f_fol``.
    """
    dalec2 = dict(zip(ALLOCATION_WEIGHT_ORDER, allocation_concentration(), strict=True))
    merged = {
        "f_fol": dalec2["f_lab"] + dalec2["f_fol"],
        "f_roo": dalec2["f_roo"],
        "f_woo": dalec2["f_woo"],
    }
    return np.array([merged[name] for name in EVERGREEN_ALLOCATION_ORDER])


def build_evergreen_priors(
    *,
    t_air: np.ndarray,
    convention: str = DEFAULT_LAI_CONVENTION,
    model: pm.Model | None = None,
) -> EvergreenPriors:
    """The evergreen prior block, inside a PyMC model context.

    Mirrors :func:`dalec.priors.build_priors` line for line, less the phenology
    and the labile pool. Variable names are DALEC2's wherever the quantity is
    the same, so a trace from either model reads the same way.
    """
    import pymc as pm
    import pytensor.tensor as pt

    lai_divisor(convention)  # raises on an unknown name before anything is built

    temperatures = np.asarray(t_air, dtype=float)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("t_air must be a non-empty one-dimensional series")
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("t_air contains non-finite values")

    bounds = evergreen_prior_sources(convention)
    model = pm.modelcontext(model)
    sampled: dict[str, Any] = {}
    derived: dict[str, Any] = {}

    with model:
        # -- sampled scalars, every bound DALEC2's ---------------------------
        for name, (lower, upper) in bounds.items():
            sampled[name] = pm.Uniform(name, lower=lower, upper=upper)

        # -- allocation: Williams (2005) Appendix p. 104,
        #    Af = (1 - t2) t3 G, Ar = (1 - t2) t4 G, Aw = (1 - t2)(1 - t3 - t4) G,
        #    with (t3, t4, 1 - t3 - t4) a Dirichlet simplex (departure 3) --------
        weights = pm.Dirichlet(
            "allocation_weights", a=evergreen_allocation_concentration()
        )
        sampled["allocation_weights"] = weights
        available = 1.0 - sampled["f_auto"]
        for index, component in enumerate(EVERGREEN_ALLOCATION_ORDER):
            derived[component] = pm.Deterministic(component, available * weights[index])

        # -- rh_ref at this draw's own Theta, exactly as DALEC2 -------------
        multiplier = pt.mean(pt.exp(sampled["temperature_exponent"] * temperatures))
        rh_ref = pm.Deterministic(
            "rh_ref", sampled["rh_annual"] / (multiplier * DAYS_PER_YEAR)
        )
        derived["rh_ref"] = rh_ref
        derived["decomposition_multiplier"] = pm.Deterministic(
            "decomposition_multiplier", multiplier
        )

        # -- initial pools, derived as in dalec.priors ----------------------
        derived["c_lit_0"] = pm.Deterministic(
            "c_lit_0", (1.0 - sampled["f_som"]) * rh_ref / sampled["theta_lit"]
        )
        derived["c_som_0"] = pm.Deterministic(
            "c_som_0", sampled["f_som"] * rh_ref / sampled["theta_som"]
        )
        litterfall = NEEDLE_LITTERFALL_G_C_M2[1]
        derived["c_fol_0"] = pm.Deterministic("c_fol_0", litterfall / sampled["c_lf"])
        derived["c_roo_0"] = pm.Deterministic(
            "c_roo_0",
            BELOWGROUND_LITTERFALL_G_C_M2 / (sampled["theta_roo"] * DAYS_PER_YEAR),
        )
        # With no labile pool, wood is the tree carbon left after foliage and
        # fine roots, so the measured total stock is preserved.
        derived["c_woo_0"] = pm.Deterministic(
            "c_woo_0", TREE_CARBON_STOCK_G_C_M2 - derived["c_fol_0"] - derived["c_roo_0"]
        )

    parameters = {
        name: sampled[name] if name in sampled else derived[name]
        for name in EVERGREEN_PARAMETER_NAMES
    }
    return EvergreenPriors(
        parameters=parameters,
        sampled=sampled,
        derived=derived,
        convention=convention,
        bounds=bounds,
    )


def build_evergreen_graph(
    *,
    parameters: dict[str, Any],
    doy: np.ndarray,
    t_air: np.ndarray,
    t_max: np.ndarray,
    t_min: np.ndarray,
    sw_in: np.ndarray,
    co2: np.ndarray,
    latitude_deg: float,
    coefficients: AcmCoefficients,
    frost_threshold_degc: float,
) -> EvergreenGraph:
    """Build the evergreen DALEC forward model over a driver record.

    Takes the same arguments as :func:`dalec.model.build_forward_graph` and
    returns the same fields, so the sampler can build either.
    """
    import pytensor
    import pytensor.tensor as pt

    drivers = {
        "doy": doy, "t_air": t_air, "t_max": t_max,
        "t_min": t_min, "sw_in": sw_in, "co2": co2,
    }
    lengths = {name: np.asarray(value).shape for name, value in drivers.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"driver arrays must share a length, got {lengths}")
    n_days = int(np.asarray(doy).size)
    if n_days == 0:
        raise ValueError("driver record is empty")

    # -- driver-only constants, exactly as dalec.model.build_forward_graph ----
    t_range = np.maximum(np.asarray(t_max) - np.asarray(t_min), 0.0)
    g_c = abs(coefficients.psi_mpa) ** coefficients.a10 / (
        0.5 * t_range + coefficients.a6 * coefficients.r_tot
    )
    day_length_factor = (
        coefficients.a2 * day_length_hours(doy, latitude_deg) + coefficients.a5
    )
    unfrozen = (~frost_mask(t_max, t_min, frost_threshold_degc)).astype(float)

    sequences = [
        pt.as_tensor_variable(np.asarray(value, dtype=float))
        for value in (t_air, t_max, sw_in, co2, g_c, day_length_factor, unfrozen)
    ]

    # float64 throughout, for the reason given in dalec.model.
    theta = {
        name: pt.cast(pt.as_tensor_variable(value), "float64")
        for name, value in parameters.items()
    }
    coef = coefficients

    # t5, the foliage turnover rate (departure 2). Time-invariant, so computed
    # once here and passed in as a non-sequence like everything else.
    # c_lf keeps its DALEC2 prior U(0.2, 0.333). Southern Finland Scots pine
    # needle biomass turnover is about 0.21 per year (Trees,
    # doi:10.1007/s00468-004-0381-4); Kolari et al. (2009) support about 0.25.
    # Both lie inside the range. Source to be verified by Hamza.
    foliage_turnover = theta["c_lf"] / DAYS_PER_YEAR

    step_parameters = (
        "lma", "ceff", "temperature_exponent",
        "f_fol", "f_roo", "f_woo", "f_auto",
        "theta_roo", "theta_woo", "theta_lit", "theta_som", "theta_min",
    )
    non_sequences = [foliage_turnover, *[theta[name] for name in step_parameters]]

    def step(
        t_air_t, t_max_t, sw_in_t, co2_t, g_c_t, dlf_t, unfrozen_t,
        c_fol, c_roo, c_woo, c_lit, c_som,
        foliage_turnover_, lma_, ceff_, temperature_exponent_,
        f_fol_, f_roo_, f_woo_, f_auto_,
        theta_roo_, theta_woo_, theta_lit_, theta_som_, theta_min_,
    ):
        # GPP: the DALEC2 ACM and frost mask, identical to dalec.model's step.
        p_d, e_0 = _acm_gpp(c_fol, lma_, ceff_, t_max_t, co2_t, g_c_t, dlf_t, coef)
        light_supply = e_0 * sw_in_t
        denominator = light_supply + p_d
        positive = pt.gt(denominator, 0.0)
        safe = pt.switch(positive, denominator, 1.0)
        p_i = pt.switch(positive, light_supply * p_d / safe, 0.0)
        gpp = unfrozen_t * p_i * dlf_t

        # The temperature term of the Appendix, T, in DALEC2's sampled form.
        temperature_rate = pt.exp(temperature_exponent_ * t_air_t)

        # Williams et al. (2005) Appendix, p. 104. With t2 = f_auto,
        # (1 - t2)(t3, t4, 1 - t3 - t4) = (f_fol, f_roo, f_woo), t5 = c_lf / 365.25,
        # t6 = theta_woo, t7 = theta_roo, t8 = theta_lit, t9 = theta_som and
        # t1 = theta_min, the fluxes are
        #   Af = f_fol G    Lf = t5 Cf    Rh1 = t8 Clit T
        #   Ar = f_roo G    Lr = t7 Cr    Rh2 = t9 Csom T
        #   Aw = f_woo G    Lw = t6 Cw    D   = t1 Clit T
        # and every pool is updated from the time-t state.
        c_fol_next = (1.0 - foliage_turnover_) * c_fol + f_fol_ * gpp   # dCf = Af - Lf
        c_roo_next = (1.0 - theta_roo_) * c_roo + f_roo_ * gpp          # dCr = Ar - Lr
        c_woo_next = (1.0 - theta_woo_) * c_woo + f_woo_ * gpp          # dCw = Aw - Lw
        c_lit_next = (                                                  # dClit = Lf + Lr - Rh1 - D
            (1.0 - (theta_lit_ + theta_min_) * temperature_rate) * c_lit
            + foliage_turnover_ * c_fol
            + theta_roo_ * c_roo
        )
        c_som_next = (                                                  # dCsom = D + Lw - Rh2
            (1.0 - theta_som_ * temperature_rate) * c_som               # (Lw here: departure 1)
            + theta_min_ * temperature_rate * c_lit
            + theta_woo_ * c_woo
        )

        ra = f_auto_ * gpp                                              # Ra = t2 G
        rh = (theta_lit_ * c_lit + theta_som_ * c_som) * temperature_rate  # Rh1 + Rh2
        nee = (ra + rh) - gpp
        return c_fol_next, c_roo_next, c_woo_next, c_lit_next, c_som_next, gpp, ra, rh, nee

    initial = [theta[f"{name}_0"] for name in EVERGREEN_POOL_NAMES]
    outputs = pytensor.scan(
        fn=step,
        sequences=sequences,
        outputs_info=[*initial, None, None, None, None],
        non_sequences=non_sequences,
        strict=True,
        name="dalec_evergreen",
        return_updates=False,
    )
    pools = pt.stack(outputs[: len(EVERGREEN_POOL_NAMES)], axis=1)
    gpp, ra, rh, nee = outputs[len(EVERGREEN_POOL_NAMES):]
    return EvergreenGraph(nee=nee, gpp=gpp, ra=ra, rh=rh, pools=pools, n_days=n_days)
