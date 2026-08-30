"""Parameter registry, prior ranges and derived phenology constants.

Single source of truth for parameter names, units and uniform prior bounds.
Nothing else in the codebase may invent a parameter name or a bound.

Source: Bloom, A. A. and Williams, M. (2015), *Constraining ecosystem carbon
dynamics in a data-limited world*, Biogeosciences 12, 1299-1315 -- the
**published** paper, Table 1 and Appendix A. The 2014 preprint differs in
several places and is not used.

Parameter count
---------------
:class:`DalecParameters` carries 24 fields but the model has **23 free
parameters**, because the five allocation fractions satisfy

    f_auto + f_lab + f_fol + f_roo + f_woo = 1

so one of them is determined by the other four. The 23 break down as:

===========================  ===  =====================================
group                          n  fields
===========================  ===  =====================================
allocation                     4  f_auto + a 4-simplex with 3 free d.o.f.
turnover / decomposition       5  theta_roo, theta_woo, theta_lit,
                                  theta_som, theta_min
temperature                    1  temperature_exponent
phenology                      5  d_onset, cr_onset, d_fall, cr_fall, c_lf
photosynthesis (ACM)           2  lma, ceff
initial pool states            6  c_lab_0 ... c_som_0
===========================  ===  =====================================

Units
-----
Carbon pools g C m-2, fluxes g C m-2 d-1, temperature degrees C, rate constants
d-1, days of year 1-366, periods in days.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import cache
from typing import Any, Final

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "ALLOCATION_CONCENTRATION",
    "ALLOCATION_PARAMETERS",
    "ALLOCATION_WEIGHT_ORDER",
    "ANNUAL_LITTER_INPUT_G_C_M2",
    "ANNUAL_RH_G_C_M2",
    "BELOWGROUND_LITTERFALL_G_C_M2",
    "DAYS_PER_YEAR",
    "DEFAULT_LAI_CONVENTION",
    "EMPIRICAL_TEMPERATURE_EXPONENT",
    "F_SOM_BOUNDS",
    "LAI_CONVENTIONS",
    "LAI_IS_PROJECTED",
    "LITTER_RESIDENCE_TIME_YEARS",
    "MEASURED_ALLOCATION_G_C_M2",
    "MEASURED_FINE_ROOT_CARBON_G_C_M2",
    "NEEDLE_LITTERFALL_G_C_M2",
    "NEEDLE_LONGEVITY_YEARS",
    "PARAMETER_NAMES",
    "PARAMETER_REGISTRY",
    "PHENOLOGY_PARAMETERS",
    "PHOTOSYNTHESIS_PARAMETERS",
    "POOL_STATE_PARAMETERS",
    "REFLEX_CEFF_BOUNDS",
    "RESPIRATION_REFERENCE_TEMPERATURE_C",
    "SIMPLEX_PARAMETERS",
    "SITE_ALL_SIDED_LAI",
    "SOIL_CARBON_STOCK_G_C_M2",
    "SOM_RESIDENCE_TIME_YEARS",
    "SUPERSEDED_ALL_SIDED_TO_PROJECTED",
    "TREE_CARBON_STOCK_G_C_M2",
    "TURNOVER_PARAMETERS",
    "DalecParameters",
    "Parameter",
    "allocation_concentration",
    "allocation_fractions",
    "canopy_bounds",
    "decomposition_multiplier",
    "derive_initial_pools",
    "derive_litter_som_pools",
    "foliar_carbon_from_litterfall",
    "implied_root_turnover",
    "lai_divisor",
    "leaf_mass_per_area_bounds",
    "phenology_psi_f",
    "prior_bounds",
    "reference_respiration_rate",
    "site_lai",
    "solve_psi",
    "som_residence_time_from_stock",
    "turnover_bounds_from_residence_time",
]

#: Tolerance on the allocation closure f_auto + f_lab + f_fol + f_roo + f_woo = 1.
#: Loose enough for accumulated floating-point error in a Dirichlet draw, tight
#: enough that a genuine modelling mistake cannot hide inside it.
ALLOCATION_CLOSURE_TOLERANCE: Final[float] = 1e-9

#: Order of the four-element allocation simplex.
#:
#: **This order is load-bearing.** The Phase 4 Dirichlet prior produces a vector
#: in this order, and :func:`allocation_fractions` consumes it in this order.
#: Changing one without the other silently swaps carbon between pools, which the
#: carbon-conservation test cannot catch -- conservation holds under any
#: permutation.
ALLOCATION_WEIGHT_ORDER: Final[tuple[str, ...]] = ("f_lab", "f_fol", "f_roo", "f_woo")

ALLOCATION_PARAMETERS: Final[tuple[str, ...]] = ("f_auto", *ALLOCATION_WEIGHT_ORDER)
SIMPLEX_PARAMETERS: Final[tuple[str, ...]] = ALLOCATION_WEIGHT_ORDER
TURNOVER_PARAMETERS: Final[tuple[str, ...]] = (
    "theta_roo",
    "theta_woo",
    "theta_lit",
    "theta_som",
    "theta_min",
)
#: Fixed on ecological grounds at FI-Hyy: the site is evergreen Scots pine, so
#: there is no leaf-out/leaf-fall cycle for NEE to inform.
PHENOLOGY_PARAMETERS: Final[tuple[str, ...]] = (
    "d_onset",
    "cr_onset",
    "d_fall",
    "cr_fall",
    "c_lf",
)
PHOTOSYNTHESIS_PARAMETERS: Final[tuple[str, ...]] = ("lma", "ceff")
POOL_STATE_PARAMETERS: Final[tuple[str, ...]] = (
    "c_lab_0",
    "c_fol_0",
    "c_roo_0",
    "c_woo_0",
    "c_lit_0",
    "c_som_0",
)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DalecParameters:
    """A complete DALEC2 parameter set.

    Field names here are the canonical parameter names for the whole project.
    Descriptions, units and prior bounds live in :data:`PARAMETER_REGISTRY`.
    """

    # -- allocation ---------------------------------------------------------
    f_auto: float
    f_lab: float
    f_fol: float
    f_roo: float
    f_woo: float

    # -- turnover and decomposition -----------------------------------------
    theta_roo: float
    theta_woo: float
    theta_lit: float
    theta_som: float
    theta_min: float

    # -- temperature response -----------------------------------------------
    temperature_exponent: float

    # -- phenology ----------------------------------------------------------
    d_onset: float
    cr_onset: float
    d_fall: float
    cr_fall: float
    c_lf: float

    # -- photosynthesis -----------------------------------------------------
    lma: float
    ceff: float

    # -- initial pool states ------------------------------------------------
    c_lab_0: float
    c_fol_0: float
    c_roo_0: float
    c_woo_0: float
    c_lit_0: float
    c_som_0: float

    def __post_init__(self) -> None:
        """Check the allocation closure for concrete numeric parameter sets.

        Skipped when the fractions are not real numbers: the Phase 6 PyTensor
        model builds them symbolically from the same simplex, where closure
        holds by construction and cannot be asserted at graph-build time.
        """
        fractions = (self.f_auto, self.f_lab, self.f_fol, self.f_roo, self.f_woo)
        if not all(isinstance(value, numbers.Real) for value in fractions):
            return
        total = math.fsum(float(value) for value in fractions)
        if not math.isclose(total, 1.0, abs_tol=ALLOCATION_CLOSURE_TOLERANCE):
            raise ValueError(
                "allocation fractions must sum to one "
                f"(f_auto + f_lab + f_fol + f_roo + f_woo = {total!r}). "
                "Build them with allocation_fractions() rather than by hand."
            )

    @classmethod
    def from_allocation_simplex(
        cls,
        *,
        f_auto: float,
        allocation_weights: Sequence[float] | np.ndarray,
        **rest: float,
    ) -> DalecParameters:
        """Construct from ``f_auto`` plus a 4-simplex, closing the fractions exactly.

        Parameters
        ----------
        f_auto
            Autotrophic respiration fraction of GPP, dimensionless.
        allocation_weights
            Four non-negative weights summing to one, ordered as
            :data:`ALLOCATION_WEIGHT_ORDER`. In Phase 4 this is a Dirichlet draw.
        **rest
            Every remaining field of :class:`DalecParameters`.
        """
        f_lab, f_fol, f_roo, f_woo = allocation_fractions(f_auto, allocation_weights)
        return cls(f_auto=f_auto, f_lab=f_lab, f_fol=f_fol, f_roo=f_roo, f_woo=f_woo, **rest)

    @property
    def initial_pools(self) -> np.ndarray:
        """Initial carbon pools as a length-6 array, g C m-2.

        Ordered as ``dalec.model_numpy.POOL_NAMES``.
        """
        return np.array(
            [
                self.c_lab_0,
                self.c_fol_0,
                self.c_roo_0,
                self.c_woo_0,
                self.c_lit_0,
                self.c_som_0,
            ],
            dtype=float,
        )

    @property
    def psi_f(self) -> float:
        """Derived leaf-fall phase offset, days. See :func:`phenology_psi_f`."""
        return phenology_psi_f(self.c_lf, self.cr_fall)

    def to_dict(self) -> dict[str, Any]:
        """Return the parameter set as a plain name to value mapping."""
        return {parameter.name: getattr(self, parameter.name) for parameter in fields(self)}


#: Canonical parameter names, in declaration order.
PARAMETER_NAMES: Final[tuple[str, ...]] = tuple(
    parameter.name for parameter in fields(DalecParameters)
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """One registry entry.

    Attributes
    ----------
    name
        Canonical parameter name; matches a :class:`DalecParameters` field.
    description
        What the parameter means.
    unit
        Physical unit, ``"1"`` for dimensionless.
    lower, upper
        Inclusive bounds of the uniform prior, from the published Table 1.
    simplex
        True for the four allocation fractions. These are **not** sampled from
        their tabulated uniform range; they come out of the Dirichlet split of
        ``1 - f_auto``. The bounds are retained because they are what the paper
        publishes and because Morris screening still needs a range to perturb
        over, but ``priors.py`` must not build a ``Uniform`` from them.
    """

    name: str
    description: str
    unit: str
    lower: float
    upper: float
    simplex: bool = field(default=False)


_REGISTRY_ENTRIES: Final[tuple[Parameter, ...]] = (
    # -- allocation ---------------------------------------------------------
    Parameter("f_auto", "Fraction of GPP respired autotrophically", "1", 0.3, 0.7),
    Parameter("f_lab", "Fraction of GPP allocated to labile", "1", 0.01, 0.5, simplex=True),
    Parameter("f_fol", "Fraction of GPP allocated to foliage", "1", 0.01, 0.5, simplex=True),
    Parameter("f_roo", "Fraction of GPP allocated to fine roots", "1", 0.01, 0.5, simplex=True),
    Parameter("f_woo", "Fraction of GPP allocated to wood", "1", 0.01, 0.5, simplex=True),
    # -- turnover and decomposition -----------------------------------------
    Parameter("theta_roo", "Fine root turnover rate to litter", "d-1", 1.0e-4, 1.0e-2),
    Parameter("theta_woo", "Wood turnover rate to soil organic matter", "d-1", 2.5e-5, 1.0e-3),
    Parameter("theta_lit", "Litter respiration rate", "d-1", 1.0e-4, 1.0e-2),
    Parameter("theta_som", "Soil organic matter respiration rate", "d-1", 1.0e-7, 1.0e-3),
    Parameter(
        "theta_min",
        "Litter mineralisation rate, i.e. the litter to SOM transfer. A "
        "transfer, not a loss: it does not appear in ecosystem respiration.",
        "d-1",
        1.0e-5,
        1.0e-2,
    ),
    # -- temperature response -----------------------------------------------
    Parameter(
        "temperature_exponent",
        "Exponential temperature dependence of decomposition, Theta in A5/A6. "
        "The decomposition multiplier is exp(temperature_exponent * T) with T "
        "the daily mean air temperature.",
        "degC-1",
        0.018,
        0.08,
    ),
    # -- phenology ----------------------------------------------------------
    Parameter("d_onset", "Day of leaf onset", "day of year", 1.0, 365.0),
    Parameter("cr_onset", "Labile release period", "day", 10.0, 100.0),
    Parameter("d_fall", "Day of leaf fall", "day of year", 1.0, 365.0),
    Parameter("cr_fall", "Leaf fall period", "day", 20.0, 150.0),
    Parameter(
        "c_lf",
        "Annual leaf fall fraction, the reciprocal of leaf lifespan in years. "
        "The published A7/A8 use c_lf directly through -log(1 - c_lf); the "
        "superseded preprint used leaf lifespan c_lspan = 1 / c_lf instead.",
        "1",
        0.125,
        1.0,
    ),
    # -- photosynthesis -----------------------------------------------------
    Parameter("lma", "Leaf mass per area", "g C m-2", 10.0, 400.0),
    Parameter("ceff", "Canopy efficiency", "1", 10.0, 100.0),
    # -- initial pool states ------------------------------------------------
    Parameter("c_lab_0", "Initial labile carbon", "g C m-2", 20.0, 2000.0),
    Parameter("c_fol_0", "Initial foliar carbon", "g C m-2", 20.0, 2000.0),
    Parameter("c_roo_0", "Initial fine root carbon", "g C m-2", 20.0, 2000.0),
    Parameter("c_woo_0", "Initial wood carbon", "g C m-2", 100.0, 1.0e5),
    Parameter("c_lit_0", "Initial litter carbon", "g C m-2", 20.0, 2000.0),
    Parameter("c_som_0", "Initial soil organic matter carbon", "g C m-2", 100.0, 2.0e5),
)

#: Name to :class:`Parameter`, in declaration order.
PARAMETER_REGISTRY: Final[dict[str, Parameter]] = {
    entry.name: entry for entry in _REGISTRY_ENTRIES
}


def prior_bounds(name: str) -> tuple[float, float]:
    """Return the inclusive uniform prior bounds for a parameter.

    Parameters
    ----------
    name
        Canonical parameter name.

    Returns
    -------
    ``(lower, upper)`` in the parameter's registered unit.

    Raises
    ------
    KeyError
        If ``name`` is not a registered parameter.

    Notes
    -----
    For the four allocation fractions these are the published marginal ranges,
    but the sampler does **not** draw from them: they come out of the Dirichlet
    split of ``1 - f_auto`` instead. See :func:`allocation_fractions`.
    """
    try:
        entry = PARAMETER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown parameter {name!r}; expected one of {sorted(PARAMETER_REGISTRY)}"
        ) from None
    return entry.lower, entry.upper


# ---------------------------------------------------------------------------
# Allocation simplex
# ---------------------------------------------------------------------------


def allocation_fractions(
    f_auto: float, weights: Sequence[float] | np.ndarray
) -> tuple[float, float, float, float]:
    """Split ``1 - f_auto`` across labile, foliage, fine root and wood.

    This is the deterministic half of the simplex reparameterisation; the
    stochastic half is a Dirichlet prior over ``weights``.

    Two things are going on here, and they are separate.

    **The closure is corrected.** Bloom & Williams (2015) Table 1, footnote 1
    gives ``f_woo = 1 - f_auto - f_fol - f_lab``, omitting ``f_roo``. That is an
    error in the paper -- identical in the preprint and the published version.
    Equations A1-A4 show four distinct allocation flows out of GPP, A10 adds
    autotrophic respiration as a fifth, and ``f_roo`` carries its own non-zero
    prior range in the same table. The closure implemented here is the corrected
    five-term one:

        f_auto + f_lab + f_fol + f_roo + f_woo = 1

    **The sampling is reparameterised.** Sampling the fractions independently
    and obtaining one by subtraction lets it go negative, which is a hard
    rejection wall in the posterior. NUTS follows gradients of the log density
    and cannot differentiate through a wall, so it stalls against it and
    produces divergences. Routing the constraint through a simplex means every
    point in the unconstrained sampling space maps to a valid, non-negative
    allocation.

    Parameters
    ----------
    f_auto
        Autotrophic respiration fraction of GPP, dimensionless, in ``[0, 1]``.
    weights
        Four non-negative weights summing to one, ordered as
        :data:`ALLOCATION_WEIGHT_ORDER`.

    Returns
    -------
    ``(f_lab, f_fol, f_roo, f_woo)``, dimensionless, summing to ``1 - f_auto``.

    Raises
    ------
    ValueError
        If ``f_auto`` is outside ``[0, 1]``, or ``weights`` is not a
        four-element non-negative simplex.
    """
    if not 0.0 <= f_auto <= 1.0:
        raise ValueError(f"f_auto must lie in [0, 1], got {f_auto!r}")

    weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != (4,):
        raise ValueError(
            f"expected 4 allocation weights ordered as {ALLOCATION_WEIGHT_ORDER}, "
            f"got shape {weight_array.shape}"
        )
    if not np.all(np.isfinite(weight_array)):
        raise ValueError(f"allocation weights must be finite, got {weight_array!r}")
    if np.any(weight_array < 0.0):
        raise ValueError(f"allocation weights must be non-negative, got {weight_array!r}")
    total = float(weight_array.sum())
    if not math.isclose(total, 1.0, abs_tol=ALLOCATION_CLOSURE_TOLERANCE):
        raise ValueError(f"allocation weights must sum to one, got {total!r}")

    remainder = 1.0 - f_auto
    f_lab, f_fol, f_roo, f_woo = (float(weight * remainder) for weight in weight_array)
    return f_lab, f_fol, f_roo, f_woo


# ---------------------------------------------------------------------------
# Derived phenology constant: psi
# ---------------------------------------------------------------------------

#: Bracket for the psi root. The root is always negative -- see :func:`solve_psi`.
_PSI_BRACKET: Final[tuple[float, float]] = (-50.0, -1e-12)


def _psi_residual(psi: float, c_lf: float) -> float:
    """Published equation A9: ``2*sqrt(pi)*log(1 - c_lf)*psi - exp(-psi**2)``."""
    return 2.0 * math.sqrt(math.pi) * math.log1p(-c_lf) * psi - math.exp(-psi * psi)


@cache
def solve_psi(c_lf: float) -> float:
    """Solve published equation A9 for ``psi``.

    Bloom & Williams resolve this with a sixth-order polynomial fitted in the
    DALEC2 source, because they sampled ``c_lf`` and needed ``psi`` millions of
    times. Here ``c_lf`` is fixed, so ``psi`` is a single startup constant and a
    numerical root is both simpler and exact.

    The root is always **negative**, unique, and monotone increasing in ``c_lf``
    over the whole prior range, so a fixed negative bracket suffices and no
    special-casing is needed. Bracketing on the positive axis fails.

    Cached: the solve happens once per distinct ``c_lf``.

    Parameters
    ----------
    c_lf
        Annual leaf fall fraction, dimensionless, in ``(0, 1)``.

    Returns
    -------
    ``psi``, dimensionless and negative.

    Raises
    ------
    ValueError
        If ``c_lf`` is outside ``(0, 1)``. At exactly 1 the equation is singular
        because ``log(1 - c_lf)`` diverges; the published prior range is
        ``[1/8, 1]``, so the upper endpoint is unusable and must be excluded.
    """
    if not 0.0 < c_lf < 1.0:
        raise ValueError(
            f"c_lf must lie strictly inside (0, 1), got {c_lf!r}. "
            "The published prior range is [1/8, 1], but log(1 - c_lf) diverges "
            "at the upper endpoint."
        )

    psi = float(brentq(_psi_residual, *_PSI_BRACKET, args=(c_lf,), xtol=1e-15, rtol=8.9e-16))
    if not math.isfinite(psi):
        raise ValueError(f"psi solve returned a non-finite root for c_lf={c_lf!r}")
    return psi


@cache
def phenology_psi_f(c_lf: float, cr_fall: float) -> float:
    """Leaf-fall phase offset ``psi_f = psi * cr_fall / sqrt(2)``, days.

    A derived constant, not a free parameter. Because ``c_lf`` and ``cr_fall``
    are both fixed at this site, this is evaluated once and injected into the
    Phase 6 PyTensor graph as a constant. If either were ever freed, the graph
    would need implicit differentiation through :func:`solve_psi` rather than a
    baked-in number.
    """
    psi_f = solve_psi(c_lf) * cr_fall / math.sqrt(2.0)
    if not math.isfinite(psi_f):
        raise ValueError(f"psi_f is not finite for c_lf={c_lf!r}, cr_fall={cr_fall!r}")
    return psi_f


# ---------------------------------------------------------------------------
# Heterotrophic respiration, reparameterised
# ---------------------------------------------------------------------------
#
# Why this exists
# ---------------
# Tasks 1 and 2 both isolated the same failure. Sampling ``theta_som`` and
# ``c_som_0`` independently over their Bloom & Williams ranges puts most of the
# prior mass on a *product* -- the soil respiration flux -- that is physically
# impossible. The prior median produces 50 g C m-2 d-1 of soil respiration
# against a site whose entire net exchange is around 0.6, 837 of 1000 draws
# failed the screen, and the two parameters ranked first and second in the OAAT
# sensitivity while being individually meaningless.
#
# The fix is to sample the flux and the partition, and derive the stocks:
#
#     sampled    rh_ref, f_som, theta_lit, theta_som
#     derived    c_lit_0 = (1 - f_som) * rh_ref / theta_lit
#                c_som_0 =      f_som  * rh_ref / theta_som
#
# ``rh_ref`` is total heterotrophic respiration at the reference temperature,
# which is **0 degrees C**, because the DALEC A5/A6 multiplier is
# ``exp(Theta * T)`` and that equals one at T = 0. ``f_som`` is the SOM share of
# it. The derived pools are exactly the stocks that produce ``rh_ref`` at steady
# state, so a draw cannot place 10^5 g C m-2 of soil behind a fast turnover rate.
#
# This is a genuine prior change and it supersedes the bounded-uniform priors on
# ``c_lit_0`` and ``c_som_0``. See DECISIONS.md section 7.

#: Reference temperature for ``rh_ref``, degrees C. Not a free choice: the
#: decomposition multiplier ``exp(Theta * T)`` is one here by construction.
RESPIRATION_REFERENCE_TEMPERATURE_C: Final[float] = 0.0

#: Annual heterotrophic respiration at FI-Hyy, g C m-2 yr-1, as (low, high).
#:
#: Ilvesniemi et al. (2009) Fig. 6, "decomposition of soil organic matter",
#: ~290-370. This is an *attribution of a measured flux*: soil CO2 efflux was
#: measured at 577-737 g C m-2 yr-1 and split roughly 50/50 between root and
#: rhizosphere respiration and decomposition on the basis of a girdling
#: experiment at the site. **It does not assume steady state**, so the fact that
#: FI-Hyy is a sink does not bias it.
ANNUAL_RH_G_C_M2: Final[tuple[float, float]] = (290.0, 370.0)

#: Annual litter input, g C m-2 yr-1, as (low, central, high). Ilvesniemi et al.
#: (2009): above-ground tree litter 142-204, below-ground tree litter ~90,
#: ground vegetation ~15, summing to 247-309 and consistent with the 280 often
#: quoted.
#:
#: **Corroboration only. Not the basis of the prior.** Equating litter input with
#: heterotrophic respiration assumes long-run steady state, and FI-Hyy is a
#: measured sink of ~206 g C m-2 yr-1. At a sink, input necessarily exceeds
#: decomposition, so the litter input is an **upper bound** on Rh, not an
#: estimate of it. The bound is not violated: 247-309 sits above the 290-370
#: flux attribution only in part, and the two overlap, which is the corroboration.
ANNUAL_LITTER_INPUT_G_C_M2: Final[tuple[float, float, float]] = (247.0, 280.0, 309.0)

#: Measured soil carbon stock at FI-Hyy, g C m-2. Ilvesniemi et al. (2009)
#: Fig. 6. Used to derive the SOM residence time below; see DECISIONS.md section
#: 7 for why that retires Check 1 as an independent validation.
SOIL_CARBON_STOCK_G_C_M2: Final[float] = 6560.0

#: Empirical temperature sensitivity, degC-1, measured from the FI-Hyy winter
#: flux record (EDA_NOTES.md, eda05). Retained as the reporting reference; the
#: prior itself now uses each draw's own sampled temperature_exponent.
EMPIRICAL_TEMPERATURE_EXPONENT: Final[float] = 0.0366

#: Residence times against the respiratory pathway, years, as (low, high).
#:
#: **Definition matters here.** ``tau = 1 / theta``, the e-folding time against
#: *respiratory loss alone*. The litter pool's total residence time is shorter
#: than ``tau_lit`` because ``theta_min`` also drains it to SOM, and that
#: transfer is not a respiratory loss. Reading a published *pool* turnover time
#: straight into ``theta_lit`` would attribute the mineralisation flux to
#: respiration and understate the litter stock.
#:
#: ``tau_lit`` 1-5 yr is supported by Yasso07 (Tuomi et al. 2009, Table 1): the
#: labile AWEN fractions decompose at 0.66, 4.3, 0.35 and 0.22 a-1, i.e.
#: residence times of 1.5, 0.23, 2.9 and 4.5 years, which bracket this range.
#:
#: ``tau_som`` 20-45 yr is derived from the site's own measurements rather than
#: from the literature, because no usable published value was found. With total
#: soil carbon S, total heterotrophic respiration R and SOM share f,
#:
#:     S / R = (1 - f) * tau_lit + f * tau_som
#:
#: **at the reference temperature**, which matters: tau = 1 / theta is the
#: e-folding time at 0 degrees C, while the field residence time is tau / M.
#: At steady state C = R * tau / M, so tau_ref = (S * M / R - (1-f) * tau_lit)/f.
#: S = 6560, R = 290-370 and M = 1.2406 (empirical Theta) to 1.3677 (prior
#: median Theta) give tau_som of 23.9-60.9 yr across f in [0.5, 0.9] and tau_lit
#: in [1, 5]. Adopted as 24-61 yr. Omitting the M factor would understate it by
#: about 25% and the implied soil stock by the same factor.
#:
#: Yasso07's humus rate alpha_H = 3.3e-3 a-1 (303 yr) is **not** used: Yasso's H
#: is the recalcitrant fraction receiving about 4% of labile mass loss, whereas
#: DALEC's SOM pool is bulk soil carbon carrying most of the heterotrophic flux.
#: They are different objects and the rate is not transferable.
LITTER_RESIDENCE_TIME_YEARS: Final[tuple[float, float]] = (1.0, 5.0)
SOM_RESIDENCE_TIME_YEARS: Final[tuple[float, float]] = (24.0, 61.0)

#: Leaf area index convention that ACM expects: **projected** (one-sided).
#:
#: Not an assumption -- two independent lines of evidence from the fitted
#: coefficients themselves. Chuter B4 half-saturates canopy quantum yield at
#: ``L = sqrt(a9)``, which is 1.45 for Loobos and 1.03 for Oregon; on an
#: all-sided basis those would be 0.6 and 0.4, far below where any canopy's light
#: capture saturates. And both sets were fitted with ``lma_reference`` near 110
#: g C m-2, which is a projected-basis needle mass for pine -- all-sided it would
#: be about 44. So ``lma`` here is leaf carbon per unit **projected** leaf area.
LAI_IS_PROJECTED: Final[bool] = True

#: Divisors taking Kolari's all-sided LAI to the basis ACM is read on.
#:
#: **Both stay live and neither is adopted** (DECISIONS §10, LIMITATIONS §15).
#: ACM aggregates SPA, parametrised for a broadleaf stand where projected,
#: hemisurface and half-total coincide, so no needle geometry entered the
#: calibration and its own provenance cannot supply the convention. The
#: calibration is run under both and the difference reported as a sensitivity.
#:
#: ``hemisurface`` -- exactly 2, by definition (Chen & Black 1992).
#: ``projected``   -- 2.5708 for Scots pine (Niinemets et al. 2001), matching
#:                    bisected-cylinder geometry (pi + 2)/2 (Grace 1987).
LAI_CONVENTIONS: Final[dict[str, float]] = {
    "hemisurface": 2.0,
    "projected": (math.pi + 2.0) / 2.0,
}

#: Which convention a caller gets if it does not say. Not an adoption: it is the
#: one with the better physical argument, and every result carries the label.
DEFAULT_LAI_CONVENTION: Final[str] = "hemisurface"

#: Superseded. The working divisor of 2.5 used before the two conventions were
#: separated; it is neither of them and is kept only to explain the lma bounds
#: recorded in DECISIONS §8 as U(144, 241).
SUPERSEDED_ALL_SIDED_TO_PROJECTED: Final[float] = 2.5

#: Seasonal maximum **all-sided** LAI at SMEAR II, dimensionless, as
#: (after thinning, before thinning). Kolari (2010) Dissertationes Forestales 99:
#: ~8.0 before the 2002 thinning and ~6.5 after, a 19% reduction.
#:
#: **This is the source of the previously unsourced "~3"**; converting it needs a
#: convention, which is what :data:`LAI_CONVENTIONS` supplies.
SITE_ALL_SIDED_LAI: Final[tuple[float, float]] = (6.5, 8.0)


def site_lai(convention: str = DEFAULT_LAI_CONVENTION) -> tuple[float, float]:
    """Seasonal maximum LAI on the requested basis, (after, before) thinning."""
    divisor = lai_divisor(convention)
    return tuple(value / divisor for value in SITE_ALL_SIDED_LAI)  # type: ignore[return-value]


def lai_divisor(convention: str = DEFAULT_LAI_CONVENTION) -> float:
    """Divisor taking all-sided LAI to ``convention``.

    Raises rather than defaulting on an unknown name: a silently wrong divisor
    is a 29% error in ``lma`` and a 37% error in GPP.
    """
    try:
        return LAI_CONVENTIONS[convention]
    except KeyError:
        raise KeyError(
            f"unknown LAI convention {convention!r}; expected one of "
            f"{sorted(LAI_CONVENTIONS)}"
        ) from None

#: Measured annual carbon flows at FI-Hyy used to set the allocation prior,
#: g C m-2 yr-1. Ilvesniemi et al. (2009) Fig. 6, all measured, none fitted:
#:
#:     foliage    needle litterfall            154
#:     fine root  below-ground tree litter      90
#:     wood       above-ground net growth 180-240 plus below-ground 34-69
#:
#: At steady state allocation equals turnover, so these are allocation fluxes.
#: They sum to about 0.49 of GPP ~ 1030, which is (1 - f_auto) at f_auto ~ 0.51 --
#: independently inside the published f_auto prior, and a useful consistency
#: check on the whole picture.
MEASURED_ALLOCATION_G_C_M2: Final[dict[str, float]] = {
    "foliage": 154.0,
    "fine_root": 90.0,
    "wood": 261.0,
}

#: ``ceff`` prior, dimensionless. Fox et al. (2009), the REFLEX model
#: intercomparison, Table 4, published range for the same ACM parameter ``a1``.
#:
#: **This is adopting a published prior, not fitting to GPP.** It is narrower
#: than the DALEC2 Table 1 U(10, 100) and it was not chosen by looking at the
#: modelled GPP; it happens to move GPP downward, which is a consequence rather
#: than the reason. Nothing here is tuned to the measured 952-1104.
REFLEX_CEFF_BOUNDS: Final[tuple[float, float]] = (5.0, 20.0)

#: Total tree carbon stock at FI-Hyy, g C m-2. Ilvesniemi et al. (2009) Fig. 6.
#: Wood is whatever remains of it after the derived labile, foliar and fine-root
#: pools are taken out.
TREE_CARBON_STOCK_G_C_M2: Final[float] = 6800.0

#: Below-ground tree litter production, g C m-2 yr-1. Ilvesniemi et al. (2009)
#: Fig. 6, ~90; the paper assumes it equals needle litter. Sets the fine-root
#: stock through the root turnover rate.
BELOWGROUND_LITTERFALL_G_C_M2: Final[float] = 90.0

#: Measured living fine-root biomass, g C m-2. Ilvesniemi et al. (2009) Table 4
#: gives 476 g m-2 dry biomass; halved for carbon. Not used in any prior -- kept
#: as an independent check on the derived ``c_roo_0``.
MEASURED_FINE_ROOT_CARBON_G_C_M2: Final[float] = 238.0

#: Total Dirichlet concentration for the allocation simplex. Sets how tightly the
#: prior holds the measured shares. 20 gives a standard deviation on the foliar
#: share of about 0.10, wide enough to contain the Fig. 6 spread without pinning
#: the allocation to a point. The superseded flat Dirichlet is concentration 4.
ALLOCATION_CONCENTRATION: Final[float] = 20.0

#: Measured above-ground needle litterfall, g C m-2 yr-1, as (low, mean, high).
#: Ilvesniemi et al. (2009): annual average 154, Fig. 6 range 142-204.
NEEDLE_LITTERFALL_G_C_M2: Final[tuple[float, float, float]] = (142.0, 154.0, 204.0)

#: Scots pine needle longevity in southern Finland, years, as (low, high). Sets
#: ``c_lf``, the annual leaf fall fraction, as its reciprocal.
NEEDLE_LONGEVITY_YEARS: Final[tuple[float, float]] = (3.0, 5.0)

#: SOM share of total heterotrophic respiration, dimensionless, as (low, high).
#: Boreal heterotrophic respiration is dominated by the humus and mineral soil
#: rather than by fresh litter, but not overwhelmingly so. A judgement, not a
#: measurement.
F_SOM_BOUNDS: Final[tuple[float, float]] = (0.5, 0.9)

#: Days per year used for every annual-to-daily conversion here.
DAYS_PER_YEAR: Final[float] = 365.25


def decomposition_multiplier(
    t_air: np.ndarray, temperature_exponent: float = EMPIRICAL_TEMPERATURE_EXPONENT
) -> float:
    """Mean of ``exp(Theta * T)`` over a driver series.

    This is ``mean(exp(Theta * T))`` and **not** ``exp(Theta * mean(T))``. The
    two differ by Jensen's inequality -- ``exp`` is convex, so the mean of the
    multiplier exceeds the multiplier of the mean whenever temperature varies.
    At FI-Hyy the daily air temperature has a standard deviation of about 9.4
    degrees C and the gap is roughly 6%. Using the naive form would inflate
    ``rh_ref`` by that much and quietly bias every derived carbon stock.

    Parameters
    ----------
    t_air
        Daily mean air temperature over the record, degrees C.
    temperature_exponent
        Theta in A5/A6, degC-1.

    Returns
    -------
    The dimensionless mean multiplier, > 1 for any varying series with mean
    above the reference temperature.
    """
    values = np.asarray(t_air, dtype=float)
    if values.size == 0:
        raise ValueError("cannot take a temperature multiplier over an empty series")
    if not np.all(np.isfinite(values)):
        raise ValueError("driver temperature series contains non-finite values")
    return float(np.mean(np.exp(float(temperature_exponent) * values)))


def reference_respiration_rate(
    annual_rh: np.ndarray | float,
    temperature_exponent: np.ndarray | float,
    t_air: np.ndarray,
) -> np.ndarray:
    """Daily heterotrophic respiration at the reference temperature, g C m-2 d-1.

    Inverts the annual total for each draw **at that draw's own Theta**:

        annual_rh = rh_ref * sum(exp(Theta * T)) = rh_ref * M(Theta) * 365.25

    so ``rh_ref = annual_rh / (M(Theta) * 365.25)``.

    Using a single multiplier computed at one fixed Theta, while
    ``temperature_exponent`` is itself sampled over U(0.018, 0.08), is an
    inconsistency: the realised annual respiration then departs from the target
    for every draw whose Theta is not that fixed value. Measured on the earlier
    construction, median realised annual Rh drifted from an intended 280 to
    299.8 g C m-2 yr-1. Computing ``M`` per draw removes the drift entirely, so
    every draw respires its sampled annual total by construction.

    ``M`` is the mean of the exponential and not the exponential of the mean.
    ``exp`` is convex, so the two differ by Jensen's inequality -- about 6% at
    FI-Hyy, where daily air temperature has a standard deviation near 9.4
    degrees C. The naive form would inflate ``rh_ref`` and every stock derived
    from it.

    Parameters
    ----------
    annual_rh
        Target annual heterotrophic respiration per draw, g C m-2 yr-1.
    temperature_exponent
        Theta per draw, degC-1.
    t_air
        Daily mean air temperature over the record, degrees C.

    Returns
    -------
    ``rh_ref`` per draw, g C m-2 d-1 at the reference temperature.
    """
    values = np.asarray(t_air, dtype=float)
    if values.size == 0:
        raise ValueError("cannot take a temperature multiplier over an empty series")
    if not np.all(np.isfinite(values)):
        raise ValueError("driver temperature series contains non-finite values")
    target = np.asarray(annual_rh, dtype=float)
    theta = np.asarray(temperature_exponent, dtype=float)
    if np.any(target <= 0.0):
        raise ValueError("annual heterotrophic respiration must be positive")
    multiplier = np.mean(np.exp(theta[..., None] * values), axis=-1)
    return target / (multiplier * DAYS_PER_YEAR)


def som_residence_time_from_stock(
    annual_rh: float,
    f_som: float,
    litter_residence_time_years: float,
    soil_carbon_stock: float = SOIL_CARBON_STOCK_G_C_M2,
) -> float:
    """SOM residence time implied by a measured stock and respiration, years.

    From ``S = (1 - f) * R * tau_lit + f * R * tau_som``, the partition of a
    measured soil carbon stock between a fast litter pool and a slow SOM pool:

        tau_som = (S / R - (1 - f) * tau_lit) / f

    Used to derive :data:`SOM_RESIDENCE_TIME_YEARS`, and retained so that
    derivation is reproducible rather than a bare pair of numbers.
    """
    if annual_rh <= 0.0:
        raise ValueError("annual heterotrophic respiration must be positive")
    if not 0.0 < f_som <= 1.0:
        raise ValueError("f_som must lie in (0, 1]")
    effective = soil_carbon_stock / annual_rh
    return (effective - (1.0 - f_som) * litter_residence_time_years) / f_som


def turnover_bounds_from_residence_time(
    residence_time_years: tuple[float, float],
) -> tuple[float, float]:
    """Convert a ``(low, high)`` residence time in years to rate bounds in d-1.

    The mapping ``theta = 1 / (tau * 365.25)`` is order-reversing, so the long
    residence time gives the lower rate.
    """
    low, high = residence_time_years
    if not 0.0 < low <= high:
        raise ValueError(f"residence times are not ordered: {residence_time_years!r}")
    return 1.0 / (high * DAYS_PER_YEAR), 1.0 / (low * DAYS_PER_YEAR)


def derive_litter_som_pools(
    rh_ref: np.ndarray | float,
    f_som: np.ndarray | float,
    theta_lit: np.ndarray | float,
    theta_som: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state litter and SOM stocks implied by a respiration draw.

    ``c_lit_0 = (1 - f_som) * rh_ref / theta_lit`` and
    ``c_som_0 = f_som * rh_ref / theta_som``: the stocks whose respiration at the
    reference temperature sums to ``rh_ref``, split by ``f_som``.

    Returns
    -------
    ``(c_lit_0, c_som_0)`` in g C m-2, broadcast to the inputs' shape.
    """
    rh = np.asarray(rh_ref, dtype=float)
    share = np.asarray(f_som, dtype=float)
    lit_rate = np.asarray(theta_lit, dtype=float)
    som_rate = np.asarray(theta_som, dtype=float)
    if np.any(share < 0.0) or np.any(share > 1.0):
        raise ValueError("f_som must lie in [0, 1]")
    if np.any(lit_rate <= 0.0) or np.any(som_rate <= 0.0):
        raise ValueError("turnover rates must be strictly positive")
    return (1.0 - share) * rh / lit_rate, share * rh / som_rate


def foliar_carbon_from_litterfall(
    litterfall: np.ndarray | float, c_lf: np.ndarray | float
) -> np.ndarray:
    """Steady-state foliar carbon, g C m-2.

    At steady state the foliage loses what it gains, so ``C_fol = litterfall /
    c_lf``: the stock a measured litterfall implies given a leaf lifespan.
    """
    fall = np.asarray(litterfall, dtype=float)
    rate = np.asarray(c_lf, dtype=float)
    if np.any(rate <= 0.0) or np.any(rate > 1.0):
        raise ValueError("c_lf must lie in (0, 1]")
    return fall / rate


def leaf_mass_per_area_bounds(
    litterfall: tuple[float, float, float] = NEEDLE_LITTERFALL_G_C_M2,
    longevity: tuple[float, float] = NEEDLE_LONGEVITY_YEARS,
    convention: str = DEFAULT_LAI_CONVENTION,
) -> tuple[float, float]:
    """``lma`` bounds implied by measured litterfall, longevity and LAI.

    ``lma = C_fol / LAI_projected`` with ``C_fol = litterfall / c_lf``. Every
    input is measured at the site; nothing is solved back from a target GPP.

    The **mean** litterfall is used, not the Fig. 6 range. That range is
    inter-annual and spatial spread in a quantity whose average is the better
    estimate; multiplying its extremes by the longevity extremes compounds two
    uncertainties and gives a needlessly wide 133-319 g C m-2. Holding litterfall
    at 154 isolates the longevity uncertainty, which is the one that matters.

    ``lma`` depends on the LAI convention and both stay live:

        hemisurface (divisor 2.000)  ->  LAI 4.00  ->  lma 116-193
        projected   (divisor 2.571)  ->  LAI 3.11  ->  lma 148-247

    The superseded working divisor of 2.5 gave LAI 3.2 and lma 144-241, which is
    what DECISIONS section 8 records; it is neither convention.
    """
    _low_fall, mean_fall, _high_fall = litterfall
    low_life, high_life = longevity
    lai = SITE_ALL_SIDED_LAI[1] / lai_divisor(convention)
    return (mean_fall * low_life / lai, mean_fall * high_life / lai)


def allocation_concentration(
    measured: dict[str, float] | None = None,
    total: float = ALLOCATION_CONCENTRATION,
) -> np.ndarray:
    """Dirichlet concentration over :data:`ALLOCATION_WEIGHT_ORDER`.

    Built from the site's measured allocation fluxes rather than left flat. The
    flat Dirichlet sends 24% of GPP to foliage and labile where the measured
    needle litterfall implies 15%, and that excess drives a feedback -- more leaf
    area gives more GPP gives more foliar allocation -- which ran modelled LAI to
    a 95th percentile of 48 against a site value near 3.

    The labile and foliar weights are set equal: DALEC's labile pool exists to
    feed foliage at bud burst, so the measurement constrains their sum, not the
    split between them.

    Returns
    -------
    Concentrations ordered as ``ALLOCATION_WEIGHT_ORDER``, summing to ``total``.
    """
    flows = dict(MEASURED_ALLOCATION_G_C_M2 if measured is None else measured)
    net_primary = sum(flows.values())
    if net_primary <= 0.0:
        raise ValueError("measured allocation fluxes must be positive")
    share = {name: value / net_primary for name, value in flows.items()}
    half_foliage = 0.5 * share["foliage"]
    weights = {
        "f_lab": half_foliage,
        "f_fol": half_foliage,
        "f_roo": share["fine_root"],
        "f_woo": share["wood"],
    }
    return np.array([total * weights[name] for name in ALLOCATION_WEIGHT_ORDER])


def canopy_bounds(
    convention: str = DEFAULT_LAI_CONVENTION,
) -> dict[str, tuple[float, float]]:
    """Site-informed bounds for the two canopy parameters, overriding Table 1.

    ``lma`` U(145, 240) g C m-2, from :func:`leaf_mass_per_area_bounds`. The
    published U(10, 400) admits 10 g C m-2, thinner than any conifer needle.

    ``c_lf`` U(0.20, 0.333), the reciprocal of a 3-5 year needle longevity. The
    published U(0.125, 1.0) spans lifespans of 1 to 8 years, and its upper half
    has an evergreen shedding its needles inside eighteen months.

    ``ceff`` U(5, 20), :data:`REFLEX_CEFF_BOUNDS`, the published REFLEX range for
    the same ACM parameter. A published prior adopted in place of a wider
    published prior -- not a value fitted to the measured GPP.

    **These are deliberately not written into the registry.** Tasks 1 and 2 are
    stated against the published priors and their numbers must stay reproducible,
    so the override lives with the sampler that uses it.
    """
    low_life, high_life = NEEDLE_LONGEVITY_YEARS
    lower, upper = leaf_mass_per_area_bounds(convention=convention)
    return {
        "lma": (float(round(lower)), float(round(upper))),
        "c_lf": (1.0 / high_life, 1.0 / low_life),
        "ceff": REFLEX_CEFF_BOUNDS,
    }


def derive_initial_pools(
    c_lf: np.ndarray | float,
    theta_roo: np.ndarray | float,
    f_lab: np.ndarray | float,
    f_fol: np.ndarray | float,
    *,
    foliar_litterfall: float = NEEDLE_LITTERFALL_G_C_M2[1],
    root_litterfall: float = BELOWGROUND_LITTERFALL_G_C_M2,
    tree_carbon: float = TREE_CARBON_STOCK_G_C_M2,
) -> dict[str, np.ndarray]:
    """Initial labile, foliar, fine-root and wood pools from measured fluxes.

    The same logic as ``rh_ref``: put the prior on a measured flux and a
    turnover rate, and let the stock follow, rather than on the stock directly.

    ``c_fol_0 = foliar_litterfall / c_lf``
        At steady state foliage loses ``c_lf * C_fol`` per year and gains
        ``(f_fol + f_lab) * GPP``; the measured litterfall is that flux, so the
        stock follows without needing GPP at all.

    ``c_lab_0 = foliar_litterfall * f_lab / (f_lab + f_fol)``
        The labile pool accumulates its share of the same foliar flux over a
        year and discharges it at bud burst, so at the 1 January start of the
        block it holds about one year's accumulation.

    ``c_roo_0 = root_litterfall / (theta_roo * 365.25)``
        Root turnover carries no temperature term in A3, so the annual flux is
        simply ``theta_roo * C_roo * 365.25``.

    ``c_woo_0 = tree_carbon - c_lab_0 - c_fol_0 - c_roo_0``
        Wood is the remainder of the measured tree carbon stock.

    Returns
    -------
    Mapping of pool name to value, g C m-2, broadcast over the inputs.
    """
    fall = np.asarray(c_lf, dtype=float)
    root_rate = np.asarray(theta_roo, dtype=float)
    labile_share = np.asarray(f_lab, dtype=float)
    foliar_share = np.asarray(f_fol, dtype=float)
    if np.any(fall <= 0.0) or np.any(fall > 1.0):
        raise ValueError("c_lf must lie in (0, 1]")
    if np.any(root_rate <= 0.0):
        raise ValueError("theta_roo must be strictly positive")
    total_share = labile_share + foliar_share
    if np.any(total_share <= 0.0):
        raise ValueError("f_lab + f_fol must be strictly positive")

    c_fol_0 = foliar_litterfall / fall
    c_lab_0 = foliar_litterfall * labile_share / total_share
    c_roo_0 = root_litterfall / (root_rate * DAYS_PER_YEAR)
    c_woo_0 = tree_carbon - c_lab_0 - c_fol_0 - c_roo_0
    if np.any(c_woo_0 <= 0.0):
        raise ValueError(
            "derived wood carbon is not positive: the labile, foliar and fine-root "
            f"pools exhaust the {tree_carbon} g C m-2 tree stock on "
            f"{int(np.count_nonzero(np.asarray(c_woo_0) <= 0.0))} draw(s). The "
            "usual cause is theta_roo running to the slow end of its range."
        )
    return {
        "c_lab_0": np.asarray(c_lab_0, dtype=float),
        "c_fol_0": np.asarray(c_fol_0, dtype=float),
        "c_roo_0": np.asarray(c_roo_0, dtype=float),
        "c_woo_0": np.asarray(c_woo_0, dtype=float),
    }


def implied_root_turnover(
    root_litterfall: float = BELOWGROUND_LITTERFALL_G_C_M2,
    fine_root_carbon: float = MEASURED_FINE_ROOT_CARBON_G_C_M2,
) -> float:
    """``theta_roo`` implied by the site's own measured root stock and flux, d-1.

    Reported, not adopted. It is a check on whether the published ``theta_roo``
    range is plausible here, and it is not currently used to narrow it.
    """
    return root_litterfall / (fine_root_carbon * DAYS_PER_YEAR)
