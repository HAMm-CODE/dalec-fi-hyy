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
    "ALLOCATION_PARAMETERS",
    "ALLOCATION_WEIGHT_ORDER",
    "PARAMETER_NAMES",
    "PARAMETER_REGISTRY",
    "PHENOLOGY_PARAMETERS",
    "PHOTOSYNTHESIS_PARAMETERS",
    "POOL_STATE_PARAMETERS",
    "SIMPLEX_PARAMETERS",
    "TURNOVER_PARAMETERS",
    "DalecParameters",
    "Parameter",
    "allocation_fractions",
    "phenology_psi_f",
    "prior_bounds",
    "solve_psi",
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
