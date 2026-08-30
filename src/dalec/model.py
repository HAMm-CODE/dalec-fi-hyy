"""DALEC2 as a differentiable PyTensor graph.

The scan is lifted from ``timing_spike.py``, which established the cost of one
gradient at this problem size. What changes here is that the arithmetic is the
**real** model rather than a shape-alike: the exact Chuter ACM from
:mod:`dalec.acm`, the published A7/A8 phenology, and the A1-A6 pool updates from
:mod:`dalec.model_numpy`. ``tests/test_model.py`` asserts agreement with the
numpy forward model to machine precision on identical inputs.

Three things are precomputed as constants because they read only drivers, never
parameters: the floored daily temperature range, the frost mask, and the day
length factor. Keeping them out of the graph matters for more than speed -- the
frost mask is a **branch**, and a parameter-dependent branch inside a NUTS
gradient is a discontinuity. It cannot become one here because nothing it
depends on is sampled.

Solving psi inside the graph
----------------------------
Equation A9, ``2*sqrt(pi)*log(1 - c_lf)*psi = exp(-psi**2)``, has no closed
form. :func:`dalec.parameters.solve_psi` uses Brent, which is not
differentiable, and its docstring notes that a freed ``c_lf`` "would need
implicit differentiation through solve_psi rather than a baked-in number".

``c_lf`` is now sampled, so this module unrolls a fixed number of Newton steps
instead. That is differentiable by construction -- the gradient flows through
the unrolled iteration and converges to the implicit-function derivative -- and
over the ``c_lf`` prior the root lies in roughly ``[-0.9, -0.6]``, so a fixed
start converges to machine precision in a handful of steps. Agreement with the
Brent solve is asserted in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from dalec.acm import (
    AcmCoefficients,
    average_daily_temperature,
    day_length_hours,
    frost_mask,
)
from dalec.model_numpy import (
    ONSET_OFFSET_CONSTANT,
    ONSET_SHAPE_CONSTANT,
    PHENOLOGY_S,
    POOL_NAMES,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "NEWTON_STEPS",
    "DalecGraph",
    "build_forward_graph",
    "psi_newton",
]

#: Newton steps for the A9 root. The residual is smooth and the start is close,
#: so this converges well past double precision; it is fixed rather than
#: tolerance-based because the graph must have a static shape.
NEWTON_STEPS: Final[int] = 24

#: Starting point for the Newton solve. The root is negative and lies near
#: -0.75 across the whole ``c_lf`` prior.
_PSI_START: Final[float] = -0.75

_SQRT2: Final[float] = math.sqrt(2.0)
_SQRT2_OVER_SQRT_PI: Final[float] = math.sqrt(2.0) / math.sqrt(math.pi)
_TWO_SQRT_PI: Final[float] = 2.0 * math.sqrt(math.pi)


@dataclass(frozen=True)
class DalecGraph:
    """The forward model as tensors, plus the constants it was built on.

    Attributes
    ----------
    nee
        Modelled net ecosystem exchange, one value per driver day, g C m-2 d-1.
    gpp, ra, rh
        The component fluxes, same shape.
    pools
        Carbon pools **after** each step, shape ``(n_days, 6)``, ordered as
        :data:`dalec.model_numpy.POOL_NAMES`.
    n_days
        Length of the driver record.
    """

    nee: Any
    gpp: Any
    ra: Any
    rh: Any
    pools: Any
    n_days: int


def psi_newton(c_lf: Any, steps: int = NEWTON_STEPS) -> Any:
    """Solve A9 for ``psi`` by unrolled Newton iteration, differentiably.

    ``f(psi) = a*psi - exp(-psi^2)`` with ``a = 2*sqrt(pi)*log(1 - c_lf)``,
    and ``f'(psi) = a + 2*psi*exp(-psi^2)``.

    Accepts a tensor or a float; with a float this is an ordinary numpy solve
    and is used by the tests to check against Brent.
    """
    import pytensor.tensor as pt

    is_tensor = hasattr(c_lf, "type")
    log1p = pt.log1p(-c_lf) if is_tensor else np.log1p(-c_lf)
    exp = pt.exp if is_tensor else np.exp

    a = _TWO_SQRT_PI * log1p
    psi = pt.as_tensor_variable(_PSI_START) if is_tensor else _PSI_START
    for _ in range(steps):
        gaussian = exp(-psi * psi)
        residual = a * psi - gaussian
        derivative = a + 2.0 * psi * gaussian
        psi = psi - residual / derivative
    return psi


def _phenology(doy, d_onset, cr_onset, cr_fall, c_lf):
    """A7 and A8 as tensors. ``d_fall`` is absent because it is inert (§2)."""
    import pytensor.tensor as pt

    onset_envelope = (
        pt.sin((doy - d_onset - ONSET_OFFSET_CONSTANT * cr_onset) / PHENOLOGY_S)
        * _SQRT2
        * PHENOLOGY_S
        / cr_onset
    )
    onset = (
        _SQRT2_OVER_SQRT_PI
        * (ONSET_SHAPE_CONSTANT / cr_onset)
        * pt.exp(-(onset_envelope**2))
    )

    psi_f = psi_newton(c_lf) * cr_fall / _SQRT2
    fall_envelope = (
        pt.sin((doy - cr_fall + psi_f) / PHENOLOGY_S) * _SQRT2 * PHENOLOGY_S / cr_fall
    )
    fall = (
        _SQRT2_OVER_SQRT_PI
        * (-pt.log1p(-c_lf) / cr_fall)
        * pt.exp(-(fall_envelope**2))
    )
    return onset, fall


def _acm_gpp(c_fol, lma, ceff, t_max, co2, g_c, day_length_factor, coef):
    """Chuter B2-B4 and Eq. 12, as tensors. ``g_c`` and the day length factor
    are driver-only and precomputed."""
    import pytensor.tensor as pt

    lai = c_fol / lma

    # B3, already divided through by conductance.
    p = ceff * lai * pt.exp(coef.a8 * t_max) / g_c

    # B2, internal CO2.
    q = coef.a3 - coef.a4
    discriminant = (co2 + q - p) ** 2 - 4.0 * (co2 * q - coef.a3 * p)
    discriminant = pt.maximum(discriminant, 0.0)
    c_i = 0.5 * (co2 + q - p + pt.sqrt(discriminant))
    p_d = g_c * (co2 - c_i)

    # B4, canopy quantum yield, written as published in c_fol and lma.
    e_0 = coef.a7 * c_fol**2 / (c_fol**2 + coef.a9 * lma**2)
    return p_d, e_0


def build_forward_graph(
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
) -> DalecGraph:
    """Build the DALEC2 forward model over a driver record.

    Parameters
    ----------
    parameters
        Name to tensor for every field of
        :class:`dalec.parameters.DalecParameters`; typically
        ``dalec.priors.build_priors(...).parameters``.
    doy, t_air, t_max, t_min, sw_in, co2
        Driver arrays, all of the same length.
    latitude_deg, coefficients, frost_threshold_degc
        Site constants, none of them sampled.
    """
    import pytensor
    import pytensor.tensor as pt

    drivers = {
        "doy": doy, "t_air": t_air, "t_max": t_max,
        "t_min": t_min, "sw_in": sw_in, "co2": co2,
    }
    lengths = {name: np.asarray(value).shape for name, value in drivers.items()}
    if len({shape for shape in lengths.values()}) != 1:
        raise ValueError(f"driver arrays must share a length, got {lengths}")
    n_days = int(np.asarray(doy).size)
    if n_days == 0:
        raise ValueError("driver record is empty")

    # -- driver-only constants -------------------------------------------------
    # The frost mask is a branch. It stays out of the parameter graph because
    # nothing it reads is sampled, which is what makes it safe under NUTS.
    t_range = np.maximum(np.asarray(t_max) - np.asarray(t_min), 0.0)
    g_c = abs(coefficients.psi_mpa) ** coefficients.a10 / (
        0.5 * t_range + coefficients.a6 * coefficients.r_tot
    )
    day_length_factor = (
        coefficients.a2 * day_length_hours(doy, latitude_deg) + coefficients.a5
    )
    unfrozen = (~frost_mask(t_max, t_min, frost_threshold_degc)).astype(float)
    _ = average_daily_temperature  # documents where the mask's temperature comes from

    sequences = [
        pt.as_tensor_variable(np.asarray(value, dtype=float))
        for value in (doy, t_air, t_max, sw_in, co2, g_c, day_length_factor, unfrozen)
    ]

    # Cast every parameter to float64. PyTensor narrows a bare Python float to
    # float32 regardless of floatX, and a 5113-step integration in float32
    # accumulates visible error -- it cost four significant figures against the
    # numpy reference before this cast was added.
    theta = {
        name: pt.cast(pt.as_tensor_variable(value), "float64")
        for name, value in parameters.items()
    }
    coef = coefficients

    def step(
        doy_t, t_air_t, t_max_t, sw_in_t, co2_t, g_c_t, dlf_t, unfrozen_t,
        c_lab, c_fol, c_roo, c_woo, c_lit, c_som,
    ):
        onset, fall = _phenology(
            doy_t, theta["d_onset"], theta["cr_onset"],
            theta["cr_fall"], theta["c_lf"],
        )
        p_d, e_0 = _acm_gpp(
            c_fol, theta["lma"], theta["ceff"], t_max_t, co2_t, g_c_t, dlf_t, coef
        )
        light_supply = e_0 * sw_in_t
        denominator = light_supply + p_d
        positive = pt.gt(denominator, 0.0)
        safe = pt.switch(positive, denominator, 1.0)
        p_i = pt.switch(positive, light_supply * p_d / safe, 0.0)
        gpp = unfrozen_t * p_i * dlf_t

        temperature_rate = pt.exp(theta["temperature_exponent"] * t_air_t)

        c_lab_next = (1.0 - onset) * c_lab + theta["f_lab"] * gpp
        c_fol_next = (1.0 - fall) * c_fol + onset * c_lab + theta["f_fol"] * gpp
        c_roo_next = (1.0 - theta["theta_roo"]) * c_roo + theta["f_roo"] * gpp
        c_woo_next = (1.0 - theta["theta_woo"]) * c_woo + theta["f_woo"] * gpp
        c_lit_next = (
            (1.0 - (theta["theta_lit"] + theta["theta_min"]) * temperature_rate) * c_lit
            + theta["theta_roo"] * c_roo
            + fall * c_fol
        )
        c_som_next = (
            (1.0 - theta["theta_som"] * temperature_rate) * c_som
            + theta["theta_woo"] * c_woo
            + theta["theta_min"] * temperature_rate * c_lit
        )

        # Respiration is formed from the time-t pools, consistent with A5/A6.
        ra = theta["f_auto"] * gpp
        rh = (
            theta["theta_lit"] * c_lit + theta["theta_som"] * c_som
        ) * temperature_rate
        nee = (ra + rh) - gpp
        return (
            c_lab_next, c_fol_next, c_roo_next, c_woo_next, c_lit_next, c_som_next,
            gpp, ra, rh, nee,
        )

    initial = [
        theta[f"c_{name}_0"] for name in ("lab", "fol", "roo", "woo", "lit", "som")
    ]
    outputs, _updates = pytensor.scan(
        fn=step,
        sequences=sequences,
        outputs_info=[*initial, None, None, None, None],
        strict=True,
        name="dalec2",
    )
    pools = pt.stack(outputs[: len(POOL_NAMES)], axis=1)
    gpp, ra, rh, nee = outputs[len(POOL_NAMES) :]
    return DalecGraph(
        nee=nee, gpp=gpp, ra=ra, rh=rh, pools=pools, n_days=n_days
    )
