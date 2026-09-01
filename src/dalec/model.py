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
    "phenology_constants",
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


def phenology_constants(d_onset, cr_onset, cr_fall, c_lf, psi_mode="hoisted"):
    """The parts of A7 and A8 that do not depend on the day of year.

    **These must be computed outside the scan.** ``psi`` solves A9, which
    involves ``c_lf`` alone, and the leaf-fall amplitude and phase are likewise
    time-invariant. Left inside the loop they are recomputed and, worse, taped
    for reverse-mode at every one of 5113 timesteps -- which is what made the
    gradient cost 44x a forward pass instead of the usual 2-5x.

    ``psi_mode`` is a profiling switch, not a modelling choice:

    ``"hoisted"``
        Solve once here. The production path.
    ``"newton_inside"``
        Return ``None`` so the caller unrolls Newton per timestep. Kept only so
        the regression can be re-measured.
    ``"constant_inside"``
        Freeze ``psi`` at its starting value. **Numerically wrong**; it exists
        to isolate the cost of the Newton solve from everything else.
    """
    import pytensor.tensor as pt

    onset_amplitude = _SQRT2_OVER_SQRT_PI * (ONSET_SHAPE_CONSTANT / cr_onset)
    onset_phase = d_onset + ONSET_OFFSET_CONSTANT * cr_onset
    fall_amplitude = _SQRT2_OVER_SQRT_PI * (-pt.log1p(-c_lf) / cr_fall)

    if psi_mode == "newton_inside":
        psi_f = None
    elif psi_mode == "constant_inside":
        psi_f = pt.as_tensor_variable(_PSI_START) * cr_fall / _SQRT2
    elif psi_mode == "hoisted":
        psi_f = psi_newton(c_lf) * cr_fall / _SQRT2
    else:
        raise ValueError(
            f"unknown psi_mode {psi_mode!r}; expected 'hoisted', "
            "'newton_inside' or 'constant_inside'"
        )
    return onset_amplitude, onset_phase, fall_amplitude, psi_f


def _phenology_day(doy, cr_onset, cr_fall, onset_amplitude, onset_phase,
                   fall_amplitude, psi_f):
    """A7 and A8 for one day, given the hoisted constants.

    ``d_fall`` is absent because it is inert as transcribed (DECISIONS §2).
    """
    import pytensor.tensor as pt

    onset_envelope = (
        pt.sin((doy - onset_phase) / PHENOLOGY_S) * _SQRT2 * PHENOLOGY_S / cr_onset
    )
    onset = onset_amplitude * pt.exp(-(onset_envelope**2))

    fall_envelope = (
        pt.sin((doy - cr_fall + psi_f) / PHENOLOGY_S) * _SQRT2 * PHENOLOGY_S / cr_fall
    )
    fall = fall_amplitude * pt.exp(-(fall_envelope**2))
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
    psi_mode: str = "hoisted",
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
    psi_mode
        Profiling switch; see :func:`phenology_constants`. Leave at
        ``"hoisted"`` for anything that is not a timing measurement.
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

    # Hoisted out of the scan and passed as **explicit non-sequences**.
    #
    # Hoisting alone is not enough, and this was measured: computing psi outside
    # the scan but referring to it through a Python closure left the gradient at
    # 2149 ms against 2165 ms, bit-for-bit identical. PyTensor pulls a closed-over
    # expression into the inner graph, so the 24 Newton steps were still being
    # taped at every one of 5113 timesteps. Passing the value through
    # ``non_sequences`` is what actually keeps it outside the loop.
    onset_amplitude, onset_phase, fall_amplitude, psi_f = phenology_constants(
        theta["d_onset"], theta["cr_onset"], theta["cr_fall"], theta["c_lf"],
        psi_mode=psi_mode,
    )
    if psi_f is None:  # the "newton_inside" profiling mode
        psi_f = psi_newton(theta["c_lf"]) * theta["cr_fall"] / _SQRT2
        invariant_names: tuple[str, ...] = ()
        inline_psi = True
    else:
        invariant_names = ()
        inline_psi = False

    # Every parameter the step needs, as an explicit non-sequence. Nothing is
    # closed over, so nothing can be silently inlined back into the loop.
    step_parameters = (
        "cr_onset", "cr_fall", "lma", "ceff", "temperature_exponent",
        "f_lab", "f_fol", "f_roo", "f_woo", "f_auto",
        "theta_roo", "theta_woo", "theta_lit", "theta_som", "theta_min",
    )
    non_sequences = [
        onset_amplitude, onset_phase, fall_amplitude, psi_f,
        *[theta[name] for name in step_parameters],
    ]
    del invariant_names, inline_psi

    def step(
        doy_t, t_air_t, t_max_t, sw_in_t, co2_t, g_c_t, dlf_t, unfrozen_t,
        c_lab, c_fol, c_roo, c_woo, c_lit, c_som,
        onset_amplitude_, onset_phase_, fall_amplitude_, psi_f_,
        cr_onset_, cr_fall_, lma_, ceff_, temperature_exponent_,
        f_lab_, f_fol_, f_roo_, f_woo_, f_auto_,
        theta_roo_, theta_woo_, theta_lit_, theta_som_, theta_min_,
    ):
        onset, fall = _phenology_day(
            doy_t, cr_onset_, cr_fall_,
            onset_amplitude_, onset_phase_, fall_amplitude_, psi_f_,
        )
        p_d, e_0 = _acm_gpp(c_fol, lma_, ceff_, t_max_t, co2_t, g_c_t, dlf_t, coef)
        light_supply = e_0 * sw_in_t
        denominator = light_supply + p_d
        positive = pt.gt(denominator, 0.0)
        safe = pt.switch(positive, denominator, 1.0)
        p_i = pt.switch(positive, light_supply * p_d / safe, 0.0)
        gpp = unfrozen_t * p_i * dlf_t

        temperature_rate = pt.exp(temperature_exponent_ * t_air_t)

        c_lab_next = (1.0 - onset) * c_lab + f_lab_ * gpp
        c_fol_next = (1.0 - fall) * c_fol + onset * c_lab + f_fol_ * gpp
        c_roo_next = (1.0 - theta_roo_) * c_roo + f_roo_ * gpp
        c_woo_next = (1.0 - theta_woo_) * c_woo + f_woo_ * gpp
        c_lit_next = (
            (1.0 - (theta_lit_ + theta_min_) * temperature_rate) * c_lit
            + theta_roo_ * c_roo
            + fall * c_fol
        )
        c_som_next = (
            (1.0 - theta_som_ * temperature_rate) * c_som
            + theta_woo_ * c_woo
            + theta_min_ * temperature_rate * c_lit
        )

        # Respiration is formed from the time-t pools, consistent with A5/A6.
        ra = f_auto_ * gpp
        rh = (theta_lit_ * c_lit + theta_som_ * c_som) * temperature_rate
        nee = (ra + rh) - gpp
        return (
            c_lab_next, c_fol_next, c_roo_next, c_woo_next, c_lit_next, c_som_next,
            gpp, ra, rh, nee,
        )

    initial = [
        theta[f"c_{name}_0"] for name in ("lab", "fol", "roo", "woo", "lit", "som")
    ]
    outputs = pytensor.scan(
        fn=step,
        sequences=sequences,
        outputs_info=[*initial, None, None, None, None],
        non_sequences=non_sequences,
        strict=True,
        name="dalec2",
        return_updates=False,
    )
    pools = pt.stack(outputs[: len(POOL_NAMES)], axis=1)
    gpp, ra, rh, nee = outputs[len(POOL_NAMES) :]
    return DalecGraph(
        nee=nee, gpp=gpp, ra=ra, rh=rh, pools=pools, n_days=n_days
    )
