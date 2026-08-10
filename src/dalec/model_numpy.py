"""Plain-NumPy DALEC2 forward model -- the readable reference implementation.

Implements Bloom & Williams (2015) Appendix A, equations A1-A6, over six carbon
pools: labile, foliage, fine root, wood, litter and soil organic matter.

This module is written *before* ``dalec.model`` on purpose. A plain Python loop
is easy to read and easy to check by hand against the appendix; the PyTensor
version then has to reproduce it to floating-point tolerance, and that
equivalence is the single most important test in the project.

Units
-----
pools           g C m-2
fluxes          g C m-2 d-1
temperature     degrees C
rate constants  d-1

The update is simultaneous, not sequential
------------------------------------------
Every right-hand side in A1-A6 is evaluated at the pool values of time ``t``;
the six pools then advance together. Mutating ``C_lab`` before using it in the
``C_fol`` equation would give different -- and wrong -- numbers, so
:func:`dalec2_step` reads the whole state first and writes it once. The
conceptual ordering in the project brief (GPP, phenology, allocation, turnover,
respiration) is the order the *flux terms* are formed, not an order of state
mutation.

Carbon conservation is an exact identity
----------------------------------------
Summing A1-A6, the internal transfers cancel pairwise: ``phi_onset * C_lab``
between labile and foliage, ``phi_fall * C_fol`` between foliage and litter,
``theta_roo * C_roo`` between root and litter, ``theta_woo * C_woo`` between
wood and soil, and ``theta_min * exp(Theta * T) * C_lit`` between litter and
soil. What remains is

    d(total C) = (f_lab + f_fol + f_roo + f_woo) * GPP
                 - (theta_lit * C_lit + theta_som * C_som) * exp(Theta * T)
               = (1 - f_auto) * GPP - Rh
               = GPP - Reco
               = -NEE

so the change in total carbon equals minus NEE every single timestep, exactly,
with no discretisation error. :attr:`DalecOutput.carbon_imbalance` measures the
residual, and it should be at floating-point noise.

Note that ``theta_min`` moves carbon from litter to soil and is therefore a
*transfer*, not a loss: only ``theta_lit`` and ``theta_som`` enter respiration.

Not yet implemented
-------------------
``F_gpp`` (the Aggregated Canopy Model) is Phase 3, still awaiting its empirical
coefficients. It is injected as a callable and defaults to a stub that raises.
Everything else -- A1-A6 and the A7/A8 phenology -- is complete, and carbon
conservation holds for *any* value ``F_gpp`` returns.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np

from dalec.parameters import DalecParameters, phenology_psi_f

__all__ = [
    "N_POOLS",
    "ONSET_OFFSET_CONSTANT",
    "ONSET_SHAPE_CONSTANT",
    "PHENOLOGY_S",
    "POOL_NAMES",
    "DalecOutput",
    "Drivers",
    "GppModel",
    "PhenologyModel",
    "dalec2_phenology",
    "dalec2_step",
    "gpp_not_implemented",
    "phi_fall",
    "phi_onset",
    "run_dalec2",
]

#: Pool ordering used by every array in this module and by ``dalec.model``.
POOL_NAMES: Final[tuple[str, ...]] = (
    "c_lab",
    "c_fol",
    "c_roo",
    "c_woo",
    "c_lit",
    "c_som",
)
N_POOLS: Final[int] = len(POOL_NAMES)


# ---------------------------------------------------------------------------
# Injected components
# ---------------------------------------------------------------------------


class Drivers(Protocol):
    """Daily meteorological drivers.

    Structural type: ``dalec.data_io.SiteData`` satisfies it, and so does any
    lightweight stand-in with the same attribute names. All arrays share one
    leading time dimension.

    Attributes
    ----------
    doy
        Day of year, 1-366.
    t_air
        Daily mean air temperature, degrees C. Drives decomposition through
        ``exp(temperature_exponent * t_air)`` in A5 and A6.
    t_day, t_night
        Daily mean daytime and nighttime air temperature, degrees C. Consumed
        by the photosynthesis routine, not by A1-A6.
    sw_in
        Daily total incoming shortwave radiation, MJ m-2 d-1.
    co2
        Atmospheric CO2 mole fraction, umol mol-1.

    Declared read-only, as properties rather than plain attributes: a Protocol
    attribute is settable by default, which no frozen dataclass can satisfy, and
    both ``SiteData`` and the test stand-ins are frozen.
    """

    @property
    def doy(self) -> np.ndarray:
        """Day of year, 1-366."""
        ...

    @property
    def t_air(self) -> np.ndarray:
        """Daily mean air temperature, degrees C."""
        ...

    @property
    def t_day(self) -> np.ndarray:
        """Daily mean daytime air temperature, degrees C."""
        ...

    @property
    def t_night(self) -> np.ndarray:
        """Daily mean nighttime air temperature, degrees C."""
        ...

    @property
    def sw_in(self) -> np.ndarray:
        """Daily total incoming shortwave radiation, MJ m-2 d-1."""
        ...

    @property
    def co2(self) -> np.ndarray:
        """Atmospheric CO2 mole fraction, umol mol-1."""
        ...


class GppModel(Protocol):
    """Daily gross primary production, ``F_gpp`` in A1-A4.

    Phase 3 supplies the real implementation in ``dalec.acm``.
    """

    def __call__(
        self,
        *,
        doy: int,
        t_day: float,
        t_night: float,
        sw_in: float,
        co2: float,
        c_fol: float,
        lma: float,
        ceff: float,
    ) -> float:
        """Return GPP for one day, g C m-2 d-1."""
        ...


class PhenologyModel(Protocol):
    """Leaf onset and leaf fall fractions, ``phi_onset`` and ``phi_fall``.

    Both are dimensionless fractions in ``[0, 1]``: the fraction of the labile
    pool released into foliage, and the fraction of the foliage pool dropped
    into litter, on this day.
    """

    def __call__(
        self,
        *,
        doy: int,
        d_onset: float,
        cr_onset: float,
        d_fall: float,
        cr_fall: float,
        c_lf: float,
    ) -> tuple[float, float]:
        """Return ``(phi_onset, phi_fall)`` for one day, both dimensionless."""
        ...


def gpp_not_implemented(
    *,
    doy: int,
    t_day: float,
    t_night: float,
    sw_in: float,
    co2: float,
    c_fol: float,
    lma: float,
    ceff: float,
) -> float:
    """Placeholder for ``F_gpp``. Signature only -- see ``dalec.acm`` (Phase 3)."""
    raise NotImplementedError(
        "F_gpp is not implemented yet. The Aggregated Canopy Model is Phase 3 and "
        "needs its empirical coefficients from Williams et al. (1997). Pass a "
        "gpp_fn= test double to exercise equations A1-A6 in the meantime."
    )


# ---------------------------------------------------------------------------
# Phenology (published equations A7 and A8)
# ---------------------------------------------------------------------------

#: The constant ``s`` shared by A7 and A8, stated immediately after A8.
#:
#: Both sine arguments are divided by ``s``, so one year advances the argument
#: by ``365.25 / s = pi``: half a period, which is what puts exactly one pulse
#: per year under the Gaussian envelope.
PHENOLOGY_S: Final[float] = 365.25 / math.pi

#: Leading constant of A7, fixed by the paper.
ONSET_SHAPE_CONSTANT: Final[float] = 6.9088
#: Phase offset of A7, in units of ``cr_onset``.
ONSET_OFFSET_CONSTANT: Final[float] = 0.6245

_SQRT2: Final[float] = math.sqrt(2.0)
_SQRT2_OVER_SQRT_PI: Final[float] = math.sqrt(2.0) / math.sqrt(math.pi)


def phi_onset(doy: int | float, d_onset: float, cr_onset: float) -> float:
    """Leaf onset fraction, published equation A7.

    Parameters
    ----------
    doy
        Day of year.
    d_onset
        Day of leaf onset, day of year.
    cr_onset
        Labile release period, days.

    Returns
    -------
    Fraction of the labile pool released into foliage today, dimensionless.
    Bounded above by ``sqrt(2/pi) * 6.9088 / cr_onset``, which is 0.55 at the
    smallest permitted ``cr_onset`` of 10 days, so it stays inside ``[0, 1]``
    across the whole prior range.
    """
    envelope = (
        math.sin((doy - d_onset - ONSET_OFFSET_CONSTANT * cr_onset) / PHENOLOGY_S)
        * _SQRT2
        * PHENOLOGY_S
        / cr_onset
    )
    return _SQRT2_OVER_SQRT_PI * (ONSET_SHAPE_CONSTANT / cr_onset) * math.exp(-(envelope**2))


def phi_fall(
    doy: int | float,
    d_fall: float,
    cr_fall: float,
    c_lf: float,
    psi_f: float | None = None,
) -> float:
    """Leaf fall fraction, published equation A8.

    The coefficient is ``-log(1 - c_lf) / cr_fall``. The superseded preprint
    form ``(log(c_lspan) - log(c_lspan - 1)) / cr_fall`` is the same quantity
    under ``c_lf = 1 / c_lspan``, and is not used here.

    .. warning::

       **The sine argument below is transcribed as** ``doy - cr_fall + psi_f``
       **, which makes** ``d_fall`` **inert.** It is accepted, documented and
       carries a prior range, but has no effect on the returned value: the pulse
       is anchored by ``cr_fall`` instead. The A7 counterpart anchors on
       ``d_onset``, so the symmetric form would be ``doy - d_fall + psi_f``.

       This is implemented as transcribed rather than silently corrected. See
       ``test_d_fall_is_inert_as_transcribed``, which asserts the behaviour so
       it cannot be forgotten, and the README. If the anchor is confirmed to be
       ``d_fall``, this is a one-token change.

    Parameters
    ----------
    doy
        Day of year.
    d_fall
        Day of leaf fall, day of year. Currently unused -- see the warning.
    cr_fall
        Leaf fall period, days.
    c_lf
        Annual leaf fall fraction, dimensionless, in ``(0, 1)``.
    psi_f
        Precomputed phase offset. Computed from ``c_lf`` and ``cr_fall`` when
        omitted; the underlying solve is cached.

    Returns
    -------
    Fraction of the foliage pool dropped into litter today, dimensionless.
    """
    if psi_f is None:
        psi_f = phenology_psi_f(c_lf, cr_fall)
    envelope = (
        math.sin((doy - cr_fall + psi_f) / PHENOLOGY_S) * _SQRT2 * PHENOLOGY_S / cr_fall
    )
    return _SQRT2_OVER_SQRT_PI * (-math.log1p(-c_lf) / cr_fall) * math.exp(-(envelope**2))


def dalec2_phenology(
    *,
    doy: int,
    d_onset: float,
    cr_onset: float,
    d_fall: float,
    cr_fall: float,
    c_lf: float,
) -> tuple[float, float]:
    """Both phenology fractions for one day, published A7 and A8.

    This is the default ``phenology_fn`` of :func:`run_dalec2`. ``psi_f`` is
    resolved through the cache in ``dalec.parameters``, so the root solve runs
    once per distinct ``(c_lf, cr_fall)`` rather than once per timestep.
    """
    return (
        phi_onset(doy, d_onset, cr_onset),
        phi_fall(doy, d_fall, cr_fall, c_lf, psi_f=phenology_psi_f(c_lf, cr_fall)),
    )


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DalecOutput:
    """Result of a forward run.

    Attributes
    ----------
    pools
        Carbon pools, ``(n_steps + 1, 6)``, g C m-2, ordered as
        :data:`POOL_NAMES`. Row 0 is the initial state, so row ``t + 1`` is the
        state after step ``t`` and the array is directly plottable as a
        trajectory.
    gpp, ra, rh, reco, nee
        Daily fluxes, ``(n_steps,)``, g C m-2 d-1. ``ra`` is autotrophic and
        ``rh`` heterotrophic respiration; ``reco = ra + rh`` and
        ``nee = reco - gpp``, so negative NEE is net uptake, matching the
        FLUXNET sign convention.
    phi_onset, phi_fall
        Realised phenology fractions, ``(n_steps,)``, dimensionless. Recorded
        for diagnostics.
    """

    pools: np.ndarray
    gpp: np.ndarray
    ra: np.ndarray
    rh: np.ndarray
    reco: np.ndarray
    nee: np.ndarray
    phi_onset: np.ndarray
    phi_fall: np.ndarray

    @property
    def n_steps(self) -> int:
        """Number of daily timesteps."""
        return int(self.gpp.shape[0])

    @property
    def total_carbon(self) -> np.ndarray:
        """Total carbon summed over all six pools, ``(n_steps + 1,)``, g C m-2."""
        return self.pools.sum(axis=1)

    @property
    def carbon_imbalance(self) -> np.ndarray:
        """Per-timestep carbon budget residual, ``(n_steps,)``, g C m-2 d-1.

        ``(C_total(t+1) - C_total(t)) - (GPP(t) - Reco(t))``, which is exactly
        zero in infinite precision. Anything above floating-point noise means a
        flux is being created or destroyed somewhere in the step.
        """
        return np.diff(self.total_carbon) - (self.gpp - self.reco)

    def pool(self, name: str) -> np.ndarray:
        """Trajectory of one named pool, ``(n_steps + 1,)``, g C m-2."""
        try:
            index = POOL_NAMES.index(name)
        except ValueError:
            raise KeyError(f"unknown pool {name!r}; expected one of {POOL_NAMES}") from None
        return self.pools[:, index]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def dalec2_step(
    state: np.ndarray,
    *,
    gpp: float,
    phi_onset: float,
    phi_fall: float,
    t_air: float,
    params: DalecParameters,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Advance the six carbon pools by one day.

    A direct transcription of equations A1-A6. Every right-hand side reads the
    time-``t`` state; nothing is mutated in place.

    Parameters
    ----------
    state
        Carbon pools at time ``t``, length 6, g C m-2, ordered as
        :data:`POOL_NAMES`.
    gpp
        Gross primary production on this day, g C m-2 d-1.
    phi_onset, phi_fall
        Leaf onset and leaf fall fractions on this day, dimensionless.
    t_air
        Daily mean air temperature, degrees C.
    params
        Parameter set.

    Returns
    -------
    next_state
        Carbon pools at time ``t + 1``, length 6, g C m-2.
    fluxes
        ``(gpp, ra, rh, nee)``, g C m-2 d-1.
    """
    c_lab, c_fol, c_roo, c_woo, c_lit, c_som = (float(value) for value in state)

    # exp(Theta * T) -- the shared decomposition temperature multiplier of A5/A6.
    temperature_rate = math.exp(params.temperature_exponent * t_air)

    # (A1) labile: drained by leaf onset, topped up by allocation.
    c_lab_next = (1.0 - phi_onset) * c_lab + params.f_lab * gpp

    # (A2) foliage: drained by leaf fall, fed by leaf onset and allocation.
    c_fol_next = (1.0 - phi_fall) * c_fol + phi_onset * c_lab + params.f_fol * gpp

    # (A3) fine root.
    c_roo_next = (1.0 - params.theta_roo) * c_roo + params.f_roo * gpp

    # (A4) wood.
    c_woo_next = (1.0 - params.theta_woo) * c_woo + params.f_woo * gpp

    # (A5) litter: loses theta_lit to respiration and theta_min to soil, both
    # temperature-scaled; gains root turnover and leaf fall.
    c_lit_next = (
        (1.0 - (params.theta_lit + params.theta_min) * temperature_rate) * c_lit
        + params.theta_roo * c_roo
        + phi_fall * c_fol
    )

    # (A6) soil organic matter: loses theta_som to respiration; gains wood
    # turnover and mineralised litter.
    c_som_next = (
        (1.0 - params.theta_som * temperature_rate) * c_som
        + params.theta_woo * c_woo
        + params.theta_min * temperature_rate * c_lit
    )

    # Respiration is formed from the time-t pools, consistent with A5/A6.
    # theta_min is a litter -> soil transfer and is deliberately absent here.
    ra = params.f_auto * gpp
    rh = (params.theta_lit * c_lit + params.theta_som * c_som) * temperature_rate
    nee = (ra + rh) - gpp

    next_state = np.array(
        [c_lab_next, c_fol_next, c_roo_next, c_woo_next, c_lit_next, c_som_next],
        dtype=float,
    )
    return next_state, (gpp, ra, rh, nee)


def run_dalec2(
    params: DalecParameters,
    drivers: Drivers,
    *,
    gpp_fn: GppModel | Callable[..., float] = gpp_not_implemented,
    phenology_fn: PhenologyModel | Callable[..., tuple[float, float]] = dalec2_phenology,
) -> DalecOutput:
    """Integrate DALEC2 forward over the full driver record.

    Parameters
    ----------
    params
        Parameter set, including the six initial pool states.
    drivers
        Daily drivers; ``dalec.data_io.SiteData`` satisfies the protocol.
    gpp_fn
        Photosynthesis routine. Defaults to :func:`gpp_not_implemented`;
        Phase 3 replaces it with the ACM implementation.
    phenology_fn
        Returns ``(phi_onset, phi_fall)`` per day. Defaults to
        :func:`dalec2_phenology`, the published A7/A8 pair.

    Returns
    -------
    DalecOutput

    Raises
    ------
    ValueError
        If the driver arrays disagree in length, are empty, or contain
        non-finite values. A single NaN driver silently contaminates every pool
        from that day onward, so it is rejected up front rather than propagated.
    """
    driver_names = ("doy", "t_air", "t_day", "t_night", "sw_in", "co2")
    arrays = {name: np.asarray(getattr(drivers, name)) for name in driver_names}

    lengths = {name: array.shape[0] for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"driver arrays disagree in length: {lengths}")

    n_steps = next(iter(lengths.values()))
    if n_steps == 0:
        raise ValueError("driver record is empty; nothing to integrate")

    for name, array in arrays.items():
        if not np.all(np.isfinite(array.astype(float))):
            raise ValueError(
                f"driver {name!r} contains non-finite values; the forward model "
                "cannot integrate through a gap"
            )

    pools = np.empty((n_steps + 1, N_POOLS), dtype=float)
    pools[0] = params.initial_pools

    gpp = np.empty(n_steps, dtype=float)
    ra = np.empty(n_steps, dtype=float)
    rh = np.empty(n_steps, dtype=float)
    nee = np.empty(n_steps, dtype=float)
    phi_onset = np.empty(n_steps, dtype=float)
    phi_fall = np.empty(n_steps, dtype=float)

    doy = arrays["doy"]
    t_air = arrays["t_air"]
    t_day = arrays["t_day"]
    t_night = arrays["t_night"]
    sw_in = arrays["sw_in"]
    co2 = arrays["co2"]

    for step in range(n_steps):
        state = pools[step]

        onset, fall = phenology_fn(
            doy=int(doy[step]),
            d_onset=params.d_onset,
            cr_onset=params.cr_onset,
            d_fall=params.d_fall,
            cr_fall=params.cr_fall,
            c_lf=params.c_lf,
        )

        # Foliar carbon enters photosynthesis at its time-t value, matching the
        # simultaneous update of A1-A6.
        daily_gpp = gpp_fn(
            doy=int(doy[step]),
            t_day=float(t_day[step]),
            t_night=float(t_night[step]),
            sw_in=float(sw_in[step]),
            co2=float(co2[step]),
            c_fol=float(state[POOL_NAMES.index("c_fol")]),
            lma=params.lma,
            ceff=params.ceff,
        )

        next_state, (step_gpp, step_ra, step_rh, step_nee) = dalec2_step(
            state,
            gpp=daily_gpp,
            phi_onset=onset,
            phi_fall=fall,
            t_air=float(t_air[step]),
            params=params,
        )

        pools[step + 1] = next_state
        gpp[step] = step_gpp
        ra[step] = step_ra
        rh[step] = step_rh
        nee[step] = step_nee
        phi_onset[step] = onset
        phi_fall[step] = fall

    return DalecOutput(
        pools=pools,
        gpp=gpp,
        ra=ra,
        rh=rh,
        reco=ra + rh,
        nee=nee,
        phi_onset=phi_onset,
        phi_fall=phi_fall,
    )
