"""Aggregated Canopy Model (Williams et al. 1997) -- daily GPP.

Source: Williams, M. et al. (1997), *Ecological Applications* 7(3), 882-894.
Equations pp. 885-886, coefficients Table 2 p. 887, calibration bounds Table 1
p. 884. DALEC2 substitutions from Bloom & Williams (2015).

Inputs: day of year, atmospheric CO2 (umol mol-1), daytime and nighttime air
temperature (degrees C), daily total shortwave radiation (MJ m-2 d-1), foliar
carbon (g C m-2), leaf mass per area (g C m-2) and canopy efficiency, plus two
fixed site constants (canopy height, leaf-to-soil water potential difference).

Output: gross primary productivity, g C m-2 d-1.

Solve order
-----------
**2, 5, 6, 4, 8, 7, 9.** The paper prints "2, 5, 6, 4, 7, 8, 9", but Eq. 7
consumes ``E_0`` from Eq. 8, so 8 must run first. Eq. 3 (``p_C``) is never
evaluated: the steady-state assumption ``p_C = p_D`` folds it into Eq. 6.

DALEC2 substitutions
--------------------
``c_eff`` replaces the nitrogen x nitrogen-use-efficiency product ``a1 * N`` of
the original Eq. 2, so ``a1`` and foliar N are not used here. ``L`` is
``C_fol / c_lma`` and comes from model state, not from drivers.

``T`` is the average daily temperature, defined by Williams et al. as the mean
of daily maximum and minimum, here ``(TA_F_DAY + TA_F_NIGHT) / 2``. Note this is
**not** ``TA_F``: decomposition in A5/A6 uses ``TA_F``, photosynthesis uses this
mean. ``D_T`` is **half** the daily temperature range, ``(T_max - T_min) / 2``
per Table 1, here ``(TA_F_DAY - TA_F_NIGHT) / 2``. Getting that factor of two
wrong is silent, so it is asserted in a test.

Frost cutoff
------------
ACM was calibrated over average daily temperature 7.0 to 30.0 degC (Table 1) and
Williams et al. state it must not extrapolate outside those bounds. FI-Hyy
reaches roughly -25 degC. Run unmodified through boreal winter, ACM produces
physically meaningless positive photosynthesis at subzero temperature -- note
that ``exp(b1 * T)`` with ``b1`` negative makes conductance *increase* as it
gets colder. Following the precedent Williams et al. applied at their Oregon
coniferous sites, no carbon is fixed below -2.0 degC.

This is safe for gradient-based sampling. The condition reads only the
temperature driver, never a sampled parameter, so it is a fixed boolean mask
over the time series: precompute once and multiply. The mask does not move as
parameters change, gradients are simply zero on masked days, and no
parameter-dependent branching enters the graph. It is applied *after* Eq. 9, so
the arithmetic always runs on finite values.

Known problems, measured
------------------------
Recorded here so they reach the thesis limitations rather than being
rediscovered late. All four were measured by the project author.

**1. Annual GPP is roughly five times too high.** Over a synthetic
Hyytiala-like year (LAI 3, T from -9 to +19 degC, I from 1 to 27 MJ m-2 d-1,
``ceff`` 40) the total was 5009 g C m-2 yr-1 against an observed Hyytiala value
near 1000-1100. ``dalec.diagnostics.gpp_magnitude_gate`` decides this
quantitatively and blocks calibration if it cannot be resolved.

**2. ACM is nearly temperature-insensitive at low irradiance.** Measured GPP
across a 40-degree swing at low ``I``:

    T (degC)     +20     +7      0      -10     -20
    GPP          2.646   2.657   2.656  2.648   2.633

A 0.5% response. The cause is structural: at low ``I`` the Eq. 7 light
limitation dominates and GPP collapses to ``E_0 * I * (d1*D_ms + d2)``, in which
``T`` does not appear at all. The frost cutoff therefore masks a **structural
insensitivity, not a mild extrapolation error**.

That floor is not small. Sustained year-round, ``2.657 * 365.25 = 970.5``
g C m-2 yr-1 -- essentially Hyytiala's entire true annual GPP, produced by a
term in which temperature does not appear.

**3. There is no parameter escape route, so the frost cutoff is load-bearing.**
Sweeping LAI and ``ceff`` across their full prior ranges (winter T = -5 degC,
D_T = 3, I = 2 MJ, doy 30; summer T = 15 degC, D_T = 5, I = 15 MJ, doy 180;
psi_d = -1.5, H = 18) the summer-to-winter ratio stays pinned near 6:

    LAI      0.5     1.0     2.0     3.0     6.0
    E_0      0.220   0.528   0.812   0.902   0.966
    winter   0.642   1.513   2.351   2.641   2.873
    summer   4.268   9.117   13.755  15.543  17.210
    ratio    6.65    6.03    5.85    5.89    5.99

Searching the whole 2D grid (LAI 0.2-6.0 x ceff 10-100) for any combination with
a summer peak in 8-11 *and* a winter floor below 0.5 yields **zero**
combinations; the maximum achievable ratio is 8.17. The ratio is structurally
bounded because LAI enters the light-limited floor only through
``E_0 = c1*L^2/(c2+L^2)``, which saturates and scales summer and winter almost
equally. Hyytiala's true summer/winter GPP ratio is effectively unbounded.

The frost cutoff is consequently the *only* mechanism available to suppress the
floor, and it acts only below -2 degC. It is not cleanup and it is not optional.

**4. The ``ceff`` reading is suspect.** The original ``a1 * N`` form reproduces
Williams et al. Fig. 2 at Harvard Forest (modelled 9.90 against a measured peak
near 11-12), so the implementation is sound. The ``ceff`` substitution does not
behave:

    ceff     10      20      40      60      80      100
    GPP      11.62   14.58   16.55   17.26   17.62   17.83

The bottom of the prior range already exceeds plausible boreal midsummer GPP of
8-11, and a tenfold change in ``ceff`` moves GPP by only 53%, nearly all of it
below ``ceff = 20``. Consistently, ``a1 * N`` at Harvard Forest is
2.95 * 1.92 = 5.7 against a ``ceff`` prior of 10-100 -- ranges that barely
overlap, which is the likely root of problems 1 and 4 both. Eq. 2 is implemented
exactly as specified: no rescaling, no correction factor. The magnitude gate
decides. The DALEC2 source has been requested from the authors.

Open question for the authors: whether the 10-100 range assumes a different LAI
convention (projected versus all-sided), a different normalisation, or an ACM
variant not described in the 2015 appendix. Until that is answered the prior is
left exactly as published -- narrowing it would be a methodological deviation,
and it is not one to make on our own measurements.

Scope: we are outside the envelope DALEC2's authors defined
--------------------------------------------------------------
Bloom & Williams (2015) Section 2.5 state that they selected sites with little
expected water stress and no more than **three months** of recorded
below-freezing soil temperature, because those criteria reflect the current
capabilities of DALEC2, hydrological processes not being explicitly
represented. FI-Hyy has roughly **four to six months** of soil frost.

Applying DALEC2 here is therefore a deliberate scope decision, not an oversight.

Bearing on RQ3
--------------
A GPP term insensitive to temperature over much of the year will push seasonal
structure onto the respiration parameters during calibration, so observed
equifinality may be partly *structural* rather than purely informational. The
Phase 7 synthetic twin separates the two: synthetic data generated by this same
GPP model should recover its parameters normally, so any gap between synthetic
and real-data recovery isolates structural error. The twin must run with the
frost mask active and over the same driver record, or the comparison is not
like-for-like.
"""

from __future__ import annotations

import warnings
from typing import Any, Final

import numpy as np

__all__ = [
    "ACM_CALIBRATION_BOUNDS",
    "ACM_COEFFICIENTS",
    "DEFAULT_FROST_THRESHOLD_DEGC",
    "MIDSUMMER_DOY",
    "AcmModel",
    "acm_from_config",
    "acm_terms",
    "average_daily_temperature",
    "daily_temperature_half_range",
    "frost_mask",
    "leaf_area_index",
    "make_acm",
]

# ---------------------------------------------------------------------------
# Empirical coefficients -- Williams et al. (1997) Table 2, p. 887
# ---------------------------------------------------------------------------
# Fixed constants. Never sampled, never fitted.

A2: Final[float] = 0.018  # temperature coefficient of NUE
THETA: Final[float] = 32.6  # canopy CO2 compensation point, umol mol-1
K: Final[float] = 576.7  # canopy CO2 half-saturation point, umol mol-1
B1: Final[float] = -0.029  # temperature coefficient of canopy conductance
B2: Final[float] = 0.315  # temperature range constant of canopy conductance
C1: Final[float] = 0.989  # maximum canopy quantum yield
C2: Final[float] = 0.873  # LAI-canopy quantum yield coefficient
D1: Final[float] = -0.0018  # day length constant
D2: Final[float] = 1.81  # midsummer coefficient

#: The NUE parameter of the original Eq. 2. **Not used**: DALEC2 replaces the
#: whole ``a1 * N`` product with the sampled ``ceff``. Retained only so the
#: 2.95 * 1.92 = 5.7 comparison in the docstring is reproducible.
A1_UNUSED: Final[float] = 2.95

#: Provenance record of every coefficient actually in use.
ACM_COEFFICIENTS: Final[dict[str, float]] = {
    "a2": A2,
    "theta": THETA,
    "k": K,
    "b1": B1,
    "b2": B2,
    "c1": C1,
    "c2": C2,
    "d1": D1,
    "d2": D2,
}

#: Reference day of year for the Eq. 9 day-length correction.
MIDSUMMER_DOY: Final[float] = 173.0

#: Frost cutoff, degrees C. Below this, no carbon is fixed. See the module
#: docstring: this is load-bearing, not cleanup -- it is the only mechanism
#: available to suppress the temperature-independent light-limited floor.
DEFAULT_FROST_THRESHOLD_DEGC: Final[float] = -2.0

#: Bounds over which ACM was calibrated (Table 1, p. 884). Driver days outside
#: these are extrapolation, and the fraction outside is a thesis limitation
#: figure -- see ``dalec.diagnostics.calibration_bound_coverage``.
#:
#: Only the average-daily-temperature bound has been supplied so far. The
#: remaining Table 1 rows (irradiance, LAI, CO2, D_T, psi_d, H) are still
#: needed; this mapping is structured to take them without code changes.
ACM_CALIBRATION_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "t_mean": (7.0, 30.0),
}

#: Below this, the Eq. 5 denominator is treated as degenerate and raises.
_MIN_CONDUCTANCE_DENOMINATOR: Final[float] = 1e-6

#: Floor applied to a negative Eq. 6 discriminant. Never applied silently.
_DISCRIMINANT_FLOOR: Final[float] = 0.0


# ---------------------------------------------------------------------------
# Driver transforms
# ---------------------------------------------------------------------------


def average_daily_temperature(t_day: Any, t_night: Any) -> Any:
    """ACM's ``T``: the mean of daily maximum and minimum, degrees C.

    Not ``TA_F``. Decomposition uses the daily mean air temperature;
    photosynthesis uses this day/night mean.
    """
    return 0.5 * (np.asarray(t_day, dtype=float) + np.asarray(t_night, dtype=float))


def daily_temperature_half_range(t_day: Any, t_night: Any) -> Any:
    """ACM's ``D_T``: **half** the daily temperature range, degrees C.

    Table 1 defines ``D_T = (T_max - T_min) / 2``. Using the full range instead
    inflates the Eq. 5 denominator and silently suppresses conductance.
    """
    return 0.5 * (np.asarray(t_day, dtype=float) - np.asarray(t_night, dtype=float))


def leaf_area_index(c_fol: Any, lma: Any) -> Any:
    """``L = C_fol / c_lma``, dimensionless (Bloom & Williams Eq. A12).

    Foliar carbon is model state, so LAI moves with the run rather than being a
    driver.
    """
    lma_array = np.asarray(lma, dtype=float)
    if np.any(lma_array <= 0.0):
        raise ValueError(f"leaf mass per area must be positive, got {lma!r}")
    return np.asarray(c_fol, dtype=float) / lma_array


def frost_mask(
    t_day: Any,
    t_night: Any,
    frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
) -> np.ndarray:
    """Boolean mask, True on days where frost suppresses all carbon fixation.

    Reads only temperature drivers, so this is **parameter-independent** and can
    be precomputed once per driver record and reused for every parameter draw.
    That is what makes it safe under NUTS: the mask never moves during sampling,
    so no parameter-dependent branch enters the gradient graph.
    """
    return np.asarray(
        average_daily_temperature(t_day, t_night) < frost_threshold_degc, dtype=bool
    )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def acm_terms(
    *,
    doy: Any,
    t_day: Any,
    t_night: Any,
    sw_in: Any,
    co2: Any,
    c_fol: Any,
    lma: float,
    ceff: float,
    canopy_height_m: float,
    psi_d_mpa: float,
    frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
) -> dict[str, np.ndarray]:
    """Evaluate ACM and return every intermediate, vectorised over days.

    Solve order 2, 5, 6, 4, 8, 7, 9. Scalars are accepted and returned as
    0-d arrays.

    Parameters
    ----------
    doy
        Day of year, 1-366.
    t_day, t_night
        Daily mean daytime and nighttime air temperature, degrees C.
    sw_in
        Daily total incoming shortwave radiation, MJ m-2 d-1 (ACM's ``I``).
    co2
        Atmospheric CO2 mole fraction, umol mol-1 (ACM's ``C_a``).
    c_fol
        Foliar carbon, g C m-2. Model state.
    lma
        Leaf mass per area, g C m-2.
    ceff
        Canopy efficiency. Replaces ``a1 * N``.
    canopy_height_m
        Canopy height ``H``, metres. Fixed site constant.
    psi_d_mpa
        Leaf-to-soil water potential difference ``psi_d``, MPa, negative. Fixed
        site constant.
    frost_threshold_degc
        Average daily temperature below which no carbon is fixed.

    Returns
    -------
    dict of numpy.ndarray
        ``t_mean``, ``d_t``, ``lai``, ``p_n``, ``g_c``, ``discriminant``,
        ``discriminant_clamped``, ``c_i``, ``p_d``, ``e_0``, ``p_i``, ``d_ms``,
        ``day_length_factor``, ``gpp_unmasked``, ``frost_masked``, ``gpp``.
        GPP terms in g C m-2 d-1.

    Raises
    ------
    ValueError
        If the Eq. 5 denominator ``b2*H + D_T`` is not safely positive.
    """
    t_mean = average_daily_temperature(t_day, t_night)
    d_t = daily_temperature_half_range(t_day, t_night)
    lai = leaf_area_index(c_fol, lma)
    irradiance = np.asarray(sw_in, dtype=float)
    c_a = np.asarray(co2, dtype=float)

    # Eq. 2 -- N-limited photosynthetic capacity.
    p_n = ceff * lai * np.exp(A2 * t_mean)

    # Eq. 5 -- canopy conductance. The denominator is dominated by b2*H with H
    # fixed and positive, so it is safe, but a bad H or an inverted day/night
    # temperature pair would break it silently.
    conductance_denominator = B2 * canopy_height_m + d_t
    if np.any(conductance_denominator < _MIN_CONDUCTANCE_DENOMINATOR):
        raise ValueError(
            "Eq. 5 denominator (b2*H + D_T) is not safely positive: minimum "
            f"{float(np.min(conductance_denominator))!r} with H={canopy_height_m!r}. "
            "Check canopy height and that TA_F_DAY exceeds TA_F_NIGHT."
        )
    g_c = (-psi_d_mpa * np.exp(B1 * t_mean)) / conductance_denominator

    # Eq. 6 -- internal CO2. Eq. 3 is folded in here by the steady-state
    # assumption p_C = p_D, so it is never evaluated separately.
    q = THETA - K
    p = p_n / g_c
    #
    # The discriminant is provably non-negative for every reachable input.
    # As a quadratic in p,
    #     D(p) = p^2 - 2p(C_a - theta - k) + (C_a - theta + k)^2
    # whose own discriminant is negative -- so D > 0 for all real p -- exactly
    # when (C_a - theta) * k > 0, i.e. C_a > theta = 32.6 umol/mol. Below that
    # compensation point a negative window does exist, but it lies entirely at
    # p < 0, and p = p_N / g_c cannot be negative because p_N >= 0 and g_c > 0.
    # The floor below is therefore unreachable in practice. It is kept anyway,
    # and kept loud, because a NaN here would poison every gradient downstream.
    discriminant = (c_a + q - p) ** 2 - 4.0 * (c_a * q - p * THETA)
    discriminant_clamped = np.asarray(discriminant < 0.0, dtype=bool)
    if np.any(discriminant_clamped):
        # Never silent: a negative discriminant would yield NaN and poison every
        # gradient downstream, so it is floored -- and reported.
        warnings.warn(
            f"Eq. 6 discriminant went negative on "
            f"{int(np.count_nonzero(discriminant_clamped))} day(s) and was floored at "
            f"{_DISCRIMINANT_FLOOR}; the resulting GPP is not trustworthy there.",
            RuntimeWarning,
            stacklevel=2,
        )
        discriminant = np.maximum(discriminant, _DISCRIMINANT_FLOOR)
    c_i = 0.5 * (c_a + q - p + np.sqrt(discriminant))

    # Eq. 4 -- diffusion-limited rate.
    p_d = g_c * (c_a - c_i)

    # Eq. 8 -- canopy quantum yield. Must precede Eq. 7, which consumes it.
    e_0 = C1 * lai**2 / (C2 + lai**2)

    # Eq. 7 -- light limitation.
    light_supply = e_0 * irradiance
    limitation_denominator = light_supply + p_d
    # Both terms vanish together when there is no foliage; the limit of the
    # harmonic combination is zero, so this is the exact value, not a fudge.
    p_i = np.divide(
        light_supply * p_d,
        limitation_denominator,
        out=np.zeros_like(np.asarray(limitation_denominator, dtype=float)),
        where=limitation_denominator > 0.0,
    )

    # Eq. 9 -- day-length correction.
    d_ms = np.abs(MIDSUMMER_DOY - np.asarray(doy, dtype=float))
    day_length_factor = D1 * d_ms + D2
    gpp_unmasked = p_i * day_length_factor

    # Frost cutoff, applied after Eq. 9 so the arithmetic above always runs on
    # finite values.
    frost_masked = frost_mask(t_day, t_night, frost_threshold_degc)
    gpp = np.where(frost_masked, 0.0, gpp_unmasked)

    return {
        "t_mean": np.asarray(t_mean),
        "d_t": np.asarray(d_t),
        "lai": np.asarray(lai),
        "p_n": np.asarray(p_n),
        "g_c": np.asarray(g_c),
        "discriminant": np.asarray(discriminant),
        "discriminant_clamped": discriminant_clamped,
        "c_i": np.asarray(c_i),
        "p_d": np.asarray(p_d),
        "e_0": np.asarray(e_0),
        "p_i": np.asarray(p_i),
        "d_ms": np.asarray(d_ms),
        "day_length_factor": np.asarray(day_length_factor),
        "gpp_unmasked": np.asarray(gpp_unmasked),
        "frost_masked": frost_masked,
        "gpp": np.asarray(gpp),
    }


class AcmModel:
    """Callable ACM satisfying ``dalec.model_numpy.GppModel``.

    Carries the two fixed site constants and the frost threshold, which are not
    DALEC2 parameters and so cannot travel through the per-day protocol
    signature.

    Attributes
    ----------
    clamp_count
        Number of days on which the Eq. 6 discriminant had to be floored. Should
        be zero; anything else invalidates those days and is warned about.
    """

    def __init__(
        self,
        *,
        canopy_height_m: float,
        psi_d_mpa: float,
        frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
    ) -> None:
        if canopy_height_m <= 0.0:
            raise ValueError(f"canopy height must be positive, got {canopy_height_m!r}")
        if psi_d_mpa >= 0.0:
            raise ValueError(
                f"psi_d is a leaf-to-soil water potential difference and must be "
                f"negative, got {psi_d_mpa!r}"
            )
        self.canopy_height_m = float(canopy_height_m)
        self.psi_d_mpa = float(psi_d_mpa)
        self.frost_threshold_degc = float(frost_threshold_degc)
        self.clamp_count = 0

    def terms(self, **kwargs: Any) -> dict[str, np.ndarray]:
        """Vectorised :func:`acm_terms` with this model's site constants applied."""
        return acm_terms(
            canopy_height_m=self.canopy_height_m,
            psi_d_mpa=self.psi_d_mpa,
            frost_threshold_degc=self.frost_threshold_degc,
            **kwargs,
        )

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
        terms = self.terms(
            doy=doy,
            t_day=t_day,
            t_night=t_night,
            sw_in=sw_in,
            co2=co2,
            c_fol=c_fol,
            lma=lma,
            ceff=ceff,
        )
        self.clamp_count += int(np.count_nonzero(terms["discriminant_clamped"]))
        return float(terms["gpp"])

    def __repr__(self) -> str:
        return (
            f"AcmModel(canopy_height_m={self.canopy_height_m!r}, "
            f"psi_d_mpa={self.psi_d_mpa!r}, "
            f"frost_threshold_degc={self.frost_threshold_degc!r})"
        )


def make_acm(
    *,
    canopy_height_m: float,
    psi_d_mpa: float,
    frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
) -> AcmModel:
    """Build the photosynthesis routine to pass as ``run_dalec2(gpp_fn=...)``."""
    return AcmModel(
        canopy_height_m=canopy_height_m,
        psi_d_mpa=psi_d_mpa,
        frost_threshold_degc=frost_threshold_degc,
    )


def acm_from_config(config: dict[str, Any]) -> AcmModel:
    """Build an :class:`AcmModel` from a loaded configuration.

    ``site.canopy_height_m`` and ``site.psi_d_mpa`` have **no defaults** and must
    be supplied from site literature; this raises rather than guessing them.
    """
    site = config.get("site") or {}
    acm_config = config.get("acm") or {}

    missing = [key for key in ("canopy_height_m", "psi_d_mpa") if site.get(key) is None]
    if missing:
        raise ValueError(
            f"site.{' and site.'.join(missing)} must be set in the config. "
            "Canopy height (m) and leaf-to-soil water potential difference (MPa, "
            "negative) are fixed site constants taken from the site literature; "
            "they are not DALEC2 parameters and have no defaults."
        )

    return make_acm(
        canopy_height_m=float(site["canopy_height_m"]),
        psi_d_mpa=float(site["psi_d_mpa"]),
        frost_threshold_degc=float(
            acm_config.get("frost_threshold_degc", DEFAULT_FROST_THRESHOLD_DEGC)
        ),
    )
