"""Aggregated Canopy Model -- daily GPP, in the form DALEC actually implements.

Primary source: Chuter, A. M., Aston, P. J., Skeldon, A. C. and Roulstone, I.
(2015), *A dynamical systems analysis of the data assimilation linked ecosystem
carbon (DALEC) models*, Chaos 25(3), 036401. Equations 12 and B1-B6; coefficient
values in Appendices A (Loobos) and D (Oregon).

Secondary source, retained below but **not used**: Williams, M. et al. (1997),
*Ecological Applications* 7(3), 882-894.

Inputs: day of year, atmospheric CO2 (umol mol-1), daily maximum and minimum air
temperature (degrees C), daily total shortwave radiation (MJ m-2 d-1), foliar
carbon (g C m-2), leaf mass per area (g C m-2) and canopy efficiency, plus one
fixed site constant, latitude.

Output: gross primary productivity, g C m-2 d-1.

Why this module was rewritten
-----------------------------
It was first implemented from Williams et al. (1997) Table 2, on the strength of
Bloom & Williams (2015) citing that paper for DALEC2's photosynthesis. **DALEC
does not use the 1997 form.** Chuter et al. write ACM out as DALEC implements it,
and it differs in three material ways. The 1997 implementation is retained at the
bottom of this module, renamed, because the comparison is thesis material.

**1. The day-length term, and this one dominates.** Williams et al. Eq. 9 applies
``p_I * (d1 * D_ms + d2)`` with ``D_ms = |173 - doy|`` -- a function of day of
year alone, *containing no latitude*. DALEC computes true day length from site
latitude and solar declination and applies ``p_I * (a2 * s + a5)``. At FI-Hyy
(61.85 N) the 1997 form overstates the factor severely, and worst in winter:

    doy              15      60     173     300     350
    Williams 1997  1.526   1.607   1.810   1.581   1.491
    DALEC          0.126   0.184   0.342   0.192   0.126
    overstatement  12.1x    8.7x    5.3x    8.3x   11.8x

A term with no latitude in it cannot know that Hyytiala gets five hours of
daylight in December. That is the origin of the winter GPP floor, and it is why
the floor was temperature-independent: the 1997 factor barely varies at all.

**2. Canopy conductance carries no canopy height.** Williams et al. Eq. 5 divides
by ``b2 * H + D_T``; Chuter B1 divides by ``0.5 * Tr + a6 * Rtot`` and raises
``|psi|`` to the power ``a10``. Canopy height does not appear anywhere in the
DALEC form, so it is no longer a required site constant.

**3. Quantum yield is 6.5x larger, and the temperature enters on Tmax.** Chuter
B4 gives ``E_0 = a7 * Cf^2 / (Cf^2 + a9 * lma^2)``, structurally the 1997 form
under ``L = Cf / lma`` but with ``a7 = 7.19`` against ``c1 = 0.989``. Chuter B3
uses ``exp(a8 * Tmax)`` -- **maximum** daily temperature, not the mean of maximum
and minimum.

Differences 1 and 3 roughly cancel at mid-latitude, which is why the error stayed
invisible until this model was run at a boreal site.

Coefficients are site-calibrated, not universal
-----------------------------------------------
Chuter publishes two complete and materially different parameter sets. That
difference is the point: these are fitted per site, not physical constants.
:data:`LOOBOS_EVERGREEN` (52 N, evergreen forest) is used here as the closest
available match to FI-Hyy by forest type; :data:`OREGON_PONDEROSA` is kept for
reference. **Neither set is boreal, and adopting Loobos is a stated limitation.**

Solve order
-----------
Conductance, then internal CO2, then the diffusion limit, then quantum yield,
then the light limitation, then day length. Eq. B4 must precede the light
limitation, which consumes ``E_0``. The steady-state assumption folds the
carboxylation rate into B2, so it is never evaluated separately.

DALEC2 substitutions
--------------------
``ceff`` replaces the ``p11 * N`` product of Chuter B3, exactly as it replaced
``a1 * N`` in the 1997 form, so foliar N is not implemented. ``L`` is
``C_fol / lma`` and comes from model state, not from drivers.

``Tmax`` and ``Tmin`` are the **true** daily maximum and minimum, derived from
the half-hourly product by ``scripts/01b_derive_tminmax.py``. ``Tr`` is the
**full** range ``Tmax - Tmin``; note B1 takes ``0.5 * Tr``, so the half-range
still appears, and getting that factor of two wrong is silent. Decomposition in
A5/A6 uses ``TA_F``, the daily mean, which is a third and distinct temperature
that never reaches this module.

``TA_F_DAY`` and ``TA_F_NIGHT`` were used as stand-ins for the extremes until the
half-hourly extremes were derived. They are a poor proxy -- the range they imply
correlates 0.641 with the truth and understates it by 4.79 degC -- and they are
retained only as the comparison baseline. See
``dalec.diagnostics.temperature_proxy_comparison``.

Frost cutoff
------------
No carbon is fixed below -2.0 degC average daily temperature, following the
precedent Williams et al. applied at their Oregon coniferous sites. ACM was
calibrated over 7-30 degC and must not extrapolate; FI-Hyy reaches roughly
-25 degC.

This is safe for gradient-based sampling. The condition reads only the
temperature driver, never a sampled parameter, so it is a fixed boolean mask over
the time series: precompute once and multiply. The mask does not move as
parameters change, gradients are simply zero on masked days, and no
parameter-dependent branching enters the graph. It is applied *after* the
day-length term, so the arithmetic always runs on finite values.

.. note::

   The cutoff was previously described as load-bearing, on the measurement that
   nothing else could suppress the winter floor. That measurement was made
   against the 1997 day-length term, which overstates midwinter by roughly 12x.
   Under the corrected form the floor is far smaller and the cutoff is no longer
   carrying that weight. It is retained on its original published justification
   -- refusing to extrapolate below the calibration range -- and its effect is
   now measured rather than assumed.

Measured behaviour of the corrected form
----------------------------------------
Recorded so that what the rewrite did and did not fix is not overstated.

**The seasonal amplitude is fixed.** Decomposed at a reference canopy (LAI 3,
``ceff`` 40), summer-to-winter: day length alone accounts for a 2.7x drop,
irradiance alone 3.9x, and the two together with realistic winter temperatures
10.5x -- before the frost mask contributes anything. The 1997 form could not
exceed 8.2x against an effectively unbounded true ratio.

**The direct temperature response is still weak, and the rewrite did not fix
it.** Above the frost threshold, at low irradiance:

    T (degC)     +20     +7      0
    GPP         3.455   3.411   3.384

About 2% across 20 degrees. Temperature reaches GPP only through
``exp(a8 * Tmax)`` with ``a8 = 0.0111``, and through the daily range in the B1
denominator; the light limitation damps both. The 1997 form gave 0.5% across
40 degrees, so this is better in kind but not in magnitude. **The seasonal cycle
here is carried by day length and irradiance, not by temperature.** That
distinction matters for RQ3 and must not be reported as resolved.

**Chuter's time origin is 21 December, not 1 January.** B5's ``t`` is days since
the winter solstice -- Section II.A shifts the time scale ten days back so the
day-length function is even around zero. Feeding it a raw calendar day of year
puts the solstice at doy 182 instead of 172 and the trough at 365 instead of 355:
a ten-day phase error through the entire seasonal cycle, and a silent one,
because the amplitude is identical and nothing looks wrong. With the origin
corrected, day length at 61.8474 N comes out 19.18 h at the solstice, 4.82 h at
midwinter and 11.88 h at both equinoxes, against true values near 19.0, 4.7 and
12.2.

**The frost mask still does something, but far less.** On a cold dim day it now
suppresses about 0.68 g C m-2 d-1, against roughly 2.66 under the 1997 form.

Resolved: the ``ceff`` prior is vindicated
------------------------------------------
``ceff`` stands in for ``p11 * N``. Chuter's two sets give ``7.4 * 4.0 = 29.6``
(Loobos evergreen) and ``2.155 * 2.7 = 5.82`` (Oregon pine). The DALEC2 prior of
10-100 brackets the evergreen value, so the prior is sound and
``dalec.parameters`` is left untouched.

An earlier measurement here suggested a workable range of 3-6, apparently
agreeing with the 1997 ``a1 * N`` value of 5.7. That agreement was an artefact:
it was the wrong ACM compensating for a light-response term inflated 5-12x by the
missing latitude. It is recorded because it is exactly the kind of coincidence
that looks like confirmation.

Scope: this site is outside the envelope DALEC2's authors defined
-----------------------------------------------------------------
Bloom & Williams (2015) Section 2.5 state that they selected sites with little
expected water stress and no more than **three months** of recorded
below-freezing soil temperature, because those criteria reflect the current
capabilities of DALEC2, hydrological processes not being explicitly represented.
FI-Hyy has roughly **four to six months** of soil frost. Applying DALEC2 here is
a deliberate scope decision, not an oversight.

Open questions for the code request to the authors
--------------------------------------------------
1. Chuter analyses DALEC EV and DALEC DE. Bloom & Williams (2015) cite Williams
   et al. (1997) for DALEC2's ACM, but DALEC2 merges the evergreen and deciduous
   versions, so it most likely inherits the implementation Chuter documents.
   **This cannot be confirmed from the papers**, and it is the single assumption
   this module rests on.
2. Whether a boreal-calibrated ``a``-parameter set exists. Neither published set
   is boreal.

Bearing on RQ3
--------------
Under the 1997 form, GPP was insensitive to temperature over most of the year,
which would have pushed seasonal structure onto the respiration parameters during
calibration and made observed equifinality partly structural. The corrected
day-length term restores a strong, latitude-driven seasonal cycle, so that
argument must be re-measured rather than carried over. The Phase 7 synthetic twin
separates structural from informational error, and must run with the frost mask
active over the same driver record or the comparison is not like-for-like.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

__all__ = [
    "ACM_CALIBRATION_BOUNDS",
    "DEFAULT_FROST_THRESHOLD_DEGC",
    "LOOBOS_EVERGREEN",
    "OREGON_PONDEROSA",
    "WILLIAMS_1997_COEFFICIENTS",
    "AcmCoefficients",
    "AcmModel",
    "acm_from_config",
    "acm_terms",
    "average_daily_temperature",
    "chuter_day_length_factor",
    "daily_temperature_half_range",
    "daily_temperature_range",
    "day_length_hours",
    "frost_mask",
    "leaf_area_index",
    "make_acm",
    "solar_declination_rad",
    "williams1997_day_length_factor",
    "williams1997_terms",
]


# ---------------------------------------------------------------------------
# Coefficient sets -- Chuter et al. (2015) Appendices A and D
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcmCoefficients:
    """One site-calibrated ACM parameter set.

    These are **fitted per site**, not physical constants. Chuter's two published
    sets differ substantially, and neither is boreal.

    Attributes
    ----------
    a2, a5
        Day-length slope and offset of Eq. 12: ``GPP = p_I * (a2 * s + a5)``,
        with ``s`` in hours.
    a3, a4
        CO2 compensation and half-saturation points of B2, umol mol-1.
        ``q = a3 - a4``.
    a6
        Hydraulic resistance coefficient of B1.
    a7, a9
        Maximum canopy quantum yield and its LAI coefficient, B4.
    a8
        Temperature coefficient of B3, applied to **maximum** daily temperature.
    a10
        Exponent on the water potential difference in B1.
    psi_mpa
        Maximum soil-to-leaf water potential difference, MPa, magnitude.
    r_tot
        Total plant-soil hydraulic resistance. Set to 1 in both published sets.
    p11, n_foliar
        Recorded for provenance only. Their product is what ``ceff`` replaces,
        so neither is used in the arithmetic.
    lma_reference
        The leaf mass per area the set was fitted with, g C m-2. Recorded for
        provenance; ``lma`` is sampled here, not fixed.
    source
        Where the set comes from.
    """

    a2: float
    a3: float
    a4: float
    a5: float
    a6: float
    a7: float
    a8: float
    a9: float
    a10: float
    psi_mpa: float
    r_tot: float
    p11: float
    n_foliar: float
    lma_reference: float
    source: str

    @property
    def ceff_equivalent(self) -> float:
        """``p11 * N`` -- the product ``ceff`` stands in for."""
        return self.p11 * self.n_foliar


#: Loobos evergreen forest, 52 N -- Chuter et al. Appendix A. **The set used
#: here**, as the closest available match to FI-Hyy by forest type. Its
#: ``ceff`` equivalent is 7.4 * 4.0 = 29.6, comfortably inside the DALEC2 prior
#: of 10-100.
LOOBOS_EVERGREEN: Final[AcmCoefficients] = AcmCoefficients(
    a2=0.0156,
    a3=4.22273,
    a4=208.868,
    a5=0.0453,
    a6=0.3783,
    a7=7.1929,
    a8=0.0111,
    a9=2.1001,
    a10=0.7897,
    psi_mpa=2.0,
    r_tot=1.0,
    p11=7.4,
    n_foliar=4.0,
    lma_reference=110.0,
    source="Chuter et al. (2015) Appendix A -- Loobos evergreen forest, 52 N",
)

#: Oregon ponderosa pine -- Chuter et al. Appendix D. Reference only. Its
#: ``ceff`` equivalent is 2.155 * 2.7 = 5.82, below the DALEC2 prior; the gap
#: between the two sets is the measure of how site-specific these are.
OREGON_PONDEROSA: Final[AcmCoefficients] = AcmCoefficients(
    a2=0.0142,
    a3=0.980,
    a4=217.9,
    a5=0.155,
    a6=2.653,
    a7=4.309,
    a8=0.060,
    a9=1.062,
    a10=0.0006,
    psi_mpa=0.8502,
    r_tot=1.0,
    p11=2.155,
    n_foliar=2.7,
    lma_reference=111.0,
    source="Chuter et al. (2015) Appendix D -- Oregon ponderosa pine",
)

#: Frost cutoff, degrees C. Below this average daily temperature, no carbon is
#: fixed. See the module docstring on why this is no longer described as the
#: only thing suppressing the winter floor.
DEFAULT_FROST_THRESHOLD_DEGC: Final[float] = -2.0

#: Bounds over which ACM was calibrated (Williams et al. Table 1, p. 884). Driver
#: days outside these are extrapolation, and the fraction outside is a thesis
#: limitation figure -- see ``dalec.diagnostics.calibration_bound_coverage``.
#:
#: Only the average-daily-temperature bound has been supplied so far. The
#: remaining rows are still needed; this mapping takes them without code changes.
ACM_CALIBRATION_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "t_mean": (7.0, 30.0),
}

#: Below this, the B1 denominator is treated as degenerate and raises.
_MIN_CONDUCTANCE_DENOMINATOR: Final[float] = 1e-6

#: Floor applied to a negative B2 discriminant. Never applied silently.
_DISCRIMINANT_FLOOR: Final[float] = 0.0

#: Days in the year used by the declination formula (Chuter B5). Deliberately
#: 365, not 365.25: it is what the published expression uses.
_DECLINATION_YEAR_DAYS: Final[float] = 365.0

#: Offset from calendar day of year to Chuter's solar time origin, days.
#:
#: Chuter Section II.A: "The time scale has been shifted ten days back, so that
#: t = 0 occurs on the 21st December, the shortest day. This was done in order to
#: ensure that the daylength function is even around zero." So ``t`` in B5 is
#: days since 21 December (doy 355), **not** calendar day of year. Feeding B5 a
#: raw doy puts the solstice at doy 182 instead of 172 -- a ten-day phase error
#: in the whole seasonal cycle, and silent, because the amplitude is unaffected.
_SOLAR_ORIGIN_OFFSET_DAYS: Final[float] = 10.0


# ---------------------------------------------------------------------------
# Driver transforms
# ---------------------------------------------------------------------------


def average_daily_temperature(t_max: Any, t_min: Any) -> Any:
    """Mean of daily maximum and minimum, degrees C.

    Used for the frost mask and for the calibration-bound diagnostics. **Not**
    used inside the DALEC form of ACM, which takes ``Tmax`` directly.
    """
    return 0.5 * (np.asarray(t_max, dtype=float) + np.asarray(t_min, dtype=float))


def daily_temperature_range(t_max: Any, t_min: Any) -> Any:
    """``Tr``: the **full** daily temperature range, degrees C.

    Chuter B1 takes ``0.5 * Tr``, so the half-range is what finally enters the
    conductance denominator. Passing the half-range here would halve it again.
    """
    return np.asarray(t_max, dtype=float) - np.asarray(t_min, dtype=float)


def daily_temperature_half_range(t_max: Any, t_min: Any) -> Any:
    """``0.5 * Tr``, degrees C -- the quantity B1 actually divides by."""
    return 0.5 * daily_temperature_range(t_max, t_min)


def leaf_area_index(c_fol: Any, lma: Any) -> Any:
    """``L = C_fol / lma``, dimensionless (Bloom & Williams Eq. A12).

    Foliar carbon is model state, so LAI moves with the run rather than being a
    driver.
    """
    lma_array = np.asarray(lma, dtype=float)
    if np.any(lma_array <= 0.0):
        raise ValueError(f"leaf mass per area must be positive, got {lma!r}")
    return np.asarray(c_fol, dtype=float) / lma_array


def solar_declination_rad(doy: Any) -> np.ndarray:
    """Solar declination, radians (Chuter B5).

    ``delta = -0.408 * cos(2*pi*t / 365)`` where ``t`` is **days since 21
    December**, not calendar day of year -- see :data:`_SOLAR_ORIGIN_OFFSET_DAYS`
    for the passage that says so. Peaks at +0.408 rad (23.4 deg) at doy 172, the
    June solstice, and troughs at doy 355.

    Passing a raw day of year here is a silent ten-day phase error: the amplitude
    is identical, so nothing looks wrong, but the whole modelled seasonal cycle
    sits ten days late against the observations it is fitted to.
    """
    solar_day = (
        np.asarray(doy, dtype=float) + _SOLAR_ORIGIN_OFFSET_DAYS
    ) % _DECLINATION_YEAR_DAYS
    return np.asarray(
        -0.408 * np.cos(2.0 * np.pi * solar_day / _DECLINATION_YEAR_DAYS),
        dtype=float,
    )


def day_length_hours(doy: Any, latitude_deg: float) -> np.ndarray:
    """Day length, hours (Chuter B6).

    ``s = 24 * arccos(-tan(lat) * tan(delta)) / pi``.

    The arccos argument is clamped to ``[-1, 1]``. Inside the polar circles it
    genuinely leaves that interval, and the clamp is what turns that into 24 h of
    daylight or 0 h rather than a NaN. At FI-Hyy (61.85 N) it stays within
    +/-0.81 all year, so the clamp never fires here -- but a NaN day length would
    silently poison every downstream gradient, so it is not left to chance.

    Returns 12.0 for every day at the equator, where ``tan(0) = 0``.
    """
    argument = -np.tan(np.radians(float(latitude_deg))) * np.tan(solar_declination_rad(doy))
    return 24.0 * np.arccos(np.clip(argument, -1.0, 1.0)) / np.pi


def chuter_day_length_factor(
    doy: Any, latitude_deg: float, coefficients: AcmCoefficients = LOOBOS_EVERGREEN
) -> np.ndarray:
    """``a2 * s + a5`` -- the DALEC day-length term of Eq. 12."""
    return coefficients.a2 * day_length_hours(doy, latitude_deg) + coefficients.a5


def frost_mask(
    t_max: Any,
    t_min: Any,
    frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
) -> np.ndarray:
    """Boolean mask, True on days where frost suppresses all carbon fixation.

    Reads only temperature drivers, so this is **parameter-independent** and can
    be precomputed once per driver record and reused for every parameter draw.
    That is what makes it safe under NUTS: the mask never moves during sampling,
    so no parameter-dependent branch enters the gradient graph.
    """
    return np.asarray(
        average_daily_temperature(t_max, t_min) < frost_threshold_degc, dtype=bool
    )


# ---------------------------------------------------------------------------
# The model -- Chuter et al. (2015), the form DALEC implements
# ---------------------------------------------------------------------------


def acm_terms(
    *,
    doy: Any,
    t_max: Any,
    t_min: Any,
    sw_in: Any,
    co2: Any,
    c_fol: Any,
    lma: float,
    ceff: float,
    latitude_deg: float,
    coefficients: AcmCoefficients = LOOBOS_EVERGREEN,
    frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
) -> dict[str, np.ndarray]:
    """Evaluate ACM and return every intermediate, vectorised over days.

    Scalars are accepted and returned as 0-d arrays.

    Parameters
    ----------
    doy
        Day of year, 1-366.
    t_max, t_min
        Daily mean daytime and nighttime air temperature, degrees C, standing in
        for the daily maximum and minimum.
    sw_in
        Daily total incoming shortwave radiation, MJ m-2 d-1 (``I``).
    co2
        Atmospheric CO2 mole fraction, umol mol-1 (``C_a``).
    c_fol
        Foliar carbon, g C m-2. Model state.
    lma
        Leaf mass per area, g C m-2.
    ceff
        Canopy efficiency. Replaces ``p11 * N``.
    latitude_deg
        Site latitude, degrees, positive north. Fixed site constant.
    coefficients
        Site-calibrated set; defaults to :data:`LOOBOS_EVERGREEN`.
    frost_threshold_degc
        Average daily temperature below which no carbon is fixed.

    Returns
    -------
    dict of numpy.ndarray
        ``t_mean``, ``t_max``, ``t_range``, ``lai``, ``g_c``, ``p``,
        ``discriminant``, ``discriminant_clamped``, ``c_i``, ``p_d``, ``e_0``,
        ``p_i``, ``declination``, ``day_length``, ``day_length_factor``,
        ``gpp_unmasked``, ``frost_masked``, ``gpp``. GPP terms in g C m-2 d-1.

    Raises
    ------
    ValueError
        If the B1 denominator ``0.5*Tr + a6*Rtot`` is not safely positive.
    """
    coef = coefficients
    t_mean = average_daily_temperature(t_max, t_min)
    t_max = np.asarray(t_max, dtype=float)
    t_range_raw = daily_temperature_range(t_max, t_min)
    lai = leaf_area_index(c_fol, lma)
    irradiance = np.asarray(sw_in, dtype=float)
    c_a = np.asarray(co2, dtype=float)

    # Tr is a *range*, so it cannot be negative. The day/night means standing in
    # for the daily maximum and minimum can invert, though -- on a warm front
    # overnight the daytime mean falls below the nighttime mean -- and at FI-Hyy
    # that happens on 14.8% of calibration days, driving the B1 denominator
    # negative on 5.0% of them. The FULLSET daily product carries no true
    # TA_F_MAX or TA_F_MIN, so this proxy is the only one available.
    #
    # Floored at zero, because a negative diurnal range is not a physical state
    # the model should integrate. Counted and reported, never silent.
    #
    # Note the direction, which is not the conservative one. Tr = 0 *minimises*
    # the B1 denominator and therefore *maximises* conductance: g_c = 4.570
    # against 1.25 at Tr = 2 and 0.32 at Tr = 10. Measured, the floor inflates
    # GPP on the affected days by 2.8% (against a plausible Tr of 1 degC) to
    # 10.9% (against 4 degC). Over the whole calibration block the effect is
    # +0.17% to +0.24%, because the affected days are mostly low-GPP winter days.
    #
    # This is an interim treatment. The real fix is to take true daily maximum
    # and minimum from the half-hourly FLUXNET product, where Tr is non-negative
    # by construction and this floor becomes unreachable.
    t_range_floored = np.asarray(t_range_raw < 0.0, dtype=bool)
    t_range = np.maximum(t_range_raw, 0.0)
    # Warn on the vectorised path only. The per-day scalar path runs once per
    # timestep, so warning there would emit hundreds of identical messages; there
    # ``AcmModel.range_floor_count`` is the reporting mechanism instead.
    if t_range_floored.ndim > 0 and np.any(t_range_floored):
        warnings.warn(
            f"daily temperature range was negative on "
            f"{int(np.count_nonzero(t_range_floored))} day(s) -- TA_F_DAY below "
            "TA_F_NIGHT -- and was floored at zero. Tr = 0 maximises canopy "
            "conductance, so this inflates GPP on those days rather than "
            "suppressing it; interim treatment pending true daily min/max.",
            RuntimeWarning,
            stacklevel=2,
        )

    # B1 -- canopy conductance. No canopy height: that was the 1997 form.
    conductance_denominator = 0.5 * t_range + coef.a6 * coef.r_tot
    if np.any(conductance_denominator < _MIN_CONDUCTANCE_DENOMINATOR):
        raise ValueError(
            "B1 denominator (0.5*Tr + a6*Rtot) is not safely positive: minimum "
            f"{float(np.min(conductance_denominator))!r}. With Tr floored at zero "
            "this can only mean a6 * Rtot is not positive."
        )
    g_c = abs(coef.psi_mpa) ** coef.a10 / conductance_denominator

    # B3 -- photosynthate, already divided through by conductance. ceff replaces
    # the p11 * N product. Note exp() takes *maximum* daily temperature.
    p = ceff * lai * np.exp(coef.a8 * t_max) / g_c

    # B2 -- internal CO2. The steady-state assumption folds the carboxylation
    # rate in here, so it is never evaluated separately.
    q = coef.a3 - coef.a4
    #
    # The discriminant is provably non-negative for every reachable input. As a
    # quadratic in p,
    #     D(p) = p^2 - 2p(C_a - a3 - a4) + (C_a - a3 + a4)^2
    # whose own discriminant is negative -- so D > 0 for all real p -- exactly
    # when (C_a - a3) * a4 > 0, i.e. C_a > a3. With a3 = 4.22 umol/mol for the
    # Loobos set and atmospheric CO2 near 380, that holds with enormous margin.
    # The floor below is therefore unreachable in practice. It is kept anyway,
    # and kept loud, because a NaN here would poison every gradient downstream.
    discriminant = (c_a + q - p) ** 2 - 4.0 * (c_a * q - coef.a3 * p)
    discriminant_clamped = np.asarray(discriminant < 0.0, dtype=bool)
    if np.any(discriminant_clamped):
        warnings.warn(
            f"B2 discriminant went negative on "
            f"{int(np.count_nonzero(discriminant_clamped))} day(s) and was floored at "
            f"{_DISCRIMINANT_FLOOR}; the resulting GPP is not trustworthy there.",
            RuntimeWarning,
            stacklevel=2,
        )
        discriminant = np.maximum(discriminant, _DISCRIMINANT_FLOOR)
    c_i = 0.5 * (c_a + q - p + np.sqrt(discriminant))

    # Diffusion-limited rate.
    p_d = g_c * (c_a - c_i)

    # B4 -- canopy quantum yield. Equivalent to a7 * L^2 / (L^2 + a9) under
    # L = Cf / lma; written as published. Must precede the light limitation.
    c_fol_array = np.asarray(c_fol, dtype=float)
    e_0 = coef.a7 * c_fol_array**2 / (c_fol_array**2 + coef.a9 * float(lma) ** 2)

    # Light limitation.
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

    # Eq. 12 -- true day length from latitude and declination. This is the term
    # the 1997 form gets wrong, by up to 12x at this latitude.
    declination = solar_declination_rad(doy)
    day_length = day_length_hours(doy, latitude_deg)
    day_length_factor = coef.a2 * day_length + coef.a5
    gpp_unmasked = p_i * day_length_factor

    # Frost cutoff, applied last so the arithmetic above always runs on finite
    # values.
    frost_masked = frost_mask(t_max, t_min, frost_threshold_degc)
    gpp = np.where(frost_masked, 0.0, gpp_unmasked)

    return {
        "t_mean": np.asarray(t_mean),
        "t_max": t_max,
        "t_range": np.asarray(t_range),
        "t_range_raw": np.asarray(t_range_raw),
        "t_range_floored": t_range_floored,
        "lai": np.asarray(lai),
        "g_c": np.asarray(np.broadcast_to(g_c, np.shape(t_range))),
        "p": np.asarray(p),
        "discriminant": np.asarray(discriminant),
        "discriminant_clamped": discriminant_clamped,
        "c_i": np.asarray(c_i),
        "p_d": np.asarray(p_d),
        "e_0": np.asarray(e_0),
        "p_i": np.asarray(p_i),
        "declination": declination,
        "day_length": day_length,
        "day_length_factor": np.asarray(day_length_factor),
        "gpp_unmasked": np.asarray(gpp_unmasked),
        "frost_masked": frost_masked,
        "gpp": np.asarray(gpp),
    }


class AcmModel:
    """Callable ACM satisfying ``dalec.model_numpy.GppModel``.

    Carries the site latitude, the coefficient set and the frost threshold, none
    of which are DALEC2 parameters and so cannot travel through the per-day
    protocol signature.

    Attributes
    ----------
    clamp_count
        Number of days on which the B2 discriminant had to be floored. Should be
        zero; anything else invalidates those days and is warned about.
    range_floor_count
        Number of days on which the daily temperature range came out negative --
        the day/night means inverting -- and was floored at zero. At FI-Hyy this
        is around 15% of days and is a property of the driver product, not a
        fault. Tr = 0 maximises conductance, so the floor inflates GPP on those
        days; see :func:`acm_terms` for the measured magnitude.
    """

    def __init__(
        self,
        *,
        latitude_deg: float,
        coefficients: AcmCoefficients = LOOBOS_EVERGREEN,
        frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
    ) -> None:
        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"latitude must lie in [-90, 90] degrees, got {latitude_deg!r}")
        self.latitude_deg = float(latitude_deg)
        self.coefficients = coefficients
        self.frost_threshold_degc = float(frost_threshold_degc)
        self.clamp_count = 0
        self.range_floor_count = 0

    def terms(self, **kwargs: Any) -> dict[str, np.ndarray]:
        """Vectorised :func:`acm_terms` with this model's site constants applied."""
        return acm_terms(
            latitude_deg=self.latitude_deg,
            coefficients=self.coefficients,
            frost_threshold_degc=self.frost_threshold_degc,
            **kwargs,
        )

    def __call__(
        self,
        *,
        doy: int,
        t_max: float,
        t_min: float,
        sw_in: float,
        co2: float,
        c_fol: float,
        lma: float,
        ceff: float,
    ) -> float:
        """Return GPP for one day, g C m-2 d-1."""
        terms = self.terms(
            doy=doy,
            t_max=t_max,
            t_min=t_min,
            sw_in=sw_in,
            co2=co2,
            c_fol=c_fol,
            lma=lma,
            ceff=ceff,
        )
        self.clamp_count += int(np.count_nonzero(terms["discriminant_clamped"]))
        self.range_floor_count += int(np.count_nonzero(terms["t_range_floored"]))
        return float(terms["gpp"])

    def __repr__(self) -> str:
        return (
            f"AcmModel(latitude_deg={self.latitude_deg!r}, "
            f"coefficients={self.coefficients.source!r}, "
            f"frost_threshold_degc={self.frost_threshold_degc!r})"
        )


def make_acm(
    *,
    latitude_deg: float,
    coefficients: AcmCoefficients = LOOBOS_EVERGREEN,
    frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
) -> AcmModel:
    """Build the photosynthesis routine to pass as ``run_dalec2(gpp_fn=...)``."""
    return AcmModel(
        latitude_deg=latitude_deg,
        coefficients=coefficients,
        frost_threshold_degc=frost_threshold_degc,
    )


def acm_from_config(config: dict[str, Any]) -> AcmModel:
    """Build an :class:`AcmModel` from a loaded configuration.

    ``site.latitude_deg`` has **no default** and must be supplied from site
    metadata; this raises rather than guessing it. Canopy height is no longer
    required -- it does not appear in the DALEC form of ACM.
    """
    site = config.get("site") or {}
    acm_config = config.get("acm") or {}

    if site.get("latitude_deg") is None:
        raise ValueError(
            "site.latitude_deg must be set in the config. Latitude is a fixed "
            "site constant, taken from the site metadata; it drives the day "
            "length term of Chuter et al. Eq. 12 and has no default."
        )

    return make_acm(
        latitude_deg=float(site["latitude_deg"]),
        coefficients=LOOBOS_EVERGREEN,
        frost_threshold_degc=float(
            acm_config.get("frost_threshold_degc", DEFAULT_FROST_THRESHOLD_DEGC)
        ),
    )


# ---------------------------------------------------------------------------
# Retained: Williams et al. (1997), the published-paper variant
#
# NOT USED by the forward model, and kept deliberately. This is the form the
# 1997 paper prints and the form this project first implemented, on the strength
# of Bloom & Williams (2015) citing that paper. DALEC does not implement it --
# see the module docstring for the three differences and the measured
# consequences. It is retained so the comparison can be reproduced rather than
# asserted, and because the discrepancy is thesis material.
#
# Do not wire this back into the forward model.
# ---------------------------------------------------------------------------

#: Williams et al. (1997) Table 2, p. 887. ``a1 = 2.95`` is recorded but was
#: never used: DALEC2 replaces the whole ``a1 * N`` product with ``ceff``.
WILLIAMS_1997_COEFFICIENTS: Final[dict[str, float]] = {
    "a1": 2.95,
    "a2": 0.018,
    "theta": 32.6,
    "k": 576.7,
    "b1": -0.029,
    "b2": 0.315,
    "c1": 0.989,
    "c2": 0.873,
    "d1": -0.0018,
    "d2": 1.81,
}

#: Reference day of year for the 1997 day-length correction.
_WILLIAMS_MIDSUMMER_DOY: Final[float] = 173.0


def williams1997_day_length_factor(doy: Any) -> np.ndarray:
    """``d1 * |173 - doy| + d2`` -- the 1997 day-length term.

    Retained for comparison. **Contains no latitude**, which is the single most
    consequential difference from the DALEC form: at 61.85 N it overstates the
    factor by 5.3x at midsummer and 12.1x in January.
    """
    coef = WILLIAMS_1997_COEFFICIENTS
    d_ms = np.abs(_WILLIAMS_MIDSUMMER_DOY - np.asarray(doy, dtype=float))
    return np.asarray(coef["d1"] * d_ms + coef["d2"], dtype=float)


def williams1997_terms(
    *,
    doy: Any,
    t_max: Any,
    t_min: Any,
    sw_in: Any,
    co2: Any,
    c_fol: Any,
    lma: float,
    ceff: float,
    canopy_height_m: float,
    psi_d_mpa: float,
    frost_threshold_degc: float = DEFAULT_FROST_THRESHOLD_DEGC,
) -> dict[str, np.ndarray]:
    """The Williams et al. (1997) form, solve order 2, 5, 6, 4, 8, 7, 9.

    Retained for comparison only -- see the block comment above. ``psi_d_mpa`` is
    negative here, unlike the magnitude Chuter uses.
    """
    coef = WILLIAMS_1997_COEFFICIENTS
    t_mean = average_daily_temperature(t_max, t_min)
    d_t = daily_temperature_half_range(t_max, t_min)
    lai = leaf_area_index(c_fol, lma)
    irradiance = np.asarray(sw_in, dtype=float)
    c_a = np.asarray(co2, dtype=float)

    # Eq. 2 -- N-limited photosynthetic capacity.
    p_n = ceff * lai * np.exp(coef["a2"] * t_mean)

    # Eq. 5 -- canopy conductance, with the canopy height DALEC does not use.
    conductance_denominator = coef["b2"] * canopy_height_m + d_t
    if np.any(conductance_denominator < _MIN_CONDUCTANCE_DENOMINATOR):
        raise ValueError(
            "Eq. 5 denominator (b2*H + D_T) is not safely positive: minimum "
            f"{float(np.min(conductance_denominator))!r} with H={canopy_height_m!r}."
        )
    g_c = (-psi_d_mpa * np.exp(coef["b1"] * t_mean)) / conductance_denominator

    # Eq. 6 -- internal CO2; Eq. 3 folded in by the steady-state assumption.
    q = coef["theta"] - coef["k"]
    p = p_n / g_c
    discriminant = (c_a + q - p) ** 2 - 4.0 * (c_a * q - p * coef["theta"])
    discriminant_clamped = np.asarray(discriminant < 0.0, dtype=bool)
    if np.any(discriminant_clamped):
        discriminant = np.maximum(discriminant, _DISCRIMINANT_FLOOR)
    c_i = 0.5 * (c_a + q - p + np.sqrt(discriminant))

    # Eq. 4, 8, 7 -- diffusion limit, quantum yield, light limitation.
    p_d = g_c * (c_a - c_i)
    e_0 = coef["c1"] * lai**2 / (coef["c2"] + lai**2)
    light_supply = e_0 * irradiance
    limitation_denominator = light_supply + p_d
    p_i = np.divide(
        light_supply * p_d,
        limitation_denominator,
        out=np.zeros_like(np.asarray(limitation_denominator, dtype=float)),
        where=limitation_denominator > 0.0,
    )

    # Eq. 9 -- the latitude-free day-length correction.
    day_length_factor = williams1997_day_length_factor(doy)
    gpp_unmasked = p_i * day_length_factor
    frost_masked = frost_mask(t_max, t_min, frost_threshold_degc)

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
        "day_length_factor": day_length_factor,
        "gpp_unmasked": np.asarray(gpp_unmasked),
        "frost_masked": frost_masked,
        "gpp": np.asarray(np.where(frost_masked, 0.0, gpp_unmasked)),
    }
