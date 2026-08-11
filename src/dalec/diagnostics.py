"""Diagnostics: GPP magnitude gate, and (Phase 8) ArviZ convergence wrappers.

**Partially built.** The GPP magnitude gate below is complete and tested. The
Phase 8 ArviZ material -- convergence summaries, trace and pair plots,
posterior-versus-prior overlays, correlation matrices, posterior predictive
bands, pool trajectories, partitioned-product consistency plots -- is not built
yet.

The GPP magnitude gate
----------------------
A pre-calibration hard gate. Before any sampling, the forward model is run over
the calibration block and its annual GPP total is compared against the FLUXNET
partitioned products. If modelled annual GPP cannot be brought within a factor
of ``tolerance`` of the observed value anywhere in the ``ceff`` prior range,
calibration must not proceed: a forward model that is several-fold wrong on
total GPP will still converge, and will still produce a posterior, and that
posterior will be meaningless.

Note what this gate is and is not. ``GPP_NT_VUT_REF`` and ``GPP_DT_VUT_REF`` are
partitioning products, not observations -- they are never assimilated, and they
carry their own structural uncertainty. Using them here is an order-of-magnitude
sanity check on the forward model, not validation, and the two products bracket
rather than pin the truth. The gate is deliberately loose for that reason.

Coverage handling: ratios are computed on **matched days only**, i.e. days where
both the model and the product are present. Summing a gappy product over a year
and dividing by a complete modelled total would bias the ratio upward in
proportion to the gaps, which is exactly the kind of quiet error this gate
exists to catch.

The gate checks LAI too, and for a reason
-----------------------------------------
The sweep varies ``ceff``, but the GPP ratio is dominated by canopy density,
which the sweep holds at whatever the caller's parameter set implies. Measured
over 1997-2005 with LAI pinned artificially: ratio 0.87 at LAI 1, 1.91 at LAI 3
and 3.10 at LAI 15.4. So a gate that reported only the ratio could be made to
pass or fail at will by the parameter set handed to it, and the reader would
have no way to see that from the output.

The mechanism is not ``E_0``, which saturates by LAI 3 and barely moves after.
It is ``p_D``: at low LAI the model is diffusion-limited and ``ceff`` has real
leverage, while at high LAI every day falls in the light-limited branch of
Eq. 7, where GPP collapses to ``E_0 * I * (d1*D_ms + d2)`` and ``ceff`` does not
appear at all. A ratio computed in that regime is not a statement about ACM.

So :func:`gpp_magnitude_gate` reports the LAI trajectory alongside the ratio and
fails independently, with a distinct message, when modelled LAI leaves a
plausible band. Both failures are real; they mean different things and the
report says which.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from dalec.acm import ACM_CALIBRATION_BOUNDS, AcmModel
from dalec.data_io import SiteData
from dalec.model_numpy import (
    DalecOutput,
    GppModel,
    PhenologyModel,
    dalec2_phenology,
    gpp_not_implemented,
    run_dalec2,
)
from dalec.parameters import DalecParameters, prior_bounds

__all__ = [
    "DEFAULT_LAI_BAND",
    "GppMagnitudeGate",
    "annual_gpp_comparison",
    "calibration_bound_coverage",
    "canopy_efficiency_sweep",
    "format_gpp_magnitude_report",
    "gpp_magnitude_gate",
    "shoulder_season_gpp",
]

#: Partitioned products compared against. Consistency check only, never
#: assimilated.
_PRODUCTS: tuple[str, str] = ("gpp_nt", "gpp_dt")

#: Plausible band for modelled projected LAI in a boreal conifer stand,
#: dimensionless. Deliberately wide: this is a sanity band, not a target, and it
#: must not quietly become a constraint on the posterior. FI-Hyy's projected LAI
#: is around 3 (provisional, pending a SMEAR II citation).
DEFAULT_LAI_BAND: tuple[float, float] = (1.0, 8.0)


def _lai_trajectory(output: DalecOutput, params: DalecParameters) -> np.ndarray:
    """Modelled projected LAI per timestep, dimensionless (Eq. A12).

    Foliar carbon enters photosynthesis at its time-``t`` value, so the final
    state is dropped: it has no corresponding timestep.
    """
    return output.pool("c_fol")[:-1] / params.lma


def _matched_ratio(modelled: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float:
    """Modelled over observed, summed on matched days only."""
    if not mask.any():
        return float("nan")
    total = float(observed[mask].sum())
    if total == 0.0:
        return float("nan")
    return float(modelled[mask].sum()) / total


def _require_products(site_data: SiteData) -> None:
    missing = [name for name in _PRODUCTS if name not in site_data.partitioned]
    if missing:
        raise ValueError(
            f"site data carries no {missing} partitioned product(s); the GPP "
            "magnitude gate needs GPP_NT_VUT_REF and GPP_DT_VUT_REF. Re-run "
            "scripts/01_prepare_data.py against a FULLSET file."
        )


def annual_gpp_comparison(output: DalecOutput, site_data: SiteData) -> pd.DataFrame:
    """Per-year modelled GPP against the FLUXNET partitioned products.

    Parameters
    ----------
    output
        Forward run over exactly the days in ``site_data``.
    site_data
        Drivers and partitioned products for the same block.

    Returns
    -------
    pandas.DataFrame
        One row per calendar year, indexed by year, all GPP columns in
        g C m-2 yr-1:

        ``days``
            Days in the year.
        ``gpp_model``
            Modelled annual GPP total, over every day.
        ``gpp_nt``, ``gpp_dt``
            Product annual totals, over the days where each is present.
        ``coverage_nt``, ``coverage_dt``
            Fraction of days on which each product is present.
        ``ratio_nt``, ``ratio_dt``
            Modelled over observed, computed on matched days only.
        ``ratio_ref``
            Modelled over the mean of the two products, on days where both are
            present. This is the gate statistic.

    Raises
    ------
    ValueError
        If the run length does not match the block, or the products are absent.
    """
    _require_products(site_data)
    if output.n_steps != site_data.n_days:
        raise ValueError(
            f"forward run has {output.n_steps} steps but the block has "
            f"{site_data.n_days} days; they must cover the same period"
        )

    modelled = output.gpp
    nt = site_data.partitioned["gpp_nt"]
    dt = site_data.partitioned["gpp_dt"]
    years = site_data.years

    rows = []
    for year in np.unique(years):
        in_year = years == year
        model_year = modelled[in_year]
        nt_year, dt_year = nt[in_year], dt[in_year]

        has_nt = np.isfinite(nt_year)
        has_dt = np.isfinite(dt_year)
        has_both = has_nt & has_dt
        mean_product = 0.5 * (nt_year + dt_year)
        rows.append(
            {
                "year": int(year),
                "days": int(in_year.sum()),
                "gpp_model": float(model_year.sum()),
                "gpp_nt": float(nt_year[has_nt].sum()),
                "gpp_dt": float(dt_year[has_dt].sum()),
                "coverage_nt": float(has_nt.mean()),
                "coverage_dt": float(has_dt.mean()),
                "ratio_nt": _matched_ratio(model_year, nt_year, has_nt),
                "ratio_dt": _matched_ratio(model_year, dt_year, has_dt),
                "ratio_ref": _matched_ratio(model_year, mean_product, has_both),
            }
        )

    return pd.DataFrame(rows).set_index("year")


def _record_ratio(table: pd.DataFrame) -> float:
    """Whole-record modelled/observed ratio, weighting years by their totals."""
    weights = table["days"].to_numpy(dtype=float)
    ratios = table["ratio_ref"].to_numpy(dtype=float)
    valid = np.isfinite(ratios)
    if not valid.any():
        return float("nan")
    observed = table["gpp_model"].to_numpy(dtype=float)[valid] / ratios[valid]
    return float(
        (table["gpp_model"].to_numpy(dtype=float)[valid] * weights[valid]).sum()
        / (observed * weights[valid]).sum()
    )


def canopy_efficiency_sweep(
    params: DalecParameters,
    site_data: SiteData,
    *,
    gpp_fn: GppModel = gpp_not_implemented,
    phenology_fn: PhenologyModel = dalec2_phenology,
    ceff_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the forward model across a range of canopy efficiencies.

    Parameters
    ----------
    params
        Parameter set; ``ceff`` is overridden per sweep point.
    site_data
        Calibration block.
    gpp_fn
        Photosynthesis routine. Required -- the default raises, because there is
        nothing useful to sweep without one.
    phenology_fn
        Phenology routine, defaulting to the published A7/A8 pair.
    ceff_values
        Canopy efficiencies to try. Defaults to 16 log-spaced points across the
        registered ``ceff`` prior range; log spacing because the response
        saturates, so most of the informative variation sits at the low end.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``ceff``, with the whole-record ``gpp_model`` total, the
        record ``ratio_ref``, the mean per-year ``ratio_nt`` / ``ratio_dt``, and
        ``lai_mean`` / ``lai_max``. The LAI columns are there because canopy
        density, not ``ceff``, is what usually moves the ratio -- reading the
        sweep without them invites the wrong conclusion.
    """
    if ceff_values is None:
        lower, upper = prior_bounds("ceff")
        ceff_values = np.geomspace(lower, upper, 16)

    rows = []
    for ceff in np.asarray(ceff_values, dtype=float):
        swept = replace(params, ceff=float(ceff))
        output = run_dalec2(swept, site_data, gpp_fn=gpp_fn, phenology_fn=phenology_fn)
        table = annual_gpp_comparison(output, site_data)
        lai = _lai_trajectory(output, swept)
        rows.append(
            {
                "ceff": float(ceff),
                "gpp_model": float(table["gpp_model"].sum()),
                "ratio_ref": _record_ratio(table),
                "ratio_nt": float(np.nanmean(table["ratio_nt"].to_numpy(dtype=float))),
                "ratio_dt": float(np.nanmean(table["ratio_dt"].to_numpy(dtype=float))),
                "lai_mean": float(lai.mean()),
                "lai_max": float(lai.max()),
            }
        )

    return pd.DataFrame(rows).set_index("ceff")


@dataclass(frozen=True)
class GppMagnitudeGate:
    """Outcome of the pre-calibration GPP magnitude check.

    Two independent checks, both of which must hold. ``passed`` is their
    conjunction; read :attr:`ratio_ok` and :attr:`lai_ok` to see which failed,
    because they mean different things.

    Attributes
    ----------
    passed
        ``ratio_ok and lai_ok``. Callers must refuse to calibrate when False.
    ratio_ok
        True when some ``ceff`` in the prior range brings the whole-record
        modelled/observed GPP ratio inside ``[1/tolerance, tolerance]``.
    lai_ok
        True when the modelled LAI trajectory at ``best_ceff`` stays inside
        ``lai_band``. Checked independently of the ratio: a ratio computed at an
        implausible canopy density is not a statement about ACM.
    tolerance
        Permitted multiplicative discrepancy.
    best_ceff, best_ratio
        The sweep point closest to a ratio of one, in log space.
    lai_band
        Plausible ``(lower, upper)`` band for projected LAI.
    lai_min, lai_mean, lai_max, lai_end
        Modelled LAI over the block at ``best_ceff``. ``lai_end`` is the last
        timestep, which is what shows drift: a run that starts at a plausible
        LAI and ends far outside the band tells you nothing about GPP.
    sweep
        Output of :func:`canopy_efficiency_sweep`.
    detail
        Per-year :func:`annual_gpp_comparison` at ``best_ceff``.
    """

    passed: bool
    ratio_ok: bool
    lai_ok: bool
    tolerance: float
    best_ceff: float
    best_ratio: float
    lai_band: tuple[float, float]
    lai_min: float
    lai_mean: float
    lai_max: float
    lai_end: float
    sweep: pd.DataFrame
    detail: pd.DataFrame


def gpp_magnitude_gate(
    params: DalecParameters,
    site_data: SiteData,
    *,
    gpp_fn: GppModel = gpp_not_implemented,
    phenology_fn: PhenologyModel = dalec2_phenology,
    tolerance: float = 1.5,
    ceff_values: np.ndarray | None = None,
    lai_band: tuple[float, float] = DEFAULT_LAI_BAND,
) -> GppMagnitudeGate:
    """Check that modelled annual GPP is the right order of magnitude.

    Sweeps ``ceff`` across its prior range and asks whether *any* value brings
    modelled annual GPP within a factor of ``tolerance`` of the partitioned
    products. Callers must refuse to calibrate when this returns ``passed=False``.

    The modelled LAI trajectory is checked separately against ``lai_band`` and
    reported either way. See the module docstring: the GPP ratio is dominated by
    canopy density rather than by ``ceff``, so a ratio quoted without its LAI is
    not interpretable.

    Parameters
    ----------
    lai_band
        Plausible ``(lower, upper)`` projected LAI. A sanity band, not a target.

    Raises
    ------
    ValueError
        If ``tolerance`` is not greater than one, ``lai_band`` is not increasing
        and positive, or the products are absent.
    NotImplementedError
        If ``gpp_fn`` is left at the default stub.
    """
    if tolerance <= 1.0:
        raise ValueError(f"tolerance must be greater than 1, got {tolerance!r}")
    if not 0.0 < lai_band[0] < lai_band[1]:
        raise ValueError(f"lai_band must be positive and increasing, got {lai_band!r}")
    _require_products(site_data)

    sweep = canopy_efficiency_sweep(
        params,
        site_data,
        gpp_fn=gpp_fn,
        phenology_fn=phenology_fn,
        ceff_values=ceff_values,
    )

    ratios = sweep["ratio_ref"].to_numpy(dtype=float)
    if not np.isfinite(ratios).any():
        raise ValueError("no finite GPP ratio was produced; check the partitioned products")

    # Closest to one in log space, so a 2x overshoot and a 2x undershoot are
    # treated as equally far off.
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.abs(np.log(ratios))
    best = int(np.nanargmin(distance))
    best_ceff = float(sweep.index[best])
    best_ratio = float(ratios[best])

    best_params = replace(params, ceff=best_ceff)
    best_output = run_dalec2(
        best_params, site_data, gpp_fn=gpp_fn, phenology_fn=phenology_fn
    )
    detail = annual_gpp_comparison(best_output, site_data)

    lai = _lai_trajectory(best_output, best_params)
    lai_min, lai_max = float(lai.min()), float(lai.max())
    ratio_ok = bool(1.0 / tolerance <= best_ratio <= tolerance)
    lai_ok = bool(lai_band[0] <= lai_min and lai_max <= lai_band[1])

    return GppMagnitudeGate(
        passed=ratio_ok and lai_ok,
        ratio_ok=ratio_ok,
        lai_ok=lai_ok,
        tolerance=tolerance,
        best_ceff=best_ceff,
        best_ratio=best_ratio,
        lai_band=lai_band,
        lai_min=lai_min,
        lai_mean=float(lai.mean()),
        lai_max=lai_max,
        lai_end=float(lai[-1]),
        sweep=sweep,
        detail=detail,
    )


def format_gpp_magnitude_report(gate: GppMagnitudeGate) -> str:
    """Render a :class:`GppMagnitudeGate` as a readable console report."""
    verdict = "PASS" if gate.passed else "FAIL"
    lines = [
        "=" * 78,
        f"  GPP magnitude gate: {verdict}",
        "=" * 78,
        f"  GPP ratio          {'ok' if gate.ratio_ok else 'FAILED'}",
        f"  canopy density     {'ok' if gate.lai_ok else 'FAILED'}",
        "",
        f"  tolerance          factor of {gate.tolerance:g}",
        f"  best ceff          {gate.best_ceff:.3g}  (prior range "
        f"{prior_bounds('ceff')[0]:g}-{prior_bounds('ceff')[1]:g})",
        f"  best ratio         {gate.best_ratio:.3f}  (modelled / partitioned mean)",
        "",
        f"  LAI at best ceff   min {gate.lai_min:.2f}   mean {gate.lai_mean:.2f}   "
        f"max {gate.lai_max:.2f}   end {gate.lai_end:.2f}",
        f"  plausible band     {gate.lai_band[0]:g} to {gate.lai_band[1]:g}",
        "",
        "  Partitioned GPP products are a magnitude sanity check, not validation,",
        "  and are never assimilated. Ratios use matched days only.",
        "",
        "  Per-year totals at the best ceff, g C m-2 yr-1:",
        gate.detail[
            ["days", "gpp_model", "gpp_nt", "gpp_dt", "ratio_nt", "ratio_dt", "ratio_ref"]
        ].to_string(
            float_format=lambda value: f"{value:9.3f}",
        ),
        "",
        "  Canopy efficiency sweep:",
        gate.sweep.to_string(float_format=lambda value: f"{value:11.3f}"),
        "",
    ]
    if not gate.passed:
        lines += ["  DO NOT PROCEED TO CALIBRATION.", ""]
    if not gate.ratio_ok:
        lines += [
            "  GPP ratio failed.",
            "  No canopy efficiency in the prior range brings modelled annual GPP",
            f"  within a factor of {gate.tolerance:g} of the partitioned products. A forward",
            "  model this far off on total GPP will still converge and will still",
            "  produce a posterior, and that posterior will be meaningless.",
            "",
        ]
    if not gate.lai_ok:
        lines += [
            "  Canopy density failed, and this invalidates the ratio above.",
            f"  Modelled LAI runs {gate.lai_min:.2f} to {gate.lai_max:.2f}, outside the "
            f"plausible band {gate.lai_band[0]:g}-{gate.lai_band[1]:g},",
            f"  ending at {gate.lai_end:.2f}. The GPP ratio is dominated by canopy density,",
            "  not by ceff, so a ratio measured at this LAI says nothing about ACM.",
            "  Fix the parameters that set the canopy -- lma, f_fol, c_lf, cr_fall and",
            "  c_fol_0 -- before reading the ratio as a verdict on photosynthesis.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ACM structural-bias diagnostics
# ---------------------------------------------------------------------------


def shoulder_season_gpp(
    params: DalecParameters,
    site_data: SiteData,
    output: DalecOutput,
    *,
    acm: AcmModel,
) -> pd.DataFrame:
    """Quantify the residual temperature-independent GPP the frost mask misses.

    ACM has effectively no temperature response where the Eq. 7 light limitation
    dominates: GPP there collapses to ``E_0 * I * (d1*D_ms + d2)``, in which
    ``T`` does not appear. The frost cutoff suppresses that floor, but only
    below its threshold. Shoulder-season days that are cold and dim yet above
    the cutoff still emit it.

    This measures how much. The third column is the number that belongs in the
    thesis limitations as a measured quantity rather than a qualitative caveat.

    Parameters
    ----------
    params
        Parameter set used for ``output``; supplies ``lma`` and ``ceff``.
    site_data
        Driver record.
    output
        Forward run over exactly that record, supplying the foliar carbon
        trajectory.
    acm
        The photosynthesis routine, carrying the site constants and threshold.

    Returns
    -------
    pandas.DataFrame
        One row per calendar year, indexed by year:

        ``days``
            Days in the year.
        ``frost_days``
            Days on which the frost mask suppressed GPP entirely.
        ``light_limited_days``
            Days where the mask was **inactive** but ACM was light-limited
            (``E_0 * I < p_D``), i.e. still emitting the floor.
        ``gpp_light_limited``
            GPP contributed by those days, g C m-2 yr-1. The residual
            structural bias.
        ``gpp_total``
            Modelled annual GPP total, g C m-2 yr-1.
        ``fraction_light_limited``
            ``gpp_light_limited / gpp_total``.
    """
    if output.n_steps != site_data.n_days:
        raise ValueError(
            f"forward run has {output.n_steps} steps but the record has "
            f"{site_data.n_days} days; they must cover the same period"
        )

    # Foliar carbon enters photosynthesis at its time-t value, so drop the final
    # state, which has no corresponding timestep.
    terms = acm.terms(
        doy=site_data.doy,
        t_day=site_data.t_day,
        t_night=site_data.t_night,
        sw_in=site_data.sw_in,
        co2=site_data.co2,
        c_fol=output.pool("c_fol")[:-1],
        lma=params.lma,
        ceff=params.ceff,
    )

    frozen = terms["frost_masked"]
    light_limited = (~frozen) & (terms["e_0"] * site_data.sw_in < terms["p_d"])
    gpp = terms["gpp"]
    years = site_data.years

    rows = []
    for year in np.unique(years):
        in_year = years == year
        total = float(gpp[in_year].sum())
        residual = float(gpp[in_year & light_limited].sum())
        rows.append(
            {
                "year": int(year),
                "days": int(in_year.sum()),
                "frost_days": int(np.count_nonzero(frozen & in_year)),
                "light_limited_days": int(np.count_nonzero(light_limited & in_year)),
                "gpp_light_limited": residual,
                "gpp_total": total,
                "fraction_light_limited": residual / total if total > 0.0 else float("nan"),
            }
        )

    return pd.DataFrame(rows).set_index("year")


def calibration_bound_coverage(site_data: SiteData) -> pd.DataFrame:
    """Fraction of the driver record falling outside each ACM calibration bound.

    ACM must not extrapolate beyond the ranges it was fitted over
    (Williams et al. Table 1). This is a thesis limitation figure, not merely a
    test: it states how much of the record is extrapolation.

    Only the average-daily-temperature bound is currently registered in
    :data:`dalec.acm.ACM_CALIBRATION_BOUNDS`; the remaining Table 1 rows are
    still outstanding, and this will pick them up without code changes.

    Returns
    -------
    pandas.DataFrame
        Indexed by quantity, with ``lower``, ``upper``, ``n_days``,
        ``n_below``, ``n_above`` and ``fraction_outside``.
    """
    from dalec.acm import average_daily_temperature

    available: dict[str, np.ndarray] = {
        "t_mean": np.asarray(average_daily_temperature(site_data.t_day, site_data.t_night)),
        "sw_in": site_data.sw_in,
        "co2": site_data.co2,
    }

    rows = []
    for quantity, (lower, upper) in ACM_CALIBRATION_BOUNDS.items():
        if quantity not in available:
            raise KeyError(
                f"calibration bound registered for {quantity!r}, but the driver "
                f"record exposes only {sorted(available)}"
            )
        values = available[quantity]
        below = int(np.count_nonzero(values < lower))
        above = int(np.count_nonzero(values > upper))
        rows.append(
            {
                "quantity": quantity,
                "lower": lower,
                "upper": upper,
                "n_days": int(values.size),
                "n_below": below,
                "n_above": above,
                "fraction_outside": (below + above) / values.size,
            }
        )

    return pd.DataFrame(rows).set_index("quantity")
