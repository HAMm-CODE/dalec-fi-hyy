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
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

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
    "GppMagnitudeGate",
    "annual_gpp_comparison",
    "canopy_efficiency_sweep",
    "format_gpp_magnitude_report",
    "gpp_magnitude_gate",
]

#: Partitioned products compared against. Consistency check only, never
#: assimilated.
_PRODUCTS: tuple[str, str] = ("gpp_nt", "gpp_dt")


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
        record ``ratio_ref``, and the mean per-year ``ratio_nt`` / ``ratio_dt``.
    """
    if ceff_values is None:
        lower, upper = prior_bounds("ceff")
        ceff_values = np.geomspace(lower, upper, 16)

    rows = []
    for ceff in np.asarray(ceff_values, dtype=float):
        swept = replace(params, ceff=float(ceff))
        output = run_dalec2(swept, site_data, gpp_fn=gpp_fn, phenology_fn=phenology_fn)
        table = annual_gpp_comparison(output, site_data)
        rows.append(
            {
                "ceff": float(ceff),
                "gpp_model": float(table["gpp_model"].sum()),
                "ratio_ref": _record_ratio(table),
                "ratio_nt": float(np.nanmean(table["ratio_nt"].to_numpy(dtype=float))),
                "ratio_dt": float(np.nanmean(table["ratio_dt"].to_numpy(dtype=float))),
            }
        )

    return pd.DataFrame(rows).set_index("ceff")


@dataclass(frozen=True)
class GppMagnitudeGate:
    """Outcome of the pre-calibration GPP magnitude check.

    Attributes
    ----------
    passed
        True when some ``ceff`` in the prior range brings the whole-record
        modelled/observed GPP ratio inside ``[1/tolerance, tolerance]``.
    tolerance
        Permitted multiplicative discrepancy.
    best_ceff, best_ratio
        The sweep point closest to a ratio of one, in log space.
    sweep
        Output of :func:`canopy_efficiency_sweep`.
    detail
        Per-year :func:`annual_gpp_comparison` at ``best_ceff``.
    """

    passed: bool
    tolerance: float
    best_ceff: float
    best_ratio: float
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
) -> GppMagnitudeGate:
    """Check that modelled annual GPP is the right order of magnitude.

    Sweeps ``ceff`` across its prior range and asks whether *any* value brings
    modelled annual GPP within a factor of ``tolerance`` of the partitioned
    products. Callers must refuse to calibrate when this returns ``passed=False``.

    Raises
    ------
    ValueError
        If ``tolerance`` is not greater than one, or the products are absent.
    NotImplementedError
        If ``gpp_fn`` is left at the default stub.
    """
    if tolerance <= 1.0:
        raise ValueError(f"tolerance must be greater than 1, got {tolerance!r}")
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

    detail = annual_gpp_comparison(
        run_dalec2(
            replace(params, ceff=best_ceff),
            site_data,
            gpp_fn=gpp_fn,
            phenology_fn=phenology_fn,
        ),
        site_data,
    )

    return GppMagnitudeGate(
        passed=bool(1.0 / tolerance <= best_ratio <= tolerance),
        tolerance=tolerance,
        best_ceff=best_ceff,
        best_ratio=best_ratio,
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
        f"  tolerance          factor of {gate.tolerance:g}",
        f"  best ceff          {gate.best_ceff:.3g}  (prior range "
        f"{prior_bounds('ceff')[0]:g}-{prior_bounds('ceff')[1]:g})",
        f"  best ratio         {gate.best_ratio:.3f}  (modelled / partitioned mean)",
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
        lines += [
            "  DO NOT PROCEED TO CALIBRATION.",
            "  No canopy efficiency in the prior range brings modelled annual GPP",
            f"  within a factor of {gate.tolerance:g} of the partitioned products. A forward",
            "  model this far off on total GPP will still converge and will still",
            "  produce a posterior, and that posterior will be meaningless.",
            "",
        ]
    return "\n".join(lines)
