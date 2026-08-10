"""Tests for the pre-calibration GPP magnitude gate."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from conftest import SyntheticFluxnet, make_parameters
from dalec.data_io import SiteData, build_site_data, load_fluxnet_dd
from dalec.diagnostics import (
    annual_gpp_comparison,
    canopy_efficiency_sweep,
    format_gpp_magnitude_report,
    gpp_magnitude_gate,
)
from dalec.model_numpy import run_dalec2

#: Roughly calibrates the fake photosynthesis routine below so that its annual
#: total lands near the synthetic partitioned products at mid-range ceff.
WELL_SCALED = 0.474


@pytest.fixture
def block(synthetic_fluxnet: SyntheticFluxnet) -> SiteData:
    """Two clean years from the synthetic FLUXNET file, with GPP products."""
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    return build_site_data(frame, start_year=2000, end_year=2001, qc_threshold=0.75)


def saturating_gpp(scale: float) -> Callable[..., float]:
    """A stand-in photosynthesis routine that saturates in ceff, as ACM does."""

    def _gpp(*, sw_in: float, ceff: float, **_: object) -> float:
        return scale * (ceff / (ceff + 20.0)) * sw_in

    return _gpp


# ---------------------------------------------------------------------------
# Annual comparison
# ---------------------------------------------------------------------------


def test_annual_comparison_reports_one_row_per_year(block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    table = annual_gpp_comparison(output, block)

    assert list(table.index) == [2000, 2001]
    assert list(table["days"]) == [366, 365]
    for column in ("gpp_model", "gpp_nt", "gpp_dt", "ratio_nt", "ratio_dt", "ratio_ref"):
        assert np.isfinite(table[column]).all(), column


def test_modelled_total_is_the_plain_annual_sum(block: SiteData) -> None:
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    table = annual_gpp_comparison(output, block)

    in_2000 = block.years == 2000
    assert table.loc[2000, "gpp_model"] == pytest.approx(output.gpp[in_2000].sum())


def test_ratio_is_computed_on_matched_days_only(block: SiteData) -> None:
    """A gappy product must not inflate the ratio in proportion to its gaps."""
    output = run_dalec2(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    complete = annual_gpp_comparison(output, block)

    # Punch out half of one year's NT product. The matched-day ratio must not
    # move; a naive full-year-over-available-days ratio would roughly double.
    gappy_nt = block.partitioned["gpp_nt"].copy()
    holes = (block.years == 2000) & (np.arange(block.n_days) % 2 == 0)
    gappy_nt[holes] = np.nan
    gappy = SiteData(
        **{
            **{
                name: getattr(block, name)
                for name in (
                    "time", "doy", "t_air", "t_day", "t_night", "sw_in", "co2",
                    "nee_obs", "nee_unc", "nee_qc", "nee_mask", "attrs",
                )
            },
            "partitioned": {**block.partitioned, "gpp_nt": gappy_nt},
        }
    )
    holed = annual_gpp_comparison(output, gappy)

    assert holed.loc[2000, "coverage_nt"] == pytest.approx(0.5)
    assert holed.loc[2000, "ratio_nt"] == pytest.approx(
        complete.loc[2000, "ratio_nt"], rel=0.05
    )
    # ...whereas the raw product total does fall by about half.
    assert holed.loc[2000, "gpp_nt"] < 0.6 * complete.loc[2000, "gpp_nt"]


def test_run_length_mismatch_raises(block: SiteData) -> None:
    frame = load_fluxnet_dd(block.attrs["source_file"])
    shorter = build_site_data(frame, start_year=2000, end_year=2000)
    output = run_dalec2(make_parameters(), shorter, gpp_fn=saturating_gpp(WELL_SCALED))

    with pytest.raises(ValueError, match="same period"):
        annual_gpp_comparison(output, block)


def test_missing_partitioned_products_raise(block: SiteData) -> None:
    stripped = SiteData(
        **{**{k: v for k, v in vars(block).items() if k != "partitioned"}, "partitioned": {}}
    )
    output = run_dalec2(make_parameters(), stripped, gpp_fn=saturating_gpp(WELL_SCALED))

    with pytest.raises(ValueError, match="partitioned product"):
        annual_gpp_comparison(output, stripped)


# ---------------------------------------------------------------------------
# Canopy efficiency sweep
# ---------------------------------------------------------------------------


def test_sweep_spans_the_ceff_prior_range(block: SiteData) -> None:
    sweep = canopy_efficiency_sweep(
        make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED)
    )

    assert sweep.index.min() == pytest.approx(10.0)
    assert sweep.index.max() == pytest.approx(100.0)
    assert sweep["gpp_model"].is_monotonic_increasing


def test_sweep_honours_explicit_ceff_values(block: SiteData) -> None:
    sweep = canopy_efficiency_sweep(
        make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED),
        ceff_values=np.array([10.0, 40.0, 100.0]),
    )
    assert list(sweep.index) == [10.0, 40.0, 100.0]


def test_sweep_overrides_ceff_rather_than_mutating_params(block: SiteData) -> None:
    params = make_parameters(ceff=15.0)
    canopy_efficiency_sweep(params, block, gpp_fn=saturating_gpp(WELL_SCALED))
    assert params.ceff == 15.0


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_gate_passes_for_a_well_scaled_forward_model(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))

    assert gate.passed
    assert 1.0 / gate.tolerance <= gate.best_ratio <= gate.tolerance
    assert 10.0 <= gate.best_ceff <= 100.0


def test_gate_fails_when_gpp_is_five_times_too_high(block: SiteData) -> None:
    """The reported failure mode: 5x on annual GPP must block calibration."""
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(5 * WELL_SCALED))

    assert not gate.passed
    assert gate.best_ratio > 1.5
    assert "DO NOT PROCEED" in format_gpp_magnitude_report(gate)


def test_gate_fails_when_gpp_is_far_too_low(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED / 10))

    assert not gate.passed
    assert gate.best_ratio < 1.0 / 1.5


def test_gate_picks_the_ceff_closest_to_a_ratio_of_one(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))

    distances = np.abs(np.log(gate.sweep["ratio_ref"].to_numpy(dtype=float)))
    assert gate.best_ceff == pytest.approx(float(gate.sweep.index[int(distances.argmin())]))


def test_gate_treats_over_and_under_shoot_symmetrically(block: SiteData) -> None:
    """Closest in log space, so 2x high and 2x low are equally far off."""
    high = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(3 * WELL_SCALED))
    low = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED / 3))

    assert not high.passed
    assert not low.passed
    assert high.best_ratio > 1.0
    assert low.best_ratio < 1.0


def test_gate_requires_a_photosynthesis_routine(block: SiteData) -> None:
    """Left at the default stub, the gate must raise rather than report nonsense."""
    with pytest.raises(NotImplementedError, match="Aggregated Canopy Model"):
        gpp_magnitude_gate(make_parameters(), block)


def test_gate_rejects_a_meaningless_tolerance(block: SiteData) -> None:
    with pytest.raises(ValueError, match="tolerance must be greater than 1"):
        gpp_magnitude_gate(
            make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED), tolerance=1.0
        )


def test_report_is_readable_and_states_what_the_products_are(block: SiteData) -> None:
    gate = gpp_magnitude_gate(make_parameters(), block, gpp_fn=saturating_gpp(WELL_SCALED))
    report = format_gpp_magnitude_report(gate)

    assert "GPP magnitude gate: PASS" in report
    assert "never assimilated" in report
    assert "matched days" in report
    assert "2000" in report and "2001" in report
