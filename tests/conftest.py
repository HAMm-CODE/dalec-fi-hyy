"""Shared pytest fixtures and test doubles.

The synthetic FLUXNET file below is deliberately small and fully specified: the
tests derive their expected counts from the same recipe constants the fixture
uses, so a change to the fixture cannot silently make an assertion vacuous.

The forward-model helpers at the bottom supply the two components that are not
implemented yet -- photosynthesis and phenology -- as controllable doubles, so
that equations A1-A6 can be tested on their own.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running the tests from a checkout without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dalec.parameters import DalecParameters  # noqa: E402

MISSING = -9999.0

# --- Recipe for the synthetic file ------------------------------------------
START = "2000-01-01"
END = "2002-12-31"

#: NEE (and its QC) set missing, as real files do when the day is unusable.
NEE_MISSING_DATES = pd.date_range("2001-01-01", "2001-01-10", freq="D")
#: NEE present but poor quality.
LOW_QC_DATES = pd.date_range("2001-02-01", "2001-02-20", freq="D")
LOW_QC_VALUE = 0.10
#: NEE present and good quality, but no random uncertainty -> no likelihood sd.
RANDUNC_MISSING_DATES = pd.date_range("2000-03-01", "2000-03-05", freq="D")
#: A driver gap. Makes 2002 unusable outright.
DRIVER_GAP_DATES = pd.date_range("2002-06-01", "2002-06-03", freq="D")

GOOD_QC_VALUE = 1.0


@dataclass(frozen=True)
class SyntheticFluxnet:
    """Path to a synthetic FLUXNET2015 FULLSET DD csv, plus its recipe."""

    path: Path
    index: pd.DatetimeIndex
    frame: pd.DataFrame


def _build_frame() -> pd.DataFrame:
    index = pd.date_range(START, END, freq="D")
    doy = index.dayofyear.to_numpy()
    rng = np.random.default_rng(20260809)

    seasonal = np.sin(2.0 * np.pi * (doy - 100) / 365.25)

    frame = pd.DataFrame(index=index)
    frame["TA_F"] = 4.0 + 14.0 * seasonal
    frame["TA_F_DAY"] = frame["TA_F"] + 2.5
    frame["TA_F_NIGHT"] = frame["TA_F"] - 2.5
    # Daily mean W m-2, always positive.
    frame["SW_IN_F"] = 110.0 + 95.0 * seasonal
    frame["CO2_F_MDS"] = 370.0 + 0.005 * np.arange(len(index))
    frame["VPD_F"] = 3.0 + 2.0 * seasonal  # present in the file, never a driver
    frame["NEE_VUT_REF"] = -2.5 * seasonal + 0.4 * rng.standard_normal(len(index))
    frame["NEE_VUT_REF_QC"] = GOOD_QC_VALUE
    frame["NEE_VUT_REF_RANDUNC"] = 0.30 + 0.10 * np.abs(seasonal)
    frame["GPP_NT_VUT_REF"] = np.clip(3.0 * seasonal + 3.0, 0.0, None)
    frame["GPP_DT_VUT_REF"] = np.clip(3.1 * seasonal + 3.0, 0.0, None)
    frame["RECO_NT_VUT_REF"] = 1.5 + 0.8 * seasonal
    frame["RECO_DT_VUT_REF"] = 1.6 + 0.8 * seasonal

    # Injected gaps, using the FLUXNET sentinel.
    frame.loc[NEE_MISSING_DATES, ["NEE_VUT_REF", "NEE_VUT_REF_QC"]] = MISSING
    frame.loc[LOW_QC_DATES, "NEE_VUT_REF_QC"] = LOW_QC_VALUE
    frame.loc[RANDUNC_MISSING_DATES, "NEE_VUT_REF_RANDUNC"] = MISSING
    frame.loc[DRIVER_GAP_DATES, "SW_IN_F"] = MISSING

    return frame


@pytest.fixture(scope="session")
def synthetic_fluxnet(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFluxnet:
    """Write a synthetic FLUXNET2015 FULLSET DD csv and return its path."""
    frame = _build_frame()
    out = frame.copy()
    out.insert(0, "TIMESTAMP", frame.index.strftime("%Y%m%d").astype("int64"))

    directory = tmp_path_factory.mktemp("fluxnet")
    path = directory / "FLX_XX-Syn_FLUXNET2015_FULLSET_DD_2000-2002_1-3.csv"
    out.to_csv(path, index=False)
    return SyntheticFluxnet(path=path, index=pd.DatetimeIndex(frame.index), frame=frame)


# ---------------------------------------------------------------------------
# Forward model (Phase 2) helpers
# ---------------------------------------------------------------------------

#: A 4-simplex over (f_lab, f_fol, f_roo, f_woo). Deliberately asymmetric so a
#: transposed allocation order shows up as a wrong answer rather than a
#: coincidence.
DEFAULT_ALLOCATION_WEIGHTS: tuple[float, float, float, float] = (0.10, 0.30, 0.20, 0.40)

#: Baseline parameter set. Every value lies inside its published prior range
#: (asserted by ``test_baseline_parameters_lie_inside_their_prior_ranges``), and
#: the rates satisfy ``(theta_lit + theta_min) * exp(temperature_exponent * T)
#: < 1`` across any plausible boreal temperature, which is the condition for
#: pools to stay non-negative under an explicit Euler step.
BASELINE_PARAMETERS: dict[str, float] = {
    "theta_roo": 2.0e-3,
    "theta_woo": 1.0e-4,
    "theta_lit": 1.0e-2,
    "theta_som": 1.0e-4,
    "theta_min": 5.0e-3,
    "temperature_exponent": 0.05,
    "d_onset": 120.0,
    "cr_onset": 30.0,
    "d_fall": 280.0,
    "cr_fall": 40.0,
    # Annual leaf fall fraction, i.e. 1 / leaf lifespan in years -- not a
    # duration in days. Prior range is [1/8, 1].
    "c_lf": 0.5,
    "lma": 60.0,
    "ceff": 15.0,
    "c_lab_0": 30.0,
    "c_fol_0": 100.0,
    "c_roo_0": 150.0,
    "c_woo_0": 8000.0,
    "c_lit_0": 200.0,
    "c_som_0": 12000.0,
}


def make_parameters(
    *,
    f_auto: float = 0.5,
    allocation_weights: tuple[float, float, float, float] = DEFAULT_ALLOCATION_WEIGHTS,
    **overrides: float,
) -> DalecParameters:
    """Build a valid :class:`DalecParameters` from the baseline, with overrides."""
    values = {**BASELINE_PARAMETERS, **overrides}
    return DalecParameters.from_allocation_simplex(
        f_auto=f_auto, allocation_weights=allocation_weights, **values
    )


@dataclass(frozen=True)
class StubDrivers:
    """Minimal stand-in satisfying ``dalec.model_numpy.Drivers``."""

    doy: np.ndarray
    t_air: np.ndarray
    t_day: np.ndarray
    t_night: np.ndarray
    sw_in: np.ndarray
    co2: np.ndarray


def make_drivers(n_days: int, *, seed: int = 0, constant_t_air: float | None = None) -> StubDrivers:
    """Seasonally varying daily drivers for ``n_days`` starting at day of year 1."""
    rng = np.random.default_rng(seed)
    doy = ((np.arange(n_days) % 365) + 1).astype(int)
    seasonal = np.sin(2.0 * np.pi * (doy - 100) / 365.25)
    t_air = (
        np.full(n_days, constant_t_air, dtype=float)
        if constant_t_air is not None
        else 3.5 + 14.0 * seasonal + 0.5 * rng.standard_normal(n_days)
    )
    return StubDrivers(
        doy=doy,
        t_air=t_air,
        t_day=t_air + 2.5,
        t_night=t_air - 2.5,
        sw_in=np.clip(9.0 + 8.0 * seasonal, 0.05, None),
        co2=380.0 + 0.005 * np.arange(n_days),
    )


def constant_gpp(value: float) -> Callable[..., float]:
    """A ``gpp_fn`` double returning the same GPP every day, g C m-2 d-1."""

    def _gpp(**_: object) -> float:
        return value

    return _gpp


def constant_phenology(onset: float, fall: float) -> Callable[..., tuple[float, float]]:
    """A ``phenology_fn`` double returning fixed onset and fall fractions."""

    def _phenology(**_: object) -> tuple[float, float]:
        return onset, fall

    return _phenology


def seasonal_phenology(
    *, onset_peak: float = 0.05, fall_peak: float = 0.02
) -> Callable[..., tuple[float, float]]:
    """A ``phenology_fn`` double with smooth spring and autumn pulses in [0, 1].

    Not the DALEC2 phenology -- just a well-behaved stand-in with the right
    shape and range for exercising long runs.
    """

    def _phenology(*, doy: int, d_onset: float, cr_onset: float, d_fall: float,
                   cr_fall: float, c_lf: float, **_: object) -> tuple[float, float]:
        onset = onset_peak * float(np.exp(-(((doy - d_onset) / cr_onset) ** 2)))
        fall = fall_peak * float(np.exp(-(((doy - d_fall) / cr_fall) ** 2)))
        return onset, fall

    return _phenology
