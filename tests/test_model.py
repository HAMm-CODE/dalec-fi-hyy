"""The PyTensor graph must reproduce the numpy forward model exactly.

Two implementations of the same equations is a real drift risk, and the whole
point of the graph is that it is the *same* model with gradients attached. So
the central test runs both on identical parameters and identical drivers and
requires agreement to machine precision, not to a tolerance chosen to pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from dalec.acm import LOOBOS_EVERGREEN, AcmModel
from dalec.model import NEWTON_STEPS, build_forward_graph, psi_newton
from dalec.model_numpy import dalec2_phenology, run_dalec2
from dalec.parameters import DalecParameters, phenology_psi_f, prior_bounds, solve_psi

pytestmark = pytest.mark.pytensor

LATITUDE = 61.8474
FROST = -2.0
N_DAYS = 5113


def _drivers(n_days: int = N_DAYS, seed: int = 11) -> dict[str, np.ndarray]:
    """A driver record with FI-Hyy's shape: 14 years, boreal seasonality."""
    rng = np.random.default_rng(seed)
    day = np.arange(n_days)
    doy = (day % 365) + 1.0
    seasonal = -12.0 * np.cos(2 * np.pi * doy / 365.25)
    t_air = 4.32 + seasonal + rng.normal(0.0, 2.0, n_days)
    t_max = t_air + np.abs(rng.normal(4.0, 1.5, n_days))
    t_min = t_air - np.abs(rng.normal(4.0, 1.5, n_days))
    sw_in = np.clip(11.0 + 9.0 * np.cos(2 * np.pi * (doy - 172) / 365.25), 0.2, None)
    return {
        "doy": doy, "t_air": t_air, "t_max": t_max,
        "t_min": t_min, "sw_in": sw_in,
        "co2": np.full(n_days, 380.0),
    }


def _parameters(seed: int = 5) -> DalecParameters:
    """A parameter set inside every registry bound, built once and reused."""
    rng = np.random.default_rng(seed)
    scalars = {
        name: float(rng.uniform(*prior_bounds(name)))
        for name in (
            "f_auto", "theta_roo", "theta_woo", "theta_lit", "theta_som",
            "theta_min", "temperature_exponent", "d_onset", "cr_onset",
            "d_fall", "cr_fall", "lma", "ceff",
        )
    }
    scalars["c_lf"] = 0.25
    weights = rng.dirichlet(np.ones(4))
    pools = {
        "c_lab_0": 80.0, "c_fol_0": 600.0, "c_roo_0": 240.0,
        "c_woo_0": 6000.0, "c_lit_0": 120.0, "c_som_0": 5800.0,
    }
    return DalecParameters.from_allocation_simplex(
        f_auto=scalars.pop("f_auto"), allocation_weights=weights, **scalars, **pools
    )


def _run_graph(parameters: DalecParameters, drivers: dict[str, np.ndarray]):
    import pytensor

    graph = build_forward_graph(
        parameters={
            name: pytensor.tensor.as_tensor_variable(float(value))
            for name, value in parameters.to_dict().items()
        },
        latitude_deg=LATITUDE,
        coefficients=LOOBOS_EVERGREEN,
        frost_threshold_degc=FROST,
        **drivers,
    )
    function = pytensor.function(
        [], [graph.nee, graph.gpp, graph.pools], on_unused_input="ignore"
    )
    return function()


def _run_numpy(parameters: DalecParameters, drivers: dict[str, np.ndarray]):
    from dalec.data_io import SiteData

    block = SiteData(
        time=np.arange(drivers["doy"].size).astype("datetime64[D]"),
        doy=drivers["doy"].astype(int),
        t_air=drivers["t_air"], t_max=drivers["t_max"], t_min=drivers["t_min"],
        t_day=drivers["t_max"], t_night=drivers["t_min"],
        sw_in=drivers["sw_in"], co2=drivers["co2"],
        nee_obs=np.zeros(drivers["doy"].size),
        nee_unc=np.ones(drivers["doy"].size),
        nee_qc=np.zeros(drivers["doy"].size, dtype=int),
        nee_mask=np.ones(drivers["doy"].size, dtype=bool),
        partitioned={},
        attrs={},
    )
    acm = AcmModel(
        latitude_deg=LATITUDE,
        coefficients=LOOBOS_EVERGREEN,
        frost_threshold_degc=FROST,
    )
    return run_dalec2(parameters, block, gpp_fn=acm, phenology_fn=dalec2_phenology)


@pytest.fixture(scope="module")
def outputs():
    """One graph run and one numpy run on identical inputs."""
    drivers = _drivers()
    parameters = _parameters()
    nee, gpp, pools = _run_graph(parameters, drivers)
    reference = _run_numpy(parameters, drivers)
    return nee, gpp, pools, reference


# ---------------------------------------------------------------------------
# The psi solve
# ---------------------------------------------------------------------------


class TestPsiNewton:
    @pytest.mark.parametrize("c_lf", [0.20, 0.25, 0.2676, 0.30, 0.3333])
    def test_it_matches_the_brent_solve(self, c_lf):
        """The unrolled Newton must land on the same root Brent finds."""
        assert psi_newton(c_lf) == pytest.approx(solve_psi(c_lf), rel=1e-14)

    def test_it_satisfies_the_published_residual(self):
        """A9: 2*sqrt(pi)*log(1 - c_lf)*psi - exp(-psi^2) == 0."""
        for c_lf in (0.20, 0.27, 0.3333):
            psi = psi_newton(c_lf)
            residual = (
                2.0 * np.sqrt(np.pi) * np.log1p(-c_lf) * psi - np.exp(-psi * psi)
            )
            assert abs(residual) < 1e-14

    def test_it_converges_well_inside_the_step_budget(self):
        """Halving the steps must not move the answer, or the budget is tight."""
        for c_lf in (0.20, 0.3333):
            assert psi_newton(c_lf, NEWTON_STEPS // 2) == pytest.approx(
                psi_newton(c_lf, NEWTON_STEPS), rel=1e-15
            )

    def test_the_symbolic_path_matches_the_float_path(self):
        import pytensor
        import pytensor.tensor as pt

        value = pt.dscalar("c_lf")
        function = pytensor.function([value], psi_newton(value))
        for c_lf in (0.21, 0.29, 0.33):
            assert float(function(c_lf)) == pytest.approx(solve_psi(c_lf), rel=1e-14)

    def test_psi_f_agrees_with_the_cached_derivation(self):
        for c_lf, cr_fall in ((0.25, 90.0), (0.31, 40.0)):
            symbolic = psi_newton(c_lf) * cr_fall / np.sqrt(2.0)
            assert symbolic == pytest.approx(
                phenology_psi_f(c_lf, cr_fall), rel=1e-14
            )

    def test_it_is_differentiable(self):
        """The reason it exists: c_lf is sampled, so psi must carry a gradient."""
        import pytensor
        import pytensor.tensor as pt

        value = pt.dscalar("c_lf")
        gradient = pytensor.function([value], pt.grad(psi_newton(value), value))
        analytic = float(gradient(0.27))
        step = 1e-6
        numeric = (psi_newton(0.27 + step) - psi_newton(0.27 - step)) / (2 * step)
        assert analytic == pytest.approx(numeric, rel=1e-6)
        assert np.isfinite(analytic)


# ---------------------------------------------------------------------------
# Equivalence with the numpy forward model
# ---------------------------------------------------------------------------


class TestGraphMatchesNumpy:
    def test_nee_agrees_to_machine_precision(self, outputs):
        nee, _gpp, _pools, reference = outputs
        difference = np.max(np.abs(nee - reference.nee))
        scale = np.max(np.abs(reference.nee))
        assert difference / scale < 1e-13, f"max abs difference {difference}"

    def test_gpp_agrees_to_machine_precision(self, outputs):
        _nee, gpp, _pools, reference = outputs
        difference = np.max(np.abs(gpp - reference.gpp))
        scale = np.max(np.abs(reference.gpp))
        assert difference / scale < 1e-13, f"max abs difference {difference}"

    def test_every_pool_agrees_to_machine_precision(self, outputs):
        _nee, _gpp, pools, reference = outputs
        # numpy carries the initial state at index 0; the scan does not.
        expected = reference.pools[1:]
        for index in range(expected.shape[1]):
            difference = np.max(np.abs(pools[:, index] - expected[:, index]))
            scale = np.max(np.abs(expected[:, index]))
            assert difference / scale < 1e-13, f"pool {index}: {difference}"

    def test_the_record_is_the_full_calibration_length(self, outputs):
        nee, _gpp, _pools, _reference = outputs
        assert nee.size == N_DAYS


class TestGraphGuards:
    def test_it_refuses_drivers_of_different_lengths(self):
        import pytensor

        drivers = _drivers(64)
        drivers["sw_in"] = drivers["sw_in"][:-1]
        with pytest.raises(ValueError, match="share a length"):
            build_forward_graph(
                parameters={
                    name: pytensor.tensor.as_tensor_variable(float(value))
                    for name, value in _parameters().to_dict().items()
                },
                latitude_deg=LATITUDE,
                coefficients=LOOBOS_EVERGREEN,
                frost_threshold_degc=FROST,
                **drivers,
            )

    def test_it_refuses_an_empty_record(self):
        import pytensor

        drivers = {name: np.array([]) for name in _drivers(4)}
        with pytest.raises(ValueError):
            build_forward_graph(
                parameters={
                    name: pytensor.tensor.as_tensor_variable(float(value))
                    for name, value in _parameters().to_dict().items()
                },
                latitude_deg=LATITUDE,
                coefficients=LOOBOS_EVERGREEN,
                frost_threshold_degc=FROST,
                **drivers,
            )


# ---------------------------------------------------------------------------
# Performance regression
# ---------------------------------------------------------------------------


class TestGradientPerformance:
    """Guard against the closure regression coming back.

    Parameters closed over inside a ``pytensor.scan`` defeat loop-invariant code
    motion, so the 24 unrolled Newton steps that solve A9 were being taped at
    every one of 5113 timesteps. That cost 2,165 ms per gradient against 178 ms
    once the parameters were passed as explicit ``non_sequences`` -- a 12.2x
    difference, with bit-for-bit identical output.

    Bit-for-bit identical output is exactly why this needs a timing test: no
    correctness test can catch it. A future refactor that reintroduces a closure
    would pass every other test in this file and only show up as a sampling run
    taking six days instead of half of one.

    The threshold is deliberately loose. The measured gradient is ~180 ms and
    this machine varies 20-30% between runs, so 400 ms is comfortably above the
    healthy path and far below the 2,165 ms regression it exists to catch.
    """

    #: Milliseconds. See the class docstring for why this number.
    BUDGET_MS = 400.0
    WARMUP = 2
    TIMED = 5

    def test_the_gradient_stays_within_budget(self):
        import time

        import pytensor
        import pytensor.tensor as pt

        from dalec.parameters import PARAMETER_NAMES

        drivers = _drivers()
        parameters = _parameters()
        theta = pt.dvector("theta")
        named = dict(
            zip(PARAMETER_NAMES, [theta[i] for i in range(len(PARAMETER_NAMES))],
                strict=True)
        )
        graph = build_forward_graph(
            parameters=named,
            latitude_deg=LATITUDE,
            coefficients=LOOBOS_EVERGREEN,
            frost_threshold_degc=FROST,
            **drivers,
        )
        observed = np.zeros(N_DAYS)
        loss = pt.sum(pt.sqr(graph.nee - observed))
        gradient = pytensor.function(
            [theta], pt.grad(loss, theta), on_unused_input="ignore"
        )
        theta0 = np.array([parameters.to_dict()[name] for name in PARAMETER_NAMES])

        for _ in range(self.WARMUP):
            gradient(theta0)
        times = []
        for _ in range(self.TIMED):
            start = time.perf_counter()
            gradient(theta0)
            times.append(time.perf_counter() - start)

        # Minimum, not mean: this is a floor test, and the minimum is the least
        # contaminated by whatever else the machine is doing.
        best_ms = 1000.0 * min(times)
        assert best_ms < self.BUDGET_MS, (
            f"gradient took {best_ms:.0f} ms at {N_DAYS} steps, budget "
            f"{self.BUDGET_MS:.0f} ms. The usual cause is a parameter closed "
            "over inside the scan instead of passed via non_sequences; see "
            "DECISIONS.md section 12."
        )
