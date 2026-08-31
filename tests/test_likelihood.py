"""The likelihood graph must reproduce the numpy log-likelihood exactly.

``dalec.diagnostics.gaussian_loglik`` is the reference: it is what Tasks 1 and 2
were computed against, so the sampler must feel the same surface those
diagnostics measured. Agreement is required to machine precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import SyntheticFluxnet, synthetic_extremes
from dalec.data_io import build_site_data, load_fluxnet_dd
from dalec.diagnostics import gaussian_loglik
from dalec.likelihood import (
    OBSERVED_NAME,
    assimilable_indices,
    build_likelihood,
    gaussian_loglik_tensor,
)

pytestmark = pytest.mark.pytensor


@pytest.fixture(scope="module")
def block(synthetic_fluxnet: SyntheticFluxnet):
    """A synthetic block carrying a realistic mix of masked and usable days.

    The conftest recipe sets NEE missing for ten days of 2001 and low-QC for
    twenty more, so the mask is genuinely mixed rather than all-True. 2002 is
    excluded: it carries deliberate SW_IN gaps the forward model refuses.
    """
    frame = load_fluxnet_dd(synthetic_fluxnet.path)
    return build_site_data(
        frame, start_year=2000, end_year=2001, qc_threshold=0.75,
        daily_extremes=synthetic_extremes(pd.DatetimeIndex(frame.index)),
    )


@pytest.fixture(scope="module")
def predicted(block):
    """A prediction that is neither the truth nor absurd.

    Finite on every day, masked ones included, because that is what the forward
    model returns -- it integrates through every day. ``block.nee_obs`` is NaN on
    the deliberately-missing days, so it is filled before the noise is added.
    """
    rng = np.random.default_rng(4)
    truth = np.nan_to_num(block.nee_obs, nan=0.0)
    return truth + rng.normal(0.0, 0.4, block.n_days)


def _evaluate(tensor):
    import pytensor

    return float(pytensor.function([], tensor, on_unused_input="ignore")())


class TestAgreementWithNumpy:
    def test_the_loglik_matches_the_numpy_reference(self, block, predicted):
        """The headline requirement: same number, to machine precision."""
        import pytensor.tensor as pt

        symbolic = _evaluate(
            gaussian_loglik_tensor(pt.as_tensor_variable(predicted), block)
        )
        reference = gaussian_loglik(predicted, block)
        assert symbolic == pytest.approx(reference, rel=1e-13, abs=1e-9)

    def test_it_matches_across_several_predictions(self, block):
        """One agreement could be luck; a spread of residual sizes could not."""
        import pytensor.tensor as pt

        rng = np.random.default_rng(9)
        for scale in (0.05, 0.5, 3.0):
            prediction = block.nee_obs + rng.normal(0.0, scale, block.n_days)
            symbolic = _evaluate(
                gaussian_loglik_tensor(pt.as_tensor_variable(prediction), block)
            )
            assert symbolic == pytest.approx(
                gaussian_loglik(prediction, block), rel=1e-13, abs=1e-9
            )

    def test_the_loglik_is_differentiable(self, block, predicted):
        import pytensor
        import pytensor.tensor as pt

        scale = pt.dscalar("scale")
        tensor = gaussian_loglik_tensor(
            scale * pt.as_tensor_variable(predicted), block
        )
        gradient = pytensor.function([scale], pt.grad(tensor, scale))
        analytic = float(gradient(1.0))
        step = 1e-6
        numeric = (
            gaussian_loglik(predicted * (1 + step), block)
            - gaussian_loglik(predicted * (1 - step), block)
        ) / (2 * step)
        assert analytic == pytest.approx(numeric, rel=1e-5)

    def test_a_nan_on_a_masked_day_poisons_the_gradient_not_the_value(
        self, block, predicted
    ):
        """The asymmetry that makes this worth asserting.

        Indexing protects the log-likelihood **value** from a non-finite
        prediction on a masked day, but reverse-mode still accumulates back
        through the selection to the full series, where ``0 * nan = nan``. So the
        number you would check stays healthy while the gradient the sampler uses
        does not.

        The forward model returns a finite series, so this cannot arise in the
        pipeline. It is pinned here so a future change that produces NaN on days
        it believes are ignored fails loudly rather than stalling NUTS.
        """
        import pytensor
        import pytensor.tensor as pt

        masked = np.flatnonzero(~block.nee_mask)
        assert masked.size > 0, "fixture must contain masked days"
        poisoned = predicted.copy()
        poisoned[masked[0]] = np.nan

        scale = pt.dscalar("scale")
        tensor = gaussian_loglik_tensor(scale * pt.as_tensor_variable(poisoned), block)

        value = pytensor.function([scale], tensor)
        assert np.isfinite(float(value(1.0))), "the value survives"

        gradient = pytensor.function([scale], pt.grad(tensor, scale))
        assert np.isnan(float(gradient(1.0))), "the gradient does not"


class TestMasking:
    def test_masked_days_are_excluded_not_downweighted(self, block, predicted):
        """Changing a prediction on a masked day must not move the logp at all.

        A zero weight would still evaluate the term; indexing makes the day
        structurally absent, which is what this asserts.
        """
        import pytensor.tensor as pt

        masked = np.flatnonzero(~block.nee_mask)
        if masked.size == 0:
            pytest.skip("synthetic block has no masked days")
        before = _evaluate(
            gaussian_loglik_tensor(pt.as_tensor_variable(predicted), block)
        )
        perturbed = predicted.copy()
        perturbed[masked] += 1e6
        after = _evaluate(
            gaussian_loglik_tensor(pt.as_tensor_variable(perturbed), block)
        )
        assert after == before

    def test_a_change_on_an_assimilable_day_does_move_the_logp(self, block, predicted):
        """The mirror of the above; without it the first test passes vacuously."""
        import pytensor.tensor as pt

        usable = np.flatnonzero(block.nee_mask)
        before = _evaluate(
            gaussian_loglik_tensor(pt.as_tensor_variable(predicted), block)
        )
        perturbed = predicted.copy()
        perturbed[usable[0]] += 1.0
        after = _evaluate(
            gaussian_loglik_tensor(pt.as_tensor_variable(perturbed), block)
        )
        assert after != before

    def test_the_selection_comes_from_the_site_data_mask(self, block):
        indices, observations, sigma = assimilable_indices(block)
        assert np.array_equal(indices, np.flatnonzero(block.nee_mask))
        assert indices.size == block.n_assimilated
        assert np.array_equal(observations, block.nee_obs[indices])
        assert np.array_equal(sigma, block.nee_unc[indices])

    def test_every_selected_sigma_is_finite_and_positive(self, block):
        _indices, _observations, sigma = assimilable_indices(block)
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0.0)

    def test_it_refuses_a_block_with_nothing_to_fit(self, block):
        from dataclasses import replace

        empty = replace(block, nee_mask=np.zeros(block.n_days, dtype=bool))
        with pytest.raises(ValueError, match="nothing to fit"):
            assimilable_indices(empty)


class TestPymcIntegration:
    def test_the_observed_variable_reproduces_the_reference_logp(
        self, block, predicted
    ):
        """PyMC's own logp must equal the numpy reference too.

        The tensor and the observed variable are two routes to the same number,
        and the sampler uses the second one.
        """
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model() as model:
            likelihood = build_likelihood(
                predicted=pt.as_tensor_variable(predicted), site_data=block
            )
            logp = float(
                model.compile_logp(sum=True)({})
            )
        assert likelihood.n_assimilated == block.n_assimilated
        assert logp == pytest.approx(
            gaussian_loglik(predicted, block), rel=1e-13, abs=1e-9
        )

    def test_it_names_the_observed_variable(self, block, predicted):
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model() as model:
            build_likelihood(predicted=pt.as_tensor_variable(predicted), site_data=block)
        assert OBSERVED_NAME in {rv.name for rv in model.observed_RVs}

    def test_the_reported_loglik_matches_the_observed_logp(self, block, predicted):
        import pymc as pm
        import pytensor.tensor as pt

        with pm.Model() as model:
            likelihood = build_likelihood(
                predicted=pt.as_tensor_variable(predicted), site_data=block
            )
            model_logp = float(model.compile_logp(sum=True)({}))
        assert _evaluate(likelihood.loglik) == pytest.approx(
            model_logp, rel=1e-13, abs=1e-9
        )
