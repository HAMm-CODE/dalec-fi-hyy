"""The Gaussian observation likelihood, as a PyTensor graph.

``NEE_VUT_REF_RANDUNC`` is the per-observation standard deviation, and days that
fail QC or lack a usable sigma contribute nothing. **Masking is a likelihood
operation, not a time-series edit**: the forward model still integrates through
every day, and the mask only decides which days are compared against data.
Dropping the days from the integration instead would change the carbon
trajectory, which is a different model.

Nothing here re-derives what already exists. The mask comes from
``dalec.data_io._likelihood_mask`` by way of
:meth:`dalec.data_io.SiteData.likelihood_arrays`, which also supplies finite
placeholders on masked days so a NaN cannot propagate into the logp.

Masked days are dropped by **indexing**, not by weighting
--------------------------------------------------------
Multiplying a masked term by zero still evaluates it, still tapes it for
reverse-mode, and still lets a non-finite value reach the sum as ``0 * inf =
nan``. Selecting the assimilable days up front costs nothing at graph-build time
-- the mask is driver-side and never moves during sampling -- and makes the
masked days structurally absent rather than numerically cancelled.

``predicted`` must be finite on **every** day, masked ones included
------------------------------------------------------------------
Indexing protects the *value* but not the *gradient*. Reverse-mode accumulates
back through the selection to the full series, so a non-finite prediction on a
masked day contributes ``0 * nan = nan`` and poisons the gradient while the
log-likelihood itself still looks perfectly healthy. That asymmetry is the
dangerous part: the number you would check is fine, and the quantity the sampler
uses is not.

The forward model integrates every day and returns a finite series, so this does
not arise in the pipeline. It is asserted in the tests so that it stays that way,
and so that a future change producing NaN on days it believes are ignored fails
loudly rather than silently stalling NUTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from dalec.data_io import SiteData

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pymc as pm

__all__ = [
    "OBSERVED_NAME",
    "GaussianLikelihood",
    "assimilable_indices",
    "build_likelihood",
    "gaussian_loglik_tensor",
]

#: Name of the observed variable in the PyMC model.
OBSERVED_NAME: Final[str] = "nee_obs"


@dataclass(frozen=True)
class GaussianLikelihood:
    """The likelihood term, plus the arrays it was built on.

    Attributes
    ----------
    loglik
        Scalar tensor, the Gaussian log-likelihood on assimilable days.
    observed
        The PyMC observed random variable, or ``None`` when built outside a
        model context.
    indices
        Positions of the assimilable days in the driver record.
    observations, sigma
        The selected observations and standard deviations, both length
        ``n_assimilated``.
    n_assimilated
        How many days actually enter the likelihood.
    """

    loglik: Any
    observed: Any
    indices: np.ndarray
    observations: np.ndarray
    sigma: np.ndarray
    n_assimilated: int


def assimilable_indices(site_data: SiteData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, observations and sigmas of the days that enter the likelihood.

    Thin wrapper over :meth:`SiteData.likelihood_arrays`, which owns the mask.
    Raises rather than returning an empty selection: a likelihood over no days is
    a configuration error, and silently returning zero would look like a
    perfectly flat posterior.
    """
    observations, sigma, mask = site_data.likelihood_arrays()
    if not mask.any():
        raise ValueError(
            "no day passes the likelihood mask, so there is nothing to fit. "
            "Check the QC threshold and the RANDUNC column."
        )
    indices = np.flatnonzero(mask)
    selected_sigma = sigma[indices]
    if not np.all(np.isfinite(selected_sigma)) or np.any(selected_sigma <= 0.0):
        raise ValueError(
            "an assimilable day has a non-positive or non-finite sigma; "
            "likelihood_arrays should have excluded it"
        )
    return indices, observations[indices], selected_sigma


def gaussian_loglik_tensor(predicted: Any, site_data: SiteData) -> Any:
    """Gaussian log-likelihood of ``predicted`` NEE, as a scalar tensor.

    The same quantity as :func:`dalec.diagnostics.gaussian_loglik`, which is the
    numpy reference the tests check this against:

        -0.5 * sum(z^2) - sum(log(sigma)) - 0.5 * n * log(2*pi)

    The normalising terms do not depend on any parameter and so contribute
    nothing to the gradient, but they are included because the value is reported
    and compared against the numpy path, and a log-likelihood missing its
    constant is a different number.
    """
    import pytensor.tensor as pt

    indices, observations, sigma = assimilable_indices(site_data)
    residual = (pt.as_tensor_variable(observations) - predicted[indices]) / (
        pt.as_tensor_variable(sigma)
    )
    return (
        -0.5 * pt.sum(residual**2)
        - float(np.sum(np.log(sigma)))
        - 0.5 * indices.size * math.log(2.0 * math.pi)
    )


def build_likelihood(
    *,
    predicted: Any,
    site_data: SiteData,
    name: str = OBSERVED_NAME,
    model: pm.Model | None = None,
) -> GaussianLikelihood:
    """Attach the Gaussian observation likelihood to a PyMC model.

    Parameters
    ----------
    predicted
        Modelled NEE over the **full** driver record, one value per day;
        typically ``dalec.model.build_forward_graph(...).nee``. Masked days are
        selected out here, so pass the whole series.
    site_data
        Block supplying the observations, the RANDUNC sigmas and the mask.
    name
        Name for the observed variable.
    model
        Model to attach to; defaults to the enclosing ``pm.Model()`` context.

    Returns
    -------
    :class:`GaussianLikelihood`.
    """
    import pymc as pm

    indices, observations, sigma = assimilable_indices(site_data)
    model = pm.modelcontext(model)
    with model:
        observed = pm.Normal(
            name,
            mu=predicted[indices],
            sigma=sigma,
            observed=observations,
        )
    return GaussianLikelihood(
        loglik=gaussian_loglik_tensor(predicted, site_data),
        observed=observed,
        indices=indices,
        observations=observations,
        sigma=sigma,
        n_assimilated=int(indices.size),
    )
