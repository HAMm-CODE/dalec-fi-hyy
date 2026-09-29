"""Assemble priors, forward model and likelihood into a PyMC model, and sample.

Nothing new is defined here. The three pieces already exist and are already
tested against their own references, so this module's only job is to connect
them and to make the connection auditable:

    dalec.priors.build_priors        -> every prior, read from its recorded source
    dalec.model.build_forward_graph  -> DALEC2, equal to the numpy model to 1e-14
    dalec.likelihood.build_likelihood -> Gaussian on RANDUNC, masked days excluded

``model_variant`` picks the model structure. ``"dalec2"`` (the default) is the
path above, unchanged. ``"evergreen"`` swaps the first two for
:mod:`dalec.model_evergreen` -- Williams et al. (2005), no labile pool and no
phenology -- and keeps the likelihood, the data and every shared prior, for the
RQ3 structure comparison (docs/evergreen.md).

The LAI convention is carried through rather than defaulted, because both
conventions stay live and the calibration is run under each (DECISIONS §10,
LIMITATIONS §15). :class:`DalecModel` records which one it was built with so a
result cannot be reported without it.

Sampling is deliberately not configured here beyond what PyMC needs. Step size,
target acceptance and tree depth are left at their defaults so that a first run
measures the problem rather than a tuning choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from dalec.acm import AcmCoefficients, AcmModel
from dalec.data_io import SiteData
from dalec.likelihood import GaussianLikelihood, build_likelihood
from dalec.model import DalecGraph, build_forward_graph
from dalec.model_numpy import POOL_NAMES
from dalec.parameters import DEFAULT_LAI_CONVENTION
from dalec.priors import DalecPriors, build_priors

if TYPE_CHECKING:  # pragma: no cover - typing only
    import arviz as az
    import pymc as pm

__all__ = [
    "DEFAULT_MODEL_VARIANT",
    "MODEL_VARIANTS",
    "DalecModel",
    "build_model",
    "sample",
]

#: Model structures the sampler can build. See docs/evergreen.md.
MODEL_VARIANTS: tuple[str, ...] = ("dalec2", "evergreen")

#: The structure every locked decision is stated against.
DEFAULT_MODEL_VARIANT: str = "dalec2"


@dataclass(frozen=True)
class DalecModel:
    """The assembled model, with the pieces it was built from kept accessible.

    Attributes
    ----------
    model
        The PyMC model.
    priors, graph, likelihood
        The three components, so a caller can inspect any intermediate quantity
        without rebuilding.
    convention
        LAI convention this model was built on. Carried because a GPP result is
        not interpretable without it.
    model_variant
        Model structure, one of :data:`MODEL_VARIANTS`. For ``"evergreen"``,
        ``priors`` and ``graph`` are the evergreen counterparts, with the same
        fields.
    """

    model: pm.Model
    priors: DalecPriors
    graph: DalecGraph
    likelihood: GaussianLikelihood
    convention: str
    model_variant: str = DEFAULT_MODEL_VARIANT


def build_model(
    *,
    site_data: SiteData,
    latitude_deg: float,
    coefficients: AcmCoefficients,
    frost_threshold_degc: float,
    convention: str = DEFAULT_LAI_CONVENTION,
    track_fluxes: bool = False,
    model_variant: str = DEFAULT_MODEL_VARIANT,
) -> DalecModel:
    """Assemble the full model over one driver block.

    Parameters
    ----------
    site_data
        Calibration block: drivers, observations, RANDUNC and the mask.
    latitude_deg, coefficients, frost_threshold_degc
        Site constants; none is sampled.
    convention
        LAI convention, ``"hemisurface"`` or ``"projected"``.
    track_fluxes
        Record GPP and the modelled NEE as deterministics. Off by default: it
        stores one array per draw per flux, which is large and is not needed for
        convergence diagnostics.
    model_variant
        ``"dalec2"`` or ``"evergreen"``. Only the priors and the forward graph
        change; the likelihood and the recorded deterministics are the same.
    """
    import pymc as pm

    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown model_variant {model_variant!r}; expected one of {MODEL_VARIANTS}"
        )
    if model_variant == "evergreen":
        from dalec.model_evergreen import (
            EVERGREEN_FOLIAGE_INDEX,
            build_evergreen_graph,
            build_evergreen_priors,
        )

        prior_builder, graph_builder = build_evergreen_priors, build_evergreen_graph
        foliage_index = EVERGREEN_FOLIAGE_INDEX
    else:
        prior_builder, graph_builder = build_priors, build_forward_graph
        foliage_index = POOL_NAMES.index("c_fol")

    with pm.Model() as model:
        priors = prior_builder(t_air=site_data.t_air, convention=convention)
        graph = graph_builder(
            parameters=priors.parameters,
            doy=site_data.doy.astype(float),
            t_air=site_data.t_air,
            t_max=site_data.t_max,
            t_min=site_data.t_min,
            sw_in=site_data.sw_in,
            co2=site_data.co2,
            latitude_deg=latitude_deg,
            coefficients=coefficients,
            frost_threshold_degc=frost_threshold_degc,
        )
        likelihood = build_likelihood(predicted=graph.nee, site_data=site_data)
        if track_fluxes:
            pm.Deterministic("nee_modelled", graph.nee)
            pm.Deterministic("gpp_modelled", graph.gpp)
        # Mean foliar carbon is one scalar per draw and is the quantity the
        # diagnostics phase predicted chains would disagree on, so it is
        # recorded unconditionally.
        pm.Deterministic("c_fol_mean", graph.pools[:, foliage_index].mean())
        pm.Deterministic("gpp_annual", graph.gpp.mean() * 365.25)
        pm.Deterministic("nee_annual", graph.nee.mean() * 365.25)

    return DalecModel(
        model=model,
        priors=priors,
        graph=graph,
        likelihood=likelihood,
        convention=convention,
        model_variant=model_variant,
    )


def model_from_config(
    config: dict[str, Any],
    site_data: SiteData,
    *,
    convention: str = DEFAULT_LAI_CONVENTION,
    track_fluxes: bool = False,
    model_variant: str = DEFAULT_MODEL_VARIANT,
) -> DalecModel:
    """Build the model from a loaded configuration and a prepared block."""
    from dalec.acm import acm_from_config

    acm = acm_from_config(config)
    if not isinstance(acm, AcmModel):  # pragma: no cover - defensive
        raise TypeError("acm_from_config did not return an AcmModel")
    return build_model(
        site_data=site_data,
        latitude_deg=acm.latitude_deg,
        coefficients=acm.coefficients,
        frost_threshold_degc=acm.frost_threshold_degc,
        convention=convention,
        track_fluxes=track_fluxes,
        model_variant=model_variant,
    )


def sample(
    dalec_model: DalecModel,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    cores: int | None = None,
    **kwargs: Any,
) -> az.InferenceData:
    """Run NUTS on an assembled model.

    Everything NUTS-specific is left at PyMC's defaults unless a caller
    overrides it through ``kwargs``: a first run should measure the problem, not
    a tuning choice. ``idata_kwargs`` requests the sampler statistics, which are
    what the divergence, tree depth and step size reporting needs.
    """
    import pymc as pm

    from dalec.compute import ensure_configured

    ensure_configured()
    options: dict[str, Any] = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "random_seed": seed,
        "progressbar": False,
        "idata_kwargs": {"log_likelihood": False},
    }
    if cores is not None:
        options["cores"] = cores
    options.update(kwargs)

    with dalec_model.model:
        return pm.sample(**options)


def initial_point_is_finite(dalec_model: DalecModel) -> tuple[bool, float]:
    """Check the model has a finite logp at its own initial point.

    Cheap, and worth doing before a long run: a non-finite starting logp means
    the sampler will fail after paying the compile cost, and the usual cause is
    a prior draw the forward model cannot integrate.
    """
    model = dalec_model.model
    point = model.initial_point()
    value = float(model.compile_logp(sum=True)(point))
    return bool(np.isfinite(value)), value
