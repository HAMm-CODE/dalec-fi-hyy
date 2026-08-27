"""Bayesian data assimilation of DALEC2 forest carbon dynamics at FI-Hyy.

Reduced DALEC2 (six carbon pools) calibrated against daily eddy-covariance
``NEE_VUT_REF`` from the Hyytiala boreal forest site, using PyMC/NUTS with a
PyTensor forward model.

Unit conventions, enforced everywhere:

* carbon pools    g C m-2
* carbon fluxes   g C m-2 d-1
* temperature     degrees C
* radiation       MJ m-2 d-1
* CO2             umol mol-1

Only the modules that are implemented are re-exported here; the rest are added
as the build phases complete.

PyTensor's execution backend is pinned and verified by ``dalec.compute``, on
first graph compilation rather than here -- importing this package does not
import PyTensor, so the NumPy forward model, the data loader and the diagnostics
all work without it installed. Anything compiling a PyTensor function must go
through ``dalec.compute.compile_function``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from dalec import compute, config, data_io, model_numpy, parameters
from dalec.compute import compile_function, configure_pytensor, ensure_configured
from dalec.config import load_config
from dalec.data_io import SiteData, coverage_table, load_fluxnet_dd, load_site_data
from dalec.model_numpy import POOL_NAMES, DalecOutput, dalec2_phenology, run_dalec2
from dalec.parameters import (
    PARAMETER_REGISTRY,
    DalecParameters,
    allocation_fractions,
    prior_bounds,
)

__all__ = [
    "PARAMETER_REGISTRY",
    "POOL_NAMES",
    "DalecOutput",
    "DalecParameters",
    "SiteData",
    "__version__",
    "allocation_fractions",
    "compile_function",
    "compute",
    "config",
    "configure_pytensor",
    "coverage_table",
    "dalec2_phenology",
    "data_io",
    "ensure_configured",
    "load_config",
    "load_fluxnet_dd",
    "load_site_data",
    "model_numpy",
    "parameters",
    "prior_bounds",
    "run_dalec2",
]
