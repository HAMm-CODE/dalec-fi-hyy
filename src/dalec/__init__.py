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
"""

from __future__ import annotations

__version__ = "0.1.0"

from dalec import config, data_io, model_numpy, parameters
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
    "config",
    "coverage_table",
    "dalec2_phenology",
    "data_io",
    "load_config",
    "load_fluxnet_dd",
    "load_site_data",
    "model_numpy",
    "parameters",
    "prior_bounds",
    "run_dalec2",
]
