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

Importing this package pins PyTensor's execution backend and asserts that
PyTensor honoured the pin -- see ``dalec.compute`` for why that check is not
optional. It costs about 1.5 s of PyTensor import and couples the pure-NumPy
paths to Numba being present, both deliberately.
"""

from __future__ import annotations

__version__ = "0.1.0"

from dalec import compute, config, data_io, model_numpy, parameters
from dalec.compute import RESOLVED_LINKER, configure_pytensor
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
    "RESOLVED_LINKER",
    "DalecOutput",
    "DalecParameters",
    "SiteData",
    "__version__",
    "allocation_fractions",
    "compute",
    "config",
    "configure_pytensor",
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
