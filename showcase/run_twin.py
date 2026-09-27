#!/usr/bin/env python
"""Synthetic twin: fit the model to fake NEE made from known parameter values.

1. The true values are one joint draw from the priors (seed from config.yaml).
2. Fake NEE = the model's NEE at those values + Gaussian noise with RANDUNC as
   the standard deviation, on the nee_mask days only.
3. The same model, drivers, mask and NUTS settings as the real calibration are
   fitted to the fake NEE. The posterior is saved with the true values beside it.

Usage, from the repository root:
    python showcase/run_twin.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pymc as pm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dalec.acm import acm_from_config  # noqa: E402
from dalec.compute import ensure_configured  # noqa: E402
from dalec.config import load_config, resolve_path  # noqa: E402
from model import build_model, load_data, sample  # noqa: E402

CONFIG = Path(__file__).with_name("config.yaml")


def main():
    config = load_config(CONFIG)
    convention = config["lai_convention"]
    first, last = config["years"]["calibration"]
    rng = np.random.default_rng(config["seed"])
    ensure_configured()

    data = load_data(config, first, last)
    acm = acm_from_config(config)

    # 1. The truth: one joint draw of every prior, what follows from it, and its NEE
    prior_model, nee = build_model(data, acm, convention)
    variables = prior_model.free_RVs + prior_model.deterministics
    *values, nee_true = pm.draw([*variables, nee], random_seed=rng)
    truth = {v.name: np.asarray(value).tolist() for v, value in zip(variables, values, strict=True)}
    print(f"true annual GPP {truth['gpp_annual']:.0f}, NEE {truth['nee_annual']:.0f} g C m-2 yr-1; "
          f"mean foliar carbon {truth['c_fol_mean']:.0f} g C m-2")

    # 2. Fake observations: true NEE + Gaussian noise (sd = RANDUNC), nee_mask days only
    days = np.flatnonzero(data.nee_mask)
    nee_fake = nee_true[days] + rng.normal(0.0, data.nee_unc[days])
    print(f"twin {first}-{last}: {data.n_days} days, {days.size} fake observations")

    # 3. Fit the same model to the fake NEE, exactly as the real calibration
    model, _ = build_model(data, acm, convention, nee_observed=nee_fake)
    idata = sample(model, config)

    idata.attrs.update(
        seed=config["seed"],
        convention=convention,
        calibration_years=[first, last],
        n_days=data.n_days,
        n_assimilated=int(days.size),
        truth="one joint draw from the priors",
    )
    out = resolve_path(config["paths"]["twin_posterior"])
    out.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(out))
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"twin posterior written to {out}, true values beside it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
