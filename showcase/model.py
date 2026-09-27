"""DALEC2 written as a PyMC model.

`build_model` is the whole calibration model: the priors, the six carbon pools
stepped day by day, NEE, and the likelihood. "B&W" is Bloom & Williams (2015),
Biogeosciences 12, Table 1. "§" is a section of DECISIONS.md.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from dalec.acm import chuter_day_length_factor, frost_mask
from dalec.compute import ensure_configured, resolved_linker_class
from dalec.config import resolve_path
from dalec.data_io import load_site_data
from dalec.model import _acm_gpp, _phenology_day, phenology_constants
from dalec.parameters import (BELOWGROUND_LITTERFALL_G_C_M2, DAYS_PER_YEAR,
                              NEEDLE_LITTERFALL_G_C_M2, TREE_CARBON_STOCK_G_C_M2,
                              allocation_concentration)
from dalec.priors import prior_sources


def load_data(config, first_year, last_year):
    """Drivers, observed NEE, RANDUNC and nee_mask for whole calendar years."""
    # Data loading and QC filtering: the project loader (see src/dalec/data_io.py)
    paths = config["paths"]
    return load_site_data(resolve_path(paths["fluxnet_file"]),
                          start_year=first_year, end_year=last_year,
                          qc_threshold=config["data"]["qc_threshold"],
                          site_code=config["site"]["code"],
                          extremes_file=resolve_path(paths["extremes_file"]))


def build_model(data, acm, convention, nee_observed=None):
    """Priors -> DALEC2 -> NEE -> likelihood. Returns the model and the daily NEE.

    nee_observed replaces the measured NEE on the nee_mask days (the synthetic twin).
    """
    bounds = prior_sources(convention)  # the U(a, b) ranges, read from src/dalec/parameters.py

    def uniform(name):
        return pm.Uniform(name, *bounds[name])

    with pm.Model() as model:
        # ---- Priors ----------------------------------------------------------
        f_auto = uniform("f_auto")              # U(0.3, 0.7)             B&W
        theta_roo = uniform("theta_roo")        # U(1e-4, 1e-2) /day      B&W
        theta_woo = uniform("theta_woo")        # U(2.5e-5, 1e-3) /day    B&W
        theta_min = uniform("theta_min")        # U(1e-5, 1e-2) /day      B&W
        temperature_exponent = uniform("temperature_exponent")  # U(0.018, 0.08) /degC   B&W
        d_onset = uniform("d_onset")            # U(1, 365) day of year   B&W
        cr_onset = uniform("cr_onset")          # U(10, 100) days         B&W
        uniform("d_fall")                       # U(1, 365) day of year   B&W; inert, §2
        cr_fall = uniform("cr_fall")            # U(20, 150) days         B&W
        c_lf = uniform("c_lf")                  # U(0.2, 0.333)   1 / needle life of 3-5 yr, §8
        # lma: needle litterfall (Ilvesniemi 2009) x needle life / site LAI (Kolari 2010), §8.
        # U(116, 192) g C m-2 on the hemisurface LAI convention, U(148, 247) on projected, §10
        lma = uniform("lma")
        ceff = uniform("ceff")                  # U(5, 20)   Fox et al. 2009 (REFLEX) Table 4, §9
        rh_annual = uniform("rh_annual")        # U(290, 370) g C m-2 yr-1   Ilvesniemi 2009, §7
        f_som = uniform("f_som")                # U(0.5, 0.9)   judgement, §6
        theta_lit = uniform("theta_lit")        # U(5.48e-4, 2.74e-3) /day   Tuomi et al. 2009, §7
        theta_som = uniform("theta_som")        # U(4.49e-5, 1.14e-4) /day   Ilvesniemi 2009, §7

        # Allocation of non-respired GPP to (labile, foliage, roots, wood): Dirichlet with
        # a = (13.06, 13.06, 15.26, 44.27), total 85.65, from measured site fluxes, §8, §11
        weights = pm.Dirichlet("allocation_weights", a=allocation_concentration())
        available = 1.0 - f_auto
        f_lab = pm.Deterministic("f_lab", available * weights[0])
        f_fol = pm.Deterministic("f_fol", available * weights[1])
        f_roo = pm.Deterministic("f_roo", available * weights[2])
        f_woo = pm.Deterministic("f_woo", available * weights[3])

        # Heterotrophic respiration at 0 degC from the sampled annual total, using this
        # draw's own temperature response: rh_annual = rh_ref * mean(exp(Theta * T)) * 365.25, §7
        multiplier = pt.mean(pt.exp(temperature_exponent * data.t_air))
        rh_ref = pm.Deterministic("rh_ref", rh_annual / (multiplier * DAYS_PER_YEAR))
        pm.Deterministic("decomposition_multiplier", multiplier)

        # Initial pools, all derived from measured site stocks and fluxes, §7, §9
        litterfall = NEEDLE_LITTERFALL_G_C_M2[1]          # 154 g C m-2 yr-1, Ilvesniemi 2009
        root_litterfall = BELOWGROUND_LITTERFALL_G_C_M2   # 90 g C m-2 yr-1, Ilvesniemi 2009
        tree_carbon = TREE_CARBON_STOCK_G_C_M2            # 6800 g C m-2, Ilvesniemi 2009
        c_lit_0 = pm.Deterministic("c_lit_0", (1.0 - f_som) * rh_ref / theta_lit)
        c_som_0 = pm.Deterministic("c_som_0", f_som * rh_ref / theta_som)
        c_fol_0 = pm.Deterministic("c_fol_0", litterfall / c_lf)
        c_lab_0 = pm.Deterministic("c_lab_0", litterfall * weights[0] / (weights[0] + weights[1]))
        c_roo_0 = pm.Deterministic("c_roo_0", root_litterfall / (theta_roo * DAYS_PER_YEAR))
        c_woo_0 = pm.Deterministic("c_woo_0", tree_carbon - c_lab_0 - c_fol_0 - c_roo_0)

        # ---- DALEC2: the six carbon pools, stepped through every day ---------
        # Leaf onset and leaf fall timing (see src/dalec/model.py)
        phenology = phenology_constants(d_onset, cr_onset, cr_fall, c_lf)
        gpp, ra, rh, c_fol = dalec2(
            data,
            acm,
            initial_pools=[c_lab_0, c_fol_0, c_roo_0, c_woo_0, c_lit_0, c_som_0],
            parameters=[
                *phenology, cr_onset, cr_fall, lma, ceff, temperature_exponent,
                f_lab, f_fol, f_roo, f_woo, f_auto,
                theta_roo, theta_woo, theta_lit, theta_som, theta_min,
            ],
        )

        # ---- NEE = Ra + Rh - GPP (positive = carbon source, as in FLUXNET) ---
        nee = ra + rh - gpp
        pm.Deterministic("c_fol_mean", c_fol.mean())
        pm.Deterministic("gpp_annual", gpp.mean() * DAYS_PER_YEAR)
        pm.Deterministic("nee_annual", nee.mean() * DAYS_PER_YEAR)

        # ---- Likelihood: Gaussian, sd = NEE_VUT_REF_RANDUNC, nee_mask days only, §1
        days = np.flatnonzero(data.nee_mask)
        observed = data.nee_obs[days] if nee_observed is None else nee_observed
        pm.Normal("nee_obs", mu=nee[days], sigma=data.nee_unc[days], observed=observed)
    return model, nee


def dalec2(data, acm, initial_pools, parameters):
    """Step DALEC2 through every day with pytensor.scan. Returns GPP, Ra, Rh and C_fol."""
    coef = acm.coefficients
    # ACM inputs that depend only on the weather, computed once (as in src/dalec/model.py)
    t_range = np.maximum(data.t_max - data.t_min, 0.0)
    g_c = abs(coef.psi_mpa) ** coef.a10 / (0.5 * t_range + coef.a6 * coef.r_tot)
    day_of_year = data.doy.astype(float)
    day_length = chuter_day_length_factor(day_of_year, acm.latitude_deg, coef)
    unfrozen = (~frost_mask(data.t_max, data.t_min, acm.frost_threshold_degc)).astype(float)
    weather = [day_of_year, data.t_air, data.t_max, data.sw_in, data.co2,
               g_c, day_length, unfrozen]

    def step(
        doy, t_air, t_max, sw_in, co2, g_c, day_length, unfrozen,  # today's weather
        c_lab, c_fol, c_roo, c_woo, c_lit, c_som,                  # today's pools
        onset_amplitude, onset_phase, fall_amplitude, psi_f,
        cr_onset, cr_fall, lma, ceff, temperature_exponent,
        f_lab, f_fol, f_roo, f_woo, f_auto,
        theta_roo, theta_woo, theta_lit, theta_som, theta_min,
    ):
        # Share of labile carbon flushed into leaves, and of leaves shed, today (A7, A8)
        onset, fall = _phenology_day(
            doy, cr_onset, cr_fall, onset_amplitude, onset_phase, fall_amplitude, psi_f
        )

        # GPP from the ACM canopy model (see src/dalec/model.py)
        p_d, e_0 = _acm_gpp(c_fol, lma, ceff, t_max, co2, g_c, day_length, coef)
        light = e_0 * sw_in
        denominator = light + p_d
        positive = pt.gt(denominator, 0.0)
        p_i = pt.switch(positive, light * p_d / pt.switch(positive, denominator, 1.0), 0.0)
        gpp = unfrozen * p_i * day_length

        # Decomposition speeds up with temperature: exp(Theta * T)
        temperature_rate = pt.exp(temperature_exponent * t_air)

        # The six carbon pools, all updated from today's state (A1-A6)
        c_lab_next = (1.0 - onset) * c_lab + f_lab * gpp
        c_fol_next = (1.0 - fall) * c_fol + onset * c_lab + f_fol * gpp
        c_roo_next = (1.0 - theta_roo) * c_roo + f_roo * gpp
        c_woo_next = (1.0 - theta_woo) * c_woo + f_woo * gpp
        c_lit_next = ((1.0 - (theta_lit + theta_min) * temperature_rate) * c_lit
                      + theta_roo * c_roo + fall * c_fol)
        c_som_next = ((1.0 - theta_som * temperature_rate) * c_som
                      + theta_woo * c_woo + theta_min * temperature_rate * c_lit)

        # Autotrophic respiration (A10); heterotrophic, from litter and soil (A5, A6)
        ra = f_auto * gpp
        rh = (theta_lit * c_lit + theta_som * c_som) * temperature_rate
        return c_lab_next, c_fol_next, c_roo_next, c_woo_next, c_lit_next, c_som_next, gpp, ra, rh

    outputs = pytensor.scan(
        fn=step,
        sequences=[pt.as_tensor_variable(np.asarray(x, dtype=float)) for x in weather],
        outputs_info=[*initial_pools, None, None, None],
        non_sequences=parameters,
        strict=True,
        name="dalec2",
        return_updates=False,
    )
    *pools, gpp, ra, rh = outputs
    return gpp, ra, rh, pools[1]  # pools[1] is foliar carbon, C_fol


def sample(model, config):
    """NUTS with the settings of the real calibration run. Returns the posterior."""
    ensure_configured()
    logp = float(model.compile_logp(sum=True)(model.initial_point()))
    print(f"linker {resolved_linker_class()}, initial logp {logp:,.1f}")
    if not np.isfinite(logp):
        raise SystemExit("initial logp is not finite; sampling would fail")
    settings = config["sampler"]
    with model:
        return pm.sample(draws=settings["draws"], tune=settings["tune"],
                         chains=settings["chains"], cores=settings["cores"],
                         random_seed=config["seed"], progressbar=False,
                         idata_kwargs={"log_likelihood": False})
