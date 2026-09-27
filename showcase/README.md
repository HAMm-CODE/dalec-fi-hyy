# DALEC2 in PyMC: a readable showcase

Results shown were produced by the full pipeline in `src/dalec/`. This folder is a readable summary of that pipeline.

## The model

DALEC2 is a daily forest carbon model with six carbon pools: labile carbon, foliage, fine roots,
wood, litter and soil organic matter (Bloom & Williams 2015). Photosynthesis (GPP) brings carbon
in, allocation, turnover and decomposition move it between the pools, and respiration returns it
to the air. Here it is calibrated against daily net ecosystem exchange (NEE) measured at
Hyytiälä, Finland (FLUXNET site FI-Hyy).

- **PyMC** is a Python library for writing a Bayesian model (priors plus a likelihood) as ordinary code.
- **NUTS** (the No-U-Turn Sampler) is the algorithm that draws samples from the posterior, using the gradient of the whole model to move efficiently.

## Files

| file | what it does |
|---|---|
| `model.py` | the model: priors, the daily step of the six pools, NEE, likelihood |
| `run_calibration.py` | builds the model, samples it with NUTS, saves the posterior (`.nc`) |
| `run_twin.py` | the synthetic twin (below) |
| `make_results.py` | reads saved `.nc` files, prints convergence checks, makes figures and tables; samples nothing |
| `config.yaml` | paths, year windows, LAI convention, sampler settings |
| `jobs/` | Slurm scripts for Roihu (CSC) |

GPP (the ACM canopy model), the timing of leaf onset and leaf fall, and data loading are imported
from `src/dalec/` rather than written out here.

## The synthetic twin

The twin fits the model to fake data that the model itself produced from parameter values we
chose, so the right answer is known. Here the true values are one random draw from the priors,
and the fake NEE is the model output at those values plus noise the size of the real measurement
error, on the same days as the real data. It comes first because a method that cannot recover
known values from perfect-model data cannot be trusted on the real forest, and any gap between
twin and real results then points to the model rather than to the method.

## How to run

From the repository root. Needs `data/raw/FLX_FI-Hyy_FLUXNET2015_FULLSET_DD_1996-2014_1-4.csv`
and `data/processed/fi_hyy_tminmax.csv` (made by `scripts/01b_derive_tminmax.py`); neither is in git.

```
python showcase/run_twin.py                                  # 1. synthetic twin
python showcase/run_calibration.py                           # 2. real data, hemisurface LAI
python showcase/run_calibration.py --convention projected    #    the sensitivity run
python showcase/make_results.py <posterior.nc> [<twin.nc>]   # 3. figures and tables
```

On Roihu, from `/scratch/project_2020170/dalec-fi-hyy/`:

```
sbatch showcase/jobs/twin.sh
sbatch showcase/jobs/calibrate.sh              # add "projected" for the sensitivity run
sbatch showcase/jobs/results.sh                # optional: <posterior.nc> <twin.nc>
```

Everything is written to `showcase/outputs/` (ignored by git): posteriors, `figures/`, tables and job logs.

## Figures

1. `fig1_nee_timeseries`: observed and modelled daily NEE, 1997–2014, calibration and held-out windows shaded. 2014 has no observed points because it has no RANDUNC values.
2. `fig2_prior_posterior`: prior against posterior for the main parameters.
3. `fig3_gpp_reco`: posterior GPP, ecosystem respiration and foliar carbon. The FLUXNET partitioned products are drawn for comparison only; they are never fitted.
4. `fig4_rank`: rank plots, a convergence check for four key parameters.
5. `fig5_twin_recovery` and `twin_recovery.csv`: the twin's posteriors against the true values.

## Decisions

Locked; the reasoning is in [DECISIONS.md](../DECISIONS.md).

- Site FI-Hyy, FLUXNET2015 FULLSET daily data; `NEE_VUT_REF` is the only quantity fitted (§1).
- Calibration window 1997–2010, held-out evaluation window 2011–2014; 1996 is excluded because its CO₂ driver has a gap (§1).
- Likelihood: Gaussian, with `NEE_VUT_REF_RANDUNC` as the standard deviation, on the `nee_mask` days only (§1).
- Allocation fractions: Dirichlet, total concentration 85.65, from measured site fluxes (§8, §11).
- Heterotrophic respiration: `rh_annual` and `f_som` are sampled; `rh_ref` and the starting litter and soil pools are derived from them (§7).
- `ceff` prior U(5, 20), Fox et al. (2009) (§9).
- No EDC constraints and no VPD (§1).
- LAI convention: hemisurface by default; projected is run as a sensitivity check (§10).
- `d_fall` is kept exactly as in the full pipeline; it is inert because Eq. A8 is used as printed (§2).
- Of the four corrections to Bloom & Williams (2015), only the five-term allocation closure changes an equation; the other three are either left as printed (`d_fall`) or sit in the paper's prose (§2).
- NUTS: 4 chains, 1000 tuning and 1000 kept draws per chain, seed 20260809; everything else at PyMC's defaults, as in the full pipeline (`jobs/calibrate.sbatch`).
