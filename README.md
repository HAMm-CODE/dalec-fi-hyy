# Bayesian data assimilation of forest carbon dynamics at FI-Hyy

Master's thesis project. A reduced DALEC2 forest carbon model is calibrated
against daily eddy-covariance net ecosystem exchange from the Hyytiälä boreal
forest site in Finland (FLUXNET2015 site code `FI-Hyy`). Inference is Bayesian,
using PyMC with the NUTS sampler; the forward model is written in PyTensor so
that NUTS can differentiate through the whole time series.

## Research questions

1. How well do NEE observations constrain the model parameters?
2. How does parameter uncertainty propagate into the carbon pools and fluxes?
3. What are the limits of parameter identifiability and equifinality when NEE is
   the only observation?

Diagnostics and figures are deliverables in their own right, not afterthoughts.

## Design decisions (locked)

| | |
|---|---|
| Assimilation target | `NEE_VUT_REF` only |
| Data product | FLUXNET2015 **FULLSET** daily (DD) — SUBSET omits the uncertainty columns |
| Drivers | `TA_F`, `TA_F_DAY`, `TA_F_NIGHT`, `SW_IN_F`, `CO2_F_MDS`, day of year |
| Likelihood | Gaussian, per-day sd from `NEE_VUT_REF_RANDUNC` |
| Priors | Bounded uniform over published DALEC2 ranges |
| Allocation fractions | Dirichlet simplex, not independent uniforms |
| Constraint handling | Reparameterisation — no EDC accept/reject |
| Sampler | NUTS |
| Design | Contiguous calibration block, forward run over a held-out contiguous block (Richardson et al. 2010) |

`TA_F_DAY` and `TA_F_NIGHT` substitute for the daily maximum and minimum air
temperature the photosynthesis routine expects.

Explicitly out of scope: VPD as a driver; any seasonal, winter-only,
nighttime-only or daytime-only filtering of the assimilation target; EDC-style
accept/reject constraints; alternative observation error distributions;
assimilating the partitioned GPP/RECO products; any sampler other than NUTS.
The partitioned products are loaded, but only for posterior **consistency**
plots — they are model output, not observations, and never enter the likelihood.

## Units

Enforced throughout, and stated in every docstring. Unit errors are the most
likely silent bug in this project.

| Quantity | Unit |
|---|---|
| Carbon pools | g C m⁻² |
| Carbon fluxes | g C m⁻² d⁻¹ |
| Temperature | °C |
| Radiation | MJ m⁻² d⁻¹ (daily total) |
| CO₂ | µmol mol⁻¹ (ppm) |

FLUXNET2015 **daily** files already report NEE, GPP and RECO in g C m⁻² d⁻¹, so
no flux conversion is applied. The one conversion on load is radiation:
`SW_IN_F` is a daily *mean* in W m⁻², and the photosynthesis routine needs a
daily *total*, so it is multiplied by **0.0864** (= 86400 s ÷ 10⁶ J MJ⁻¹).

## Installation

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux
pip install -e ".[dev]"
```

`pyproject.toml` rather than `requirements.txt`: it makes `src/dalec` an
installable package, so scripts, tests and Slurm jobs all import `dalec`
identically without `sys.path` manipulation, and it keeps the dependency
pins, pytest, ruff and mypy configuration in one file. The scripts also carry a
small `sys.path` fallback so they run from a bare checkout.

## Data

Place the FLUXNET2015 FULLSET daily csv in `data/raw/`:

```
data/raw/FLX_FI-Hyy_FLUXNET2015_FULLSET_DD_1996-2014_1-3.csv
```

It is gitignored — the FLUXNET data policy governs redistribution, and
everything downstream is reproducible from it. Adjust `paths.fluxnet_file` in
`config/default.yaml` if your filename differs.

## Usage

```bash
# Per-year coverage table -- decides the calibration / evaluation split
python scripts/00_qc_coverage.py --csv results/qc_coverage.csv

# ...then fill years.calibration and years.evaluation in config/default.yaml

# Raw csv -> processed NetCDF arrays
python scripts/01_prepare_data.py
```

Scripts `02`–`05` arrive with their build phases.

Days failing the QC threshold (`data.qc_threshold`, default 0.75) are **masked
out of the likelihood but kept in the time series** — the forward model has to
integrate carbon pools continuously through them. Drivers, by contrast, must be
gap-free; `build_site_data` raises rather than interpolating, because a single
NaN driver silently contaminates every pool for the rest of the run.

## Configuration

All paths, seeds, year blocks, screening thresholds, fixed parameters and
sampler settings live in `config/default.yaml`. Nothing in `src/dalec`
hard-codes any of them, and no parameter name appears outside the
`dalec.parameters` registry.

Everything stochastic derives its seed from the single master `seed` and records
it in the output. Sampler output is written as NetCDF through ArviZ, never
pickled.

## Forward model

Six carbon pools — labile, foliage, fine root, wood, litter, soil organic
matter — stepped daily by Bloom & Williams (2015) equations A1–A6. The update is
**simultaneous**: every right-hand side reads the time-`t` state, and the six
pools advance together. A sequential implementation that overwrote `C_lab`
before evaluating the `C_fol` equation would give different, wrong numbers.

Summing A1–A6 the internal transfers cancel pairwise, which makes carbon
conservation an exact identity rather than an approximation:

```
ΔC_total = (1 − f_auto)·GPP − (θ_lit·C_lit + θ_som·C_som)·e^{ΘT}
         = GPP − Reco  =  −NEE
```

`θ_min` moves carbon from litter to soil and is therefore a *transfer*; only
`θ_lit` and `θ_som` appear in respiration. `DalecOutput.carbon_imbalance`
measures the residual, which sits at floating-point noise (~1e-11 g C m⁻² d⁻¹
against pools of ~2×10⁴ g C m⁻²).

### Allocation closure

The five allocation fractions are closed by construction: `f_auto` is sampled on
its own range and `(1 − f_auto)` is split by a 4-simplex ordered
`(f_lab, f_fol, f_roo, f_woo)`. **That order is load-bearing** — permuting it
swaps carbon between pools without breaking conservation, so no test can catch
it for you.

Bloom & Williams Table 1 footnote 1 gives `f_woo = 1 − f_auto − f_fol − f_lab`,
omitting `f_roo`. That is an error in the paper — identical in the preprint and
the published version — and the corrected five-term closure is used here.

The Dirichlet support is a *superset* of the published marginal ranges for
`f_lab`…`f_woo` (0.01–0.5): under the simplex an individual fraction may reach
`1 − f_auto`, up to 0.7. Enforcing the published marginals would require a
truncated Dirichlet, which reintroduces exactly the rejection wall the
reparameterisation exists to remove. The registry keeps the tabulated bounds
(Morris screening needs a range to perturb over) and flags those four as
`simplex=True` so `priors.py` cannot build a `Uniform` from them.

### Phenology

Published equations A7 and A8, with `s = 365.25/π`. The leaf-fall phase offset
`ψ_f = ψ·cr_fall/√2` comes from solving A9 numerically — the root is negative,
unique and monotone in `c_lf`, so a fixed negative bracket suffices. Because
`c_lf` and `cr_fall` are fixed at this site, `ψ_f` is a cached startup constant
and Phase 6 can bake it into the PyTensor graph. Note `c_lf` is the **annual
leaf fall fraction** (1/leaf-lifespan-in-years, range 1/8–1), not a duration.

> **⚠ Open question — `d_fall` is currently inert.** The A8 sine argument is
> transcribed as `doy − cr_fall + ψ_f`, so `d_fall` is accepted and carries a
> prior range but has no effect: sweeping it across 1–365 produces an identical
> leaf-fall pulse, peaking at day 50 regardless. The A7 counterpart anchors on
> `d_onset`, so the symmetric form would be `doy − d_fall + ψ_f`, which makes the
> pulse track `d_fall + 10`. Implemented as transcribed rather than silently
> corrected; `test_d_fall_is_inert_as_transcribed` pins the behaviour. If the
> anchor is confirmed to be `d_fall`, it is a one-token change in
> [model_numpy.py](src/dalec/model_numpy.py).

## Tests

```bash
pytest
```

Beyond the per-module tests, these invariants are enforced:

- carbon is conserved each timestep: inputs − outputs = change in total pool size ✅
- allocation fractions sum to one for any Dirichlet draw ✅
- pools never go negative over a long run ✅
- unit conversions round-trip ✅
- the NumPy and PyTensor forward models agree to floating-point tolerance — Phase 6

## Build status

| Phase | Contents | Status |
|---|---|---|
| 1 | Skeleton, config, `data_io`, QC coverage script | **done** |
| 2 | `model_numpy.py` — DALEC2 A1–A6 plus A7/A8/A9 phenology | **done** |
| 3 | `acm.py` — photosynthesis | blocked: ACM empirical coefficients |
| 4 | `parameters.py` registry and prior ranges | **done** |
| 4 | `priors.py` — PyMC prior construction | after Phase 3 |
| 5 | Morris screening | after Phase 4 |
| 6 | `model.py` (PyTensor), `inference.py` | after Phase 5 |
| 7 | Synthetic twin | after Phase 6 |
| 8 | Real-data calibration, diagnostics | blocked: chosen year blocks |
| 9 | Slurm jobs (Tampere, CSC) | after Phase 8 |

Unbuilt modules raise `NotImplementedError` rather than returning
plausible-looking numbers. One component of the forward model is still injected
as a callable defaulting to a stub that raises:

- **`F_gpp`** — the ACM photosynthesis routine, Phase 3, awaiting its full
  specification: the ten empirical coefficients, the equations, the solve
  order, the `D_T` definition, the frost cutoff and the numerical guards.

### GPP magnitude gate

A hard pre-calibration check lives in
[diagnostics.py](src/dalec/diagnostics.py). Once ACM is wired in,
`gpp_magnitude_gate()` sweeps `ceff` across its prior range and compares annual
modelled GPP against `GPP_NT_VUT_REF` and `GPP_DT_VUT_REF`. **If no `ceff`
brings the ratio within a factor of 1.5, calibration must not proceed** — a
forward model several-fold wrong on total GPP still converges and still yields
a posterior, and that posterior is meaningless.

Ratios are computed on **matched days only**, so a gappy product cannot inflate
them. The partitioned products remain a magnitude sanity check, never
validation and never assimilated.

Known ACM problems already measured, recorded in [acm.py](src/dalec/acm.py):
annual GPP roughly 5× too high; near-total temperature insensitivity at low
irradiance (0.5% across a 40 °C swing, because the light limitation drops `T`
out entirely); and a `ceff` prior of 10–100 that barely overlaps the original
`a1·N` value of ~5.7.

Everything else is complete. Carbon conservation holds for *any* value `F_gpp`
returns, so A1–A8 are fully testable with a double in its place.

## References

- Bloom, A. A. and Williams, M. (2015). Constraining ecosystem carbon dynamics
  in a data-limited world. *Biogeosciences* 12, 1299–1315.
- Williams, M. et al. (1997). Predicting gross primary productivity in
  terrestrial ecosystems. *Ecological Applications* 7(3), 882–894.
- Richardson, A. D. et al. (2010). Estimating parameters of a forest ecosystem C
  model with measurements of stocks and fluxes as joint constraints. *Oecologia*
  164, 25–40.
- Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux
  processing pipeline. *Scientific Data* 7, 225.
