# Decisions, corrections and known problems

A durable record of everything that would be expensive to rediscover: the design
decisions that are settled, the four places where the published papers are
wrong, the constants that were resolved once, and the structural problems that
are recorded rather than solved.

Nothing here is new. It is collected from the module docstrings, code comments,
`config/default.yaml` and `README.md`, which remain the authoritative sources.
Where this file and the code disagree, the code is right and this file is stale.

---

## 1. Locked design decisions

These are settled. They are not options, and they should not be re-opened
without a deliberate decision recorded here.

| | |
|---|---|
| Assimilation target | `NEE_VUT_REF` only |
| Data product | FLUXNET2015 **FULLSET** daily (DD) — SUBSET omits `NEE_VUT_REF_RANDUNC` |
| Drivers | `TA_F`, `TA_F_DAY`, `TA_F_NIGHT`, `SW_IN_F`, `CO2_F_MDS`, day of year |
| Likelihood | Gaussian, per-day sd from `NEE_VUT_REF_RANDUNC` |
| Priors | Bounded uniform over published DALEC2 ranges |
| Allocation fractions | Dirichlet simplex, not independent uniforms |
| Constraint handling | Reparameterisation — no EDC accept/reject |
| Sampler | NUTS |
| Design | Contiguous calibration block, forward run over a held-out contiguous block (Richardson et al. 2010) |

**Explicitly out of scope.** VPD as a driver — `VPD_F` is present in the file and
is never read. Any seasonal, winter-only, nighttime-only or daytime-only
filtering of the assimilation target. EDC-style accept/reject constraints, which
break gradient-based sampling. Alternative observation error distributions.
Assimilating the partitioned GPP/RECO products. Any sampler other than NUTS.

The partitioned products are loaded, but only as a magnitude sanity check and
for posterior consistency plots. They are model output, not observations, and
never enter the likelihood.

**Year blocks.** Calibration 1997–2005 (3287 days, 2781 assimilable, 84.6%);
evaluation 2006–2009 (1461 days, 1390 assimilable, 95.1%). The blocks abut, so
the forward run integrates from the start of calibration to the end of
evaluation with no un-assimilated bridge year. The calibration block is nine
years rather than the four of Richardson et al.; this was a deliberate choice.
Excluded years and the reasons are recorded in `config/default.yaml`. Note 2014:
356 QC-passing days at mean QC 0.974 and **zero** `RANDUNC` values, so not one
day of it can enter the likelihood.

---

## 2. Corrections to the published papers

Four errors were found in Bloom & Williams (2015) Appendix A, each by internal
consistency against the paper's own equations. **Do not "fix" these back to the
published form.**

**1. Table 1 footnote 1 omits `f_roo` from the allocation closure.** The
footnote gives `f_woo = 1 - f_auto - f_fol - f_lab`. Equations A1–A4 show four
distinct allocation flows out of GPP, A10 adds autotrophic respiration as a
fifth, and `f_roo` carries its own non-zero prior range in the same table. The
error is identical in the preprint and the published version. The corrected
five-term closure is implemented:

```
f_auto + f_lab + f_fol + f_roo + f_woo = 1
```

Recorded in `parameters.allocation_fractions`.

**2. Eq. A8's sine argument prints `cr_fall` where `d_fall` belongs — NOT
corrected in code.** This one differs from the other three in status, and the
difference matters. The A8 argument is implemented **as transcribed**, `doy -
cr_fall + psi_f`, which leaves `d_fall` **inert**: it is accepted, documented and
carries a 1–365 prior, but sweeping it across that whole range produces an
identical leaf-fall pulse, peaking at day 50 regardless. The A7 counterpart
anchors on `d_onset`, so the symmetric form would be `doy - d_fall + psi_f`,
making the pulse track `d_fall + 10`.

Implemented as transcribed rather than silently corrected, with
`test_d_fall_is_inert_as_transcribed` pinning the behaviour so it cannot be
forgotten. If the anchor is confirmed to be `d_fall`, it is a one-token change in
`model_numpy.py`. **Consequence for the results: `d_fall` is a sampled parameter
that currently cannot affect the likelihood, so it will return a flat marginal
posterior — an artefact of the transcription, not a finding about
identifiability.** This must not be reported as evidence for RQ1 or RQ3.

**3. The derivation text writes `Phi_onset` where `Phi_fall` is meant.**

**4. The same sentence mixes `cr_onset` and `cr_fall`.**

Corrections 3 and 4 are errors in the paper's prose rather than in the
equations, and they do not reach the code: `psi` is obtained by solving A9
numerically rather than by following the derivation.

**Use the published 2015 version only.** The 2014 preprint differs. `phi_fall`'s
coefficient is `-log(1 - c_lf) / cr_fall`; the preprint's
`(log(c_lspan) - log(c_lspan - 1)) / cr_fall` is the same quantity under
`c_lf = 1 / c_lspan` and is not used. `cr_fall` range is 20–150 d, and the pools
`C_lab`, `C_fol`, `C_roo`, `C_lit` are 20–2000 g C m⁻².

---

## 3. Resolved constants

**Phenology.** `s = 365.25 / pi`, stated after Eq. A8.

`psi` solves Eq. A9 numerically at startup rather than through the paper's
sixth-order polynomial, which exists only because the authors sampled `c_lf` and
needed `psi` millions of times. Here `c_lf` is fixed, so `psi` is a single cached
startup constant and a numerical root is simpler and exact. The root is always
negative, unique and monotone increasing in `c_lf` over the whole prior range,
so a fixed negative bracket suffices; bracketing on the positive axis fails.
Reference values, reproduced by `parameters.solve_psi`:

```
c_lf = 0.125  ->  psi = -0.91483330
c_lf = 0.25   ->  psi = -0.64600950
c_lf = 0.333  ->  psi = -0.52742954
c_lf = 0.5    ->  psi = -0.35801684
```

`c_lf` is the **annual leaf fall fraction** — the reciprocal of leaf lifespan in
years, prior range 1/8 to 1 — not a duration in days. At exactly 1 the equation
is singular because `log(1 - c_lf)` diverges, so the upper endpoint is unusable.

The leaf-fall phase offset is `psi_f = psi * cr_fall / sqrt(2)`, a derived
constant rather than a free parameter. Both `c_lf` and `cr_fall` are fixed at
this evergreen site, so it is evaluated once and can be baked into the Phase 6
PyTensor graph.

**ACM coefficients** — from Chuter et al. (2015), *not* Williams et al. (1997).
See §5a for why. These are **site-calibrated, not universal**: Chuter publishes
two complete and materially different sets, and neither is boreal. The Loobos
evergreen set (52 N, Appendix A) is used here as the closest available match by
forest type, and that adoption is a stated limitation.

```
Loobos evergreen (used):     a2=0.0156  a3=4.22273  a4=208.868  a5=0.0453
                             a6=0.3783  a7=7.1929   a8=0.0111   a9=2.1001
                             a10=0.7897 psi=2.0     Rtot=1.0
                             p11=7.4    N=4.0       lma_ref=110.0

Oregon ponderosa (reference): a2=0.0142  a3=0.980   a4=217.9   a5=0.155
                              a6=2.653   a7=4.309   a8=0.060   a9=1.062
                              a10=0.0006 psi=0.8502 Rtot=1.0
                              p11=2.155  N=2.7      lma_ref=111.0
```

`ceff` replaces the `p11 * N` product, so foliar N is not implemented.

**Solve order**: conductance, internal CO₂, diffusion limit, quantum yield, light
limitation, day length. B4 must precede the light limitation, which consumes
`E_0`. The steady-state assumption folds the carboxylation rate into B2, so it is
never evaluated separately.

**Day length** comes from latitude and solar declination (Chuter B5, B6):

```
delta = -0.408 * cos(2*pi*doy / 365)
s     = 24 * arccos(-tan(lat) * tan(delta)) / pi        # hours
GPP   = p_I * (a2 * s + a5)
```

The arccos argument is clamped to [-1, 1]; inside the polar circles it genuinely
leaves that interval, and the clamp turns that into 24 h or 0 h rather than a
NaN. At FI-Hyy it stays within ±0.81 all year, so the clamp never fires here.

**The published declination formula runs about ten days late.** It carries no
phase offset, so it peaks at doy 182.5 where the June solstice falls near 172.
Implemented as published, pinned by a test. It shifts the modelled seasonal cycle
late; it does not affect the amplitude.

**Substitutions and conventions.** `L = C_fol / c_lma` (Eq. A12), from model
state rather than drivers. `Tmax` is the daily maximum, here `TA_F_DAY`;
B3 applies `exp(a8 * Tmax)` to the **maximum**, not to a mean. `Tr` is the
**full** daily temperature range, and B1 takes `0.5 * Tr` — so the half-range is
what finally divides, and getting that factor of two wrong is silent.
Decomposition in A5/A6 uses `TA_F`, a third and distinct temperature.

**The daily range is floored at zero, counted and warned.** `Tr` is a range and
cannot be negative, but the day/night means standing in for maximum and minimum
do invert: at FI-Hyy on **14.8%** of calibration days, driving the B1 denominator
negative on **5.0%** of them. The FULLSET daily product carries no `TA_F_MAX` or
`TA_F_MIN` — only day and night means — so the proxy is the only option
available. On floored days conductance takes its minimum rather than a measured
value. `AcmModel.range_floor_count` reports the tally.

**Units.** Pools g C m⁻², fluxes g C m⁻² d⁻¹, temperature °C, radiation
MJ m⁻² d⁻¹ (daily total), CO₂ µmol mol⁻¹, rate constants d⁻¹. FLUXNET daily
files already report fluxes in g C m⁻² d⁻¹, so the only conversion on load is
radiation: `SW_IN_F` is a daily mean in W m⁻² and is multiplied by **0.0864**
(86400 s ÷ 10⁶ J MJ⁻¹).

---

## 4. Structural properties that are load-bearing

**The pool update is simultaneous, not sequential.** Every right-hand side in
A1–A6 is evaluated at the time-`t` state and the six pools advance together.
Mutating `C_lab` before using it in the `C_fol` equation gives different and
wrong numbers.

**Carbon conservation is an exact identity, not an approximation.** Summing
A1–A6, the internal transfers cancel pairwise, leaving
`d(total C) = (1 - f_auto)*GPP - Rh = GPP - Reco = -NEE` every timestep with no
discretisation error. `theta_min` moves carbon from litter to soil and is
therefore a *transfer*, not a loss — only `theta_lit` and `theta_som` enter
respiration. Measured residual sits at floating-point noise.

**`ALLOCATION_WEIGHT_ORDER` is load-bearing and no test can protect it.**
Conservation is invariant under permuting the allocation order, so a transposed
simplex swaps carbon between pools while every conservation test still passes.
The order is `(f_lab, f_fol, f_roo, f_woo)`.

**The Dirichlet support is a superset of the published marginal ranges.** Under
the simplex an individual fraction may reach `1 - f_auto`, up to 0.7, against
published marginals of 0.01–0.5. Enforcing the marginals would need a truncated
Dirichlet, reintroducing exactly the rejection wall that reparameterisation
exists to remove. The registry keeps the tabulated bounds — Morris screening
needs a range to perturb over — and flags those four as `simplex=True` so
`priors.py` cannot build a `Uniform` from them.

**The frost cutoff is load-bearing, not cleanup.** No carbon is fixed below
−2.0 °C, following the precedent Williams et al. applied at their Oregon
coniferous sites. It reads only the temperature driver, never a sampled
parameter, so it is a fixed boolean mask over the time series — precomputed once,
gradients simply zero on masked days, no parameter-dependent branch in the
graph. It is applied *after* Eq. 9 so the arithmetic always runs on finite
values. Do not raise it towards zero and do not make it easy to disable; see
problem 3 below for why.

---

## 5a. The ACM source was wrong, and this is what fixed it

ACM was first implemented from **Williams et al. (1997) Table 2**, on the
strength of Bloom & Williams (2015) citing that paper for DALEC2's
photosynthesis. **DALEC does not use that form.** Chuter et al. (2015) write ACM
out as DALEC implements it, and it differs in three material ways. The 1997
implementation is retained in `acm.py`, renamed and marked, because the
comparison is thesis material — it is not a code path.

**1. The day-length term, and this one dominates.** Williams Eq. 9 applies
`d1 * |173 - doy| + d2`, a function of day of year alone *containing no
latitude*. DALEC computes true day length from latitude and declination. At
FI-Hyy the 1997 form overstates the factor severely, and worst in winter:

| doy | 15 | 60 | 173 | 300 | 350 |
|---|---:|---:|---:|---:|---:|
| Williams 1997 | 1.526 | 1.607 | 1.810 | 1.581 | 1.491 |
| DALEC @61.85N | 0.126 | 0.184 | 0.342 | 0.192 | 0.126 |
| overstatement | 12.1× | 8.7× | 5.3× | 8.3× | 11.8× |

A term with no latitude in it cannot know that Hyytiälä gets five hours of
daylight in December. That is the origin of the winter GPP floor, and it is why
the floor was temperature-independent: the 1997 factor barely varies at all.

**2. Canopy conductance carries no canopy height.** Chuter B1 divides by
`0.5*Tr + a6*Rtot` and raises `|psi|` to the power `a10`. Canopy height does not
appear in the DALEC form at all, so it is no longer a required site constant.

**3. Quantum yield is 6.5× larger and the temperature enters on Tmax.** Chuter B4
gives `E_0 = a7 * Cf² / (Cf² + a9 * lma²)` with `a7 = 7.19` against `c1 = 0.989`.

Differences 1 and 3 roughly cancel at mid-latitude, which is why the error stayed
invisible until this model was run at a boreal site.

**Consequences.** Problems 1, 2 and 3 as previously recorded — annual GPP several
times too high, near-total temperature insensitivity, no parameter escape route,
a summer/winter ratio capped near 8.2 — were **artefacts of the wrong ACM**, not
properties of DALEC. They are retained here only as the record of what the wrong
source produced. The frost cutoff was described as load-bearing on the strength
of those measurements; under the corrected form it no longer carries that weight,
and it is retained on its original published justification instead.

**4. The `ceff` prior is resolved and vindicated — do not change it.** `ceff`
stands in for `p11 * N`: Chuter's sets give `7.4 * 4.0 = 29.6` (Loobos evergreen)
and `2.155 * 2.7 = 5.82` (Oregon pine). The DALEC2 prior of 10–100 brackets the
evergreen value. An earlier measurement here suggested a workable range of 3–6,
apparently agreeing with the 1997 `a1 * N` value of 5.7; that agreement was an
artefact of the wrong ACM compensating for a light-response term inflated 5–12×
by the missing latitude. `parameters.py` is left untouched. Recorded because it is
exactly the kind of coincidence that looks like confirmation.

## 5. Known structural problems — recorded, not solved

These are thesis limitations. They live in the `acm.py` docstring, measured
against the corrected form.

**1. The direct temperature response is weak, and the rewrite did not fix it.**
Above the frost threshold at low irradiance, GPP moves from 3.455 to 3.384
g C m⁻² d⁻¹ between +20 °C and 0 °C — about 2% across 20 degrees. Temperature
reaches GPP only through `exp(a8 * Tmax)` with `a8 = 0.0111` and through the
daily range in the B1 denominator; the light limitation damps both. **The
seasonal cycle here is carried by day length and irradiance, not by
temperature.** What the rewrite fixed is the seasonal *amplitude*.

**2. The coefficient sets are site-calibrated and neither is boreal.** Adopting
Loobos is the closest available match, not a fitted choice.

**3. 59.3% of calibration days fall below ACM's 7 °C lower calibration bound**,
so the model extrapolates on most of the record. `ACM_CALIBRATION_BOUNDS`
currently holds only the temperature row.

**4. The day/night temperature proxy inverts on 14.8% of days**, floored and
counted — see §3.

**Scope: this site is outside the envelope DALEC2's authors defined.** Bloom &
Williams (2015) §2.5 selected sites with little expected water stress and no more
than **three months** of recorded below-freezing soil temperature, those criteria
reflecting DALEC2's capabilities — hydrological processes are not explicitly
represented. FI-Hyy has roughly **four to six months** of soil frost. Applying
DALEC2 here is a deliberate scope decision, not an oversight.

**Open questions for the code request to the authors.** Chuter analyses DALEC EV
and DALEC DE. Bloom & Williams cite Williams et al. (1997) for DALEC2's ACM, but
DALEC2 merges the evergreen and deciduous versions, so it most likely inherits
the implementation Chuter documents. **This cannot be confirmed from the
papers**, and it is the single assumption `acm.py` rests on. Also open: whether a
boreal-calibrated `a`-parameter set exists.

**Scope: this site is outside the envelope DALEC2's authors defined.** Bloom &
Williams (2015) §2.5 selected sites with little expected water stress and no more
than **three months** of recorded below-freezing soil temperature, those criteria
reflecting DALEC2's capabilities — hydrological processes are not explicitly
represented. FI-Hyy has roughly **four to six months** of soil frost. Applying
DALEC2 here is a deliberate scope decision, not an oversight.

**Bearing on RQ3.** Under the 1997 form, GPP was insensitive to temperature over
most of the year, which would have pushed seasonal structure onto the respiration
parameters during calibration and made observed equifinality partly structural.
The corrected day-length term restores a strong, latitude-driven seasonal cycle,
so **that argument must be re-measured rather than carried over**. The Phase 7
synthetic twin separates structural from informational error — synthetic data
generated by this same GPP model should recover its parameters normally, so any
gap between synthetic and real-data recovery isolates structural error. The twin
must run with the frost mask active and over the same driver record, or the
comparison is not like-for-like.

---

## 6. Provisional values — not sourced, do not cite

Flagged here so they cannot quietly become fact:

- `site.latitude_deg` is `61.8474` in config, against `61.8475` in the working
  notes. Confirm against the site metadata: a latitude error is silent, and day
  length is what carries the seasonal cycle of photosynthesis at 62 N.
- The target projected LAI of ≈ 3 for FI-Hyy is provisional pending a citation
  from the SMEAR II literature.
- `lma = 110` g C m⁻² is Chuter's Loobos reference value, used as a working value
  in the gate re-run. It is a sampled parameter, not a constant, and the Loobos
  figure is not a FI-Hyy measurement.
- `ACM_CALIBRATION_BOUNDS` currently holds only the `t_mean` row (7–30 °C). The
  remaining Williams et al. Table 1 rows — irradiance, LAI, CO₂ and the rest —
  have not been supplied, so `calibration_bound_coverage()` reports on
  temperature alone.

Removed, and deliberately: `site.canopy_height_m` and `site.psi_d_mpa`. Both were
required only by the Williams et al. (1997) form. Canopy height does not appear
in the DALEC conductance equation, and the water potential is a coefficient of
the site-calibrated parameter set rather than a config value.

---

## References

- Bloom, A. A. and Williams, M. (2015). Constraining ecosystem carbon dynamics
  in a data-limited world. *Biogeosciences* 12, 1299–1315.
- Chuter, A. M., Aston, P. J., Skeldon, A. C. and Roulstone, I. (2015). A
  dynamical systems analysis of the data assimilation linked ecosystem carbon
  (DALEC) models. *Chaos* 25(3), 036401. **The ACM source actually used.**
- Williams, M. et al. (1997). Predicting gross primary productivity in
  terrestrial ecosystems. *Ecological Applications* 7(3), 882–894. Cited by
  Bloom & Williams for ACM, but not the form DALEC implements — see §5a.
- Richardson, A. D. et al. (2010). Estimating parameters of a forest ecosystem C
  model with measurements of stocks and fluxes as joint constraints. *Oecologia*
  164, 25–40.
- Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux
  processing pipeline. *Scientific Data* 7, 225.
