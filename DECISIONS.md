# Decisions, corrections and known problems

A durable record of everything that would be expensive to rediscover: the design
decisions that are settled, the four places where the published papers are
wrong, the constants that were resolved once, and the structural problems that
are recorded rather than solved.

Nothing here is new. It is collected from the module docstrings, code comments,
`config/default.yaml` and `README.md`, which remain the authoritative sources.
Where this file and the code disagree, the code is right and this file is stale.

Limitations — what is wrong, whether the literature has solved it, and our
position on each — live separately in [LIMITATIONS.md](LIMITATIONS.md).

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

**Year blocks — AUTHORITATIVE, decided 2026-08-26.**

| block | years | inclusive |
|---|---|---|
| **calibration** | **1997–2010** | 14 years |
| **prediction** | **2011–2014** | 4 years |

This is the Richardson et al. (2010) split: a long calibration record followed by
a forward run over a held-out contiguous block.

| | total days | QC ≥ 0.75 | usable sigma | **assimilable** | driver gaps |
|---|---:|---:|---:|---:|---:|
| calibration 1997–2010 | 5113 | 4662 | 4748 | **4413 (86.3%)** | 0 |
| prediction 2011–2014 | 1461 | 1418 | 844 | **810 (55.4%)** | 0 |

"Assimilable" is the intersection — days that both pass QC and carry a usable
sigma — and is the number that matters, since either condition alone is not
enough to enter the likelihood. Every year of the record except 1996 is used;
nothing is held in reserve.

**Justification.** DALEC's slow pools — wood and soil organic matter — turn over
on timescales of years to decades. A short record cannot identify them at all,
because the observation window never sees enough of a turnover cycle for the
data to distinguish one rate from another. Length is therefore not a convenience
here; it is a precondition for the parameters being identifiable in principle.
The split also follows published design rather than being chosen to suit this
record, which matters because the alternative — picking the window that makes
the results look best — is exactly the kind of choice a reviewer should not have
to take on trust.

This supersedes the earlier calibration 1997–2005 / evaluation 2006–2009 split
and the reasoning recorded against it. `years.evaluation` is **retired**: the key
is gone from the config, and `scripts/01_prepare_data.py` and
`tests/test_config.py` now refer to `prediction`.

**Why 1996 is excluded, and why that reason is not coverage.** `CO2_F_MDS` is
missing for **1996-01-01 to 1996-01-19**, 19 consecutive days at the very start
of the record. `data_io._check_drivers_complete` raises on any driver gap rather
than interpolating, because one NaN driver silently contaminates every carbon
pool from that day onward. **1997–2014 is entirely gap-free** — no driver, in any
year, is missing on any day.

The consequence is worth stating plainly, because it is the kind of claim a
reviewer will want and it is rarely available: **the calibration block contains
no imputed driver values.** Every driver on every one of its 5113 days is as the
FLUXNET2015 product supplies it. Backfilling 19 days of CO2 to buy one more year
would have forfeited that, for a year that is in any case only 50% assimilable.

1996 is therefore excluded on driver completeness, not on the coverage grounds
that drove the earlier and now-superseded year choices.

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

**`t` is days since 21 December, not calendar day of year.** Chuter §II.A shifts
the time scale ten days back so that `t = 0` falls on the shortest day and the
day-length function is even around zero. Feeding B5 a raw doy puts the solstice
at doy 182 instead of 172 — a ten-day phase error through the whole seasonal
cycle, and silent, because the amplitude is unaffected. Corrected:

```python
t_solar = (doy + 10) % 365
delta   = -0.408 * np.cos(2*np.pi*t_solar / 365.0)
```

Verified at 61.8474 N: 19.18 h at the solstice, 4.82 h at midwinter, 11.88 h at
both equinoxes, against true values near 19.0, 4.7 and 12.2.

**Substitutions and conventions.** `L = C_fol / c_lma` (Eq. A12), from model
state rather than drivers. `Tmax` is the daily maximum, here `TA_F_DAY`;
B3 applies `exp(a8 * Tmax)` to the **maximum**, not to a mean. `Tr` is the
**full** daily temperature range, and B1 takes `0.5 * Tr` — so the half-range is
what finally divides, and getting that factor of two wrong is silent.
Decomposition in A5/A6 uses `TA_F`, a third and distinct temperature.

### Daily extremes come from the half-hourly product, not from the daily one

`Tr` is a range and cannot be negative. The FULLSET **daily** product carries no
`TA_F_MAX` or `TA_F_MIN` — only daytime and nighttime *means* — and those means
invert on **14.8%** of calibration days, driving the B1 denominator negative on
**5.0%** of them. So `scripts/01b_derive_tminmax.py` derives the true extremes
from the half-hourly `TA_F` by a groupby, and `data_io` joins them onto the
driver record. All 6940 days are complete at 48 valid half-hours; no day has a
negative range.

**The proxy was not merely occasionally inverted — it was systematically wrong.**
Measured over 1997–2005:

| | correlation | mean bias | RMSE | proxy min | truth min |
|---|---:|---:|---:|---:|---:|
| `t_max` vs `TA_F_DAY` | 0.989 | −2.51 °C | 2.91 | −29.44 | −26.32 |
| `t_min` vs `TA_F_NIGHT` | 0.989 | +2.28 °C | 2.65 | −29.07 | −32.13 |
| **`t_range`** | **0.641** | **−4.79 °C** | 5.39 | −5.54 | 0.20 |

**The correlation split is the finding.** Each *level* tracks its true
counterpart almost perfectly, at 0.989 — but the biases run in opposite
directions, the daytime mean sitting 2.51 °C below the true maximum and the
nighttime mean 2.28 °C above the true minimum, and those two biases **compound in
the difference**. The range therefore correlates only 0.641 and averages
1.55 °C against a true 6.34: understated fourfold across the whole record, not
merely on the 486 days that inverted. On those 486 days the true range averages
4.63 °C and reaches 17.8. This justifies the extra preprocessing step and belongs
in the thesis Methods.

**Sensitivity, and what it says about the regime.** Substituting the true
extremes changes annual GPP by **−1.2% at the gate's operating point** but
**−16.8% at `ceff` = 100**. Small where the model is light-limited, in which
regime conductance barely matters; large where it is diffusion-limited, where it
matters directly. The model sits in the light-limited regime at its working
`ceff`, which is why the correction is modest there.

**The interim floor is retained but now unreachable.** `AcmModel.range_floor_count`
still counts days where `Tr < 0` and is asserted to be zero over the calibration
block; if it ever fires again, something upstream has broken. Note the direction
it had, since it is easy to assume the reassuring one: `Tr = 0` minimises the B1
denominator and therefore *maximises* conductance — `g_c = 4.570` against 1.25 at
`Tr = 2` and 0.32 at `Tr = 10` — so the floor **inflated** GPP on the affected
days by 2.8–10.9%, and the whole block by 0.17–0.24%.

**The day/night proxy path is retained behind an explicit flag**,
`temperature_source="day_night_proxy"`, unreachable by default. It is the
comparison baseline for `diagnostics.temperature_proxy_comparison`, not a
substitute.

### Three temperatures, and they must not be confused

The driver record now carries four temperature quantities, of which three reach
the model and they reach different parts of it:

| quantity | source | consumed by |
|---|---|---|
| `t_air` | `TA_F`, daily mean | decomposition and respiration, A5/A6/A10 |
| `t_max`, `t_min` | derived from half-hourly | photosynthesis only |
| `t_day`, `t_night` | `TA_F_DAY`, `TA_F_NIGHT` | **nothing** — comparison baseline |

Mixing them changes numbers without raising anything, so the separation is
asserted from both directions: a spy `gpp_fn` confirms photosynthesis receives
`t_max`/`t_min` and never `t_air` by name or by value, and moving `t_day`/
`t_night` by 25 °C is confirmed to change neither GPP nor respiration at all.

**Half-range convention, verified.** B1 applies `0.5 * Tr` internally, so the
`Tr` passed in must be the **full** range `T_max − T_min`, not the half-range
Williams et al. Table 1 calls `D_T`. Checked directly: at `t_day = 20`,
`t_night = 10`, `terms["t_range"] = 10.0` and `g_c = 0.321424`, which matches a
denominator built from `0.5 × 10`. Halving twice would give 0.600603. Not
halving twice.

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

**3. 59.0% of calibration days fall below ACM's 7 °C lower calibration bound**,
so the model extrapolates on most of the record. `ACM_CALIBRATION_BOUNDS`
currently holds only the temperature row.

**4. The modelled growing season is about 41 days too long, and its peak about a
quarter too low.** Caught by `seasonal_timing` on its first real run, at
`ceff` 11.7 with LAI near 3: the seasonal *peak* lands on the observed day
exactly (doy 185), but onset is at doy 79 against 110–111 for the two products
and cessation at 287 against 277, giving a season of 208 days against 166–167.
Peak daily GPP is 7.56 g C m⁻² d⁻¹ at the 99th percentile against 10.43 (NT) and
11.52 (DT). The annual total is right — the gate passes at 1.032 — because the
model spreads the correct amount of carbon over a longer, flatter season.

**This is exactly the failure the magnitude gate cannot see**, and the reason the
timing diagnostic exists. Whether it is an ACM property or a phenology-parameter
artefact is not yet established: `d_onset`, `cr_onset` and `cr_fall` were set by
hand for this run, not fitted, and `d_fall` is inert (§2, correction 2). Do not
attribute it to ACM before Morris screening.

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

- `site.latitude_deg` is `61.8474`. Still to be confirmed against site metadata:
  a latitude error is silent, and day length is what carries the seasonal cycle
  of photosynthesis at 62 N.
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
