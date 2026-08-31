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
- ~~The target projected LAI of ≈ 3~~ **RESOLVED**: Kolari (2010) gives all-sided
  seasonal maximum ~8.0 before the 2002 thinning and ~6.5 after, which over a
  2.5 all-sided-to-projected ratio is projected 3.2 and 2.6. See §8.
- ~~`lma = 110` g C m⁻², Chuter's Loobos reference value~~ **RESOLVED**: `lma`
  now takes U(144, 241) g C m⁻² derived from measured needle litterfall,
  longevity and projected LAI. See §8. The Loobos 110 is retained only as
  coefficient-set provenance.
- **`REFLEX_CEFF_BOUNDS` U(5, 20) is load-bearing**: the conclusion that GPP has
  no detectable magnitude bias depends on it, and on the DALEC2 range the
  residual is 2.0–2.9× instead of 0.90–1.19×. Fox et al. (2009) Table 4, for the
  same ACM parameter `a1`. **Verified at source by the user; not independently
  checked in this session** — ScienceDirect is paywalled and the Reading and
  Edinburgh repository copies are access-restricted. Record the table and page in
  the thesis.
- Fox et al. (2009) §4.5 is cited in §9 on initial pool states. **Not read in
  full.** Carvalhais et al. (2008) has been downgraded to a see-also; the
  initial-state argument no longer depends on it.
- **The LAI area basis is load-bearing and OPEN.** The ratio is now sourced —
  total/projected = 2.57 for Scots pine (Niinemets et al. 2001), matching
  (π+2)/2 = 2.5708 (Grace 1987) — but *which basis ACM expects* is not settled,
  and the calibration cannot settle it because ACM derives from SPA parameterised
  on a broadleaf stand where all three bases coincide. Hemisurface gives a
  divisor of exactly 2 and a residual of **1.03×**; projected gives 2.571 and
  **0.66×**. The magnitude conclusion turns on this. **Not resolvable by citation
  — the citation hunt is closed** (§10, LIMITATIONS §15). Both conventions stay
  live and the calibration will be run under both as a sensitivity.
- `F_SOM_BOUNDS`, the SOM share of heterotrophic respiration, U(0.5, 0.9). A
  judgement about boreal soils, not a measurement. It is now the only unsourced
  input to the respiration prior: τ7 records sources for everything else.
- `ACM_CALIBRATION_BOUNDS` currently holds only the `t_mean` row (7–30 °C). The
  remaining Williams et al. Table 1 rows — irradiance, LAI, CO₂ and the rest —
  have not been supplied, so `calibration_bound_coverage()` reports on
  temperature alone.

Removed, and deliberately: `site.canopy_height_m` and `site.psi_d_mpa`. Both were
required only by the Williams et al. (1997) form. Canopy height does not appear
in the DALEC conductance equation, and the water potential is a coefficient of
the site-calibrated parameter set rather than a config value.

---

## 7. Heterotrophic respiration is reparameterised — site-informed prior

**Decided 2026-08-28, corrected the same day.** Supersedes the bounded-uniform
priors on `c_lit_0`, `c_som_0`, `theta_lit` and `theta_som` in §1. Everything
else in §1 stands.

Source for every site number below: **Ilvesniemi, H., Levula, J., Ojansuu, R.,
Kolari, P., Kulmala, L., Pumpanen, J., Launiainen, S., Vesala, T. and Nikinmaa,
E. (2009). Long-term measurements of the carbon balance of a boreal Scots pine
dominated forest ecosystem. *Boreal Environment Research* 14(4), 731–753.**
Figure 6, p. 747. **Verified against the full paper**, not taken on report.

### The problem this fixes

Sampling `theta_som` and `c_som_0` as independent uniforms places the prior on
two stocks whose *product* is a flux, and does not constrain that flux:

| | superseded | reparameterised |
|---|---:|---:|
| median Rh at T = 0, g C m⁻² d⁻¹ | 42.4 | ≈ 0.7 |
| maximum Rh at T = 0 | 211.9 | bounded by construction |
| median `c_som_0`, g C m⁻² | 99,468 | ≈ 5,800 |

### The design

Sampled:

| parameter | prior | basis |
|---|---|---|
| `rh_annual` | U(290, 370) g C m⁻² yr⁻¹ | Fig. 6, decomposition of SOM |
| `f_som` | U(0.5, 0.9) | judgement, §6 |
| `theta_lit` | U(5.476×10⁻⁴, 2.738×10⁻³) d⁻¹ | τ_lit = 1–5 yr |
| `theta_som` | U(4.488×10⁻⁵, 1.141×10⁻⁴) d⁻¹ | τ_som = 24–61 yr |

Derived:

```
rh_ref  = rh_annual / (M(Theta_draw) * 365.25)
c_lit_0 = (1 - f_som) * rh_ref / theta_lit
c_som_0 =      f_som  * rh_ref / theta_som
```

`rh_ref` is defined at a reference temperature of **0 °C**, not a free choice:
the A5/A6 multiplier `exp(Θ·T)` is one there.

### Correction 1 — the prior rests on Fig. 6, not on the mass balance

**`rh_annual` comes from the measured flux, not from litter input.** Figure 6
gives *decomposition of soil organic matter* directly as ~290–370 g C m⁻² yr⁻¹,
obtained by splitting a measured soil CO₂ efflux of 577–737 g C m⁻² yr⁻¹ roughly
50/50 between root-and-rhizosphere respiration and decomposition, on the basis of
a girdling experiment at the site. **That is an attribution of a measured flux
and assumes nothing about steady state**, so the sink objection recorded below
does not touch it.

**The 280 g C m⁻² yr⁻¹ litter input is corroboration only.** Figure 6's
components — above-ground tree litter 142–204, below-ground tree litter ~90,
ground vegetation ~15 — sum to 247–309, consistent with the 280 usually quoted.
It is retained as `ANNUAL_LITTER_INPUT_G_C_M2` and used nowhere in the
derivation, because **equating litter input with Rh assumes long-run steady
state, which makes it an upper bound at this site, not an estimate**: FI-Hyy is
a measured sink, so input necessarily exceeds decomposition. The two ranges
overlap, which is the corroboration.

**An earlier claim here was wrong and is withdrawn.** It said that if 280 is
litter input then the implied Rh is 280 − 217 ≈ 63, making `rh_ref` more than
four times too high. That does not hold: the accumulation goes overwhelmingly
into **tree biomass**, which the paper puts at 242 g C m⁻² yr⁻¹ against an NEE of
206, and it states that soil carbon accumulation could not be measured. Nothing
licenses subtracting the whole ecosystem accumulation from the soil flux.

### Correction 2 — the temperature conversion is now per draw

At steady state, over a year,

```
rh_annual = rh_ref * Σ exp(Θ·T) = rh_ref * M(Θ) * 365.25
```

**`M` is now computed from each draw's own sampled `temperature_exponent`**,
rather than once at a fixed Θ. Under the fixed-Θ construction the realised annual
respiration drifted off target for every draw whose Θ differed: median realised
Rh was **299.8** g C m⁻² yr⁻¹ against an intended 280.

**That 299.8 is itself a result worth keeping.** It falls inside Fig. 6's
independently measured 290–370, so the mass-balance estimate and the girdling-
based flux attribution — two different methods, one assuming steady state and one
not — agree at this site. That is a second corroboration, not a nuisance.

With the fix the inversion is exact: realised annual Rh matches the sampled
`rh_annual` to within 6×10⁻¹⁴ g C m⁻² yr⁻¹ on every draw.

**A third, independent estimate agrees.** Kolari (2010) gives heterotrophic
respiration as 35–40% of total ecosystem respiration at the 75-year-old stand,
about half the soil CO₂ efflux. With Re ≈ 850 g C m⁻² yr⁻¹ that is **300–340**,
inside `rh_annual`'s U(290, 370). Three routes now agree on the same quantity:

| route | estimate, g C m⁻² yr⁻¹ | assumes steady state? |
|---|---:|---|
| litter mass balance (Ilvesniemi) | 280, realised 299.8 | yes |
| girdling split of soil efflux (Ilvesniemi Fig. 6) | 290–370 | no |
| share of Re (Kolari 2010) | 300–340 | no |

Two of the three make no equilibrium assumption, which is what makes the
agreement worth something.

`M` remains the **mean of the exponential**, never `exp` of the mean. `exp` is
convex, the FI-Hyy daily temperature has sd 9.4 °C, and the gap is 5.6%:
M = 1.2406 against a naive 1.1715 at Θ = 0.0366. The naive form would inflate
every derived stock by that much.

### Correction 3 — the residence times, and what was actually found

**τ_lit = 1–5 yr is now sourced.** Yasso07 (Tuomi, M. et al. (2009), *Leaf litter
decomposition — estimates of global variability based on Yasso07 model*,
Ecological Modelling 220, 3362–3371, Table 1) gives labile AWEN decomposition
rates of α_A = 0.66, α_W = 4.3, α_E = 0.35, α_N = 0.22 a⁻¹ — residence times of
1.5, 0.23, 2.9 and 4.5 yr, which bracket the adopted range. Value read from the
paper.

**Yasso07's humus rate is deliberately not used.** α_H = 3.3×10⁻³ a⁻¹ implies 303
yr, but Yasso's H is the recalcitrant fraction receiving ~4% of labile mass loss,
whereas DALEC's SOM pool is bulk soil carbon carrying most of the heterotrophic
flux. Different objects; the rate is not transferable. Adopting it would have put
soil respiration an order of magnitude too low.

**Liski & Westman: nothing usable found.** Both *Carbon storage in forest soil of
Finland* papers (Biogeochemistry 36, 1997) are paywalled, and no residence or
turnover time for southern-Finland forest soil surfaced in the accessible
literature. Recorded as a null result rather than filled with a guess.

**τ_som = 24–61 yr is therefore derived from the site's own measurements.** The
derivation, which is why it is not simply a wider guess:

> `tau` in this model is `1/theta`, the e-folding time **at the 0 °C reference**.
> The field residence time is shorter by the mean multiplier, because at steady
> state `C = R·tau/M`. So from a measured stock `S`, respiration `R` and share
> `f`:
>
> ```
> tau_som = (S * M / R - (1 - f) * tau_lit) / f
> ```
>
> With S = 6560 g C m⁻² (Fig. 6, measured), R = 290–370, M = 1.2406 (empirical Θ)
> to 1.3677 (prior-median Θ), f ∈ [0.5, 0.9] and τ_lit ∈ [1, 5], this gives
> **23.9–60.9 yr**. Adopted as 24–61.

**Omitting the `M` factor understates τ_som by about 25%** and the implied soil
stock by the same factor — a real trap, since `tau` reads naturally as a field
residence time but is defined at the reference temperature.

The adopted range was taken from this derivation, **not** chosen by scoring
candidate ranges against Check 1. A scan over candidates did independently favour
28–62 yr as best-centring the stock, which corroborates the derivation; it did
not select it.

### Check 1 is no longer an independent validation

Stated plainly because it changes what the check means. `c_som_0` is now derived
through a relation that takes the measured soil carbon stock as an **input**. The
same number cannot be both a prior input and a validation target. Under the old
unsourced τ_som, Check 1 was a weak consistency check; it is now close to
tautological, and the implied total soil carbon landing near 6560 confirms the
arithmetic rather than the science.

This is a deliberate trade: a well-sourced prior built on a direct measurement is
worth more than a weak test against a supplied range. What replaces Check 1 as a
test is the prior predictive itself — Checks 2 and 3.

### Known residual inconsistency

`rh_annual` is a **steady-state** relation: it assumes the stocks that produce
`rh_annual` are the stocks the model starts with. FI-Hyy is a sink, so the true
pools are growing and the initial condition is only approximately at steady
state. The magnitude is bounded by the accumulation the paper attributes to soil,
which it says could not be measured — so this cannot presently be quantified. It
is recorded, not solved.

### The three acceptance checks — results after the corrections, 2026-08-28

`scripts/10_reparameterised_prior.py`, 1000 draws, seed 20260809, block
1997–2010. Full output in `reports/prior_diagnostics/reparameterised_checks.txt`.

| check | before corrections | after | Task 1 baseline |
|---|---:|---:|---:|
| 1. `c_som_0` inside 5,000–10,000 | 39.6% | **58.3%** | 2.5% |
| 1. `c_som_0` median, g C m⁻² | 5,257 | **5,784** | 99,468 |
| 1. total soil C median | — | **5,932** (measured 6,560) | — |
| 2a. draw failure rate | 2.9% | **1.7%** | 83.7% |
| 2b. coverage of the 90% band | 0.724 | **0.703** | 0.651 |
| 3. median annual NEE | +543.8 | **+575.8** | — |

Rh at T = 0 is bounded by construction: median 0.66, maximum 0.92 g C m⁻² d⁻¹,
against the superseded prior's 42.50 and 208.51.

**Checks 2b and 3 moved slightly the wrong way, and that is expected.** The
corrections raised the respiration target from an effective ~300 to a sampled
290–370, median 330, because Fig. 6's measured flux attribution is higher than
the mass-balance figure the prior previously used. More Rh means a more positive
NEE. The corrections were made for correctness of sourcing, not to improve
Check 3, and **Check 3 cannot improve while GPP is wrong** — see below.

**Site type, and the right comparison row.** All Kolari study sites are
**Vaccinium type** (Cajander classification), Scots pine dominant. The correct
Viskari Table 4 row is therefore **VT_SP**, measured SOC **5.73 kg C m⁻²
(SD 0.71)** — not CT_SP, which was the wrong forest type. Against it:

| | g C m⁻² |
|---|---:|
| derived `c_som_0`, median | **5,784** |
| VT_SP measured SOC | **5,730 ± 710** |

The derived median sits 54 g C m⁻² from the measured mean, well inside one
standard deviation. **This is a consistency check, not independent validation**:
the measured soil stock informs τ_som and therefore `c_som_0`, so the same
measurement cannot also test it. It confirms the arithmetic, not the science.

Check 1 is no longer an independent validation; see the subsection above.

### Why check 3 fails: GPP, not respiration

Full attribution in `reports/prior_diagnostics/FINDINGS_gpp.md`
(`scripts/11_gpp_investigation.py`).

| flux | modelled median | measured | ratio |
|---|---:|---:|---:|
| GPP | 2,570 | 952–1,104 (Fig. 6) | **2.50×** |
| NEE | +575.8 | −215.8 | wrong sign |

Only 0.8% of draws produce a GPP inside the measured range; 68.3% sit above it.

**`ceff` is not the cause, contrary to what this section previously recorded.**
Across its full tenfold prior range median GPP moves by 1.32×, and the lowest
quintile still overshoots by more than double; Spearman correlation is +0.241.
ACM's light interception saturates, so canopy efficiency has little leverage in
a dense canopy.

**Leaf area is the cause**, correlating +0.962 with annual GPP. Modelled mean LAI
has median 5.09 and a 95th percentile of 48.26 against a site value near 3,
driven by a positive feedback — more leaf area gives more GPP gives more foliar
allocation. The measurable root cause is allocation: prior draws send 24.0% of
GPP to foliage and labile, where the site's measured needle litterfall of 154
g C m⁻² yr⁻¹ over GPP ≈ 1030 implies 15.0%.

**Priors alone will not close it.** At the site's own LAI of about 3 the model
still yields ~1,800 g C m⁻² yr⁻¹; the measured range corresponds to LAI ≈ 1–2.
A residual factor of about 1.7 survives any canopy correction and belongs to
model structure — LIMITATIONS §1 (no high-latitude temperature limitation on
photosynthesis) and §2 (no boreal ACM coefficient set). Not fixed here, and not
to be fixed by fitting `ceff` to the measured GPP.

### The steady-state premise, and its actual scope

The derivation assumes the initial stocks are those that produce `rh_annual` at
steady state. **FI-Hyy is not at steady state**: it is a measured sink of about
217 g C m⁻² yr⁻¹ over the calibration block (`Reco` − `GPP` = 886 − 1103 = −217,
matching the observed NEE of −215.8 to 1.2 g).

**This bears on the litter-input route, which is why that route is corroboration
only.** At a sink, litter input necessarily exceeds decomposition, so equating
them makes the litter input an upper bound on Rh rather than an estimate of it.

**It does not bear on the Fig. 6 route the prior actually uses.** That number is
an attribution of a *measured* soil CO₂ efflux between root respiration and
decomposition, via girdling. It makes no steady-state assumption at all.

**Kolari (2010) makes the steady-state relation defensible here.** He reports no
trend in soil carbon stock across the chronosequence, and that changes over a
rotation are generally very small, citing Liski and Westman (1995). So using a
steady-state relation to inform the *soil* prior is sound at this site even
though the *stand* is plainly not at equilibrium: the accumulation goes into tree
biomass, which the paper puts at 242 g C m⁻² yr⁻¹, and not into the soil.

The distinction matters and is easy to blur. **The stand is not at equilibrium;
the soil approximately is.** The prior only ever needed the second.

What remains is second-order: the initial pools are set to their steady-state
values while the true pools are slowly growing. The accumulation attributable to
soil is what would size that error, and the paper states soil carbon accumulation
could not be measured — the 217 goes overwhelmingly into tree biomass, which the
paper puts at 242 g C m⁻² yr⁻¹. So the residual is real, unquantified, and
recorded rather than solved.

An alternative that needs no literature value at all: `reco_nt`/`reco_dt` are
already loaded, and Rh = `Reco` − `f_auto`·GPP is a per-draw quantity. It is
model output rather than observation (§1), so it cannot enter the likelihood, but
it could inform a prior. Not adopted; recorded.

### Where it lives

`dalec.parameters` holds the constants and the four conversion functions;
`dalec.diagnostics.sample_reparameterised_parameters` draws from them.
`scripts/10_reparameterised_prior.py` runs the three acceptance checks and
reports, and changes nothing.

The superseded path is retained and still tested:
`dalec.diagnostics.sample_prior_parameters` continues to draw the published
priors, because Tasks 1 and 2 are stated against it and their numbers must stay
reproducible.

---

## 8. The canopy: LAI convention, `lma`, `c_lf` and allocation

**Decided 2026-08-28.** Source: **Kolari, P. (2010). Dissertationes Forestales 99,
doi 10.14214/df.99**, with the allocation fluxes from Ilvesniemi et al. (2009)
Fig. 6. Supersedes the published priors on `lma` and `c_lf` and the flat
Dirichlet on allocation, for the reparameterised sampler only.

### The LAI convention is projected, and this is now settled

**ACM expects projected (one-sided) LAI**, so `lma` is leaf carbon per unit
projected leaf area. This was assumed throughout the code but never established;
two independent lines of evidence from the fitted coefficients settle it.

1. **Where quantum yield saturates.** Chuter B4 is
   `E_0 = a7·L²/(L² + a9)`, half-saturating at `L = sqrt(a9)`: **1.45** for
   Loobos, **1.03** for Oregon. On an all-sided basis those become 0.58 and 0.41,
   far below where any canopy's light capture saturates. On a projected basis
   they are exactly where it does.
2. **What the sets were fitted with.** Both carry `lma_reference` ≈ **110**
   g C m⁻². That is a projected-basis needle mass for pine; all-sided it would be
   about 44, which no conifer has.

Recorded as `LAI_IS_PROJECTED`. Anyone changing `lma` must check they are on this
basis, because the error is a silent factor of 2.5.

### The site value, and the end of the unsourced "≈ 3"

Kolari gives the **seasonal maximum all-sided** LAI at SMEAR II as ~8.0 before
the 2002 thinning and ~6.5 after, a 19% reduction. Over the conifer all-sided to
projected ratio of ~2.5:

| | all-sided | **projected** |
|---|---:|---:|
| before 2002 | 8.0 | **3.2** |
| after 2002 | 6.5 | **2.6** |

**This is the source of the ≈ 3 that DECISIONS §6 carried as provisional**, and
it also tells us that figure is projected. Removed from §6.

### `lma` — derived, not fitted

```
C_fol = needle litterfall / c_lf          = 154 / (1/5 .. 1/3)  = 462 .. 770
lma   = C_fol / LAI_projected             = 462..770 / 3.2      = 144 .. 241
```

Adopted **U(144, 241) g C m⁻²**, against the published U(10, 400) which admits
10 g C m⁻², thinner than any conifer needle. Every input is measured; nothing is
solved back from a target GPP.

The **mean** litterfall of 154 is used rather than the Fig. 6 range of 142–204.
That range is inter-annual and spatial spread in a quantity whose average is the
better estimate, and multiplying its extremes by the longevity extremes compounds
two uncertainties into a needlessly wide 133–319. Holding litterfall at its mean
isolates the longevity uncertainty, which is the one that matters.

The pairing is slightly inconsistent and is recorded as such: litterfall 154 is a
multi-year average spanning the 2002 thinning, while LAI 3.2 is the pre-thinning
value. Using the block-weighted projected LAI of 2.81 — five years at 3.2 and
nine at 2.6 — would give 164–274 instead. The two overlap across most of their
range and the difference is not material at this stage.

### `c_lf` — needle longevity

Scots pine in southern Finland retains 3–5 needle age classes, so
`c_lf = 1/longevity` gives **U(0.20, 0.333)**. The published U(0.125, 1.0) spans
lifespans of 1 to 8 years and its upper half has an evergreen conifer shedding
its needles inside eighteen months.

**Direction matters here.** `c_lf` correlates *negatively* with GPP (−0.628), so
tightening it alone more than doubles the steady-state foliar pool and makes the
GPP overestimate worse. It is only safe alongside the allocation constraint
below, which cuts the inflow. Applied together, never separately.

### Allocation — a Dirichlet concentration from measured fluxes

The flat Dirichlet sent **24.0%** of GPP to foliage and labile where the measured
needle litterfall over GPP implies **15.0%**, and that excess drove a feedback —
more leaf area gives more GPP gives more foliar allocation — which ran modelled
LAI to a 95th percentile of 48.

At steady state allocation equals turnover, so Fig. 6's measured flows are
allocation fluxes:

| pool | measured flux, g C m⁻² yr⁻¹ | share of NPP |
|---|---:|---:|
| foliage | 154 (needle litterfall) | 0.306 |
| fine root | 90 (below-ground litter) | 0.178 |
| wood | 261 (growth, 180–240 above + 34–69 below) | 0.516 |

These sum to 505, which over GPP ≈ 1030 is 0.49 — that is `1 - f_auto` at
`f_auto` ≈ 0.51, **independently inside the published f_auto prior**. A useful
consistency check that the picture holds together.

The concentration is these shares times a total of 20, giving a standard
deviation on the foliar share near 0.10: informative without pinning allocation
to a point. Labile and foliar weights are set equal, because DALEC's labile pool
exists to feed foliage at bud burst and the measurement constrains their sum, not
the split.

Realised in the sampler: foliar share median **0.144** against the measured
0.150, fine root **0.081** against 0.087, wood **0.250** against 0.253.

### What the canopy priors did to GPP, and what it proved

> **Superseded by §9.** The 2.00× ACM productivity bias asserted at the end of
> this subsection is **withdrawn**, along with the claim that the ACM bias was
> "measured twice". It was measuring the DALEC2 `ceff` prior U(10, 100), and it
> compared an annual-mean modelled LAI against a seasonal-maximum measured one.
> The subsection is kept because the arithmetic in it is correct and the sequence
> matters; the conclusion is not. Current position: §9 and
> `reports/prior_diagnostics/FINDINGS_gpp.md`.

Measured, not predicted: applying §8 made prior predictive GPP **worse**, from a
median of 2,570 to **2,972** g C m⁻² yr⁻¹ against a measured 952–1104, with mean
LAI rising from 5.09 to 7.49.

The mechanism is arithmetic. Steady-state LAI is `share/(c_lf·lma) · GPP`; the
coefficient goes from 0.00208 to 0.00281, a 35% rise, because cutting `c_lf` by
53% outweighs cutting allocation by 40%. The two were applied together, as
required, and together they still point the wrong way.

**The priors are nonetheless correct, and this is the point.** At the *measured*
GPP of 1030 the new coefficient implies LAI **2.89**, against Kolari's measured
3.2 before thinning and 2.6 after. The old priors implied 2.14. What the priors
pin is the ratio LAI/GPP; they cannot pin the level, because ACM sets GPP from
LAI in the other direction.

**That appeared to isolate a residual.** Among draws sitting at mean LAI 2.8–3.6
the median GPP was 2,056 against 1,030, read at the time as ACM being 2.00× too
productive at the correct canopy. ~~The ACM bias is measured twice, in magnitude
and in seasonal distribution.~~ **Both readings are withdrawn (§9).** The band was
not the correct canopy — it is a seasonal maximum, not an annual mean — and the
`ceff` prior was DALEC2's, not REFLEX's. Only the seasonal defect survives.

`ceff`'s correlation with GPP rose from +0.241 to +0.707 once the canopy was
pinned, so the earlier claim that it is a weak lever was true of the old prior
only. **Do not fit it to GPP** — that remains right, and for the original reason:
a canopy-efficiency parameter tuned to a flux hides whatever structural error is
actually present. Adopting REFLEX's published range instead is what §9 does.

Full analysis in `reports/prior_diagnostics/FINDINGS_gpp.md`.

### Not written into the registry, deliberately

`lma` and `c_lf` keep their registry names but take these bounds inside
`sample_reparameterised_parameters` only. Tasks 1 and 2 are stated against the
published priors and their numbers must stay reproducible, so
`sample_prior_parameters` and the registry are untouched.


## 9. `ceff` from REFLEX, and every initial pool derived

**Decided 2026-08-28.** Supersedes the published `ceff` prior and all six initial
pool priors, for the reparameterised sampler only.

### `ceff` — a published prior adopted in place of a wider published prior

`ceff` takes **U(5, 20)**, Fox et al. (2009), the REFLEX model intercomparison,
Table 4, published range for the same ACM parameter `a1`.

**This is not fitting to GPP.** The range was taken from the intercomparison, not
chosen by inspecting modelled output. It happens to move GPP downward, which is a
consequence rather than the reason.

Note it is **not a narrowing**: REFLEX's range extends *below* DALEC2's
U(10, 100), and the measured GPP is reached at `ceff` ≈ 11–15, inside REFLEX and
near the bottom of DALEC2. That is consistent with the GPP magnitude gate having
passed at `ceff` ≈ 11.7 (§5a) and with Task 2's one-at-a-time optimum near 30
being an artefact of the then-uncontrolled canopy.

### Every initial pool is now derived

The goal was that no initial state is left on a generic published range. All six:

| pool | derivation | source of the flux |
|---|---|---|
| `c_lab_0` | litterfall · `f_lab`/(`f_lab`+`f_fol`) | Ilvesniemi Fig. 6, 154 |
| `c_fol_0` | litterfall / `c_lf` | Ilvesniemi Fig. 6, 154 |
| `c_roo_0` | below-ground litterfall / (`theta_roo`·365.25) | Ilvesniemi Fig. 6, ~90 |
| `c_woo_0` | tree carbon − the other three | Ilvesniemi Fig. 6, 6,800 |
| `c_lit_0` | (1−`f_som`)·`rh_ref` / `theta_lit` | §7 |
| `c_som_0` | `f_som`·`rh_ref` / `theta_som` | §7 |

Root and wood turnover carry no temperature term in A3/A4, so those annual fluxes
are simply `theta · C · 365.25`; only litter and SOM decomposition take the
`exp(Θ·T)` multiplier.

### Why this was necessary

`c_fol_0`'s published U(20, 2000) has median **1,010** against the **462–770**
the canopy priors imply, and on that prior the modelled foliar trajectory fell
monotonically from **1,497 in 1997 to 1,041 in 2010 and was still falling**. The
model never reaches steady state inside the calibration window, so **every
calibration year was a relaxation transient away from a wrong initial state**.

This is a known failure mode, not a novel concern: REFLEX (Fox et al. 2009)
§4.5 treats initial pool states as part of what must be estimated rather than
assumed. **Not read in full**; see §6.

**Carvalhais et al. (2008) is downgraded to a see-also.** It was cited here for
initial-state error biasing parameters and inflating confidence intervals, but
the argument no longer needs it: the case for deriving the pools rests on the
site's own measurements and on the trajectory this project measured directly —
`c_fol_0`'s published median of 1,010 against the 462–770 the canopy priors
imply, and a monotone fall from 1,497 to 1,041 still descending at 2010. That is
first-hand evidence from this model on this block, and it does not require a
second-hand citation to carry it.

### It did not deliver a stationary trajectory

Reported because the goal was explicit. After the change the trajectory falls
from **562 in 1997 to 177 in 2010**, a 69% decline — in relative terms *worse*
than the 30% before.

**The initial value is nonetheless correct.** `c_lf · c_fol_0` returns the
measured litterfall of 154 exactly, by construction and by test. What the drift
now shows is that the *model's own fixed point* under most prior draws lies below
the measured canopy: with `ceff` spanning 5–20, many draws produce GPP too low to
sustain 579 g C m⁻² of foliage.

**The drift is no longer an initial-state error**, and the diagnosis is sharper
than first stated. Cycling the block to convergence (`scripts/13_foliage_fixed_point.py`,
40 draws, up to 60 cycles) shows the model is **bistable**: 24 of 40 draws
collapse to a bare canopy and 16 run away to a dense one, and **none settles at
the measured leaf area**. The site's own canopy sits at an *unstable*
equilibrium, because the LAI → GPP → allocation → LAI loop has gain above one
there.

On the converged branch the litterfall excess factorises cleanly — allocation
1.27× times GPP 1.62× gives 2.05× against an observed 2.07× — and `c_lf` at 0.271
(3.7 yr) sits inside the measured 3–5 yr, so **turnover is not the culprit**. The
answer to "allocation too low, turnover too high, or GPP too low" is none of the
three: on the surviving branch both allocation and GPP are too *high*.

The stated goal of a stationary trajectory is not met, and now it is clear that no
initial condition could have met it.

### `c_roo_0` inherits a prior that is too wide

Deriving it from `theta_roo` moves the problem rather than solving it. The
published `theta_roo` U(10⁻⁴, 10⁻²) spans fine-root residence times of 0.27 to 27
years, and its median puts the stock at **49 g C m⁻² against a measured 238**
(Ilvesniemi Table 4, 476 g m⁻² dry biomass halved for carbon).

The site's own numbers pin it: 90 g C m⁻² yr⁻¹ through 238 g C m⁻² gives
**`theta_roo` = 1.04×10⁻³ d⁻¹, a 2.6 year residence time**. Available as
`implied_root_turnover()`, **reported and not adopted** — narrowing `theta_roo`
was outside this scope and is the obvious next step.

### The result

| | before | after | measured |
|---|---:|---:|---:|
| median annual GPP | 2,972 | **956** | 952–1,104 |
| ratio | 2.89× | **0.93×** | — |
| mean LAI | 7.49 | **2.48** | ~2.1–2.3 |
| GPP at mean LAI 2.8–3.6 | 2,056 (2.00×) | **1,219 (1.19×)** | — |

**The structural error estimate for the thesis: no detectable magnitude bias,
and the residual depends on an unsourced assumption.** Across defensible canopy
bands the residual runs 0.90× to 1.19× and zero is inside the range. It is firmly
no longer a factor of two.

But the all-sided-to-projected LAI ratio moves it further than the canopy band
does: **1.03× at ratio 2.0, 0.73× at 2.5, 0.58× at 3.0**, with zero inside the
IQR only at 2.0 (§6, `FINDINGS_gpp.md`). On the adopted 2.5 the model
*under*-produces by 27% — the sign is opposite to the withdrawn claims. Quote the
conclusion as "no detectable magnitude bias, conditional on `ceff` U(5, 20) and
an all-sided-to-projected ratio of 2.0–2.5", not as a single number.

**A correction to how the canopy band is read.** Kolari's 3.2 and 2.6 are
seasonal *maxima*; the diagnostic's `lai_mean` is an annual *mean*, and comparing
them directly was an error in the previous write-up. Kolari et al. (2009) give
the seasonal minimum all-sided LAI as 4.5–4.9, projected 1.8–2.0, against maxima
of 6.0–6.5 all-sided, projected 2.4–2.6 — so the annual mean projected LAI is
about **2.1–2.3**, and the modelled 2.48 is slightly high rather than low.

The same paper independently corroborates `c_lf`: "the annual turnover of foliage
was **25% of the maximum of foliage mass each year**", i.e. `c_lf` = 0.25, the
midpoint of the adopted U(0.20, 0.333).

**The seasonal asymmetry is not fixed and was not expected to be.** October/April
GPP improved from 0.290 to **0.397** against a measured 0.76 — still 0.52×, with
**100%** of draws below. Magnitude and seasonal distribution are separate defects
and only the first responded to the priors. See LIMITATIONS §1a.


## 10. The LAI area basis: three bases, and the calibration is silent

**Recorded 2026-08-28.** `lma` and the GPP magnitude conclusion both rest on the
divisor taken from Kolari's all-sided LAI. That divisor is now sourced, but the
**convention question is not settled**, and this section records why.

### There are three bases, not two

| basis | definition | ratio to total |
|---|---|---:|
| **total** (all-sided) | the whole needle surface | 1 |
| **hemisurface** | half the total | **exactly 2, by definition** |
| **projected** | the shadow area | **2.571 for Scots pine** |

- **Total / projected = 2.57** for Scots pine, Niinemets et al. (2001). This
  matches bisected-cylinder geometry exactly: (π + 2)/2 = **2.5708**. Grace, J. C.
  (1987), *NZ J For Sci* 17: 292–294, "Theoretical ratio between one-sided and
  total surface area for pine needles".
- **Total / hemisurface = 2** is a definition, not a measurement.
- Therefore **hemisurface / projected = 1.285**.

Chen, J.-M. and Black, T. A. (1992), *Plant Cell Environ* 15: 421–429, established
**hemisurface** as the standard for radiation work, arguing that projection has
neither physical nor biological significance whereas total intercepting area has
physical meaning for radiation interception.

### Which basis was ACM calibrated on? The calibration cannot say

**This is the finding, and it is a negative one.**

ACM (Williams et al. 1997) was not fitted to observations directly. Its own
abstract states the starting point was "a fine-scale, multilayer model of
half-hourly canopy processes that has been **parametrized for Harvard Forest,
Massachusetts**" — that is SPA, Williams et al. (1996), built for a
**Quercus–Acer deciduous broadleaf stand**.

**For flat leaves the three bases collapse**: projected = hemisurface = total/2,
identically. No needle geometry entered the calibration of either model, so
**there is no fact of the matter recoverable from it**. ACM's coefficients are
equally consistent with either reading, because the data that produced them could
not distinguish the two.

Internal evidence is likewise silent. Canopy quantum yield half-saturates at
`L = sqrt(c2)` = 0.93 for Williams 1997 and `sqrt(a9)` = 1.45 and 1.03 for
Chuter's two sets — all on a basis where, for the stand each was fitted to, the
distinction either does not arise or was never stated.

### What each reading implies

| basis | divisor | max projected LAI | mean projected LAI | `lma` | median GPP | residual | zero in IQR |
|---|---:|---:|---|---|---:|---:|---|
| **hemisurface** | 2.000 | 4.00 | 2.62–2.85 | 116–192 | 1,061 | **1.03×** | **YES** (0.89–1.19) |
| **projected** | 2.571 | 3.11 | 2.04–2.22 | 148–247 | 683 | **0.66×** | **NO** (0.58–0.99) |

**The two readings differ by 37% in GPP and they disagree on whether the model
has a magnitude bias at all.**

### The recommendation, and the confound stated plainly

**On physical grounds the hemisurface reading is the principled extension.** ACM's
radiation scheme was fitted to flat-leaf geometry, where the intercepting area per
unit LAI equals one side of the leaf. Carrying that scheme to needles without
changing what it means requires the basis that preserves intercepting area, which
is hemisurface — precisely Chen & Black's argument. The divisor is then exactly 2.

**But the basis that physics favours is also the one that closes the residual,
and that is a reason for more scrutiny rather than less.** It is recorded here as
the better-argued reading, **not adopted on the strength of the fit**, and the
projected reading is kept live in §6 as the alternative. Anyone writing this up
must present both and say which they chose and why.

### Nothing can settle it, and the citation hunt is closed

An earlier version of this section named **Williams et al. (1997) Table 1** as
the thing that would decide the basis. **It cannot, and neither can any other
document.** A flat-leaf calibration carries no needle basis *whatever LAI range
it spans*: the table would report the magnitudes tested, not the convention to
apply to a geometry that never appeared in the fitting data. The ambiguity is
**structural, not bibliographic** — there is no answer to find because the
calibration never contained one.

**Decision: both conventions stay live. Neither is adopted.** The calibration
will be **run under both and the difference reported as a sensitivity**. That is
the honest treatment of an ambiguity that cannot be resolved by looking harder.

Recorded as a limitation in its own right: LIMITATIONS §15.

Practical consequence: any statement about GPP magnitude bias must carry the
convention with it. "No detectable magnitude bias" holds on the hemisurface
reading and fails on the projected one, where the model under-produces by 34%.


## 11. `priors.py`, and what the collapse rate turned out to be

**Built 2026-08-28.** `src/dalec/priors.py` constructs the PyMC prior block by
reading `dalec.parameters` and nothing else. **No bound is typed in it**; a prior
can only be changed by changing its recorded source, and `tests/test_priors.py`
checks each bound against an independent lookup rather than against a copy.

Contents: bounded uniforms from the registry (§1), the canopy overrides `lma`,
`c_lf` and `ceff` (§8, §9), the four reparameterised respiration parameters (§7),
the allocation Dirichlet (§8), all six derived initial pools (§7, §9), and the LAI
convention as a switch that changes `lma` and nothing else (§10).

`LAI_CONVENTIONS` maps hemisurface to 2.0 and projected to (π+2)/2 = 2.5708. The
site LAI is now recorded **all-sided**, as Kolari published it, and `lai_divisor`
raises on an unknown name rather than defaulting — a silently wrong divisor is a
29% error in `lma`. The superseded working divisor of 2.5, which is neither
convention, is kept only to explain the `lma` bounds recorded in §8.

### The 60% collapse rate is ours, and it is the allocation Dirichlet

The foliage sustainability condition is
`R = (f_fol + f_lab)·(dGPP/dLAI)/(lma·c_lf) > 1`. In logs the decomposition is
exact and additive:

| term | share of Var(log R) |
|---|---:|
| **log(f_fol + f_lab)** | **69.2%** |
| log(dGPP/dLAI) | 15.3% |
| −log(lma) | **7.9%** |
| −log(c_lf) | 7.6% |

**The expectation was `lma`. It is not `lma`** — 7.9% against the allocation
share's 69.2%. Every term enters with unit coefficient, so influence is decided
by spread, and the allocation share spans **18.6×** where `lma` and `c_lf` span
1.67× each. Those two are tightly bounded by measurement; the Dirichlet's
concentration of 20 is a judgement with no measured uncertainty behind it.

**So 64.2% collapsing is a property of one of our priors, not of DALEC's
published ones.** At the measured central values `R` = 1.112 — the site's own
canopy is sustainable, and the collapse is a tail of the spread we chose.

### The concentration is now sourced, not judged

**Decided 2026-08-28.** The concentration was the last unsourced prior in the
model, and it carried 69% of the variance above. It is now **derived** from the
same Fig. 6 ranges that set the mean.

The measured foliar share of GPP is litterfall / GPP, and both carry ranges:

| | value | share |
|---|---|---:|
| low | 142 / 1104 | 0.1286 |
| central | 154 / 1028 | 0.1498 |
| high | 204 / 952 | 0.2143 |

The Dirichlet marginal of a component group is `Beta(a_group, a0 - a_group)`
with `Var(w) = p(1-p)/(a0+1)`, so `a0 = p(1-p)/Var(w) - 1`. The simplex is over
shares of NPP while the measurement is a share of GPP, so the propagated standard
deviation is divided by `1 - f_auto` = 0.5 at the prior mean.

| reading of the Fig. 6 range | sd(share) | implied `a0` |
|---|---:|---:|
| **full support (uniform)** | 0.02473 | **85.7** |
| 95% interval (±2 sd) | 0.02142 | 114.5 |

**Adopted: `a0` = 85.65**, the more conservative of the two. The Fig. 6 range is
interannual and spatial spread, not a stated confidence interval, and what a
fixed parameter needs is the uncertainty in the *long-run mean*, which is
narrower still. Reading the range as full support therefore errs toward a wider
prior. Implemented as `sourced_allocation_concentration()`, with the constant
pinned to its output and a test asserting they agree, so the two cannot drift.

**Consequences, reported and not targeted:**

| | judgement, `a0` = 20 | **sourced, `a0` = 86** | measured |
|---|---:|---:|---:|
| foliar share, median | 0.1431 | **0.1490** | 0.1498 |
| foliar share, 95% | 0.056–0.295 | **0.082–0.243** | 0.129–0.214 |
| inside the measured range | 43.5% | **57.1%** | — |
| **collapse rate** | 64.2% | **44.3%** | — |
| allocation share of Var(log R) | 69.2% | **51.6%** | — |

The sourced value is **4.3× the judgement value**, so the prior tightened. The
collapse rate falling from 64.2% to 44.3% is a consequence of sourcing, not a
target: nothing was tuned toward it, and the derivation would have been adopted
had it come out below 20 and raised the rate instead.

The allocation share remains the largest single contributor at 51.6%, but it no
longer dominates, and `Var(log R)` fell from 0.2720 to 0.1731. **Every prior in
the model is now sourced.**

---

## 12. Methods result: closures inside `pytensor.scan` defeat loop-invariant code motion

**Measured 2026-08-28.** Recorded as a methods finding rather than a bug fix,
because it is not specific to this model and it cost a 12.2× factor that no
correctness test could have caught.

### The finding

**Parameters closed over inside a `pytensor.scan` are pulled into the inner
graph, defeating loop-invariant code motion.** In this model `psi` solves
equation A9 from `c_lf` alone, so it is time-invariant, but the 24 unrolled
Newton steps that solve it were being evaluated *and taped for reverse-mode* at
every one of 5113 timesteps.

**Passing the parameters as explicit `non_sequences` recovers the hoisting**,
which PyTensor then performs by itself. The gradient went from **2,165 ms to 178
ms — 12.2×** — with **bit-for-bit identical** output.

### The three diagnostic steps

Each one was measured before the next was attempted.

| step | gradient | ratio to forward | what it established |
|---|---:|---:|---|
| 1. constant-`psi` control | **189 ms** | **4.06×** | the cost is the Newton chain |
| 2. hoisted via closure | **2,149 ms** | 43× | PyTensor pulls closures back inside |
| 3. explicit `non_sequences` | **178 ms** | **3.49×** | matches the control — fixed |

**Step 1 located the cost.** Freezing `psi` at a constant *inside* the scan —
numerically wrong, but a clean timing control — dropped the ratio from 37.6× to
4.06×, squarely in the normal reverse-mode range. That put the entire overhead in
the Newton chain and nowhere else.

**Step 2 refuted the obvious fix.** Computing `psi` outside the scan and
referring to it through a Python closure changed nothing: 2,149 ms against 2,165
ms, bit-for-bit identical. Hoisting the *expression* is not enough; what matters
is how the value crosses the scan boundary.

**Step 3 confirmed the mechanism.** With every parameter passed as an explicit
`non_sequences` argument rather than closed over as `theta[i]`, the
`newton_inside` variant — which still writes the Newton solve inside the step
function — runs at 178 ms and 3.49×, *identical to the constant-`psi` control*.
That identity is the proof: the chain is now genuinely outside the loop, hoisted
by PyTensor's own optimiser.

### Implicit differentiation was considered and is not needed

`solve_psi`'s docstring anticipated that a freed `c_lf` "would need implicit
differentiation through `solve_psi` rather than a baked-in number", and a custom
`Op` supplying `dpsi/dtheta = -(df/dtheta)/(df/dpsi)` would indeed have removed
the taped chain — one derivative evaluation regardless of iteration count.

**It is not implemented, because it is not necessary.** An argument-passing
change achieves the same result exactly, with no custom `Op`, no hand-written
gradient to keep correct, and no new surface to test. The unrolled Newton is
kept: exact, differentiable, and now costing one solve per gradient rather than
5113.

### Why this needs a timing test

The regression is **bit-for-bit invisible**. Every correctness test in the suite
passed in the slow configuration, because the arithmetic was identical — only the
graph was wasteful. A future refactor reintroducing a closure would show up
nowhere except as a sampling run taking six days instead of half of one.

`tests/test_model.py::TestGradientPerformance` asserts the gradient stays under
400 ms at 5113 steps, marked `pytensor` so it stays out of the fast loop. The
budget is loose on purpose: ~180 ms measured, 20–30% machine variation, and
2,165 ms is what it exists to catch.

### Consequence for the schedule

159 hours was withdrawn as a schedule number on the strength of this. The
projection is now 12.7 h for one run and 25.3 h for the two-convention
sensitivity (§10, LIMITATIONS §10).

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
- Ilvesniemi, H., Levula, J., Ojansuu, R., Kolari, P., Kulmala, L., Pumpanen,
  J., Launiainen, S., Vesala, T. and Nikinmaa, E. (2009). Long-term measurements
  of the carbon balance of a boreal Scots pine dominated forest ecosystem.
  *Boreal Environment Research* 14(4), 731–753. **Fig. 6 verified against the
  full paper.** Source of every FI-Hyy carbon-balance number in §7.
- Tuomi, M., Thum, T., Järvinen, H., Fronzek, S., Berg, B., Harmon, M.,
  Trofymow, J. A., Sevanto, S. and Liski, J. (2009). Leaf litter decomposition —
  estimates of global variability based on Yasso07 model. *Ecological Modelling*
  220, 3362–3371. Table 1 AWEN rates; source of τ_lit.
- Kolari, P. (2010). *Dissertationes Forestales* 99, doi 10.14214/df.99.
  Source of the projected LAI convention and site value, the third Rh estimate,
  the spring/autumn asymmetry, the 2002 thinning and the Vaccinium site type.
- Fox, A. et al. (2009). The REFLEX project: comparing different algorithms and
  implementations for the inversion of a terrestrial ecosystem model against
  eddy covariance data. *Agricultural and Forest Meteorology* 149, 1597–1615.
  Table 4 `a1` range and §4.5. **Not read in full.**
- Carvalhais, N. et al. (2008). Implications of the carbon cycle steady state
  assumption for biogeochemical modeling performance and inverse parameter
  retrieval. *Global Biogeochemical Cycles* 22, GB2007. **See-also only, not read
  in full, and nothing in §9 depends on it.**
- Niinemets, U. et al. (2001). Total-to-projected leaf area ratio for Scots
  pine, 2.57. **Supplied, not read.**
- Grace, J. C. (1987). Theoretical ratio between one-sided and total surface
  area for pine needles. *NZ J For Sci* 17: 292–294. Bisected-cylinder
  geometry, (π+2)/2 = 2.5708. **Supplied, not read.**
- Chen, J.-M. and Black, T. A. (1992). Defining leaf area index for non-flat
  leaves. *Plant Cell Environ* 15: 421–429. Establishes hemisurface as the
  radiation standard. Abstract and secondary sources checked; **full paper not
  read.**
- Williams, M. et al. (1996). Modelling the soil-plant-atmosphere continuum in
  a Quercus-Acer stand at Harvard Forest. *Plant Cell Environ* 19, 911–927.
  SPA, the model ACM aggregates. **Not read; broadleaf parameterisation
  established from the Williams 1997 abstract.**
- Kolari, P. et al. (2009). CO2 exchange and component CO2 fluxes of a boreal
  Scots pine forest. *Boreal Environment Research* 14, 761–783. Paper V of the
  dissertation. **Verified against the full paper** for the chamber-based
  photosynthetic capacity, the LAI seasonal range and the 25% foliage turnover.
- Liski, J. and Westman, C. J. (1995). Cited via Kolari (2010) for the absence
  of soil carbon trend over a rotation. **Not read directly.**
- Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux
  processing pipeline. *Scientific Data* 7, 225.
