# Limitations register

Every known limitation, whether it can be solved, and what the literature says.
Keep this current — it becomes the thesis limitations chapter.

> **⚠ Citations in this file are UNVERIFIED.** Several came from abstracts,
> review comments or search snippets rather than from the full papers. Verify
> every one against the full text before it reaches the thesis. Nothing here has
> been checked by reading the source. Items dated 2026 are the highest risk.
>
> Measured numbers are a separate matter: those marked *measured* were produced
> by this repository and are reproducible from it.

---

## 1. ACM has no temperature limitation on photosynthesis at high latitude

**What it is.** Above the frost cutoff, modelled GPP moves about 2% across a
20-degree temperature change *(measured)*. The light-limitation branch dominates
and temperature drops out. At FI-Hyy this puts a temperature-independent floor
under GPP for much of the year.

**Solved in the literature?** Yes, twice over.

- *Acknowledged:* López-Blanco et al. (2019), *Earth System Dynamics* 10,
  233–255. On their pan-Arctic CARDAMOM analysis, they attribute low bias in LAI
  and biomass to the ACM photosynthesis prior, which they state lacks a
  temperature acclimation for high latitudes in that implementation. This is the
  model developers naming the problem in print.
- *Fixed:* "Resolving temperature limitation on spring productivity in an
  evergreen conifer forest using a model–data fusion framework," *Biogeosciences*
  19, 541 (2022). Explicitly states that baseline CARDAMOM limits seasonal GPP
  primarily by shortwave radiation, which works for deciduous forests but fails
  in evergreen forests exposed to high sunlight and below-freezing spring
  temperature. They add a temperature sensitivity. **Get the full citation.**

### 1a. What was withdrawn, and what stands

> **Withdrawn.** This entry previously reported a **2.00× structural ACM
> productivity bias** at the measured canopy, and before that a 2.3× GPP
> overestimate driven by leaf area. **Both are withdrawn.** They were measuring
> the DALEC2 `ceff` prior U(10, 100), whose median near 55 the site does not
> support, and they compared an annual-mean modelled LAI against a
> seasonal-maximum measured one. See `reports/prior_diagnostics/FINDINGS_gpp.md`
> for the full sequence.

**There is no detectable GPP magnitude bias.** With `ceff` on REFLEX's published
U(5, 20) and the annual-mean canopy comparison corrected, the residual runs
**0.90× to 1.19×** across defensible canopy bands, and zero bias is inside that
range. This conclusion is load-bearing on the `ceff` prior: on the DALEC2 range
the residual is 2.0–2.9×.

**The seasonal distribution defect stands, and did not respond to any prior
change.**

| | October / April GPP |
|---|---:|
| measured (Kolari 2010, EC) | **0.76** |
| cross-check, `gpp_nt` / `gpp_dt` | 0.705 / 0.747 |
| modelled, flat priors | 0.290 |
| modelled, canopy priors | 0.290 |
| **modelled, current** | **0.397** |
| **ratio** | **0.52×** |
| **draws below the measured ratio** | **100.0%** |

Correcting the GPP magnitude by a factor of three moved this from 0.38× to 0.52×
and left **every** draw below the measured value. Magnitude and seasonal
distribution are separate defects; only the first was ever a prior problem.

**The mechanism.** April and October have almost identical mean temperature in
this block — 3.61 °C against 4.02 °C, matching Kolari's 3.5 and 4.1 — but April
irradiance is **12.95** against October's **3.09**, a factor of 4.2. ACM is
radiation-driven and returns approximately the irradiance ratio of 0.24. The real
canopy achieves 0.76 on a quarter of the light because its photosynthetic
capacity is still summer-acclimated. **ACM is not missing a modest autumn
deficit; it is missing a large autumn enhancement.**

**Validation or consistency check?** Both, in different parts.

Kolari et al. (2009), paper V, derives the seasonal course of photosynthetic
capacity twice: once from a delayed temperature history, and once **"directly
from the measured shoot CO₂ exchange"** using automated 1 dm³ chambers closed
70–100 times a day, with **no eddy covariance involved**. So the
delayed-acclimation mechanism is established independently of the NEE signal
DALEC is calibrated against, and the model's failure to reproduce it is not
circular. **The mechanism is validated.**

The specific **0.76 is EC-derived**, and our cross-check against `gpp_nt` and
`gpp_dt` uses the same partitioned data, so it is not independent either. A fully
independent number would require October/April computed from the chamber-based
GPP series, published only as a figure. **The ratio is a consistency check.**
State it that way in the thesis.

**Our position.** Do not implement the fix. It adds parameters, which directly
worsens the identifiability question RQ3 asks about. Cite both papers, state the
limitation, quantify it with the shoulder-season diagnostic
(`diagnostics.shoulder_season_gpp`). Raise at supervisor meeting.

---

## 2. ACM coefficients are site-calibrated and neither published set is boreal

**What it is.** Chuter et al. (2015) give two coefficient sets — Loobos (52 N,
Netherlands, evergreen) and Oregon ponderosa pine. They differ substantially. We
adopt Loobos as the closest match by forest type and latitude. FI-Hyy is at
61.85 N.

**Solved?** Not directly. Boreal CARDAMOM analyses exist (López-Blanco et al.
2019; pan-Arctic and permafrost CARDAMOM datasets at
datashare.ed.ac.uk/handle/10283/864) but coefficient sets are not published in a
form we can lift.

**Our position.** State the substitution and its justification. Sensitivity to
the choice is tested by re-running the gate with the Oregon set — see §2a.

### 2a. Measured sensitivity to the coefficient set

*(measured)*

Both sets, same drivers, same true daily extremes, LAI re-solved to 3 at every
`ceff`. Annual modelled GPP ÷ the mean of the two partitioned products:

| `ceff` | Loobos | Oregon |
|---:|---:|---:|
| 10.0 | 0.919 | 1.372 |
| 11.5 | **1.013** | 1.434 |
| 15.0 | 1.206 | 1.540 |
| 29.6 | 1.745 | 1.724 |
| 100.0 | 2.475 | 1.851 |

**The headline conclusion survives: both sets pass the gate**, so "a `ceff`
inside the published 10–100 prior brings annual GPP to the measured value with a
stable canopy" is not an artefact of choosing Loobos. That is the robustness
claim worth reporting.

**But the two sets are not interchangeable, in three ways worth stating.**

1. *Where they land in the prior.* Loobos reaches ratio 1.0 near `ceff` 11.4,
   comfortably inside 10–100. Oregon's ratio is monotonically increasing and
   already 1.372 at the bottom of the prior, so its best achievable is 1.305 at
   `ceff` = 10 — it passes only because the tolerance is a factor of 1.5, and it
   cannot reach 1.0 anywhere in the range. Oregon is the flatter response:
   tenfold in `ceff` moves it 1.37 → 1.85, against Loobos's 0.92 → 2.48.
2. *They disagree about the peak.* Oregon's peak daily GPP is 11.36 g C m⁻² d⁻¹
   at the 99th percentile against measured 10.43 (NT) and 11.52 (DT) — closer
   than Loobos's 8.49. So the set that is worse on the annual total is better on
   the peak. The two sets trade off total against shape, and neither gets both.
3. *The over-long season is not a coefficient artefact.* Season length is 207
   days under Loobos and 200 under Oregon, against 166–167 measured. **It
   survives the substitution**, which is evidence against attributing §11 to the
   ACM coefficients and in favour of the phenology parameters or DALEC's
   structure. Not proof — Morris screening is still what settles it — but it
   removes one candidate.

**A caveat on `best_ceff`.** The gate's own `best_ceff` moved from 11.7 to 13.6
between two runs of the same configuration, because the gate sweeps `ceff` while
holding the foliage allocation at whatever the caller's parameter set implies, so
the answer depends on the `ceff` the caller happened to solve that share at. The
LAI-re-solved table above is the trustworthy one; treat a single `best_ceff` from
the gate as indicative only. This is the same design limitation recorded in
[DECISIONS.md](DECISIONS.md) — the gate reports canopy density precisely because
the ratio alone is not self-interpreting.

---

## 3. DALEC2 is applied outside its stated scope

**What it is.** Bloom & Williams (2015) Section 2.5 selected sites with no more
than three months of below-freezing soil temperature, because hydrological
processes are not explicitly represented in DALEC2. FI-Hyy has four to six months
of soil frost.

**Solved?** Partially, in later DALEC versions. Yang et al. (2022),
*Geoscientific Model Development* 15, 1789–1802, introduced plant available and
unavailable water concepts into the DALEC family (CARDAMOM-FluxVal). That is a
different model version from the one we implement.

**Our position.** Deliberate, stated scope decision. Frame as "we test DALEC2
beyond its stated envelope and quantify the structural consequences," not as an
oversight.

---

## 4. NEE alone cannot constrain wood and soil turnover — this is already known

**What it is.** The slow pools receive almost no information from a fast flux
observation.

**Solved?** Not solved — *established*. This is settled literature and must not
be presented as a new finding:

- Fox et al. (2009), *Agric. For. Meteorol.* 149, 1597–1615 — little constraint
  on wood allocation and turnover from fast observations, regardless of
  algorithm.
- Williams et al. (2005), *Glob. Change Biol.* 11, 89–105 — identified the need
  for long-term woody biomass measurements.
- Williams et al. (2009), *Biogeosciences* 6, 1341–1359.
- Smallman et al. (2017), *JGR Biogeosciences* — repeated woody biomass
  observations constrain decadal uncertainty in aggrading forests.
- Smallman et al. (2021), *Earth System Dynamics* 12, 1191–1237 — quantified the
  covariance structure between wood dynamics and fast/slow observables.

In a May 2026 peer review, T. L. Smallman writes of exactly this finding: "This
is not 'Notable', it is the expected result."

**Our position.** RQ3 must not be answered as "NEE cannot constrain the slow
pools." Reframe around the *structure* of the unidentifiability:

- Chuter et al. (2015) proves analytically that `p2` and `p3` are recoverable
  only as the product `p3(1 - p2)`, and that the foliar pool decouples from the
  others in DALEC EV. Combine that proof with the empirical posterior to show
  *which* combinations the data informs.
- NUTS divergences and effective sample size are geometric diagnostics
  unavailable to random-walk samplers. Report where the geometry is pathological
  and why.
- Report the Morris screening result as a prior statement of what cannot be
  informed, then confirm against the posterior.

---

## 5. Single site, single data stream, no biometric validation

**What it is.** One site, NEE only, no independent wood or soil measurements.

**Solved?** Yes, by adding data streams — Richardson et al. (2010) added soil
respiration, litterfall and biomass; Smallman et al. (2017) added repeated
biomass. Not available to us in this scope.

**Our position.** This is the premise of RQ3, not an accident. State it as a
deliberate design choice with a named cost.

---

## 6. Structural model error is absorbed into parameters

**What it is.** A well-constrained posterior does not mean a correct parameter.
Structural error in the forward model gets compensated by parameter values that
fit the data.

**Solved?** Partially. Gaussian-process discrepancy terms and explicit
model-error terms exist — Safta et al. (2015) used a statistical model error term
with DALEC. Adds complexity we cannot absorb in this timeline.

**Our position.** The synthetic twin experiment is the mitigation. Synthetic data
generated by the same flawed forward model will recover parameters normally, so
any gap between synthetic and real recovery isolates structural error. Make that
comparison explicit in the results.

---

## 7. Fixed parameters across a growing stand

**What it is.** The FI-Hyy stand was sown in 1962; across 1996–2014 it goes from
age 34 to 52. We assume one time-invariant parameter set.

**Solved?** Not simply. Time-varying parameters multiply the dimension and worsen
identifiability.

**Our position.** State it.

> **⚠ The mitigation as originally drafted does not apply.** The register's note
> that "the four-year calibration block limits the exposure" is out of date: the
> calibration block is **nine years**, 1997–2005, over which the stand ages from
> 35 to 43, and the evaluation block carries it to 47. Exposure to this
> limitation is therefore larger than the note assumed, not smaller. Either
> argue the mitigation from the nine-year block honestly, or revisit the block
> length — but the four-year framing cannot be used.

---

## 8. Observation error model

**What it is.** Gaussian likelihood with per-day sigma from
`NEE_VUT_REF_RANDUNC`. Two known issues: `RANDUNC` captures only random
measurement error, not structural error; and flux errors are known to be
non-Gaussian.

**Solved?** Documented, not solved here.

- Hollinger & Richardson (2005), *Tree Physiology* 25, 873–885 — flux measurement
  error follows a double exponential more closely than a normal distribution.
- Richardson & Hollinger (2005), *Agric. For. Meteorol.* 131, 191–208 — given the
  stochastic uncertainty in nighttime flux measurements, fitting respiration
  models by ordinary least squares is incorrect.

**Our position.** Stated limitation. The heavier-tailed comparison was scoped out
of this thesis.

---

## 9. Day/night temperature proxy (RESOLVED)

**What it was.** `TA_F_DAY` and `TA_F_NIGHT` substituting for daily maximum and
minimum. Measured against true extremes from the half-hourly file *(measured)*:
`t_max` and `t_min` each correlate at 0.989, but the *range* only at 0.641,
because the biases run in opposite directions (−2.51 and +2.28 °C) and compound
in the difference. True range averages 6.34 °C against a proxy 1.55 —
understated fourfold across the whole record.

**Resolved** by deriving true daily extremes from the FULLSET HH product
(`scripts/01b_derive_tminmax.py`). Keep the comparison table as a methods result;
it is in [DECISIONS.md](DECISIONS.md) §3.

---

## 10. Computational feasibility — now MEASURED, and it is the schedule risk

**Measured 2026-08-28** on the real model at the real problem size, replacing the
`timing_spike.py` estimate. `scripts/18_model_equivalence_and_timing.py`, 5113
steps, 20 runs after 3 warmups, ARM64, NumbaLinker.

| | median | min | max |
|---|---:|---:|---:|
| forward pass | **50.7 ms** | 46.3 | 59.8 |
| gradient | **2,239.6 ms** | 2,078.8 | 2,812.6 |

**The gradient costs 44× the forward pass.** Reverse-mode AD normally costs 2–5×,
so this is not the expected overhead: the scan's backward pass is far heavier
than its forward one under Numba. That ratio, not the absolute time, is the thing
worth attacking if this needs to be faster.

**Projection: ~159 hours of gradient evaluation** for 4 chains × (1000 tune +
1000 draws) at a nominal 32 leapfrog steps per iteration. Treat it as an order of
magnitude — the step count adapts, and this machine's timings have been unstable
across runs by 20–30%.

**This is a schedule risk against a 31 December 2026 submission**, and it is now
a number rather than an unknown. Options, none taken here: fewer chains, a
shorter calibration block (which trades against slow-pool identifiability, §1a
and DECISIONS §11), cluster time, or attacking the 44× ratio directly.

**One thing checked and found not to matter.** The first timing ran on PyTensor's
default `linker="auto"`, which DECISIONS records as unsafe because it silently
resolves differently per machine. Re-run through `dalec.compute.compile_function`,
which pins Numba, the gradient came out at 2,240 ms against 2,496 ms — the same
number within this machine's run-to-run spread, because `auto` was already
resolving to `NumbaLinker` here (no C++ compiler present). The pin is still
correct and the script now uses it; it just was not the explanation.

---

## 10b. The original untested estimate

**What it is.** NUTS must differentiate through a PyTensor `scan` of roughly 1460
daily steps with 12–23 free parameters. This has not been timed.

**Solved?** Precedent is thin. DifferLand uses JAX with hardware acceleration for
exactly this reason. Dhulipala, Che & Shields (2023), *J. Computational Physics*,
use latent Hamiltonian neural networks to cut gradient evaluations by one to two
orders of magnitude — a fallback if cost is prohibitive.

**Our position.** Time it early on a short slice. This is the single largest
threat to finishing by December, and the mitigations (fewer parameters, shorter
calibration block) all require knowing the number first.

> **Note on the step count.** 1460 steps is a four-year block. The chosen
> calibration block is nine years, 3287 days, so the scan is **2.25× longer**
> than this estimate assumes. Time it at 3287 steps, not 1460.

---

## 11. The modelled growing season is about six weeks too long (OPEN)

*(measured)*

**What it is.** At `ceff` 11.7 with LAI near 3, the seasonal *peak* lands on the
observed day exactly (doy 185), but onset is at doy 79 against 110–111 for the two
partitioned products and cessation at 287 against 277 — a 208-day season against
166–167. Peak daily GPP is 7.56 g C m⁻² d⁻¹ at the 99th percentile against 10.43
(NT) and 11.52 (DT). The annual total is correct, and the magnitude gate passes at
1.032, because the model spreads the right amount of carbon over a longer,
flatter season.

**Not yet attributed.** `d_onset`, `cr_onset` and `cr_fall` were hand-set for that
run rather than fitted, and `d_fall` is inert as transcribed
([DECISIONS.md](DECISIONS.md) §2, correction 2). It may be ACM, it may be the
phenology parameters, and Morris screening is what separates them. **Do not
attribute it to ACM before then.**

One candidate is already eliminated: the over-long season survives substituting
the Oregon coefficient set for Loobos — 200 days against 207, both far from the
measured 166–167 — so it is not an artefact of the ACM coefficients (§2a).

Caught by `diagnostics.seasonal_timing`, which exists because an annual total is
an integral and an integral is nearly blind to phase — the magnitude gate cannot
see this failure mode at all.

---

## 12. 2014 has no random-uncertainty estimate, and it is a product defect

*(measured)*

**Scope of this claim.** It concerns **FI-Hyy in the FLUXNET2015 FULLSET product,
release `1-4`** — the daily and half-hourly files in `data/raw/`. It is *not* a
claim about ONEFlux in general, about other sites, or about other releases. We
have checked one site in one release and nothing else.

**What it is.** `NEE_VUT_REF_RANDUNC` is absent for all 365 days of 2014, so no
day of 2014 can enter the Gaussian likelihood, which takes its per-day sigma from
that column. 2014 is otherwise one of the cleanest years in the record: 365/365
valid `NEE_VUT_REF`, QC median 1.000, 333 days at QC = 1.0, 356 days passing
QC ≥ 0.75, and a physically ordinary range of −6.10 to 2.90 g C m⁻² d⁻¹.

**It is not a data-quality problem.** Three independent lines of evidence:

1. **Fourteen independent uncertainty columns stop on the same day.** Every
   flux `*_RANDUNC` column in the daily file — VUT and CUT, REF and USTAR50,
   NIGHT and DAY, and `LE_RANDUNC` and `H_RANDUNC`, which are latent and sensible
   heat rather than carbon — has its last valid value on **2013-10-14** (two on
   10-13). Meanwhile `NEE_VUT_REF`, `GPP_NT_VUT_REF`, `RECO_NT_VUT_REF`,
   `LE_F_MDS`, `H_F_MDS`, `TA_F` and `SW_IN_F` all run complete to 2014-12-31. A
   simultaneous stop across physically unrelated variables, while the variables
   they derive from continue, is a pipeline event and not a sensor one.

2. **The pipeline records zero inputs, not missing inputs.**
   `NIGHT_RANDUNC_N` and `DAY_RANDUNC_N` — the counts of records entering the
   uncertainty estimate — are **present for all 365 days of 2014 and equal to
   0.0** throughout. The stage ran and reported that nothing reached it.

3. **The measurements it would have used are there, in normal quantity.** The
   daily QC flag cannot settle this, because it counts "measured *or good-quality
   gap-fill*". The half-hourly flag distinguishes them (`0` = measured), and by
   that measure 2014 is unremarkable — **8385 measured half-hours, 47.9%**,
   against **7135 (40.6%) in 2012** and 6362 (36.3%) in 2010, running through to
   2014-12-31. 2014 carries *more* genuinely measured data than 2012, a year
   whose RANDUNC is 99.7% complete.

**The failure is upstream of daily aggregation.** In the FULLSET **half-hourly**
file the uncertainty columns stop at the same point — `NEE_VUT_REF_RANDUNC` has
its last valid value at **2013-10-14 18:00**, with `LE_RANDUNC` at 22:00 and
`H_RANDUNC` at 23:30 the same day, and the `_N` and `_METHOD` companions ending
on the identical half-hour — so half-hourly uncertainty estimation itself
stopped, and the daily product is faithfully aggregating an input that is not
there rather than losing it in the aggregation step.

**Consequence for this thesis.** 2014 contributes **0 assimilable days**. It sits
in the prediction block 2011–2014, which is 810 of 1461 days assimilable (55.4%);
the forward run integrates 2014 normally, but none of it can be scored against
observed NEE. The loss is a limitation of the published product for this site and
release, not of the site or of the measurements.

**Not investigated, deliberately.** Whether a later FLUXNET release or an ICOS
reprocessing supplies the missing uncertainties is a question for supervision,
not something to resolve by trying files.

---

## 13. 1996 winter NEE is gap-fill, and it reads as sustained midwinter uptake

*(measured)*

**Scope.** FI-Hyy in the FLUXNET2015 FULLSET daily product, release `1-4`, as for
§12. Not a claim about other sites, other years, or ONEFlux in general.

**What it is.** The first **163 consecutive days of 1996 — doy 1 to 163,
1996-01-01 to 1996-06-11 — carry `NEE_VUT_REF_QC = 0.0`**: not one measured or
good-quality gap-filled half-hour in nearly six months. `NEE_VUT_REF` is
nevertheless populated for every one of those days, by gap-filling, and it is
populated with a **repeating three-value cycle**: −5.38, −4.66 and −5.11
g C m⁻² d⁻¹, each appearing exactly 31 times, covering 93 of the 163 days. Mean
NEE across the QC = 0 block is **−4.26 g C m⁻² d⁻¹**.

Read at face value that is five months of vigorous, unbroken net carbon uptake
through a boreal midwinter, at a site whose January air temperature climatology
sits near −8 °C. It is fill, not flux.

**Why it matters beyond simply being wrong.** Unscreened, it moves a result. The
spring-onset estimate in `eda02` — the first ten consecutive days of net uptake —
is dragged to **doy 46** for 1996, against a median of doy 96 across the other
eighteen years. Any analysis that reads `NEE_VUT_REF` without checking
`NEE_VUT_REF_QC` will inherit this, and the failure is silent: the column is
fully populated, finite, and of plausible magnitude.

**1996 cannot date its own spring.** It has **zero** QC-passing days between
doy 60 and 150 — 0 of 91 — and its first day at QC ≥ 0.75 is doy 165. So its
apparent onset is not a late spring but the first moment real data resumes. It is
excluded outright from the onset panel, and the exclusion is stated in the panel
title rather than left to the reader.

**This is independent of, and additional to, the CO₂ driver gap.** §1 of
[DECISIONS.md](DECISIONS.md) excludes 1996 from the calibration block because
`CO2_F_MDS` is missing for doy 1–19. That is a *driver* problem, it spans 19 days,
and it would in principle be repairable by backfilling. This is an *observation*
problem, it spans 163 days, and backfilling is exactly what caused it. Either one
alone would justify excluding 1996; they are not the same finding and neither
substitutes for the other.

**Our position.** 1996 stays excluded from calibration, now on two independent
grounds. More generally: **screen on `NEE_VUT_REF_QC` before using
`NEE_VUT_REF` for anything**, including exploratory work, because the gap-filled
values are neither missing nor obviously wrong. `dalec.data_io._likelihood_mask`
already enforces this for the likelihood; the exposure is in ad-hoc analysis that
reads the raw column directly.

**Not investigated.** Whether the same pattern appears in other years, at other
sites, or in other releases. §12's cutoff and this one are both artefacts of the
same product and release, but nothing here establishes a common cause.

---

## 14. The stand was thinned in 2002, inside the calibration window

**What it is.** Kolari (2010) records a thinning at SMEAR II in early 2002 that
removed about **19% of the leaf area** — all-sided seasonal maximum LAI falls from
~8.0 to ~6.5, projected 3.2 to 2.6. The calibration block is 1997–2010, so the
event sits five years into fourteen.

**Why it matters.** DALEC has no management event. Carbon leaves the foliar pool
only through the phenology term, and there is no mechanism by which a fifth of
the canopy disappears in one winter. The model must therefore represent a stand
whose canopy stepped down partway through as a single stationary canopy.

**What the model does instead.** The modelled foliar trajectory shows nothing at
2002 — measured, not assumed; see `scripts/12_acm_asymmetry.py`. The step is
absorbed into whatever stationary canopy best fits the whole block, so the
posterior will describe a compromise between the pre- and post-thinning stand
rather than either.

**Consequence for the inference.** The calibration and prediction blocks straddle
different canopies: 1997–2001 at projected LAI ~3.2, 2002 onward at ~2.6. Since
the prediction block (2011–2014) is entirely post-thinning while the calibration
block is 64% post-thinning, the held-out forward run is being asked to predict a
canopy the calibration only partly saw. This is a **structural** contribution to
prediction error, not a parameter one.

**Not fixed.** Adding a management event means a discontinuity in the state, which
breaks the smoothness NUTS needs. The alternative — restricting calibration to
2002–2010 — costs five of fourteen years and is a live option, but the year blocks
are locked (DECISIONS §1) and this is a supervision question.

---

## 15. ACM's leaf-area convention cannot be recovered from its own provenance

**What it is.** ACM needs a leaf area index. For a conifer canopy there are three
defensible bases — projected, hemisurface (half total), and total — and **ACM's
own derivation cannot say which it expects.**

ACM (Williams et al. 1997) was not fitted to observations. It aggregates SPA
(Williams et al. 1996), parametrised for a **Quercus–Acer deciduous broadleaf
stand at Harvard Forest**. For flat leaves projected, hemisurface and half-total
are **identical**, so **no needle geometry entered either calibration**. Applying
ACM to a conifer canopy therefore requires an area convention that its own
provenance cannot supply.

**The size of it.** The two defensible readings of Kolari's all-sided LAI:

| basis | divisor | `lma`, g C m⁻² | median GPP | residual | zero bias in IQR |
|---|---:|---|---:|---:|---|
| hemisurface | 2.000 (exact, by definition) | 116–192 | 1,061 | **1.03×** | **YES** |
| projected | 2.571 (Scots pine) | 148–247 | 683 | **0.66×** | **NO** |

They differ by **37% in GPP** and **disagree on whether the model has a magnitude
bias at all**. This is not a rounding question; it decides a headline result.

Ratio sources: total/projected = 2.57 for Scots pine (Niinemets et al. 2001),
matching bisected-cylinder geometry (π+2)/2 = 2.5708 (Grace 1987);
total/hemisurface = 2 by definition, with hemisurface established as the
radiation-work standard by Chen & Black (1992).

**The citation hunt is closed, and this is why.** Williams et al. (1997) Table 1
was previously named here as the thing that would settle it. **It cannot.** A
flat-leaf calibration carries no needle basis *whatever LAI range it spans* — the
table would tell us the magnitudes tested, not which convention to use for a
geometry that never appeared in the fitting data. No document can supply an
answer the underlying calibration never contained. Chasing further citations here
is wasted effort.

**Our position — both conventions stay live.** Neither is adopted. **The
calibration will be run under both and the difference reported as a
sensitivity.** That is the honest treatment of an ambiguity that is structural
rather than empirical: it cannot be resolved by looking harder, so it is carried
through to the results and shown.

The hemisurface reading has the better physical argument — ACM's radiation scheme
was fitted where intercepting area per unit LAI is one side of a flat leaf, and
carrying it to needles unchanged requires the basis that preserves intercepting
area. **It is also the reading that closes the GPP residual, which is grounds for
scrutiny rather than confidence.** Both facts are stated wherever the result is.

**Consequence for the thesis.** Any statement about GPP magnitude bias must be
conditioned on the convention. "No detectable magnitude bias" is true on the
hemisurface reading and false on the projected one, where the model
under-produces by 34%.

---

## Novelty claim — narrowed

Do not claim gradients through DALEC are new. They are not:

- Delahaies, Roulstone & Nichols (2017), *GMD* 10, 2635–2650 — 4D-Var with an
  adjoint.
- Wu et al. (2026), EGUsphere preprint 2026-2241 — DifferLand, a JAX-based
  differentiable TBM with a differentiable DALEC (Fang & Gentine, 2024).

Both use gradients for **optimisation**. Neither samples a posterior — a reviewer
of the latter notes that its five-member ensemble is not a robust
characterisation of uncertainty.

Defensible claim: *gradient-based posterior sampling (HMC/NUTS) has not been
applied to DALEC.* Supported by a Scopus search of
`DALEC AND (Hamiltonian OR NUTS OR "No-U-Turn" OR PyMC)` returning zero documents
on 10 August 2026, and a companion search on gradient terms returning nine, none
of which sample posteriors.

> **⚠ A null search result decays.** Date-stamp the search in the thesis, and
> re-run it before submission. State the database, the exact query string and the
> date, so the claim is reproducible rather than asserted.
