# GPP after the REFLEX `ceff` prior and derived initial pools

`scripts/11_gpp_investigation.py`, 400 draws, seed 20260809, block 1997–2010.
Measured: Ilvesniemi et al. (2009) Fig. 6, GPP (EC) **952–1104** g C m⁻² yr⁻¹.
Nothing is fitted to it.

## Result

| | flat priors | canopy priors | **+ REFLEX `ceff`, derived pools** | measured |
|---|---:|---:|---:|---:|
| median annual GPP | 2,570 | 2,972 | **956** | 952–1,104 |
| ratio to midpoint | 2.50× | 2.89× | **0.93×** | — |
| inside the measured range | 0.8% | 0.3% | **7.2%** | — |
| above it | 68.3% | 97.7% | **43.2%** | — |
| mean LAI, median | 5.09 | 7.49 | **2.48** | ~2.1–2.3 |
| peak LAI, median | — | 11.14 | **3.70** | 2.4–3.2 |

The draws that land inside the measured GPP range are physically ordinary:
`ceff` 14.2, `lma` 223, mean LAI 2.41, `f_auto` 0.49.

## The structural error estimate

This is the number for the thesis. GPP at the measured canopy:

| band | n | median GPP | ratio |
|---|---:|---:|---:|
| **mean LAI 2.8–3.6 (as specified)** | 44 | **1,219** | **1.19×** |
| mean LAI 2.4–3.2 | 53 | 1,137 | 1.11× |
| mean LAI 2.1–2.6 | 40 | 921 | 0.90× |

**Before this change the same 2.8–3.6 band gave 2,056, a ratio of 2.00×.**

**Take the structural error as ≈ 1.2×, and treat its sign as undetermined.**
Across defensible canopy bands the residual runs from 0.90× to 1.19×, so zero is
inside the range. What can be said firmly is that it is no longer a factor of
two.

### A correction to how that band should be read

Kolari's 3.2 and 2.6 are **seasonal maxima**, while this diagnostic's `lai_mean`
is an annual mean — comparing them directly was an error in the earlier write-up.
Kolari et al. (2009) give the seasonal *minimum* all-sided LAI as 4.5–4.9,
projected 1.8–2.0, against maxima of 6.0–6.5 all-sided, projected 2.4–2.6. The
annual **mean** projected LAI is therefore about **2.1–2.3**.

On that basis the modelled mean LAI of 2.48 is slightly high but close, and the
2.1–2.6 row above is the more appropriate comparison — which is where the
residual reads 0.90×. The specified 2.8–3.6 band is retained as the headline
because it is what was asked for, but it sits above the site's actual mean
canopy.

## `ceff`

`REFLEX_CEFF_BOUNDS` U(5, 20), Fox et al. (2009) Table 4, for the same ACM
parameter `a1`. **This is adopting a published prior in place of a wider
published prior.** It was not selected by looking at modelled GPP. Note it is not
a narrowing: REFLEX's range extends *below* DALEC2's U(10, 100), and the lower
half of it is where the measured GPP is reached.

| `ceff` quintile | median GPP | inside measured |
|---|---:|---:|
| 5.1 – 8.4 | 369 | 5.0% |
| 8.4 – 11.4 | 703 | 6.2% |
| **11.4 – 14.8** | **972** | **10.0%** |
| 14.8 – 17.5 | 1,287 | 7.5% |
| 17.5 – 20.0 | 1,748 | 7.5% |

The measured range is reached at `ceff` ≈ 11–15, which is inside REFLEX's prior
and near the bottom of DALEC2's. Consistent with the GPP magnitude gate passing
at `ceff` ≈ 11.7.

## Initial pools: all six now derived

None is left on a generic published range.

| pool | derivation | median | independent check |
|---|---|---:|---|
| `c_lab_0` | litterfall · f_lab/(f_lab+f_fol) | 77 | — |
| `c_fol_0` | litterfall / `c_lf` | 579 | — |
| `c_roo_0` | 90 / (`theta_roo`·365.25) | 49 | measured 238 — **0.20×** |
| `c_woo_0` | 6800 − the other three | 6,063 | — |
| `c_lit_0` | (1−`f_som`)·`rh_ref`/`theta_lit` | 119 | — |
| `c_som_0` | `f_som`·`rh_ref`/`theta_som` | 5,810 | VT_SP 5,730 ± 710 |

**`c_roo_0` is the weak one.** Deriving it from `theta_roo` only moves the
problem: the published `theta_roo` U(10⁻⁴, 10⁻²) spans fine-root residence times
of 0.27 to 27 years, and its median puts the stock at 49 g C m⁻² against a
measured 238. The site's own numbers pin it — 90 g C m⁻² yr⁻¹ through 238
g C m⁻² is **`theta_roo` = 1.04×10⁻³ d⁻¹, a 2.6 year residence time**. Narrowing
`theta_roo` to that is the obvious next step and was outside this scope.

## The `c_fol_0` trajectory is still not stationary

**Asked directly, and the answer is no.**

| year | median `c_fol` |
|---|---:|
| 1997 | 562 |
| 2000 | 448 |
| 2002 | 371 |
| 2006 | 251 |
| 2010 | **177** |

It falls monotonically by **69%** over the block — in relative terms *worse* than
the 30% fall (1,497 → 1,041) it showed on the published prior. Fourteen years is
still not long enough, and the run is still descending at 2010.

**The interpretation is different now, and better.** The initial value 579 is
correct: it reproduces the measured litterfall exactly, `c_lf · c_fol_0` = 154 by
construction, and that is asserted in the tests. What the trajectory shows is
that the *model's own fixed point* under most prior draws lies below the measured
canopy — with `ceff` now spanning 5–20, a large share of draws produce GPP too
low to sustain 579 g C m⁻² of foliage. The drift is no longer an initial-state
error; it is the GPP–LAI fixed point disagreeing with the measurement.

So the initial-state work did what it should — the start is now right and the
remaining drift is diagnostic of something else — but the stated goal of a
stationary trajectory is **not** met.

## The seasonal asymmetry is not fixed, and was not expected to be

| | October / April GPP |
|---|---:|
| measured | 0.76 |
| modelled, before | 0.290 |
| **modelled, after** | **0.397** |
| ratio | **0.52×** |
| draws below measured | **100.0%** |

Correcting the magnitude improved the seasonal ratio from 0.38× to 0.52× but did
not close it, and now *every* draw falls below the measured value. This is a
distribution error, not a magnitude error: ACM tracks radiation, and April
irradiance is 12.95 against October's 3.09.

**The magnitude problem is largely solved; the seasonal one is not.** They are
separate defects and only the first responded to the priors.

## Item 3 — is the seasonal test validation or a consistency check?

**Partly validation. The mechanism is independent of eddy covariance; the
specific 0.76 is not.**

Kolari et al. (2009), *Boreal Env. Res.* 14:761–783 — paper V of the
dissertation — measures shoot CO₂ exchange with automated chambers (1 dm³, closed
70–100 times a day) and derives the seasonal course of photosynthetic capacity
β **twice**, by two independent routes:

1. predicted from a delayed temperature history `S`, using parameters from Kolari
   et al. (2007) — paper II, the acclimation paper;
2. **"estimation of daily β directly from the measured shoot CO₂ exchange"** —
   chamber-based, with no eddy covariance involved.

The paper then compares three GPP series in its Fig. 6: eddy covariance,
chamber-based upscaling, and the SPP prediction. It states plainly that "in boreal
evergreen conifers the seasonal cycle of photosynthetic capacity can be described
accurately as a delayed response to temperature" (Pelkonen & Hari 1980; Mäkelä
et al. 2004).

**What this establishes.** The delayed-acclimation mechanism rests on shoot-level
chamber gas exchange, which is independent of the NEE signal DALEC is calibrated
against. So the 0.52× result is not circular: the model is failing to reproduce
behaviour that has been measured independently at the shoot.

**What it does not establish.** The specific 0.76 ratio is EC-derived, and the
cross-check against `gpp_nt`/`gpp_dt` (0.705, 0.747) uses the same
EC-partitioned data, so it is not independent either. A fully independent test
would compute October/April from the chamber-based GPP series, which is published
only as a figure. **The mechanism is validated; the number is a consistency
check.** State it that way in the thesis.

## Item 4 — a pre-sampling prediction for RQ3

**Recorded before sampling so the posterior tests it rather than rationalises
it.**

If ACM over-produces GPP by a factor `k` and the sampler must reproduce an
observed NEE of about −206 g C m⁻² yr⁻¹, then respiration has to absorb the
excess: the posterior will inflate Reco to `GPP − 206` rather than correct GPP.
Both quantities are independently measurable from the chamber data, so this is a
falsifiable prediction and not a hedge.

| premise | predicted posterior GPP | predicted posterior Reco |
|---|---:|---:|
| `k` = 2.00 (as of the previous run) | ~2,060 | ~1,850 |
| **`k` = 1.19 (current, after `ceff`)** | **~1,225** | **~1,020** |
| measured | 1,030 | 850 |

**The premise moved inside this session**, which is why both rows are recorded.
Item 4 was written when the residual was 2.00×; adopting the REFLEX prior cut it
to about 1.19×, so the second row is the live prediction and the first is kept to
show what the `ceff` change bought.

**What would falsify it.** A posterior with GPP near 1,030 and Reco near 850 —
that is, one that fits NEE without inflating either gross flux. **What would
confirm it.** Posterior GPP and Reco both biased high by roughly the same
absolute amount, leaving NEE right for the wrong reasons. The second is the
classic compensating-error signature and is exactly what RQ3 asks about.

A secondary prediction, from the seasonal result: even a posterior with the right
annual totals should show a **systematic autumn deficit and spring surplus**,
because the 0.52× seasonal ratio cannot be fixed by any parameter in the model.
Check the posterior predictive residuals by month.

## Open

1. **`theta_roo` should be narrowed** to the measured 1.04×10⁻³ d⁻¹ so `c_roo_0`
   stops inheriting a 100-fold range.
2. **The trajectory is still not stationary.** Now attributable to the GPP–LAI
   fixed point rather than the initial state.
3. **The seasonal asymmetry is untouched** and belongs to LIMITATIONS §1/§1a.
4. **Task 2 must be re-run** against these priors; the existing ranking was
   computed against the superseded ones.
