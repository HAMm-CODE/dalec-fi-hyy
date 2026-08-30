# GPP: two claims withdrawn, one finding standing

`scripts/11_gpp_investigation.py`, 400 draws, seed 20260809, block 1997–2010.
Measured: Ilvesniemi et al. (2009) Fig. 6, GPP (EC) **952–1104** g C m⁻² yr⁻¹,
verified against the paper.

> **Withdrawn.** Earlier versions of this file claimed a **2.3× GPP overestimate
> driven by excess leaf area**, and later a **2.00× structural ACM bias at the
> measured canopy**. LIMITATIONS §1 carried an entry built on the second. **Both
> are withdrawn.** The sequence that retired them is set out below, because the
> numbers were quoted before they were stable and the record should show why.

## What was claimed, and what changed

| stage | median GPP | claim made | status |
|---|---:|---|---|
| 1. flat priors | 2,570 | "2.3× overestimate, driven by leaf area" | **withdrawn** |
| 2. canopy priors | 2,972 | "2.00× structural ACM bias at LAI 2.8–3.6" | **withdrawn** |
| 3. + REFLEX `ceff`, derived pools | **956** | magnitude bias not detectable | current |

Two things changed between stage 2 and stage 3.

**`ceff` moved to REFLEX's published U(5, 20).** Stages 1 and 2 used the DALEC2
Table 1 prior U(10, 100), whose median near 55 is far above anything the site
supports. With `ceff` on the REFLEX range the measured GPP is reached at
`ceff` ≈ 11–15. Both withdrawn claims were therefore measuring the DALEC2 `ceff`
prior, not ACM.

**The canopy comparison was wrong on its own terms.** Kolari's projected LAI of
3.2 and 2.6 are **seasonal maxima**; the diagnostic's `lai_mean` is an **annual
mean**. Comparing them directly overstated the target canopy by roughly 40%.
Kolari et al. (2009) give the seasonal *minimum* all-sided LAI as 4.5–4.9 against
maxima of 6.0–6.5, so the annual mean projected LAI is about **2.1–2.3**, not
3.2. Stage 2's "2.00× at LAI 2.8–3.6" was evaluated at a canopy denser than the
site's.

## Current position: the magnitude bias is not detectable

| | value | measured |
|---|---:|---:|
| median annual GPP | **956** | 952–1,104 |
| ratio to midpoint | **0.93×** | — |
| draws inside the measured range | 7.2% | — |
| mean LAI, median | 2.48 | ~2.1–2.3 |
| peak LAI, median | 3.70 | 2.4–3.2 |

GPP at the measured canopy, across defensible bands:

| band | n | median GPP | residual |
|---|---:|---:|---:|
| mean LAI 2.8–3.6 (stage 2's band) | 44 | 1,219 | **1.19×** |
| mean LAI 2.4–3.2 | 53 | 1,137 | 1.11× |
| mean LAI 2.1–2.6 (annual-mean basis) | 40 | 921 | **0.90×** |

**The residual runs 0.90× to 1.19× and zero bias is inside that range.** There is
no detectable GPP magnitude bias. It is not a factor of two, and its sign cannot
be determined from these data.

Draws landing inside the measured GPP range are physically ordinary: `ceff` 14.2,
`lma` 223, mean LAI 2.41, `f_auto` 0.49.

## The `ceff` prior is load-bearing

**The magnitude conclusion depends entirely on it.** On the DALEC2 prior the
residual is 2.0–2.9×; on the REFLEX prior it is 0.90–1.19×. Nothing else in the
analysis moves it that far, and anyone revisiting the `ceff` range revisits this
conclusion with it.

`ceff` = **U(5, 20)**, Fox et al. (2009), *Agricultural and Forest Meteorology*
149, 1597–1615, Table 4, published range for the same ACM parameter `a1`.

**Provenance, precisely.** This is a published prior adopted in place of a wider
published prior — not a value fitted to GPP; it was not selected by inspecting
modelled output. Note it is *not* a narrowing: REFLEX's range extends **below**
DALEC2's U(10, 100), and the measured GPP is reached in its lower half.
**Verified at source by the user. Not independently checked in this session** —
the paper is paywalled at ScienceDirect and the Reading and Edinburgh repository
copies are access-restricted. Given the prior is load-bearing, that verification
should be recorded in the thesis with the table and page.

## The surviving finding: the seasonal distribution defect

This one did not move and is stated separately because it is what remains.

| | October / April GPP |
|---|---:|
| measured (Kolari 2010, EC) | 0.76 |
| cross-check, `gpp_nt` / `gpp_dt` | 0.705 / 0.747 |
| modelled, flat priors | 0.290 |
| modelled, canopy priors | 0.290 |
| **modelled, current** | **0.397** |
| **ratio** | **0.52×** |
| **draws below the measured ratio** | **100.0%** |

**It did not respond to any prior change.** Correcting the GPP magnitude by a
factor of three moved the seasonal ratio from 0.38× to 0.52× and left every
single draw below the measured value. Magnitude and seasonal distribution are
separate defects, and only the first was ever a prior problem.

**The mechanism.** April and October have almost identical mean temperature —
3.61 °C against 4.02 °C in this block, matching Kolari's 3.5 and 4.1 — but April
irradiance is **12.95** against October's **3.09**, a factor of 4.2. ACM is
radiation-driven, so it returns approximately the irradiance ratio of 0.24. The
real canopy achieves 0.76 on a quarter of the light, because its photosynthetic
capacity is still summer-acclimated. **ACM is not missing a modest autumn
deficit; it is missing a large autumn enhancement.**

### The mechanism is independently measured; the ratio is not

Kolari et al. (2009), *Boreal Env. Res.* 14:761–783, paper V of the dissertation,
derives the seasonal course of photosynthetic capacity β **twice**:

1. predicted from a delayed temperature history, using parameters from Kolari et
   al. (2007), paper II;
2. **"estimation of daily β directly from the measured shoot CO₂ exchange"** —
   automated 1 dm³ chambers, closed 70–100 times a day, **no eddy covariance
   involved**.

The paper compares three GPP series in Fig. 6 — EC, chamber-based upscaling, and
the model prediction — and states that in boreal evergreen conifers the seasonal
cycle of photosynthetic capacity "can be described accurately as a delayed
response to temperature" (Pelkonen & Hari 1980; Mäkelä et al. 2004).

**Mechanism: validated.** It rests on shoot-level chamber gas exchange, which is
independent of the NEE signal DALEC is calibrated against. The model's failure to
reproduce it is therefore not circular.

**Ratio: a consistency check.** The 0.76 is EC-derived, and our cross-check
against `gpp_nt`/`gpp_dt` uses the same EC-partitioned data. A fully independent
number would need October/April computed from the chamber-based GPP series, which
is published only as a figure. **Say it that way in the thesis.**

## Initial pools

All six are derived from measured fluxes and turnover; none is left on a generic
published range. See DECISIONS §9.

| pool | derivation | median | independent check |
|---|---|---:|---|
| `c_lab_0` | litterfall · f_lab/(f_lab+f_fol) | 77 | — |
| `c_fol_0` | litterfall / `c_lf` | 579 | — |
| `c_roo_0` | 90 / (`theta_roo`·365.25) | 49 | measured 238 — **0.20×** |
| `c_woo_0` | 6,800 − the other three | 6,063 | — |
| `c_lit_0` | (1−`f_som`)·`rh_ref`/`theta_lit` | 119 | — |
| `c_som_0` | `f_som`·`rh_ref`/`theta_som` | 5,810 | VT_SP 5,730 ± 710 |

`c_roo_0` inherits a `theta_roo` prior spanning 0.27–27 year residence times; the
site's own stock and flux pin it at **1.04×10⁻³ d⁻¹**, a 2.6 year residence time.
Reported, not adopted.

The foliar trajectory is **still not stationary** — it falls 562 → 177 over the
block. The initial value is correct by construction (`c_lf · c_fol_0` = 154
exactly, asserted in tests). The cause is diagnosed below: the model is
**bistable**, and the measured canopy sits at an unstable equilibrium between a
runaway branch and a collapsing one.

## The foliar fixed point: the model is bistable, and neither branch is the site

`scripts/13_foliage_fixed_point.py`, 40 draws, block cycled to convergence on the
end-of-cycle foliar pool (tolerance 10⁻³, up to 60 cycles ≈ 840 years).

**The headline is the outcome distribution, not the terms.**

| outcome | draws |
|---|---:|
| converged to a stable canopy | **16 / 40** |
| **collapsed** (foliage → < 5 g C m⁻²) | **24 / 40** |
| still drifting after 60 cycles | 0 |

**No draw settles at the measured canopy.** The model runs away to a dense canopy
or it collapses. The 14-year trajectory falling 562 → 177 is the collapsing
majority; the converged minority goes the other way.

That is bistability with an **unstable equilibrium near the measured leaf area**:
the LAI → GPP → foliar allocation → LAI loop has gain greater than one there, so
a canopy at the site's own LAI is pushed away from it in whichever direction it
is nudged. It is not a slow relaxation from a bad initial state — the initial
state is right by construction — it is a fixed point the model cannot hold.

### The identity, on the converged branch

At a fixed point annual foliar loss equals annual input, so
`(f_fol + f_lab) · GPP` **is** the litterfall the model sustains. That form is
exact; `c_lf × mean(C_fol)` is not, because leaf fall is a Gaussian pulse rather
than spread evenly through the year.

| term | median | IQR | measured |
|---|---:|---|---:|
| GPP | 1,666 | 1,430 – 2,495 | 1,028 |
| allocation share `f_fol+f_lab` | 0.190 | 0.170 – 0.244 | 0.150 |
| **litterfall = share · GPP** | **318** | 243 – 616 | **154** |
| `C_fol` | 890 | 589 – 1,709 | ~568 |
| LAI (projected) | 4.10 | 3.28 – 9.10 | ~2.1–2.3 |
| `c_lf` | 0.271 | 0.241 – 0.297 | 0.20–0.333 |

### Which term carries it

| term | ratio to measured |
|---|---:|
| allocation share | **1.27×** |
| GPP | **1.62×** |
| product | 2.05× |
| litterfall | **2.07×** |

**The excess factorises cleanly**: 1.27 × 1.62 = 2.05 against an observed 2.07.

**Turnover is not the culprit.** `c_lf` at the fixed point is 0.271, a 3.7 year
needle residence time, sitting inside the measured 3–5 years and independently
corroborated by Kolari's 25% annual foliage turnover. Nothing is wrong with the
leaf-fall parameter.

**Nor is GPP too low — it is too high**, by 1.62× on this branch, which is the
opposite of what a drifting-down trajectory suggests. And allocation is not too
low but 1.27× too high, despite the Dirichlet concentration having been set from
the measured fluxes: the *prior* share is 0.144, but the draws that survive to a
stable canopy are selectively those with a higher share.

**So the answer to "allocation too low, turnover too high, or GPP too low" is
none of the three.** On the branch that survives, allocation and GPP are both too
high and reinforce one another; on the branch that does not, the canopy collapses
entirely. The measured canopy lies between two branches the model does not
occupy.

### What this means

The GPP–LAI feedback, not any single parameter, is the remaining structural
problem. It also sharpens the RQ3 prediction: a sampler forced to reproduce
observed NEE will have to sit near an unstable equilibrium, which is a plausible
source of poor mixing and of posterior mass split between branches. **Watch for
multimodality in the foliage-related parameters.**

Caveats: 40 draws, 16 on the converged branch, and the IQRs are wide. The
direction and the bistability are robust; the specific ratios are indicative.

## The 2.5 all-sided-to-projected LAI ratio is also load-bearing

**No Scots pine measurement was found**, so the ratio remains a **conventional
assumption**, recorded as such in DECISIONS §6. It is not innocuous:

| ratio | `lma` bounds | mean projected LAI band | n | median GPP | residual | 1.00× inside IQR |
|---:|---|---|---:|---:|---:|---|
| 2.0 | 116–192 | 2.62–2.85 | 10 | 1,061 | **1.03×** | **YES** (0.89–1.19) |
| **2.5 (adopted)** | 144–241 | 2.10–2.28 | 12 | 753 | **0.73×** | **NO** (0.58–0.99) |
| 3.0 | 173–289 | 1.75–1.90 | 17 | 598 | **0.58×** | **NO** (0.47–0.78) |

**Zero bias does not stay inside the residual range in all three cases.** It sits
inside only at ratio 2.0. At 2.5 and 3.0 the model under-produces by 27% and 42%.

**This qualifies the "no detectable magnitude bias" conclusion.** That conclusion
is safe at ratio 2.0 and weakens as the ratio rises, and the *sign* flips relative
to the withdrawn claims: the model now under-produces rather than over-produces
on the adopted ratio. Both `ceff` and this ratio are load-bearing, and the ratio
has no source at all.

Small samples: n = 10–17 per band, wide IQRs. The 0.73× here and the 0.90×
reported for "mean LAI 2.1–2.6" elsewhere in this file are the same quantity on a
narrower band, not a contradiction.

## Pre-sampling prediction for RQ3

Recorded before sampling so the posterior tests it rather than rationalises it.

If ACM over-produces GPP by a factor `k` and the sampler must reproduce an
observed NEE near −206 g C m⁻² yr⁻¹, respiration absorbs the excess rather than
GPP being corrected.

| premise | predicted posterior GPP | predicted posterior Reco |
|---|---:|---:|
| `k` = 2.00 (stage 2, withdrawn) | ~2,060 | ~1,850 |
| **`k` ≈ 1.0–1.2 (current)** | **~1,030–1,230** | **~830–1,030** |
| measured | 1,030 | 850 |

**With the magnitude bias no longer detectable, the compensating-error prediction
weakens accordingly** — the earlier ~2,000/~1,800 figures were a consequence of
the DALEC2 `ceff` prior and are withdrawn with it.

**Falsified by** a posterior with GPP near 1,030 and Reco near 850 and no
seasonal structure in the residuals. **Confirmed by** both gross fluxes biased
high by the same absolute amount, leaving NEE right for the wrong reasons.

The seasonal prediction is the sharper one and does not depend on the magnitude:
**expect a systematic autumn deficit and spring surplus in the monthly posterior
predictive residuals**, because no parameter in the model can fix a 0.52×
October/April ratio. Check residuals by month.

## Open

1. **`theta_roo` should be narrowed** to the measured 1.04×10⁻³ d⁻¹.
2. **The seasonal defect** is the standing GPP problem, LIMITATIONS §1/§1a.
3. **Task 2 must be re-run** against these priors; its ranking was computed
   against superseded ones.
