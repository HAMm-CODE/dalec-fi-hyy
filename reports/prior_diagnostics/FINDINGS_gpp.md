# Why prior predictive GPP is 2.5× the measured value

`scripts/11_gpp_investigation.py`, 400 draws, seed 20260809, calibration block
1997–2010. Reference: Ilvesniemi et al. (2009) Fig. 6, GPP (EC) **952–1104**
g C m⁻² yr⁻¹, verified against the paper.

Nothing here is fitted. No parameter was tuned to the measured GPP, and this
script writes no prior.

## The distribution

| | g C m⁻² yr⁻¹ |
|---|---:|
| prior predictive median | **2,570** |
| IQR | 542 – 3,162 |
| 5–95% | 84 – 3,353 |
| measured (Fig. 6) | 952 – 1,104 |
| ratio to measured midpoint | **2.50×** |
| draws inside the measured range | **0.8%** |
| draws above it | 68.3% |

The prior is not merely wide, it is displaced: two thirds of draws sit above a
range that fewer than one in a hundred reach.

## `ceff` is not the lever

**This overturns the earlier suspicion**, which was that the `ceff` prior
U(10, 100) drove the overestimate because the GPP gate passed at `ceff` ≈ 11.7
and Task 2's optimum was ≈ 30.

| `ceff` quintile | median GPP | inside measured |
|---|---:|---:|
| 10.5 – 30.5 | 2,216 | 2.5% |
| 30.5 – 48.3 | 2,333 | 0.0% |
| 48.3 – 68.6 | 2,517 | 0.0% |
| 68.6 – 84.8 | 2,932 | 0.0% |
| 84.8 – 99.8 | 2,934 | 1.3% |

**A tenfold span of `ceff` moves median GPP by 1.32×**, and the bottom quintile
still overshoots by more than double. Spearman correlation with annual GPP is
only **+0.241**. ACM's light interception saturates, so once the canopy is dense
the canopy-efficiency parameter has little left to do. Narrowing `ceff` cannot
fix this, and fitting it to the measured GPP would buy a number without a
mechanism.

## Leaf area is the lever

| parameter | Spearman ρ with annual GPP |
|---|---:|
| **mean LAI** | **+0.962** |
| `c_lf` | −0.628 |
| `lma` | −0.538 |
| `ceff` | +0.241 |
| `f_lab` | +0.230 |
| `f_fol` | +0.139 |
| `f_auto` | −0.103 |
| `c_fol_0` | +0.046 |

`c_lf` and `lma` rank where they do precisely because LAI = `C_fol` / `lma` and
`c_lf` sets the foliage turnover. `c_fol_0` is near zero because the pool
equilibrates away from its initial value over fourteen years — this is a
steady-state problem, not an initialisation one.

Modelled LAI, against a site value of about 3 (provisional, DECISIONS §6):

| | |
|---|---:|
| mean LAI, median | **5.09** |
| mean LAI, 5–95% | 0.18 – 48.26 |
| peak LAI, median | **10.60** |

An upper tail reaching LAI 48 is a runaway, and its mechanism is a **positive
feedback**: more leaf area → more GPP → more carbon allocated to foliage → more
leaf area. Nothing in the prior closes that loop.

## The measurable cause: too much carbon goes to foliage

| | share of GPP to foliage + labile |
|---|---:|
| prior draws, median | **0.240** |
| implied by measurement | **0.150** |

The site's measured above-ground needle litterfall is 154 g C m⁻² yr⁻¹ (Fig. 6
range 142–204) against GPP ≈ 1030. At steady state, allocation to foliage equals
foliage litterfall, so the measured share is 154/1030 = **0.150**. The flat
Dirichlet over the four allocation components delivers 0.240 — **60% too much**,
before the feedback amplifies it.

## What the site's own numbers imply

Steady-state foliar carbon is litterfall / `c_lf`, and LAI is that over `lma`:

| needle longevity | `c_lf` | `C_fol` | LAI at `lma`=150 | LAI at `lma`=205 |
|---|---:|---:|---:|---:|
| 3 yr | 0.33 | 462 | 3.1 | 2.3 |
| 4 yr | 0.25 | 616 | 4.1 | 3.0 |
| 5 yr | 0.20 | 770 | 5.1 | 3.8 |

Scots pine in southern Finland retains roughly 3–5 needle age classes, so
`c_lf` ≈ 0.20–0.33 and `lma` ≈ 200 g C m⁻² put LAI at 2.3–3.8 — around the site
value, and reached without reference to the measured GPP.

## Proposed fix, on physical grounds

Three changes, in the order they matter. **They must go together**; see the
caveat below.

1. **Constrain allocation to foliage from measured litterfall.** `f_fol + f_lab`
   should carry 0.13–0.21 of GPP (Fig. 6 litterfall 142–204 over GPP 952–1104),
   against the flat Dirichlet's 0.240. This attacks the feedback at its source —
   it limits the *flux* into the foliar pool rather than rescaling the pool
   afterwards.
2. **Constrain `c_lf` to needle longevity.** The present U(0.125, 1.0) spans
   lifespans of 1 to 8 years; its upper half implies an evergreen conifer
   shedding its needles inside eighteen months. 3–5 years gives 0.20–0.33.
3. **Narrow `lma` to a physically possible range.** U(10, 400) g C m⁻² admits
   10 g C m⁻², thinner than any conifer needle. **This one still needs a
   source** — no Scots pine LMA measurement has been verified for this project,
   and it must not be set by back-solving from a target LAI.

**The caveat that makes the ordering matter.** `c_lf` correlates *negatively*
with GPP (−0.628), so change 2 **on its own moves GPP the wrong way**: tightening
`c_lf` from a prior median of 0.56 down to 0.25 more than doubles the
steady-state foliar pool. It is only safe alongside change 1, which cuts the
inflow. Applying the physically correct needle longevity without the allocation
constraint would make the overestimate substantially worse.

## Priors alone will not close this gap

At the site's own LAI of about 3, the model still produces roughly **1,800**
g C m⁻² yr⁻¹:

| mean LAI | n | median GPP |
|---|---:|---:|
| 0–1 | 103 | 218 |
| 1–2 | 24 | 921 |
| 2–3 | 32 | 1,807 |
| 3–4 | 24 | 2,392 |
| 4–6 | 29 | 2,698 |
| 6–10 | 49 | 3,009 |
| >10 | 133 | 3,240 |

The measured 952–1104 corresponds to **LAI ≈ 1–2**, not 3. So even after the
canopy is corrected, ACM over-predicts by about **1.7×** at the right leaf area.

That residual is a model-structure problem, not a prior problem, and it points at
two limitations already on the register: ACM has **no temperature limitation on
photosynthesis at high latitude** (LIMITATIONS §1) and its **coefficients are
site-calibrated with neither published set boreal** (§2). Correcting the
allocation priors is necessary and will not be sufficient.

## Open questions

1. **Is the provisional LAI ≈ 3 right?** It is unsourced (DECISIONS §6). The GPP
   evidence points to an effective LAI nearer 2, but that inference is
   contaminated by the ACM bias above, so it is not evidence against 3.
2. **Which `lma`?** The one proposed change with no source behind it.
3. **Should the allocation constraint be a Dirichlet concentration or a bound?**
   A concentration keeps the simplex reparameterisation intact and stays
   differentiable; a hard bound would not.
4. **Does the ACM residual survive a boreal coefficient set?** Untested, and the
   cheapest next check on §2.
