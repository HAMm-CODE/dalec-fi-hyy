# GPP: the canopy priors are now right, and GPP got worse

`scripts/11_gpp_investigation.py`, 400 draws, seed 20260809, block 1997–2010.
Measured reference: Ilvesniemi et al. (2009) Fig. 6, GPP (EC) **952–1104**
g C m⁻² yr⁻¹. Nothing is fitted to it.

## The headline: the fix moved the number the wrong way

| | before the canopy priors | **after** | measured |
|---|---:|---:|---:|
| median annual GPP | 2,570 | **2,972** | 952–1,104 |
| ratio to measured midpoint | 2.50× | **2.89×** | — |
| draws inside the measured range | 0.8% | **0.3%** | — |
| draws above it | 68.3% | **97.7%** | — |
| mean LAI, median | 5.09 | **7.49** | 3.2 / 2.6 |

**This was the risk flagged in advance and it is worth being precise about why it
landed.** `c_lf` correlates negatively with GPP, so tightening it from a prior
median of 0.5625 to 0.267 more than doubles the steady-state foliar pool. The
allocation constraint was supposed to offset that by cutting the inflow from 24%
of GPP to 15%. It was not enough:

```
steady-state LAI = share / (c_lf * lma) * GPP      call the coefficient k

  old priors   k = 0.240 / (0.5625 * 205) = 0.00208
  new priors   k = 0.144 / (0.2670 * 192) = 0.00281      35% larger
```

Cutting allocation by 40% while cutting `c_lf` by 53% is a net **increase** in
foliage. The two changes were applied together, as specified, and together they
still point the wrong way.

## But the priors are not wrong — they are right, and that is the finding

Evaluate the same coefficient at the **measured** GPP:

| | LAI implied at GPP = 1030 |
|---|---:|
| old priors | 2.14 |
| **new priors** | **2.89** |
| measured (Kolari 2010) | 3.2 before thinning, 2.6 after |

**The new canopy priors reproduce the site's measured leaf area almost exactly,
given the site's measured GPP.** The old ones did not. The priors are correct.

What they pin is the **ratio** LAI/GPP. They cannot pin the **level**, because
LAI and GPP are jointly determined: allocation sets LAI from GPP, and ACM sets
GPP from LAI. Constraining the first relation leaves the second free, and the
second is where the error lives.

## ACM is a clean factor of two too productive per unit leaf area

| mean LAI | n | median GPP |
|---|---:|---:|
| 0–2 | 8 | 671 |
| 2–3 | 20 | 1,904 |
| 3–4 | 19 | 1,985 |
| 4–6 | 76 | 2,757 |
| 6–10 | 174 | 3,019 |
| >10 | 95 | 3,215 |

Restricted to the draws that actually sit at the measured leaf area:

> **draws with mean LAI 2.8–3.6 (n = 16): median GPP 2,056 against a measured
> 1,030 — ACM is 2.00× too productive at the right canopy.**

That is the whole residual, and it is now measured rather than inferred. The
joint system therefore has no fixed point near the truth: feed LAI 3.2 in and ACM
returns twice the GPP, which allocation converts back into more leaf area, until
the prior settles at LAI ≈ 7.5 and GPP ≈ 2,970.

## `ceff`: the earlier verdict needs qualifying

| | before | after |
|---|---:|---:|
| Spearman ρ with GPP | +0.241 | **+0.707** |
| median GPP, lowest `ceff` quintile | 2,216 | 2,179 |
| median GPP, highest quintile | 2,934 | 3,191 |

**`ceff` matters much more than it appeared.** With the canopy loosely determined
its signal was swamped; with the canopy pinned it is the second-strongest
predictor. The earlier statement that it moves GPP by only 1.32× across its range
was true of the old prior and is not true now — it moves it 1.46×.

What has *not* changed is the conclusion: the lowest `ceff` decile still gives
**1,898**, still **1.85×** the measured midpoint. `ceff` cannot reach the measured
range from inside its published prior, so narrowing it does not fix this, and
fitting it to the measured GPP would hide a structural error in a parameter.

## Two independent measurements of the same ACM bias

| test | measured | modelled | gap |
|---|---:|---:|---:|
| GPP at the site's leaf area | 1,030 | 2,056 | **2.00×** |
| October / April GPP ratio | 0.76 | 0.290 | **0.38×** |

The second is LIMITATIONS §1a. Both say ACM's productivity is wrong in a way no
prior can absorb — too much in absolute terms, and distributed wrongly through
the season because it tracks radiation instead of acclimated capacity.

## What is left, and what should happen next

1. **`c_fol_0` is now inconsistent with its own steady state.** It still carries
   the published U(20, 2000), median 1,010, while the canopy priors imply 462–770.
   The foliar trajectory falls monotonically from 1,497 in 1997 to 1,041 in 2010
   and is still falling — fourteen years is not long enough to relax. Deriving
   `c_fol_0` from litterfall and `c_lf`, exactly as `c_lit_0` and `c_som_0` are
   derived from `rh_ref`, is the obvious completion and was outside this scope.
2. **The ACM residual is the real blocker** and belongs to LIMITATIONS §1 and §2,
   not to the priors. It is now quantified twice.
3. **Do not fit `ceff`.** It would absorb a factor of two of structural error into
   a canopy-efficiency parameter and make the posterior look healthy while the
   model stayed wrong.

## What this exercise established

The canopy priors were changed on measured evidence and the headline number got
worse. That is not a failed change: it converted a vague "GPP is 2.5× too high,
cause unclear" into a specific, quantified statement — **the allocation and
canopy parameters are right, and ACM over-produces by a factor of two at the
correct leaf area.** The prior can no longer be blamed, which is what makes the
structural problem visible.
