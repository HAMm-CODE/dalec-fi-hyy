# The evergreen DALEC, and how it differs from DALEC2

FI-Hyy is an evergreen Scots pine forest. DALEC2 carries a labile carbon pool and
a leaf onset and leaf fall calendar that an evergreen forest may not need, and
some of those parameters already misbehave: `d_fall` cannot affect the fit at
all, and the phenology is involved in the two-mode posterior. The evergreen
model tests, for research question 3, whether a simpler structure is better
identified at this site.

It is an **addition, not a replacement**. DALEC2 stays the default everywhere
(`model_variant: dalec2` in `config/default.yaml`) and none of its code, numbers
or priors changed.

## What the model is

The evergreen DALEC of **Williams et al. (2005)**, *Global Change Biology* 11,
89–105. The equations are in its Appendix, "Carbon dynamic model equations"
(p. 104), and the assumptions behind them on p. 95. Code:
`src/dalec/model_evergreen.py`.

Five carbon pools:

| pool | what it holds |
|---|---|
| `c_fol` | foliage |
| `c_roo` | fine roots |
| `c_woo` | wood: stems and coarse roots |
| `c_lit` | fresh leaf and fine-root litter |
| `c_som` | soil organic matter plus woody debris |

Each day, photosynthesis (GPP) is split: a fixed fraction is respired straight
away, and the rest is shared between foliage, roots and wood in fixed
proportions. Every plant pool loses a fixed fraction of itself each day as
litter. Litter either decomposes into soil organic matter or is respired, and
soil organic matter is respired, both faster when it is warm. Net ecosystem
exchange is respiration minus GPP.

In symbols (Williams et al. 2005, Appendix, p. 104), with `G` = GPP and `T` the
temperature factor:

```
Ra = t2 G                          autotrophic respiration
Af = (1 - t2) t3 G                 allocation to foliage
Ar = (1 - t2) t4 G                 allocation to fine roots
Aw = (1 - t2)(1 - t3 - t4) G       allocation to wood
Lf = t5 Cf    Lw = t6 Cw    Lr = t7 Cr            litterfall
Rh1 = t8 Clit T    Rh2 = t9 Csom T    D = t1 Clit T  respiration, decomposition

dCf = Af - Lf     dCw = Aw - Lw     dCr = Ar - Lr
dClit = Lf + Lr - Rh1 - D            (see correction 1: Lw is not here)
dCsom = D + Lw - Rh2                 (see correction 1: Lw is here)
NEE = Ra + Rh1 + Rh2 - G
```

## Every difference from DALEC2

**Structure. These are the differences being tested.**

1. **No labile pool.** DALEC2 stores carbon and releases it to the leaves in
   spring. Here allocation to foliage goes straight to foliage.
2. **No leaf onset or leaf fall calendar.** DALEC2's Eqs. A7 and A8 are gone.
   Foliage is shed at the same small rate every day of the year.
3. **Therefore four fewer sampled parameters:** `d_onset`, `cr_onset`, `d_fall`
   and `cr_fall`. The labile allocation fraction `f_lab` and the labile starting
   pool `c_lab_0` go too.

**Kept identical to DALEC2, so the comparison isolates the structure:**

- **GPP**: the same ACM canopy model (Chuter et al. 2015, Loobos coefficients),
  the same frost cutoff and the same code. The ACM printed in the Williams
  (2005) Appendix is *not* used; its parameter table lists a3 = 217.9 and
  a4 = 0.980, which look swapped against Chuter et al. (2015) Appendix D.
- **Data, windows, QC, likelihood and sampler**: FI-Hyy FLUXNET2015 FULLSET
  daily NEE, calibration 1997–2010, held-out prediction 2011–2014, Gaussian
  likelihood with RANDUNC as the standard deviation on `nee_mask` days, NUTS
  with the same chains, draws, tuning and seed, hemisurface LAI.
- **Every shared parameter keeps exactly its DALEC2 prior.** They are read from
  the same source (`dalec.priors.prior_sources`); nothing is retyped.
- **The initial pools are derived exactly as in DALEC2** (DECISIONS §7 and §9),
  except that without a labile pool the wood pool is the measured tree carbon
  less foliage and fine roots, so the total is unchanged.

## Three departures from the printed Williams (2005) Appendix

Each was decided before any code was written, and each is marked in the code.

### Correction 1 — wood litter goes to the soil pool

The Appendix prints `dClit = Lf + Lw + Lr - Rh1 - D`, sending wood litter into
the fresh-litter pool. The same paper (p. 95) defines that pool as "fresh leaf
and fine root litter" and the soil pool as "soil organic matter plus woody
debris"; Fox et al. (2009) §2.1 describes the same pools the same way. The
equation and the pool definitions cannot both hold. **Implemented:** wood
litter enters the soil pool, `dCsom = D + Lw - Rh2`, which is also where DALEC2
sends wood turnover. Following the printed equation instead would have added a
second structural difference to the comparison.

### Decision 2 — the temperature response stays DALEC2's

Williams (2005) fixes the temperature sensitivity of decomposition at
Q10 = 2, `T = 0.5 exp(0.0693 A)`. **Implemented:** DALEC2's `exp(Theta T)` with
`Theta` sampled on its DALEC2 prior, U(0.018, 0.08) per °C. Williams' 0.0693
lies inside that range. Fixing it would remove a parameter and confound the
structural comparison.

### Decision 3 — foliage turnover comes from `c_lf`

Williams (2005) has a free foliage turnover rate `t5`. **Implemented:**
`t5 = c_lf / 365.25`, with `c_lf` on its DALEC2 prior U(0.2, 0.333), the
reciprocal of a 3–5 year needle life. At steady state the annual litterfall is
then `c_lf × Cf`, the same relation DALEC2's starting foliage
`c_fol_0 = litterfall / c_lf` rests on. A cohort-based reading,
`t5 = -ln(1 - c_lf) / 365.25`, would give a rate 11–22% faster; it was not used.

### And allocation is a Dirichlet, as locked in DECISIONS §1

Williams' `t3` and `t4` are the fractions of NPP going to foliage and to fine
roots, with wood taking the rest: a three-way split. It is sampled as a
Dirichlet over (foliage, roots, wood) whose concentration is DALEC2's with the
labile and foliar weights added together: (26.12, 15.26, 44.27), total 85.65.
Merging two parts of a Dirichlet gives a Dirichlet with the summed
concentration, so the prior on the foliage share here is exactly DALEC2's prior
on labile plus foliage, and the root and wood shares are unchanged.

## Parameters, side by side

| DALEC2 | evergreen | why |
|---|---|---|
| `f_auto` | kept, same prior | Williams' t2 |
| `theta_roo`, `theta_woo` | kept, same priors | t7, t6 |
| `theta_lit`, `theta_som`, `theta_min` | kept, same priors | t8, t9, t1 |
| `temperature_exponent` | kept, same prior | decision 2 |
| `rh_annual`, `f_som` | kept, same priors | respiration reparameterisation, DECISIONS §7 |
| `lma`, `ceff` | kept, same priors | same ACM |
| `c_lf` | kept, same prior | now sets the daily foliage turnover (decision 3) |
| `d_onset`, `cr_onset`, `d_fall`, `cr_fall` | removed | no phenology |
| allocation, 4 components | 3 components | labile and foliar merged |
| `f_lab`, `c_lab_0` | removed | no labile pool |
| `c_woo_0` | tree carbon − foliage − roots | no labile pool to subtract |

Free dimensions: 19 in DALEC2, 14 here.

## What this should and should not change

**Likely to fix.** `d_fall` is gone, so its flat posterior is gone. The two
modes of the DALEC2 posterior differ mainly in `cr_fall` and in whether
allocation runs through the labile pool (`reports/arviz/summary_per_chain.csv`),
and both of those routes disappear here.

**Not guaranteed.** The canopy feedback that made DALEC2 bistable — more leaf
area, more GPP, more allocation to leaves (DECISIONS §9) — is still here, so a
collapse-or-runaway pair of modes could reappear.

**Will not fix, and may make worse: the spring.** The modelled growing season
starts about a month early because GPP follows day length and sunlight, with no
gradual cold limitation on photosynthesis (DECISIONS §5; Stettz et al. 2022 fix
exactly this with a cold-temperature scaling on GPP). DALEC2's posterior partly
hides it by storing carbon in the labile pool and releasing it in one sharp
flush in late April. The evergreen model has no such lever, so its spring fit
may well be worse even if its parameters are better identified. The comparison
reports spring skill separately so that this trade-off is visible.

**Also unchanged:** parameters pinned at prior bounds for reasons unrelated to
phenology (`ceff`, `lma`, `rh_annual`, `theta_woo`), which point to tension in
the size of GPP and respiration.

## Running it

```
python scripts/25_evergreen_prior_predictive.py                  # prior checks, evergreen
python scripts/25_evergreen_prior_predictive.py --model-variant dalec2   # same checks, same engine
python scripts/04_calibrate.py --model-variant evergreen         # calibration
python scripts/compare_structures.py results/calibration_dalec2_hemisurface.nc \
                                     results/calibration_evergreen_hemisurface.nc
```

On Roihu, `jobs/submit_comparison.sh` submits both calibrations side by side
and the comparison after both succeed. Outputs are named
`results/calibration_<variant>_<convention>.*`; nothing existing is overwritten.
