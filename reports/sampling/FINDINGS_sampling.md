# Sampling-stage diagnostics: what the plumbing run's machinery shows

`scripts/22_sampling_figures.py`, reading `plumbing_hemisurface.nc`. Plumbing
run: 2 chains, 200 tune, 200 draws, block 1997–1998, hemisurface convention,
730 days of which 622 assimilable. Seed 20260809.

> **This is not a posterior.** 400 post-warmup draws over two years is far too
> short for any statement about parameter values, and nothing below should be
> read as one. What these figures are for is the machinery: whether the chains
> mixed, which parameters the prior is truncating, and which pairs the data
> cannot separate. Every claim here is about the sampler, not about FI-Hyy.

Sampling was **not** re-run to produce these. The text and CSV were regenerated
from the stored trace with `python scripts/20_plumbing_run.py --from-trace`.

---

## The convergence check was reporting a rounded number, and it hid a failure

**`az.summary(idata, round_to=None)` does not disable rounding.** Only the
string `"none"` does; Python `None` falls through to ArviZ's default
two-significant-figure display. The consequence, measured:

| | worst r-hat |
|---|---:|
| `round_to=None` (what was reported) | **1.000** |
| `round_to="none"` (the true value) | **1.0199** |

So the plumbing run's headline "worst r_hat 1.000" was a **failed** convergence
check displayed as a clean one. Two variables exceed the 1.01 threshold —
`d_onset` at 1.0199 and `c_lit_0` at 1.0138 — and the old table showed none,
because 1.0199 and 1.0138 both round to 1.0 at two significant figures.

Fixed in `scripts/20_plumbing_run.py`, and the text and CSV regenerated. **The
same call is still present at `scripts/04_calibrate.py:559`**, so every r-hat and
ESS figure reported from the full calibration is likewise rounded to two
significant figures. Not changed here; recorded so it is not forgotten.

---

## fig20 — convergence

`fig20_convergence`. r-hat per variable against the 1.01 threshold, and bulk and
tail ESS against 400.

**0 / 400 divergences and 0 / 400 max-treedepth hits.** Tree depth median 5,
max 6, against a limit of 10. Whatever is wrong here, the integrator is not
struggling: no divergence to cluster, no trajectory truncated.

**Two variables exceed r-hat 1.01**, both labelled on the figure. Neither is
alarming on its own — 1.02 at 400 draws is marginal rather than broken — but
the point is that the previous report said there were none.

**Bulk ESS runs 228 to 607; 18 of 35 variables sit below 400.** Lowest bulk is
`theta_lit` at 228, lowest tail is `theta_min` at 114. Read the 400 line with
the run length in mind: there are only 400 draws in total, so this run cannot
comfortably clear that threshold and the ESS panel mostly says *the run is
short*, which is what it was designed to be.

---

## fig21 — which parameters are pinned against their prior

`fig21_prior_position`. Each sampled scalar's 89% posterior interval rescaled so
its own prior maps to [0, 1], sorted by distance to the nearest bound. Bounds are
read from `PARAMETER_REGISTRY`, `canopy_bounds("hemisurface")` and
`reparameterised_bounds()` — no bound is typed into the figure script, and the
composed set is asserted against `dalec.priors.prior_sources`, which is what the
model itself built the priors from.

**Eight of sixteen sampled scalars have their 89% interval within 0.02 of a
bound:**

| parameter | distance to nearest bound | side |
|---|---:|---|
| `c_lf` | 0.0000 | upper |
| `ceff` | 0.0001 | upper |
| `f_som` | 0.0001 | lower |
| `theta_min` | 0.0003 | upper |
| `theta_lit` | 0.0011 | upper |
| `theta_roo` | 0.0011 | upper |
| `lma` | 0.0017 | upper |
| `theta_som` | 0.0045 | upper |

**A posterior pressed against a bound is the prior truncating the answer**, not
an estimate of the parameter. The data wanted to go further and the range would
not let it. Seven of the eight are at the *upper* bound, `f_som` alone at the
lower.

This matters most for `ceff`, whose U(5, 20) is load-bearing for the GPP
magnitude conclusion (DECISIONS §9, `FINDINGS_gpp.md`): the posterior sits on the
ceiling of the range that conclusion rests on. It is not evidence that the range
is wrong — 400 draws over two years cannot support that — but it is the first
direct sign that the likelihood pulls against it.

The figure draws a median marker on every row deliberately: several of these
intervals are narrower than a pixel, and without it the most pinned parameters
would render as blank rows.

---

## fig22 — phenology, and the one parameter that provably cannot be informed

`fig22_phenology`. `d_onset` and `d_fall` share the prior U(1, 365), drawn flat
behind both.

| | median | sd |
|---|---:|---:|
| `d_onset` | 122.6 | **0.36** |
| `d_fall` | 177 | **104.0** |

**`d_onset`'s entire posterior is narrower than one day**, on a prior spanning a
year. **`d_fall` recovers its prior**, sd 104.0 against the uniform's 105.1.

**That is the correct behaviour, not a convergence failure.** `d_fall` is inert
as transcribed — Eq. A8's sine argument prints `cr_fall` where `d_fall` belongs,
and the code implements it as published (DECISIONS §2, correction 2) — so
`d_fall` cannot affect the likelihood and its posterior *must* equal its prior.
The figure is the visual confirmation that the transcription behaves as
documented.

The two must be plotted on separate y-scales, because their densities differ by
a factor of about 400: drawn together, `d_fall` is indistinguishable from the
prior line and reads as missing rather than as lying on it.

**Do not report `d_fall`'s flat marginal as an identifiability finding for RQ1
or RQ3.** It is an artefact of the transcription and is recorded as such.

---

## fig23 — the four worst-mixing variables

`fig23_traces`. Trace and rank plots for the four highest r-hat, chains
coloured separately.

| variable | r-hat | bulk ESS |
|---|---:|---:|
| `d_onset` | 1.0199 | 553 |
| `c_lit_0` | 1.0138 | 490 |
| `c_fol_0` | 1.0073 | 458 |
| `c_lf` | 1.0073 | 458 |

**`c_fol_0` and `c_lf` are the same variable.** `c_fol_0` = 154 / `c_lf` by
construction (DECISIONS §9), their posterior correlation is **−1.000**, and their
r-hat and ESS are identical to three decimals. The four worst-mixing variables
are really three.

The traces are visually stationary and the chains overlap throughout; the rank
plots are close to flat with mild departures. That is what r-hat ≈ 1.02 looks
like — marginal, not pathological. **No trend, no stuck chain, no excursion.**

Worth noting how narrow these are: `c_lf` moves only over 0.3327–0.3333 against
a prior of 0.20–0.3333. Its poor mixing is partly a consequence of being pressed
into a sliver at the bound, where the posterior has almost no room.

---

## fig24 — equifinality: the eight most-correlated pairs

`fig24_pairs`. All pairwise posterior correlations over the sampled scalars and
the four allocation weights; the eight largest by absolute value.

| pair | r | |
|---|---:|---|
| `w[f_lab]` – `w[f_woo]` | −0.846 | simplex, structural |
| **`f_auto` – `lma`** | **−0.677** | |
| **`theta_som` – `theta_woo`** | **−0.606** | |
| `rh_annual` – `temperature_exponent` | +0.416 | |
| `w[f_lab]` – `w[f_roo]` | −0.407 | simplex, structural |
| `f_auto` – `temperature_exponent` | −0.389 | |
| `cr_onset` – `d_onset` | +0.314 | |
| `f_auto` – `w[f_lab]` | +0.288 | |

**Two of the eight are correlated by construction and are marked as such on the
figure.** The allocation weights live on a simplex and must sum to one, so
negative correlation among them is a property of the parameterisation, not
evidence about the site. Reporting them as equifinality would be wrong.

**`theta_som` – `theta_woo` at −0.606 is the one that speaks to RQ3.** These are
the two slowest turnover rates in the model, and NEE alone trading one against
the other is exactly the structure LIMITATIONS §4 says is already established in
the literature — Fox et al. (2009), Williams et al. (2005). The value here is
not that it happens but that the geometry is now measurable: this is the
covariance NUTS gives access to and a random-walk sampler does not.

**`f_auto` – `lma` at −0.677 is the largest non-structural correlation** and is
not one the literature review anticipated. Both control how much of GPP reaches
the canopy — `f_auto` by removing it to respiration before allocation, `lma` by
setting how much leaf area a unit of foliar carbon buys — so a compensating pair
is mechanically sensible. Both also sit at or near a prior bound in fig21, so
part of what the scatter shows is the corner of the prior box rather than a ridge
the data selected.

**Caveat, and it is a large one.** 400 draws over two years, with eight of
sixteen parameters against a bound. These correlations describe the geometry of a
short plumbing run inside a truncated prior. They are the right diagnostic to
compute and the wrong run to conclude from. **Re-measure on the full calibration
before any of this reaches RQ3.**

---

## Files

| figure | what it shows |
|---|---|
| `fig20_convergence` | r-hat and ESS per variable against their thresholds |
| `fig21_prior_position` | 89% intervals rescaled to each prior's own range |
| `fig22_phenology` | `d_onset` informed, `d_fall` inert on its prior |
| `fig23_traces` | traces and rank plots, four worst by r-hat |
| `fig24_pairs` | the eight most-correlated parameter pairs |

Both `.png` and `.pdf`, via `dalec.plotting.save_figure`, the same style helper
the prior-diagnostics figures use.
