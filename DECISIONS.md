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

**ACM coefficients**, Williams et al. (1997) Table 2 p. 887, never sampled:

```
a2 = 0.018    theta = 32.6    k  = 576.7
b1 = -0.029   b2    = 0.315
c1 = 0.989    c2    = 0.873
d1 = -0.0018  d2    = 1.81
```

`a1 = 2.95` is **not used**: DALEC2 replaces the whole `a1 * N` product with the
sampled `ceff`, so foliar N is not implemented.

**Solve order 2, 5, 6, 4, 8, 7, 9.** The paper prints "2, 5, 6, 4, 7, 8, 9", but
Eq. 7 consumes `E_0` from Eq. 8, so 8 must run first. Eq. 3 is never evaluated
separately — the steady-state assumption `p_C = p_D` folds it into Eq. 6.

**Substitutions and conventions.** `L = C_fol / c_lma` (Eq. A12), from model
state rather than drivers. `T` is the mean of daily maximum and minimum,
`(TA_F_DAY + TA_F_NIGHT) / 2` — *not* `TA_F`, which is what decomposition in
A5/A6 uses. `D_T` is **half** the daily temperature range, `(T_max - T_min) / 2`
per Table 1; using the full range is a silent factor-of-two error and is
asserted against in a test.

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

## 5. Known structural problems — recorded, not solved

These are thesis limitations. They live in the `acm.py` docstring, measured.

**1. Annual GPP is several-fold too high.** Over a synthetic Hyytiälä-like year
the total was 5009 g C m⁻² yr⁻¹ against an observed value near 1000–1100.

**2. ACM is nearly temperature-insensitive at low irradiance.** Across a
40-degree swing at low `I`, GPP moves from 2.646 to 2.633 g C m⁻² d⁻¹ — a 0.5%
response. The cause is structural: at low `I` the Eq. 7 light limitation
dominates and GPP collapses to `E_0 * I * (d1*D_ms + d2)`, in which `T` does not
appear at all. Sustained year-round that floor integrates to
`2.657 * 365.25 = 970.5` g C m⁻² yr⁻¹ — essentially Hyytiälä's entire true annual
GPP, from a term with no temperature in it.

**3. There is no parameter escape route.** Searching the full 2D grid
(LAI 0.2–6.0 × `ceff` 10–100) for any combination with a summer peak in 8–11
*and* a winter floor below 0.5 yields **zero** combinations; the maximum
achievable summer/winter ratio is 8.17 against a true ratio that is effectively
unbounded. LAI enters the light-limited floor only through
`E_0 = c1*L²/(c2+L²)`, which saturates and scales summer and winter almost
equally. The frost cutoff is consequently the only mechanism available to
suppress the floor.

**4. The `ceff` reading is suspect.** The original `a1 * N` form reproduces
Williams et al. Fig. 2 at Harvard Forest (modelled 9.90 against a measured peak
near 11–12), so the implementation is sound. `a1 * N` there is
`2.95 * 1.92 = 5.7`, against a `ceff` prior of 10–100 — ranges that barely
overlap. Eq. 2 is implemented exactly as specified, with no rescaling and no
correction factor; the magnitude gate decides. **Open question for the authors:**
whether the 10–100 range assumes a different LAI convention (projected versus
all-sided), a different normalisation, or an ACM variant not described in the
2015 appendix. The DALEC2 source has been requested.

**Scope: this site is outside the envelope DALEC2's authors defined.** Bloom &
Williams (2015) §2.5 selected sites with little expected water stress and no more
than **three months** of recorded below-freezing soil temperature, those criteria
reflecting DALEC2's capabilities — hydrological processes are not explicitly
represented. FI-Hyy has roughly **four to six months** of soil frost. Applying
DALEC2 here is a deliberate scope decision, not an oversight.

**Bearing on RQ3.** A GPP term insensitive to temperature over much of the year
pushes seasonal structure onto the respiration parameters during calibration, so
observed equifinality may be partly *structural* rather than purely
informational. The Phase 7 synthetic twin separates the two — synthetic data
generated by this same GPP model should recover its parameters normally, so any
gap between synthetic and real-data recovery isolates structural error. The twin
must run with the frost mask active and over the same driver record, or the
comparison is not like-for-like.

---

## 6. Provisional values — not sourced, do not cite

Flagged here so they cannot quietly become fact:

- `site.canopy_height_m` (`H`) and `site.psi_d_mpa` (`psi_d`) are **unset** in
  config. `acm_from_config()` raises rather than guessing. Values of 18 m and
  −1.5 MPa have been used in exploratory sweeps; they come from a working
  estimate, not from the site literature, and are deliberately not written into
  config.
- The target projected LAI of ≈ 3 for FI-Hyy is provisional pending a citation
  from the SMEAR II literature.
- `ACM_CALIBRATION_BOUNDS` currently holds only the `t_mean` row (7–30 °C). The
  remaining Williams et al. Table 1 rows — irradiance, LAI, CO₂, `D_T`, `psi_d`,
  `H` — have not been supplied, so `calibration_bound_coverage()` reports on
  temperature alone.

---

## References

- Bloom, A. A. and Williams, M. (2015). Constraining ecosystem carbon dynamics
  in a data-limited world. *Biogeosciences* 12, 1299–1315.
- Williams, M. et al. (1997). Predicting gross primary productivity in
  terrestrial ecosystems. *Ecological Applications* 7(3), 882–894.
- Richardson, A. D. et al. (2010). Estimating parameters of a forest ecosystem C
  model with measurements of stocks and fluxes as joint constraints. *Oecologia*
  164, 25–40.
- Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux
  processing pipeline. *Scientific Data* 7, 225.
