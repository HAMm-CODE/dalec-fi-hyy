# Spec: pre-sampling diagnostics for DALEC / FI-Hyy

**Goal.** Four diagnostics that run *before* any MCMC sampling, to establish (a) whether the
priors are consistent with the observed NEE record, (b) which parameters the NEE likelihood
can plausibly constrain, (c) where in the seasonal cycle each parameter acts and which
parameters compete for the same signal, and (d) what `NEE_VUT_REF_RANDUNC` actually is.

**Hard constraint: no posterior sampling anywhere in this work.** No `pm.sample()`. Forward
model evaluations and `pm.sample_prior_predictive` only. If a task seems to need the posterior,
stop and say so rather than running NUTS.

**Reuse, do not reimplement.** The DALEC forward model, the prior specification, the driver
loading, the QC screening and the date handling already exist in the repo. Import them. If you
find yourself writing a second copy of the ACM routine or a second prior dictionary, stop — the
whole point is that these diagnostics describe *the model that will actually be sampled*.

Repo layout assumed: `src/`, `scripts/`, `data/`, `results/`, `tests/`. Adapt to what is
actually there; do not restructure the repo.

---

## 0. Shared infrastructure

Create `src/diagnostics/` with:

- `config.py` — one dataclass holding: calibration window start/end dates, RNG seed (fixed,
  default 20260825), number of prior draws, number of sweep points, figure output directory,
  QC threshold (0.75), and the blow-up thresholds from §1.4.
- `forward.py` — a thin wrapper exposing `run_forward(theta_dict, drivers, dates) -> dict`
  returning at minimum `nee` (daily array), and the pool trajectories. This must call the
  existing forward model. It exists so the four tasks share one entry point.
- `plotting.py` — shared matplotlib style. One style, applied everywhere: no seaborn theme,
  serif-free sans font, 300 dpi, colourblind-safe palette, every figure saved as both `.png`
  and `.pdf`. Axis labels always carry units (`g C m⁻² d⁻¹`).

Every script writes to `results/prior_diagnostics/` and every script is runnable standalone
from the repo root as `python -m scripts.<name>`.

**Dates, not indices.** Every array that indexes time must be aligned on an actual
`DatetimeIndex`. Where days have been dropped by QC screening, the forward model must still
step through the missing days internally (the carbon pools evolve whether or not anyone
measured that day) and the likelihood must then select only the observed days by date match.
If any code path uses positional row numbers as a time coordinate, that is a bug — write a test
that catches it (§5).

---

## 1. Task 1 — Prior predictive check

**Script:** `scripts/prior_predictive.py`

### 1.1 Draws

Use the existing PyMC model. Call `pm.sample_prior_predictive(draws=N, random_seed=seed)` with
N = 1000 on the *real* model object, with observed data attached. Reasons this is the required
route rather than hand-rolling numpy draws:

- it exercises the actual PyTensor graph, so it doubles as a smoke test of the forward model
- it guarantees the priors used here are byte-identical to the priors that will be sampled
- it returns both the latent model prediction and the observation-level prediction in one object

Draw **both** process parameters and initial carbon pool states. The initial pools are unknown
quantities with priors; fixing them at point values here would understate the prior spread and
make the check look better than it is.

### 1.2 Two bands, reported separately

- **Latent band** — the model's predicted NEE, `M_t(θ, u)`, with no observation noise. This
  describes what the model can produce.
- **Observation band** — `M_t(θ, u) + ε_t`, `ε_t ~ N(0, RANDUNC_t²)`. This is what should be
  compared against the observed points, because the observations carry that noise too.

Plot both. Coverage numbers (§1.4) are computed against the observation band only.

### 1.3 Figures

1. **`fig01_prior_pred_daily.pdf`** — full calibration window. Observed NEE as small grey dots.
   Prior predictive 5–95% ribbon (light) and 25–75% ribbon (darker), plus the prior median line.
   Also emit a 3-year zoom (`fig01b`) since the full record is too dense to read.
2. **`fig02_prior_pred_seasonal.pdf`** — mean seasonal cycle. Collapse both observed and
   predicted onto day-of-year, averaging across years. Observed mean ± 1 SD as a band; prior
   predictive 5–95% and median. **This is the most informative single panel** — it is where a
   spring-onset timing error becomes visible, which a daily plot hides in the noise.
3. **`fig03_prior_pred_annual.pdf`** — annual NEE. Only use years with ≥ 90% day coverage after
   QC screening; incomplete years produce biased sums. State the excluded years in the caption.
   Observed annual value as a point, prior predictive as a box or violin per year.
4. **`fig04_prior_pred_rank_hist.pdf`** — for each observed day, find the quantile at which it
   falls within that day's observation-level prior predictive distribution. Histogram those
   quantiles. Flat/uniform means the prior is well calibrated. A U-shape means the prior is too
   narrow (data falls in the tails more often than it should). A central hump means the prior is
   far too wide. Say which of the three you see.

### 1.4 Numbers to report

Write these into `FINDINGS.md` as actual computed values, not placeholders:

- fraction of observed days falling inside the 90% observation-level band (a well-specified
  prior gives roughly 0.90 or a bit above; substantially below 0.90 means the priors and the
  data disagree and the posterior will be fighting the prior)
- median width of the 90% latent band, in `g C m⁻² d⁻¹`, compared to the observed NEE range
- prior predictive median annual NEE vs observed annual NEE
- day-of-year at which prior predictive median NEE first crosses zero going negative
  (spring onset proxy), and the same for the observations, and the difference in days

### 1.5 Failed draws — count them, do not hide them

Some prior draws will produce unusable trajectories. Classify each draw as failed if any of:
NaN or inf anywhere in the NEE or pool trajectories; any carbon pool going negative; any pool
growing by more than a factor of 100 over the run; `|NEE_t| > 30 g C m⁻² d⁻¹` on any day.

Do not silently drop them. Report the failure rate, save the offending parameter vectors to
`results/prior_diagnostics/failed_draws.csv`, and produce a figure showing the marginal
distribution of each parameter among failed vs successful draws. Exclude failures from the
ribbons but state clearly in the caption how many were excluded.

This is a result in its own right, not housekeeping: it quantifies exactly the pathology that
Ecological Dynamical Constraints were designed to eliminate, which is the empirical support for
the argument that dropping EDCs is a deliberate methodological choice with a measurable cost.

---

## 2. Task 2 — One-at-a-time sensitivity

**Script:** `scripts/sensitivity_oaat.py`

### 2.1 Sweep design

For each parameter in the vector, in turn:

- hold every other parameter at its **prior median**, computed on the parameter's own natural
  scale (for a log-uniform prior the median is the geometric mean of the bounds, not the
  arithmetic mean — get this right, it matters for turnover rates spanning orders of magnitude)
- hold initial pools at their prior medians too
- sweep the target parameter across K = 15 points from the 1st to the 99th prior percentile,
  spaced evenly on the prior's natural scale (log-spaced for log-uniform priors)
- run the forward model at each point, keep the full daily NEE trajectory

Allocation fractions are Dirichlet-reparameterised, so they cannot be swept independently —
moving one moves the others. Handle these as a separate block: sweep along the simplex edges
(each fraction from near-0 to near-1 with the remainder split proportionally among the others)
and label the figure clearly so the difference in treatment is visible.

### 2.2 Metrics per parameter

- `d_nee_annual` — range of annual mean NEE across the sweep
- `d_rmse` — range of RMSE against observed NEE across the sweep
- **`d_loglik`** — max minus min of the Gaussian log-likelihood across the sweep, using
  `RANDUNC_t` as σ_t and only QC-passing days. **This is the headline metric.**
- `grad_scaled` — `|∂ log p(y|θ) / ∂θ_i|` evaluated at the prior median, obtained by PyTensor
  automatic differentiation, multiplied by the prior standard deviation of θ_i

Two of these need justifying because the choice is not obvious:

**Why log-likelihood and not ΔNEE.** A parameter can move NEE substantially and still be
invisible to the data, if it moves it during a period when `RANDUNC` is large. The likelihood
weights each day by `1/σ_t²`. Only the log-likelihood metric reflects what the sampler will
actually feel. Report `d_nee_annual` alongside it, because a large ΔNEE with a small Δlog-lik
is itself the interesting finding — that is a parameter the ecosystem cares about and the
measurements cannot see.

**Why scale the gradient by the prior SD.** A raw partial derivative carries the units of the
parameter, so gradients for a turnover rate in d⁻¹ and a leaf mass in g C m⁻² are not
comparable. Multiplying by the prior SD makes it dimensionless and gives a directly readable
quantity: log-likelihood units gained per one prior-standard-deviation of movement. This is
essentially free given the PyTensor graph already exists, and it previews the gradient-based
diagnostics that RQ3 rests on.

### 2.3 Figures

1. **`fig05_oaat_loglik_panels.pdf`** — small-multiples grid, one panel per parameter, x = the
   parameter value (log axis where the prior is log-uniform), y = log-likelihood relative to its
   maximum across that sweep. A flat panel is a parameter the data cannot see. Use a shared
   y-axis across panels so flatness is comparable by eye.
2. **`fig06_oaat_ranking.pdf`** — horizontal bar chart, parameters ordered by `d_loglik`,
   log-scaled x-axis. Add a second bar series for `grad_scaled` so the two rankings can be
   compared; where they disagree, the response is non-linear across the prior range.
3. **`fig07_oaat_trajectories.pdf`** — for the top ~6 parameters by `d_loglik`, overlay the
   mean-seasonal-cycle NEE across the sweep, coloured by parameter value. Seasonal-cycle form
   rather than daily — the daily version is unreadable.

Write the full metric table to `results/prior_diagnostics/oaat_metrics.csv`.

### 2.4 Interpretation guidance for FINDINGS.md

Report the numbers; do not hard-code a verdict. As a reading aid: a `d_loglik` below roughly
2–3 units across the entire prior range means the data cannot distinguish the ends of that range
from one another, and the posterior for that parameter should be expected to come back close to
its prior. Above roughly 10 units, expect visible contraction. These are rules of thumb for
orienting yourself, not thresholds to report as results.

**State the limitation explicitly in FINDINGS.md**, in these terms: one-at-a-time sweeps hold
every other parameter fixed at one point in the space. They therefore cannot detect constraint
that only exists in combination, and they cannot detect trade-offs where two parameters are
jointly identified but individually flat. This is a screening tool that predicts which
posteriors will look like their priors; it is not a proof of non-identifiability. The synthetic
twin experiment and the posterior itself are what settle that.

---

## 3. Task 3 — Seasonal attribution and parameter overlap

**Script:** `scripts/seasonal_attribution.py`

Reuses the Task 2 sweeps. Cache the sweep trajectories in Task 2 (`.npz` under
`results/prior_diagnostics/cache/`) so this does not re-run the forward model.

### 3.1 Sensitivity profiles

For each parameter `p`, from its sweep trajectories:

- daily profile `S_p(t) = max_sweep(NEE_t) − min_sweep(NEE_t)`
- day-of-year profile `S_p(doy)` — average `S_p(t)` across years by day of year, then smooth
  with a 15-day centred rolling mean
- monthly profile `S_p(month)`

Produce each profile in three forms:

- **absolute** — raw `g C m⁻² d⁻¹`, shows which parameters matter most overall
- **normalised** — each parameter's profile scaled to sum to 1, shows *shape* only, i.e. *when*
  in the year the parameter acts regardless of how strongly
- **σ-weighted** — `S_p(t) / σ_t` before aggregating, where `σ_t` is `RANDUNC_t`

The σ-weighted version is the one that answers the scientific question. A parameter that swings
NEE hard in midsummer, when `RANDUNC` is also large, delivers less information to the likelihood
than the raw profile suggests. The raw and σ-weighted heatmaps side by side make that visible,
and they connect Task 4 back to Task 3.

### 3.2 Figures

1. **`fig08_seasonal_heatmap.pdf`** — parameters (rows, ordered by total sensitivity) × month
   (columns), cell colour = normalised sensitivity. Emit both the raw and σ-weighted versions as
   panels of the same figure so they can be compared directly.
2. **`fig09_winter_summer_split.pdf`** — for each parameter, a diverging bar showing the share
   of its total sensitivity falling in the winter window (Nov–Mar) versus the growing season.
   This is the figure that turns the winter-as-respiration-window argument from an assertion into
   evidence: respiration parameters should sit clearly on the winter side.
3. **`fig10_overlap_matrix.pdf`** — cosine similarity between every pair of normalised
   day-of-year profiles, as a heatmap with hierarchical clustering applied to the row/column
   order. Cosine similarity near 1 means two parameters push NEE in the same direction at the
   same time of year, so NEE can only see their combination, not each of them separately.

### 3.3 Equifinality candidate list

From the overlap matrix, extract all pairs with cosine similarity above 0.95, and report them in
`FINDINGS.md` as a table with the pair, the similarity, and each parameter's `d_loglik`. These
are the pairs to watch for ridge structure in the posterior, and they are the concrete
prediction this whole exercise buys: named pairs, before any sampling has been run.

Add the caveat that a high cosine similarity in a one-at-a-time profile is suggestive, not
conclusive — the true correlation structure is a property of the joint posterior.

---

## 4. Task 4 — Characterising RANDUNC

**Script:** `scripts/randunc_characterisation.py`

Purely descriptive analysis of the data file. No model involved.

### 4.1 Data audit — do this first and report it

From the FLUXNET2015 FULLSET DD file for FI-Hyy, using `TIMESTAMP`, `NEE_VUT_REF`,
`NEE_VUT_REF_RANDUNC`, `NEE_VUT_REF_QC`:

- convert the `-9999` sentinel to NaN before anything else
- total days in record; days with valid NEE; days passing QC ≥ 0.75
- **days with valid NEE but missing or zero RANDUNC** — report this count prominently. Every
  such day is a day the Gaussian likelihood cannot evaluate as specified, because σ_t would be
  NaN or zero. A rule is needed before sampling: drop those days, or impute σ from a binned
  median. State which was chosen and why.
- the date ranges of any contiguous blocks of missing data, with block lengths

### 4.2 Figures

Single multi-panel figure, `fig11_randunc.pdf`:

1. **RANDUNC vs |NEE|** — faint scatter of all valid days, coloured by season. Overlay binned
   medians with interquartile range in |NEE| bins. Report the OLS slope, intercept and R² of
   RANDUNC on |NEE| in the panel.
2. **RANDUNC vs day of year** — binned median and IQR. Answers whether uncertainty has a
   seasonal structure beyond what flux magnitude alone explains.
3. **RANDUNC vs QC flag** — boxplot of RANDUNC grouped by binned QC value. If RANDUNC rises as
   the measured fraction falls, RANDUNC is tracking how much of the day was actually measured.
4. **RANDUNC vs 1/√n**, where `n = QC × 48` is the approximate number of measured half-hours in
   the day. **This is the panel that settles the open question.** If RANDUNC is the standard
   error of a daily mean built from `n` half-hourly values, it should scale as `1/√n` and this
   panel should be close to linear through the origin. Fit it and report the R². A strong linear
   relationship here is direct evidence for the reading that RANDUNC is an estimate of
   within-day variability rather than a known measurement error.
5. **Histogram of RANDUNC / |NEE|** — the relative uncertainty. Note the median and the tail.

### 4.3 The circularity caveat — must appear in FINDINGS.md

The regression of RANDUNC on |NEE| is descriptive only. Do **not** fit `σ = a + b|NEE|` and then
substitute the fitted function into the likelihood: that would make each observation's weight a
function of the observation itself, which is circular and would bias the posterior. `RANDUNC` is
used as supplied, per observation. The regression exists solely to characterise the error
structure and to provide the empirical justification for a time-varying σ_t over a constant σ.

---

## 5. Tests

Add to `tests/`, in the style of the existing suite:

- `test_forward_wrapper_uses_dates` — construct drivers with a deliberate gap, confirm that two
  observations separated by a 3-day gap are propagated through 3 forward steps, not 1. This is
  the failure mode the supervisor specifically flagged; it must be caught by a test, not by eye.
- `test_prior_medians_log_scale` — for a log-uniform prior on [a, b], the median returned is
  `sqrt(a*b)`, not `(a+b)/2`.
- `test_sweep_endpoints_within_prior_support` — every swept value lies inside the prior's support.
- `test_failed_draw_classifier` — hand-built trajectories containing NaN, a negative pool, and a
  30+ magnitude NEE spike are each classified as failures; a clean trajectory is not.
- `test_loglik_ignores_qc_failed_days` — days below the QC threshold contribute zero to the
  log-likelihood sum.
- `test_randunc_sentinel_conversion` — `-9999` becomes NaN and is excluded from every statistic.

All existing tests must still pass.

---

## 6. Deliverable

`results/prior_diagnostics/FINDINGS.md`, written last, containing:

- every number listed in §1.4, §2.2, §3.3 and §4.1, as computed values
- each figure embedded with a caption that states what it shows and what it implies
- a ranked list of parameters by expected constraint, with the explicit prediction of which
  posteriors will contract and which will return close to their priors
- the named equifinality candidate pairs from §3.3
- a plain statement of what RANDUNC appears to be, based on §4.2 panel 4
- the OAAT limitation paragraph from §2.4 and the circularity caveat from §4.3
- a short "open questions" section listing anything the diagnostics could not resolve

Written for a supervisor to read in ten minutes. Figures carry the argument; prose is brief and
states conclusions plainly.

---

## 7. Explicitly out of scope

- No `pm.sample()`, no NUTS, no posterior anything.
- Do not adjust the priors to make Task 1 look better. If the prior predictive check fails, that
  is the finding — report the mismatch with numbers and propose specific changes as a separate
  recommendation. A prior tuned to fit the data it is about to be updated with is no longer a
  prior.
- Do not add EDC constraints to suppress the failed draws in §1.5. Counting them is the point.
- No new dependencies beyond what is already in the environment (PyMC, PyTensor, ArviZ, numpy,
  pandas, matplotlib, scipy). If something seems to need `seaborn` or `SALib`, implement it
  directly or ask first.
- Do not refactor the existing forward model. If a bug is found in it, report it; do not silently
  fix it, because the existing test suite encodes assumptions that a silent fix would break.
