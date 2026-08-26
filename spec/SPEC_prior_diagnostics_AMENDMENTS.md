# Amendments to SPEC_prior_diagnostics.md

The original spec was written without access to the repository. These amendments supersede it
wherever the two conflict. They follow a review pass that identified the mismatches below;
that review is accepted except where noted in A2.

---

## A1 — Priors are bounded uniform, not log-uniform

The spec's instruction to use the geometric mean as the prior median is **wrong for this repo**
and would have put `theta_som`'s median two orders of magnitude off. The locked decision is
bounded uniform over the published Bloom & Williams ranges; `Parameter` has no log flag.

- Prior median = `(lower + upper) / 2`.
- Sweep points are linearly spaced between the 1st and 99th percentile of the uniform prior.
- Delete `test_prior_medians_log_scale` from §5. Replace it with `test_prior_median_uniform`.

**Additional requirement, not a change to the locked decision.** For every parameter whose prior
range spans more than two orders of magnitude, compute and report in FINDINGS.md the fraction of
prior mass falling in each decade of the range. A uniform prior on `[1e-7, 1e-3]` places roughly
99% of its mass above `1e-5`, which is a strong substantive claim rather than an expression of
ignorance. Report this as an open question for supervision. Do not alter the priors.

---

## A2 — No PyMC model is required for Tasks 1–3

The review concluded that Tasks 1–3 are blocked because `src/dalec/priors.py`, `model.py`,
`likelihood.py` and `inference.py` are stubs. **This conclusion is rejected.** PyMC was one
implementation route, not a requirement of the analysis.

Replace §1.1 in full:

- Draw prior samples directly with numpy: `rng.uniform(lower, upper)` per scalar parameter,
  `rng.dirichlet(...)` for the allocation-fraction simplex block, and the same for initial pool
  states.
- **Binding constraint:** the sampler must read bounds from the existing `Parameter` registry.
  Hand-typed bounds anywhere in the diagnostics code is a defect. The point of the original
  `sample_prior_predictive` instruction was that the priors checked here are identical to the
  priors later sampled; that guarantee is preserved by reading the same objects, and is lost if
  the bounds are retyped.
- The observation-level band of §1.2 is formed by adding `rng.normal(0, randunc_t)` to the
  latent trajectory, drawn once per prior sample.
- Seed everything from the master seed in `config/default.yaml` (`20260809`), per the existing
  rule. The spec's `20260825` is void.

Everything downstream of §1.1 stands unchanged.

---

## A3 — `grad_scaled` via central finite differences

Autodiff is unavailable, but the metric does not depend on autodiff. For each parameter i:

```
grad_i ≈ ( loglik(θ + h·e_i) − loglik(θ − h·e_i) ) / (2h)
h = 1e-3 × (upper_i − lower_i)
grad_scaled_i = |grad_i| × prior_sd_i
```

Cost is two extra forward runs per parameter, roughly 30 s in total. Verify the step size is
stable by recomputing at `h` and `h/10` for three parameters and confirming agreement to two
significant figures; report if it does not. Do not drop this metric.

---

## A4 — Paths, naming and conventions

| Spec said | Use instead |
|---|---|
| `specs/` | `spec/` |
| create `src/diagnostics/` | extend the existing `dalec.diagnostics` module |
| `src/diagnostics/plotting.py` | `dalec/plotting.py` |
| `python -m scripts.<name>` | `python scripts/NN_name.py` with the existing `sys.path` shim |
| seed `20260825` | master seed from `config/default.yaml` |
| explicit calibration start/end dates | `years.calibration` via `require_year_block()` |

Do not create a second diagnostics package. Do not restructure `scripts/`.

---

## A5 — Forward model signature

`run_forward(theta_dict, drivers, dates)` does not exist. The real entry point is:

```
run_dalec2(params: DalecParameters, drivers, *, gpp_fn, phenology_fn)
```

The wrapper must construct a `DalecParameters` dataclass, not pass a dict, and must pass
`gpp_fn` and `phenology_fn` explicitly. Drivers carry their own dates; no separate `dates`
argument.

---

## A6 — `d_fall` is inert and must be handled explicitly

`d_fall` has no effect on the forward model as currently transcribed (see `DECISIONS.md` §2,
correction 2). Consequences:

- Its `d_loglik` will be exactly zero and its §3.1 sensitivity profile identically zero.
- It must **not** appear in the §2.3 ranking as an unconstrained parameter. Reporting it that way
  would misattribute a transcription artefact to a property of the data.
- Report it in a separate, clearly labelled subsection of FINDINGS.md as a known documented
  issue, with the cross-reference to `DECISIONS.md`.
- The §3.2 cosine similarity matrix must guard against zero-norm vectors. Any parameter with an
  identically zero profile is excluded from the matrix, not assigned a similarity of 0 or NaN.

Apply the same guard generically: any parameter with a zero profile is excluded and listed.

---

## A7 — RANDUNC: record the decided rule, do not re-open it

§4.1's phrasing implied an open decision. It is closed and encoded. `_likelihood_mask` requires
`isfinite(unc) & (unc > 0.0)`; days failing that are dropped, not imputed. FINDINGS.md records
this with the code reference.

Corrections to §4.1:

- RANDUNC is never zero in this record, only missing. Drop "or zero" from the audit description
  but keep the zero check in code as a guard.
- Report all of: total days, days with valid NEE, days passing QC ≥ 0.75, days with valid NEE and
  missing RANDUNC, QC-passing days with no usable sigma, and how many of those fall inside the
  calibration block. The last figure is the one that matters for the likelihood.

§4.2 panel 4: `n = QC × 48` is exact, not approximate — QC takes 49 unique values, all multiples
of 1/48. State this in the figure caption; it strengthens the panel's argument rather than
weakening it.

---

## A8 — Test reframe: observation gaps, not driver gaps

`_check_drivers_complete` raises on incomplete drivers, so `test_forward_wrapper_uses_dates`
cannot be built around a driver gap. Reframe it around the case that actually occurs: a run where
`nee_mask` excludes a block of days, asserting that the forward model integrates through the
masked days and that the likelihood selects the correct surviving days by date.

Note in FINDINGS.md that the date-versus-index concern is already satisfied architecturally —
`build_site_data` never drops rows, `SiteData.time` is a real datetime array, and the mask is
applied at the likelihood rather than by deletion. Cite the functions. This is a direct answer to
a supervision question and should be findable.

---

## A9 — Output location

- `results/` remains gitignored: caches, `.npz` sweep archives, intermediate arrays.
- Create `reports/prior_diagnostics/`, **tracked in git**: `FINDINGS.md`, final `.pdf` and `.png`
  figures, `oaat_metrics.csv`, `failed_draws.csv`.

Rule: anything that would be shown to a supervisor or cited in the thesis is tracked.

---

## A10 — Task 4 findings are standalone

Task 4's write-up is emitted as `reports/prior_diagnostics/FINDINGS_task4.md`, self-contained,
including the §4.3 circularity caveat. It merges into the combined FINDINGS.md when Tasks 1–3
land. Do not defer the caveat.

---

## A11 — Morris screening: deferred, not cancelled

SALib is installed and `sensitivity.py` is planned around it. Morris elementary-effects screening
would directly address the acknowledged weakness of one-at-a-time sweeps, since it samples across
the whole prior space rather than at a single point, and at roughly 250 forward runs it costs
about 3 minutes.

Do not build it until Tasks 1–4 are complete and their figures exist. If they land with time to
spare, add it as Task 2b. If not, FINDINGS.md notes it as the planned next step.

---

## A12 — Build order

1. Task 4 (unblocked, model-free)
2. Task 1 (numpy prior sampling per A2)
3. Task 2 (including A3 finite-difference gradients)
4. Task 3 (reuses Task 2 cache)
5. FINDINGS.md
6. Task 2b Morris, only if time remains

One task per session. Stop and report at the end of each.
