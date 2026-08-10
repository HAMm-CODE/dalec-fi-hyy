#!/usr/bin/env python
"""Synthetic twin experiment -- can the pipeline recover known parameters?

**Phase 7. Not yet implemented.**

Picks known parameter values, runs the forward model to generate synthetic NEE,
adds Gaussian noise with the per-day sd taken from the real RANDUNC series, then
runs the full inference. Reports, per parameter: true value, posterior mean, 94%
interval, and whether the truth falls inside it.

This validates the pipeline before any real data is touched.
"""

raise SystemExit("Phase 7 has not been built yet.")
