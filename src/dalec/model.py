"""DALEC2 forward model in PyTensor -- differentiable for NUTS.

**Phase 6. Not yet implemented.**

The same model as ``dalec.model_numpy``, expressed with ``pytensor.scan`` so
that gradients flow through the whole time series. Equivalence with the NumPy
twin to floating-point tolerance is the project's most important test.

Things to get right, and to leave comments on where handled:

* No Python control flow inside the scan may depend on a parameter value --
  that would bake one branch into the compiled graph.
* Pools must be kept non-negative without hard clipping. ``clip`` produces a
  zero-gradient region that NUTS reads as a flat wall; use softplus or an
  exponential parameterisation instead.
* Make the number of timesteps a compile-time constant where possible, so scan
  can unroll and specialise.
"""

from __future__ import annotations

raise NotImplementedError("Phase 6 has not been built yet.")
