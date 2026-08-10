"""PyMC model assembly and NUTS sampling.

**Phase 6. Not yet implemented.**

Assembles priors from the registry, the PyTensor forward model and the masked
Gaussian likelihood into one ``pm.Model``, then samples with NUTS. NUTS only --
no alternative samplers, by design. Sampler output is written as NetCDF through
ArviZ, never pickled, with the seed recorded in the trace attributes.
"""

from __future__ import annotations

raise NotImplementedError("Phase 6 has not been built yet.")
