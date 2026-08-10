"""Gaussian likelihood on daily NEE.

**Phase 6. Not yet implemented.**

Observations: ``NEE_VUT_REF``, g C m-2 d-1.
Standard deviation: ``NEE_VUT_REF_RANDUNC``, per day, g C m-2 d-1.

Only days where ``SiteData.nee_mask`` is True contribute. The forward model
still runs through the masked days -- masking is a likelihood operation, not a
time-series edit. Use ``SiteData.likelihood_arrays()`` for finite-safe inputs.
"""

from __future__ import annotations

raise NotImplementedError("Phase 6 has not been built yet.")
