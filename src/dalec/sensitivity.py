"""Morris elementary-effects screening over the free parameters.

**Phase 5. Not yet implemented.**

Ranks parameters by their influence on daily NEE using the NumPy forward model
(no gradients needed) via SALib. Produces a ranked mu*/sigma table and an
elementary-effects plot, and writes a candidate fixed-parameter list back to
config. Target: reduce 23 parameters to roughly 12-14 before the expensive
sampling.

The screening output is a thesis result in its own right -- table and figure
are publication-quality, not diagnostics.
"""

from __future__ import annotations

raise NotImplementedError("Phase 5 has not been built yet.")
