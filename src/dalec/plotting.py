"""Shared matplotlib style for every figure this project emits.

One style, applied everywhere, so that figures from different diagnostics sit
together in a thesis chapter without looking like they came from different
projects.

Choices, and why
----------------
* **No seaborn.** It is not a dependency of this project and its themes override
  rcParams globally in ways that are hard to undo. Everything here is plain
  matplotlib.
* **Sans-serif faces.** DejaVu Sans ships with matplotlib, so a figure renders
  identically on a laptop and on a cluster node with no font packages installed.
* **Okabe-Ito palette.** Eight colours distinguishable under the three common
  forms of colour blindness. Ordered so the first few are the ones used most.
* **300 dpi, and both formats.** ``.pdf`` for the thesis, ``.png`` for pasting
  into a message or a slide.
* **Units on every axis label.** A flux without a unit is not a result; see the
  unit table in ``dalec.data_io``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "FIGURE_DPI",
    "OKABE_ITO",
    "SEASON_COLOURS",
    "apply_style",
    "save_figure",
]

#: Okabe & Ito (2008) qualitative palette, colourblind-safe.
OKABE_ITO: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

#: Meteorological seasons, in a fixed order with fixed colours so that any
#: figure splitting by season uses the same mapping.
SEASON_COLOURS: Final[dict[str, str]] = {
    "DJF": "#0072B2",
    "MAM": "#009E73",
    "JJA": "#D55E00",
    "SON": "#E69F00",
}

FIGURE_DPI: Final[int] = 300


def apply_style() -> None:
    """Install the project style into the global rcParams.

    Call once at the top of a plotting script. Idempotent.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.titleweight": "semibold",
            "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO)),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.4,
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(figure: plt.Figure, directory: str | Path, stem: str) -> list[Path]:
    """Write ``figure`` as both ``.pdf`` and ``.png``. Returns the paths written.

    Parameters
    ----------
    figure
        The figure to save.
    directory
        Output directory; created if absent.
    stem
        Filename without extension, e.g. ``"fig11_randunc"``.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        figure.savefig(path)
        written.append(path)
    return written


def season_of(month: Any) -> Any:
    """Map a month number (1-12) to its meteorological season label."""
    import numpy as np

    months = np.asarray(month)
    labels = np.full(months.shape, "DJF", dtype="<U3")
    labels[(months >= 3) & (months <= 5)] = "MAM"
    labels[(months >= 6) & (months <= 8)] = "JJA"
    labels[(months >= 9) & (months <= 11)] = "SON"
    return labels
