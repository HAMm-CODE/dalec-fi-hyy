"""The single place PyTensor's execution backend is pinned, and checked.

Why this module exists
----------------------
``pytensor.config.linker`` ships as ``"auto"``, and auto does not mean one thing.
It resolves to the C backend where a C++ compiler exists and to Numba where one
does not, so identical code silently runs on a different backend depending on the
machine. Worse, asking for the C backend explicitly does **not** fail when no
compiler is present: measured on this machine, ``linker="cvm"`` returns a plain
``VMLinker`` with no error and no warning, and runs a 1000-step scan gradient in
231 ms against Numba's 1.7 ms. A 130x slowdown, reported as success.

That is the failure this module is here to make loud. The backend is pinned by
name, and the resolved linker is read back and asserted at import.

What is pinned, and why Numba
-----------------------------
``numba``. There is no C++ compiler in this environment -- ``config.cxx`` is
empty and neither g++, clang++ nor cl is on PATH -- so the C backend is not
available to pin even if it were wanted. Numba is not a fallback in any
meaningful sense: on the real calibration graph it stays in nopython mode
throughout, with no op dropping into object mode.

Pinning rather than leaving ``auto`` also means that adding a compiler to a
machine later cannot silently move the project onto a different backend and
change its numbers.

Cost of the check
-----------------
Importing this module imports PyTensor, roughly 1.5 s, and the assertion itself
compiles nothing -- it reads the resolved default mode. Note the coupling this
creates: ``import dalec`` now requires PyTensor and Numba even for the pure-NumPy
forward model. That is deliberate. Discovering that a machine has no Numba at
import is far cheaper than discovering it from a sampling run that is 130x slower
than budgeted.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "EXPECTED_LINKER_CLASS",
    "PYTENSOR_LINKER",
    "LinkerConfigurationError",
    "assert_expected_linker",
    "configure_pytensor",
    "resolved_linker_class",
]

#: The linker pinned for this project. Set by name, never left at ``"auto"``.
PYTENSOR_LINKER: Final[str] = "numba"

#: The class name ``PYTENSOR_LINKER`` must resolve to. Checked, not assumed:
#: ``config.linker`` accepts a value and then resolves it, and the two need not
#: agree -- ``"cvm"`` is accepted and resolves to ``VMLinker`` when no C compiler
#: is present.
EXPECTED_LINKER_CLASS: Final[str] = "NumbaLinker"


class LinkerConfigurationError(RuntimeError):
    """Raised when PyTensor did not resolve to the pinned backend."""


def resolved_linker_class() -> str:
    """Class name of the linker PyTensor would actually use right now.

    Reads the resolved default mode rather than ``config.linker``, because the
    requested value and the resolved one differ precisely in the case worth
    catching. Compiles nothing.
    """
    import pytensor

    return type(pytensor.compile.mode.get_default_mode().linker).__name__


def assert_expected_linker() -> str:
    """Raise unless PyTensor resolved to :data:`EXPECTED_LINKER_CLASS`.

    Returns
    -------
    The resolved linker class name, when it is the expected one.

    Raises
    ------
    LinkerConfigurationError
        Naming what was requested, what it resolved to, and what that costs.
    """
    import pytensor

    resolved = resolved_linker_class()
    if resolved == EXPECTED_LINKER_CLASS:
        return resolved

    raise LinkerConfigurationError(
        f"PyTensor resolved to {resolved!r}, not {EXPECTED_LINKER_CLASS!r}.\n"
        f"  requested linker : {PYTENSOR_LINKER!r}\n"
        f"  config.linker    : {pytensor.config.linker!r}\n"
        f"  config.cxx       : {pytensor.config.cxx!r}\n"
        "\n"
        "This is the failure mode this check exists for: PyTensor accepts a "
        "linker it cannot provide and silently substitutes a slower one rather "
        "than raising. A VMLinker or PerformLinker runs the forward model and "
        "its gradient roughly 130x slower than Numba, which turns a feasible "
        "sampling run into an infeasible one while reporting success.\n"
        "\n"
        "Most likely cause: Numba is not installed or failed to import. Check "
        "`import numba` before anything else."
    )


def configure_pytensor(linker: str = PYTENSOR_LINKER) -> str:
    """Pin the linker and confirm PyTensor honoured it.

    Called once on import of this module. Safe to call again; setting the same
    value twice is a no-op.

    Parameters
    ----------
    linker
        PyTensor linker name. Defaults to :data:`PYTENSOR_LINKER`; overriding it
        is for diagnostics, not for production runs.

    Returns
    -------
    The resolved linker class name.
    """
    import pytensor

    pytensor.config.linker = linker
    return assert_expected_linker()


# Startup check. Import of this module is what makes the guarantee.
RESOLVED_LINKER: Final[str] = configure_pytensor()
