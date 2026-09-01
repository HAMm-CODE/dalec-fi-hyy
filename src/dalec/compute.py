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
name, and the resolved linker is read back and asserted.

What is pinned, and why Numba
-----------------------------
``numba``. Numba is not a fallback in any meaningful sense: on the real
calibration graph it stays in nopython mode throughout, with no op dropping into
object mode.

Pinning rather than leaving ``auto`` also means that adding a compiler to a
machine later cannot silently move the project onto a different backend and
change its numbers.

**The justification for the pin is now weaker than it was, and deliberately
unchanged.** This docstring previously said the C backend was "not available to
pin even if it were wanted". That is true of the development laptop, where
``config.cxx`` is empty and no g++ is on PATH. It is **false on Roihu**, where
``g++`` resolves to the Tykky container's own compiler rather than the system
GCC, ``config.cxx`` is populated, and the C backend genuinely exists. So on the
cluster the pin is a *choice between two available backends*, not a statement
about which ones exist, and a choice wants a measurement behind it.

The pin is not being changed on that basis alone. ``scripts/21_cluster_timing.py``
times both backends on the real graph; see DECISIONS §13.

The check is lazy, and that is deliberate
-----------------------------------------
Nothing here imports PyTensor at module import. The pin and the assertion run on
**first graph compilation**, through :func:`compile_function`, which is the
sanctioned way for this project to build a PyTensor function.

The alternative -- asserting at ``import dalec`` -- was tried and reverted. It
made PyTensor and Numba hard dependencies of the pure-NumPy forward model, the
data loader and the diagnostics, none of which touch PyTensor, and it added
roughly 25 s to the test suite for a guarantee none of those code paths need.
Deferring costs nothing in safety: the guarantee is needed exactly when a graph
is compiled, and that is exactly when it is now made.

Anything compiling a PyTensor function must go through :func:`compile_function`
rather than calling ``pytensor.function`` directly. That is the one rule this
module asks for, and it is what makes "first compilation" a place that exists.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "EXPECTED_LINKER_CLASS",
    "PYTENSOR_LINKER",
    "LinkerConfigurationError",
    "assert_expected_linker",
    "compile_function",
    "configure_pytensor",
    "ensure_configured",
    "is_configured",
    "resolved_linker_class",
]

#: The linker pinned for this project. Set by name, never left at ``"auto"``.
PYTENSOR_LINKER: Final[str] = "numba"

#: The class name ``PYTENSOR_LINKER`` must resolve to. Checked, not assumed:
#: ``config.linker`` accepts a value and then resolves it, and the two need not
#: agree -- ``"cvm"`` is accepted and resolves to ``VMLinker`` when no C compiler
#: is present.
EXPECTED_LINKER_CLASS: Final[str] = "NumbaLinker"

#: Set once :func:`ensure_configured` has run. Not a cache of the answer -- the
#: answer is re-derivable at any time by :func:`resolved_linker_class` -- only a
#: record that the one-time pin has been applied.
_configured: bool = False


class LinkerConfigurationError(RuntimeError):
    """Raised when PyTensor did not resolve to the pinned backend."""


def resolved_linker_class() -> str:
    """Class name of the linker PyTensor would actually use right now.

    Reads the resolved default mode rather than ``config.linker``, because the
    requested value and the resolved one differ precisely in the case worth
    catching. Compiles nothing. Imports PyTensor, so this is not free on the
    first call.
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

    Unconditional: use this to repair a linker something else has changed.
    :func:`ensure_configured` is the once-per-process form.

    Parameters
    ----------
    linker
        PyTensor linker name. Defaults to :data:`PYTENSOR_LINKER`; overriding it
        is for diagnostics, not for production runs.

    Returns
    -------
    The resolved linker class name.
    """
    global _configured

    import pytensor

    pytensor.config.linker = linker
    resolved = assert_expected_linker()
    _configured = True
    return resolved


def is_configured() -> bool:
    """Whether the one-time pin has been applied in this process."""
    return _configured


def ensure_configured() -> str:
    """Pin and assert once per process; cheap on every call after the first.

    Called by :func:`compile_function` before any graph is built, which is what
    makes the check happen on first compilation rather than at import.
    """
    if _configured:
        return EXPECTED_LINKER_CLASS
    return configure_pytensor()


def compile_function(*args: Any, **kwargs: Any) -> Any:
    """``pytensor.function``, with the backend pinned and verified first.

    The sanctioned way to compile a PyTensor function in this project. Takes and
    returns exactly what ``pytensor.function`` does; the only difference is that
    the linker is guaranteed, and loudly so, before the graph is built.
    """
    ensure_configured()

    import pytensor

    return pytensor.function(*args, **kwargs)
