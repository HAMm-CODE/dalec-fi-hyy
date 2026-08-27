"""Tests for the pinned PyTensor backend and its first-compilation assertion.

The point of these is that the guard *fires*. A check that cannot fail is worse
than no check, because it reads like protection.

Everything that imports PyTensor in-process carries ``@pytest.mark.pytensor``, so
``pytest -m "not pytensor"`` gives a loop that touches neither PyTensor nor Numba.
Two tests are deliberately left unmarked: the constants check, which is free, and
``test_importing_dalec_does_not_import_pytensor``, which needs neither library
and guards the very property that keeps the fast loop fast.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import dalec
from dalec.compute import (
    EXPECTED_LINKER_CLASS,
    PYTENSOR_LINKER,
    LinkerConfigurationError,
    assert_expected_linker,
    compile_function,
    configure_pytensor,
    ensure_configured,
    resolved_linker_class,
)

_SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture
def restore_linker():
    """Put the linker back however the test leaves it."""
    import pytensor

    original = pytensor.config.linker
    yield
    pytensor.config.linker = original


def test_the_linker_is_pinned_by_name_not_left_on_auto() -> None:
    """"auto" resolves to C where a compiler exists and Numba where it does not."""
    assert PYTENSOR_LINKER == "numba"
    assert PYTENSOR_LINKER != "auto"


def test_importing_dalec_does_not_import_pytensor() -> None:
    """The NumPy paths must not depend on PyTensor or Numba being installed.

    Checked in a fresh interpreter, since this one has already imported both.
    """
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, r'{_SRC}'); import dalec; "
            "print('pytensor' in sys.modules, 'numba' in sys.modules)",
        ],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False False", result.stdout


@pytest.mark.pytensor
def test_first_compilation_pins_and_verifies_the_linker() -> None:
    """The guarantee is made where it is needed: when a graph is built."""
    import pytensor.tensor as pt

    x = pt.dvector("x")
    fn = compile_function([x], x.sum())

    assert type(fn.maker.mode.linker).__name__ == EXPECTED_LINKER_CLASS
    assert dalec.compute.is_configured()
    assert resolved_linker_class() == EXPECTED_LINKER_CLASS


@pytest.mark.pytensor
def test_the_assertion_actually_fires_on_the_silent_fallback(restore_linker) -> None:
    """`cvm` is accepted and silently resolves to VMLinker with no compiler.

    This is the exact case the guard exists for, so it is exercised directly
    rather than trusted.
    """
    import pytensor

    pytensor.config.linker = "cvm"
    assert resolved_linker_class() != EXPECTED_LINKER_CLASS, (
        "expected cvm to degrade silently; if it no longer does, this "
        "environment has gained a C compiler and the guard needs rethinking"
    )
    with pytest.raises(LinkerConfigurationError, match="not 'NumbaLinker'"):
        assert_expected_linker()


@pytest.mark.pytensor
def test_the_error_names_what_it_resolved_to_and_what_it_costs(restore_linker) -> None:
    """An error that does not say what to do is a slower way to be confused."""
    import pytensor

    pytensor.config.linker = "py"
    with pytest.raises(LinkerConfigurationError) as excinfo:
        assert_expected_linker()

    message = str(excinfo.value)
    assert "PerformLinker" in message, "must name what it actually resolved to"
    assert "numba" in message.lower()
    assert "130x" in message, "must say what the silent fallback costs"


@pytest.mark.pytensor
def test_configure_is_idempotent() -> None:
    assert configure_pytensor() == EXPECTED_LINKER_CLASS
    assert configure_pytensor() == EXPECTED_LINKER_CLASS
    assert ensure_configured() == EXPECTED_LINKER_CLASS


@pytest.mark.pytensor
def test_configure_restores_a_bad_pin_rather_than_leaving_it(restore_linker) -> None:
    """configure_pytensor() repairs; ensure_configured() is the once-only form.

    The distinction matters: after the first compilation ``ensure_configured``
    short-circuits, so it will not notice a linker something else has changed.
    Repairing is what ``configure_pytensor`` is for.
    """
    import pytensor

    ensure_configured()
    pytensor.config.linker = "vm"
    assert resolved_linker_class() == "VMLinker"

    assert ensure_configured() == EXPECTED_LINKER_CLASS
    assert resolved_linker_class() == "VMLinker", (
        "ensure_configured short-circuits once configured -- it is not a repair"
    )

    assert configure_pytensor() == EXPECTED_LINKER_CLASS
    assert resolved_linker_class() == EXPECTED_LINKER_CLASS
