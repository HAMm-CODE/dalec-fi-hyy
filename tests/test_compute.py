"""Tests for the pinned PyTensor backend and its startup assertion.

The point of these is that the guard *fires*. A check that cannot fail is worse
than no check, because it reads like protection.
"""

from __future__ import annotations

import pytest

import dalec
from dalec.compute import (
    EXPECTED_LINKER_CLASS,
    PYTENSOR_LINKER,
    LinkerConfigurationError,
    assert_expected_linker,
    configure_pytensor,
    resolved_linker_class,
)


def test_the_linker_is_pinned_by_name_not_left_on_auto() -> None:
    """"auto" resolves to C where a compiler exists and Numba where it does not."""
    assert PYTENSOR_LINKER == "numba"
    assert PYTENSOR_LINKER != "auto"


def test_importing_dalec_leaves_pytensor_on_numba() -> None:
    import pytensor

    assert pytensor.config.linker == PYTENSOR_LINKER
    assert dalec.RESOLVED_LINKER == EXPECTED_LINKER_CLASS
    assert resolved_linker_class() == EXPECTED_LINKER_CLASS


def test_the_assertion_actually_fires_on_the_silent_fallback() -> None:
    """`cvm` is accepted and silently resolves to VMLinker with no compiler.

    This is the exact case the guard exists for, so it is exercised directly
    rather than trusted. The linker is restored afterwards whatever happens.
    """
    import pytensor

    original = pytensor.config.linker
    try:
        pytensor.config.linker = "cvm"
        assert resolved_linker_class() != EXPECTED_LINKER_CLASS, (
            "expected cvm to degrade silently; if it no longer does, this "
            "environment has gained a C compiler and the guard needs rethinking"
        )
        with pytest.raises(LinkerConfigurationError, match="not 'NumbaLinker'"):
            assert_expected_linker()
    finally:
        pytensor.config.linker = original

    assert resolved_linker_class() == EXPECTED_LINKER_CLASS, "linker not restored"


def test_the_error_names_what_it_resolved_to_and_what_it_costs() -> None:
    """An error that does not say what to do is a slower way to be confused."""
    import pytensor

    original = pytensor.config.linker
    try:
        pytensor.config.linker = "py"
        with pytest.raises(LinkerConfigurationError) as excinfo:
            assert_expected_linker()
    finally:
        pytensor.config.linker = original

    message = str(excinfo.value)
    assert "PerformLinker" in message, "must name what it actually resolved to"
    assert "numba" in message.lower()
    assert "130x" in message, "must say what the silent fallback costs"


def test_configure_is_idempotent() -> None:
    assert configure_pytensor() == EXPECTED_LINKER_CLASS
    assert configure_pytensor() == EXPECTED_LINKER_CLASS


def test_configure_restores_a_bad_pin_rather_than_leaving_it() -> None:
    """Calling configure_pytensor() repairs a linker someone else changed."""
    import pytensor

    pytensor.config.linker = "vm"
    assert resolved_linker_class() == "VMLinker"
    assert configure_pytensor() == EXPECTED_LINKER_CLASS
