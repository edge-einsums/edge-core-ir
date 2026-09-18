"""Tests for IRBase"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edge_ir.ir.base import IRBase


class _Sample(IRBase):
    """Simple subclass to test IRBase behaviour."""

    name: str
    value: int


def test_sample_constructs_with_valid_data() -> None:
    # define an instance
    s = _Sample(name="foo", value=42)
    # the fields should set correctly
    assert s.name == "foo"
    assert s.value == 42


def test_sample_is_frozen() -> None:
    # define an instance,
    s = _Sample(name="foo", value=42)
    # try to change a field, Error!
    with pytest.raises(ValidationError):
        s.name = "bar"


def test_sample_rejects_extra_fields() -> None:
    # make sure we don't add extra fields
    with pytest.raises(ValidationError):
        _Sample(name="foo", value=42, extra_field="oops")  # type: ignore[call-arg]


def test_sample_strict_rejects_string_for_int() -> None:
    # make sure PyDantic knows to raise an error
    with pytest.raises(ValidationError):
        _Sample(name="foo", value="42")  # type: ignore[arg-type]


def test_sample_strict_rejects_int_for_string() -> None:
    with pytest.raises(ValidationError):
        # we use "type: ignore" so that `mypy` won't complain to me
        # this is a test after all.
        _Sample(name=42, value=42)  # type: ignore[arg-type]
