"""Tests for grammar-derived regex patterns."""

from __future__ import annotations

import re

from edge_ir.ir import patterns


def test_rank_name_accepts_uppercase() -> None:
    assert re.fullmatch(patterns.REGEX_RANK_NAME, "S")
    assert re.fullmatch(patterns.REGEX_RANK_NAME, "ABC")


def test_rank_name_rejects_lowercase() -> None:
    assert re.fullmatch(patterns.REGEX_RANK_NAME, "s") is None
    assert re.fullmatch(patterns.REGEX_RANK_NAME, "abc") is None


def test_rank_name_rejects_digits() -> None:
    assert re.fullmatch(patterns.REGEX_RANK_NAME, "S1") is None


def test_tensor_name_accepts_valid() -> None:
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "G")
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "BFS")
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "State1")


def test_tensor_name_accepts_underscore_in_body() -> None:
    """Underscores in the body are permitted so parser-synthesized
    literal names like Lit_0, Lit_c can live alongside user-declared
    tensors. First character must still be uppercase."""
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "Lit_0")
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "Lit_c")
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "F_temp")


def test_tensor_name_rejects_lowercase_start() -> None:
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "lowercase") is None
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "g") is None
    # Leading underscore still rejected.
    assert re.fullmatch(patterns.REGEX_TENSOR_NAME, "_Lit_0") is None


def test_user_defined_name_accepts_underscore_start() -> None:
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "_private")
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "my_func")


def test_user_defined_name_accepts_hyphen_in_body() -> None:
    """Paper Section 7 UDF examples must match the regex now that hyphens
    are allowed in the body."""
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "select-any-s")
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "select-min-s")
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "user-rand-func")


def test_user_defined_name_rejects_leading_hyphen() -> None:
    """Hyphens are body-only; the first character must still be a letter
    or underscore."""
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "-foo") is None


def test_user_defined_name_rejects_digit_start() -> None:
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "1function") is None


def test_user_defined_name_rejects_space() -> None:
    """Whitespace is never permitted in a user-defined name."""
    assert re.fullmatch(patterns.REGEX_USER_DEFINED_NAME, "foo bar") is None


def test_coordinate_accepts_alphanumeric() -> None:
    assert re.fullmatch(patterns.REGEX_COORDINATE, "a")
    assert re.fullmatch(patterns.REGEX_COORDINATE, "vertex42")
    assert re.fullmatch(patterns.REGEX_COORDINATE, "0")


def test_binary_label_accepts_digits_only() -> None:
    assert re.fullmatch(patterns.REGEX_BINARY_LABEL, "0")
    assert re.fullmatch(patterns.REGEX_BINARY_LABEL, "42")


def test_binary_label_rejects_non_digits() -> None:
    assert re.fullmatch(patterns.REGEX_BINARY_LABEL, "1a") is None


def test_rank_variable_accepts_lowercase_start() -> None:
    assert re.fullmatch(patterns.REGEX_RANK_VARIABLE, "m")
    assert re.fullmatch(patterns.REGEX_RANK_VARIABLE, "sm")
    assert re.fullmatch(patterns.REGEX_RANK_VARIABLE, "m1")


def test_rank_variable_rejects_uppercase_start() -> None:
    assert re.fullmatch(patterns.REGEX_RANK_VARIABLE, "M") is None
    assert re.fullmatch(patterns.REGEX_RANK_VARIABLE, "Sx") is None


def test_rank_variable_rejects_digit_start() -> None:
    assert re.fullmatch(patterns.REGEX_RANK_VARIABLE, "1m") is None
