"""Tests for operator categories in edge_ir.ir.op."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    BuiltinUnaryOp,
    ComputeOp,
    CoordinateOp,
    MergeOp,
    UnaryOp,
    UserDefinedComputeOp,
    UserDefinedMergeOp,
    UserDefinedUnaryOp,
)

# All 16 valid built-in merge symbols, per Appendix A.
_BUILTIN_MERGE_SYMBOLS: list[str] = [
    "pass_through",
    "no_pass",
    "intersect",
    "take_left_only",
    "take_left",
    "take_right_only",
    "take_right",
    "xor",
    "union",
    "nor",
    "xnor",
    "not_right",
    "not_left",
    "implies_left",
    "implies_right",
    "nand",
]


################################################################################
# BuiltinComputeOp
################################################################################


@pytest.mark.parametrize("symbol", ["+", "-", "*", "/"])
def test_builtin_compute_op_accepts_valid_symbol(symbol: str) -> None:
    """All four arithmetic compute symbols must construct successfully."""
    op = BuiltinComputeOp(symbol=symbol)  # type: ignore[arg-type]
    assert op.kind == "builtin"
    assert op.symbol == symbol


def test_builtin_compute_op_rejects_modulo_symbol() -> None:
    """`%` is not a compute operator (it appears in RankArith, not here)."""
    with pytest.raises(ValidationError):
        BuiltinComputeOp(symbol="%")  # type: ignore[arg-type]


def test_builtin_compute_op_rejects_empty_symbol() -> None:
    with pytest.raises(ValidationError):
        BuiltinComputeOp(symbol="")  # type: ignore[arg-type]


def test_builtin_compute_op_rejects_user_defined_kind() -> None:
    """The discriminator must be 'builtin'; supplying 'user_defined' here
    contradicts the symbol field."""
    with pytest.raises(ValidationError):
        BuiltinComputeOp(kind="user_defined", symbol="+")  # type: ignore[arg-type]


def test_builtin_compute_op_is_frozen() -> None:
    op = BuiltinComputeOp(symbol="+")
    with pytest.raises(ValidationError):
        op.symbol = "-"  # type: ignore[misc]


def test_builtin_compute_op_round_trips() -> None:
    op = BuiltinComputeOp(symbol="*")
    serialized = op.model_dump_json()
    deserialized = BuiltinComputeOp.model_validate_json(serialized)
    assert op == deserialized


################################################################################
# UserDefinedComputeOp
################################################################################


def test_user_defined_compute_op_accepts_simple_name() -> None:
    op = UserDefinedComputeOp(name="min")
    assert op.kind == "user_defined"
    assert op.name == "min"


def test_user_defined_compute_op_accepts_underscore_start() -> None:
    op = UserDefinedComputeOp(name="_private")
    assert op.name == "_private"


def test_user_defined_compute_op_accepts_alphanumeric_with_underscore() -> None:
    op = UserDefinedComputeOp(name="f1_2")
    assert op.name == "f1_2"


def test_user_defined_compute_op_rejects_digit_start() -> None:
    with pytest.raises(ValidationError):
        UserDefinedComputeOp(name="1foo")


def test_user_defined_compute_op_accepts_hyphen() -> None:
    """Hyphens are allowed in UDF bodies (paper Section 7 examples like
    `select-any-s`, `select-min-s`, `user-rand-func`)."""
    op = UserDefinedComputeOp(name="select-any-s")
    assert op.kind == "user_defined"
    assert op.name == "select-any-s"


def test_user_defined_compute_op_accepts_double_hyphen_in_body() -> None:
    """The relaxed regex permits adjacent hyphens in the body; this test
    pins down that decision so future readers know it is deliberate."""
    op = UserDefinedComputeOp(name="foo--bar")
    assert op.name == "foo--bar"


def test_user_defined_compute_op_accepts_trailing_hyphen() -> None:
    """A trailing hyphen is permitted by the regex `[A-Za-z_][A-Za-z0-9_-]*`."""
    op = UserDefinedComputeOp(name="foo-")
    assert op.name == "foo-"


def test_user_defined_compute_op_rejects_leading_hyphen() -> None:
    """The first character must be a letter or underscore; a leading hyphen
    is not allowed, even with the relaxed body."""
    with pytest.raises(ValidationError):
        UserDefinedComputeOp(name="-foo")


def test_user_defined_compute_op_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        UserDefinedComputeOp(name="")


def test_user_defined_compute_op_round_trips() -> None:
    op = UserDefinedComputeOp(name="min")
    serialized = op.model_dump_json()
    deserialized = UserDefinedComputeOp.model_validate_json(serialized)
    assert op == deserialized


################################################################################
# ComputeOp discriminated union
################################################################################


def test_compute_op_dispatches_builtin() -> None:
    """A `kind=builtin` payload must deserialize as BuiltinComputeOp."""
    adapter: TypeAdapter[ComputeOp] = TypeAdapter(ComputeOp)
    parsed = adapter.validate_json('{"kind": "builtin", "symbol": "*"}')
    assert isinstance(parsed, BuiltinComputeOp)
    assert parsed.symbol == "*"


def test_compute_op_dispatches_user_defined() -> None:
    """A `kind=user_defined` payload must deserialize as UserDefinedComputeOp."""
    adapter: TypeAdapter[ComputeOp] = TypeAdapter(ComputeOp)
    parsed = adapter.validate_json('{"kind": "user_defined", "name": "min"}')
    assert isinstance(parsed, UserDefinedComputeOp)
    assert parsed.name == "min"


def test_compute_op_rejects_missing_kind() -> None:
    adapter: TypeAdapter[ComputeOp] = TypeAdapter(ComputeOp)
    with pytest.raises(ValidationError):
        adapter.validate_json('{"symbol": "+"}')


def test_compute_op_rejects_unknown_kind() -> None:
    adapter: TypeAdapter[ComputeOp] = TypeAdapter(ComputeOp)
    with pytest.raises(ValidationError):
        adapter.validate_json('{"kind": "weird", "symbol": "+"}')


################################################################################
# BuiltinMergeOp
################################################################################


@pytest.mark.parametrize("symbol", _BUILTIN_MERGE_SYMBOLS)
def test_builtin_merge_op_accepts_all_appendix_a_symbols(symbol: str) -> None:
    """Every one of the 16 merge symbols from Appendix A must construct."""
    op = BuiltinMergeOp(symbol=symbol)  # type: ignore[arg-type]
    assert op.kind == "builtin"
    assert op.symbol == symbol


def test_builtin_merge_op_rejects_uppercase_alias() -> None:
    """Symbols are exact strings; 'AND' is not a valid merge symbol."""
    with pytest.raises(ValidationError):
        BuiltinMergeOp(symbol="AND")  # type: ignore[arg-type]


def test_builtin_merge_op_rejects_close_misspelling() -> None:
    """'intersection' is close to 'intersect' but is not a valid symbol."""
    with pytest.raises(ValidationError):
        BuiltinMergeOp(symbol="intersection")  # type: ignore[arg-type]


def test_builtin_merge_op_round_trips() -> None:
    op = BuiltinMergeOp(symbol="intersect")
    serialized = op.model_dump_json()
    deserialized = BuiltinMergeOp.model_validate_json(serialized)
    assert op == deserialized


################################################################################
# UserDefinedMergeOp
################################################################################


def test_user_defined_merge_op_accepts_valid_name() -> None:
    op = UserDefinedMergeOp(name="my_merge")
    assert op.kind == "user_defined"
    assert op.name == "my_merge"


def test_user_defined_merge_op_accepts_hyphen() -> None:
    """Hyphens are allowed in UDF bodies (paper Section 7 example
    `select-min-s`)."""
    op = UserDefinedMergeOp(name="select-min-s")
    assert op.kind == "user_defined"
    assert op.name == "select-min-s"


def test_user_defined_merge_op_rejects_digit_start() -> None:
    with pytest.raises(ValidationError):
        UserDefinedMergeOp(name="1foo")


def test_user_defined_merge_op_round_trips() -> None:
    op = UserDefinedMergeOp(name="custom_merge")
    serialized = op.model_dump_json()
    deserialized = UserDefinedMergeOp.model_validate_json(serialized)
    assert op == deserialized


################################################################################
# MergeOp discriminated union
################################################################################


def test_merge_op_dispatches_builtin() -> None:
    adapter: TypeAdapter[MergeOp] = TypeAdapter(MergeOp)
    parsed = adapter.validate_json('{"kind": "builtin", "symbol": "union"}')
    assert isinstance(parsed, BuiltinMergeOp)
    assert parsed.symbol == "union"


def test_merge_op_dispatches_user_defined() -> None:
    adapter: TypeAdapter[MergeOp] = TypeAdapter(MergeOp)
    parsed = adapter.validate_json('{"kind": "user_defined", "name": "my_merge"}')
    assert isinstance(parsed, UserDefinedMergeOp)
    assert parsed.name == "my_merge"


################################################################################
# CoordinateOp (no discriminator, single class)
################################################################################


def test_coordinate_op_accepts_valid_name() -> None:
    op = CoordinateOp(name="label_select")
    assert op.name == "label_select"


def test_coordinate_op_accepts_hyphen() -> None:
    """Hyphens are allowed in UDF bodies (paper Section 7 example
    `select-any-s` used as a coordinate-mapping function)."""
    op = CoordinateOp(name="select-any-s")
    assert op.name == "select-any-s"


def test_coordinate_op_rejects_digit_start() -> None:
    with pytest.raises(ValidationError):
        CoordinateOp(name="1foo")


def test_coordinate_op_rejects_whitespace_in_name() -> None:
    with pytest.raises(ValidationError):
        CoordinateOp(name="foo bar")


def test_coordinate_op_rejects_kind_field() -> None:
    """CoordinateOp has NO `kind` discriminator. With extra=forbid, supplying
    a `kind` field must raise — this guards against the union pattern leaking
    into the coordinate category."""
    with pytest.raises(ValidationError):
        CoordinateOp(name="label_select", kind="user_defined")  # type: ignore[call-arg]


def test_coordinate_op_round_trips() -> None:
    op = CoordinateOp(name="label_select")
    serialized = op.model_dump_json()
    deserialized = CoordinateOp.model_validate_json(serialized)
    assert op == deserialized


################################################################################
# BuiltinUnaryOp
################################################################################


def test_builtin_unary_op_accepts_not() -> None:
    op = BuiltinUnaryOp(symbol="not")
    assert op.kind == "builtin"
    assert op.symbol == "not"


def test_builtin_unary_op_rejects_neg() -> None:
    """'neg' is not the canonical IR symbol; the only built-in unary is 'not'."""
    with pytest.raises(ValidationError):
        BuiltinUnaryOp(symbol="neg")  # type: ignore[arg-type]


def test_builtin_unary_op_round_trips() -> None:
    op = BuiltinUnaryOp(symbol="not")
    serialized = op.model_dump_json()
    deserialized = BuiltinUnaryOp.model_validate_json(serialized)
    assert op == deserialized


################################################################################
# UserDefinedUnaryOp
################################################################################


def test_user_defined_unary_op_accepts_valid_name() -> None:
    op = UserDefinedUnaryOp(name="sqrt")
    assert op.kind == "user_defined"
    assert op.name == "sqrt"


def test_user_defined_unary_op_accepts_hyphen() -> None:
    """Hyphens are allowed in UDF bodies (paper Section 7 example
    `user-rand-func`)."""
    op = UserDefinedUnaryOp(name="user-rand-func")
    assert op.kind == "user_defined"
    assert op.name == "user-rand-func"


def test_user_defined_unary_op_rejects_digit_start() -> None:
    with pytest.raises(ValidationError):
        UserDefinedUnaryOp(name="1sqrt")


def test_user_defined_unary_op_round_trips() -> None:
    op = UserDefinedUnaryOp(name="sqrt")
    serialized = op.model_dump_json()
    deserialized = UserDefinedUnaryOp.model_validate_json(serialized)
    assert op == deserialized


################################################################################
# UnaryOp discriminated union
################################################################################


def test_unary_op_dispatches_builtin() -> None:
    adapter: TypeAdapter[UnaryOp] = TypeAdapter(UnaryOp)
    parsed = adapter.validate_json('{"kind": "builtin", "symbol": "not"}')
    assert isinstance(parsed, BuiltinUnaryOp)
    assert parsed.symbol == "not"


def test_unary_op_dispatches_user_defined() -> None:
    adapter: TypeAdapter[UnaryOp] = TypeAdapter(UnaryOp)
    parsed = adapter.validate_json('{"kind": "user_defined", "name": "sqrt"}')
    assert isinstance(parsed, UserDefinedUnaryOp)
    assert parsed.name == "sqrt"
