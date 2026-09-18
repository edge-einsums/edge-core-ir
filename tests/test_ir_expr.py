"""Tests for rank expressions and tensor projections."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    Expression,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankFunction,
    RankVariable,
    TensorProjection,
    UnaryApp,
)
from edge_ir.ir.op import (
    BuiltinUnaryOp,
    UserDefinedUnaryOp,
)

################################################################################
# RankVariable
################################################################################


def test_rank_variable_simple() -> None:
    v = RankVariable(name="m")
    assert v.kind == "var"
    assert v.name == "m"


def test_rank_variable_multi_letter() -> None:
    """Rank variables can be multi-letter, e.g., 'sm', 'dm' for source/dest."""
    v = RankVariable(name="sm")
    assert v.name == "sm"


def test_rank_variable_with_digits() -> None:
    """Rank variables can have digits after the initial lowercase letters."""
    v = RankVariable(name="m1")
    assert v.name == "m1"


def test_rank_variable_rejects_uppercase_start() -> None:
    """Uppercase starts are reserved for rank names, not variables."""
    with pytest.raises(ValidationError):
        RankVariable(name="M")


def test_rank_variable_rejects_digit_start() -> None:
    with pytest.raises(ValidationError):
        RankVariable(name="1m")


def test_rank_variable_rejects_special_chars() -> None:
    with pytest.raises(ValidationError):
        RankVariable(name="m_2")


def test_rank_variable_round_trips() -> None:
    v = RankVariable(name="m")
    serialized = v.model_dump_json()
    deserialized = RankVariable.model_validate_json(serialized)
    assert v == deserialized


################################################################################
# RankConstantLiteral
################################################################################


def test_rank_constant_literal_int() -> None:
    c = RankConstantLiteral(value=5)
    assert c.kind == "literal"
    assert c.value == 5

    # Round-trips through JSON.
    deserialized = RankConstantLiteral.model_validate_json(c.model_dump_json())
    assert deserialized == c


def test_rank_constant_literal_zero() -> None:
    c = RankConstantLiteral(value=0)
    assert c.value == 0


def test_rank_constant_literal_negative_int() -> None:
    c = RankConstantLiteral(value=-3)
    assert c.value == -3


def test_rank_constant_literal_label() -> None:
    """A literal coordinate label (e.g. `"a"` from a CoordSetEnum) is a
    RankConstantLiteral with a string value, NOT a shape symbol."""
    c = RankConstantLiteral(value="a")
    assert c.kind == "literal"
    assert c.value == "a"

    deserialized = RankConstantLiteral.model_validate_json(c.model_dump_json())
    assert deserialized == c


def test_rank_constant_literal_rejects_float() -> None:
    """A float isn't a valid coordinate value -- int or string only.
    Pins the readable two-branch ValidationError today's Pydantic gives."""
    with pytest.raises(ValidationError) as exc_info:
        RankConstantLiteral(value=3.14)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "int" in msg.lower() or "integer" in msg.lower()
    assert "str" in msg.lower() or "string" in msg.lower()
    assert "3.14" in msg


################################################################################
# RankConstantShapeSym
################################################################################


def test_rank_constant_shape_sym() -> None:
    """Shape-DSL placeholders like '|V|' use the shape_sym variant."""
    c = RankConstantShapeSym(name="|V|")
    assert c.kind == "shape_sym"
    assert c.name == "|V|"

    deserialized = RankConstantShapeSym.model_validate_json(c.model_dump_json())
    assert deserialized == c


def test_rank_constant_shape_sym_single_letter() -> None:
    """Single-letter shape parameters like 'M' or 'N'."""
    c = RankConstantShapeSym(name="M")
    assert c.kind == "shape_sym"
    assert c.name == "M"


def test_rank_constant_shape_sym_round_trips() -> None:
    c = RankConstantShapeSym(name="N")
    serialized = c.model_dump_json()
    deserialized = RankConstantShapeSym.model_validate_json(serialized)
    assert c == deserialized


################################################################################
# RankArith
################################################################################


def test_rank_arith_simple_addition() -> None:
    """m + 1: a basic shift expression."""
    expr = RankArith(
        op="+",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=1),
    )
    assert expr.kind == "arith"
    assert expr.op == "+"


def test_rank_arith_subtraction() -> None:
    expr = RankArith(
        op="-",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=1),
    )
    assert expr.op == "-"


def test_rank_arith_multiplication() -> None:
    expr = RankArith(
        op="*",
        lhs=RankConstantLiteral(value=2),
        rhs=RankVariable(name="k"),
    )
    assert expr.op == "*"


def test_rank_arith_with_shape_dsl_constant() -> None:
    """k * |V|: variable times a shape parameter."""
    expr = RankArith(
        op="*",
        lhs=RankVariable(name="k"),
        rhs=RankConstantShapeSym(name="|V|"),
    )
    assert isinstance(expr.rhs, RankConstantShapeSym)
    assert expr.rhs.name == "|V|"


def test_rank_arith_modulo() -> None:
    """k % N: useful for periodic/wrapping coordinates."""
    expr = RankArith(
        op="%",
        lhs=RankVariable(name="k"),
        rhs=RankConstantShapeSym(name="N"),
    )
    assert expr.op == "%"


def test_rank_arith_floor_division() -> None:
    """m // 2: floor division is supported."""
    expr = RankArith(
        op="//",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=2),
    )
    assert expr.op == "//"


def test_rank_arith_division() -> None:
    expr = RankArith(
        op="/",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=2),
    )
    assert expr.op == "/"


def test_rank_arith_nested() -> None:
    """(m + 1) * 2: nested arithmetic via recursion."""
    inner = RankArith(
        op="+",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=1),
    )
    outer = RankArith(
        op="*",
        lhs=inner,
        rhs=RankConstantLiteral(value=2),
    )
    assert isinstance(outer.lhs, RankArith)
    assert outer.lhs.op == "+"


def test_rank_arith_deeply_nested() -> None:
    """((m + 1) * 2) % N: arbitrary nesting depth."""
    expr = RankArith(
        op="%",
        lhs=RankArith(
            op="*",
            lhs=RankArith(
                op="+",
                lhs=RankVariable(name="m"),
                rhs=RankConstantLiteral(value=1),
            ),
            rhs=RankConstantLiteral(value=2),
        ),
        rhs=RankConstantShapeSym(name="N"),
    )
    assert expr.op == "%"
    assert isinstance(expr.lhs, RankArith)
    assert isinstance(expr.lhs.lhs, RankArith)


def test_rank_arith_rejects_unknown_operator() -> None:
    with pytest.raises(ValidationError):
        RankArith(
            op="**",  # exponentiation not supported
            lhs=RankVariable(name="m"),
            rhs=RankConstantLiteral(value=2),
        )  # type: ignore[arg-type]


def test_rank_arith_rejects_unknown_field() -> None:
    """RankArith uses extra='forbid'. The old ``is_affine`` field was
    removed when the affine analysis moved to ``edge_ir.analysis.affine``;
    passing it (or any other unknown field) must raise."""
    with pytest.raises(ValidationError, match="is_affine"):
        RankArith(
            op="+",
            lhs=RankVariable(name="m"),
            rhs=RankConstantLiteral(value=1),
            is_affine=True,  # type: ignore[call-arg]
        )


def test_rank_arith_round_trips() -> None:
    expr = RankArith(
        op="+",
        lhs=RankVariable(name="m"),
        rhs=RankConstantShapeSym(name="|V|"),
    )
    serialized = expr.model_dump_json()
    deserialized = RankArith.model_validate_json(serialized)
    assert expr == deserialized


def test_rank_arith_round_trips_when_nested() -> None:
    expr = RankArith(
        op="*",
        lhs=RankArith(
            op="+",
            lhs=RankVariable(name="m"),
            rhs=RankConstantLiteral(value=1),
        ),
        rhs=RankConstantLiteral(value=2),
    )
    serialized = expr.model_dump_json()
    deserialized = RankArith.model_validate_json(serialized)
    assert expr == deserialized


################################################################################
# RankFunction
################################################################################


def test_rank_function_one_arg() -> None:
    """A simple shift function: add1(m). is_affine defaults to None."""
    f = RankFunction(
        func_name="add1",
        args=[RankVariable(name="m")],
    )
    assert f.kind == "func"
    assert f.func_name == "add1"
    assert len(f.args) == 1
    assert f.is_affine is None


def test_rank_function_multi_arg() -> None:
    """Non-affine: min(a, w) from the connected components example."""
    f = RankFunction(
        func_name="min",
        args=[RankVariable(name="a"), RankVariable(name="w")],
    )
    assert len(f.args) == 2


def test_rank_function_arg_can_be_rank_constant_int() -> None:
    """RankFunction args are RVEs: an integer RankConstantLiteral arg is valid.

    min(a, 0): the second arg is a RankConstantLiteral(value=0), not a variable.
    The discriminated union must round-trip it back to RankConstantLiteral, not
    coerce it into a RankVariable or drop the kind.
    """
    f = RankFunction(
        func_name="min",
        args=[RankVariable(name="a"), RankConstantLiteral(value=0)],
    )
    assert f.kind == "func"
    arg = f.args[1]
    assert isinstance(arg, RankConstantLiteral)
    assert arg.kind == "literal"
    assert arg.value == 0

    deserialized = RankFunction.model_validate_json(f.model_dump_json())
    assert deserialized == f
    round_tripped = deserialized.args[1]
    assert isinstance(round_tripped, RankConstantLiteral)
    assert round_tripped.value == 0


def test_rank_function_arg_can_be_shape_dsl_constant() -> None:
    """A shape-DSL RankConstantShapeSym (e.g. '|V|') is valid as an arg.

    f(m, |V|): the '|V|' must survive as a RankConstantShapeSym through
    the round-trip, distinct from a literal coordinate value.
    """
    f = RankFunction(
        func_name="f",
        args=[RankVariable(name="m"), RankConstantShapeSym(name="|V|")],
    )
    arg = f.args[1]
    assert isinstance(arg, RankConstantShapeSym)
    assert arg.kind == "shape_sym"
    assert arg.name == "|V|"

    deserialized = RankFunction.model_validate_json(f.model_dump_json())
    assert deserialized == f
    round_tripped = deserialized.args[1]
    assert isinstance(round_tripped, RankConstantShapeSym)
    assert round_tripped.name == "|V|"


def test_rank_function_arg_can_be_rank_arith() -> None:
    """A RankArith arg is valid: f(m + 1).

    The structured-arithmetic tree must be preserved in arg position and
    dispatch back to RankArith (not RankFunction) through the union.
    """
    f = RankFunction(
        func_name="f",
        args=[
            RankArith(
                op="+",
                lhs=RankVariable(name="m"),
                rhs=RankConstantLiteral(value=1),
            )
        ],
    )
    arg = f.args[0]
    assert isinstance(arg, RankArith)
    assert arg.kind == "arith"
    assert arg.op == "+"
    assert isinstance(arg.lhs, RankVariable)
    assert arg.lhs.name == "m"
    assert isinstance(arg.rhs, RankConstantLiteral)
    assert arg.rhs.value == 1

    deserialized = RankFunction.model_validate_json(f.model_dump_json())
    assert deserialized == f
    round_tripped = deserialized.args[0]
    assert isinstance(round_tripped, RankArith)
    assert round_tripped.op == "+"


def test_rank_function_arg_can_be_nested_rank_function() -> None:
    """A nested RankFunction arg is valid: f(g(x)).

    The discriminated union must resolve the inner func arg back to a
    RankFunction, recursively, rather than flattening it.
    """
    f = RankFunction(
        func_name="f",
        args=[
            RankFunction(
                func_name="g",
                args=[RankVariable(name="x")],
            )
        ],
    )
    arg = f.args[0]
    assert isinstance(arg, RankFunction)
    assert arg.kind == "func"
    assert arg.func_name == "g"
    assert len(arg.args) == 1
    assert isinstance(arg.args[0], RankVariable)
    assert arg.args[0].name == "x"

    deserialized = RankFunction.model_validate_json(f.model_dump_json())
    assert deserialized == f
    round_tripped = deserialized.args[0]
    assert isinstance(round_tripped, RankFunction)
    assert round_tripped.func_name == "g"
    assert isinstance(round_tripped.args[0], RankVariable)
    assert round_tripped.args[0].name == "x"


def test_rank_function_with_affine_tag() -> None:
    """is_affine can be set at construction for hand-built IR."""
    f = RankFunction(
        func_name="add1",
        args=[RankVariable(name="m")],
        is_affine=True,
    )
    assert f.is_affine is True


def test_rank_function_rejects_zero_args() -> None:
    """A rank function with no rank-variable arguments isn't a valid RVE."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        RankFunction(func_name="something", args=[])


def test_rank_function_rejects_invalid_name() -> None:
    with pytest.raises(ValidationError, match="should match pattern"):
        RankFunction(
            func_name="1bad",  # can't start with digit
            args=[RankVariable(name="m")],
        )


def test_rank_function_round_trips() -> None:
    f = RankFunction(
        func_name="min",
        args=[RankVariable(name="a"), RankVariable(name="w")],
    )
    serialized = f.model_dump_json()
    deserialized = RankFunction.model_validate_json(serialized)
    assert f == deserialized


################################################################################
# RankExpression discriminated dispatch from JSON
################################################################################


def test_rank_expression_dispatches_to_var() -> None:
    """A bare rank variable should deserialize as RankVariable."""
    raw_json = json.dumps(
        {
            "tensor": "A",
            "ranks": [{"kind": "var", "name": "m"}],
        }
    )
    proj = TensorProjection.model_validate_json(raw_json)
    assert len(proj.ranks) == 1
    assert isinstance(proj.ranks[0], RankVariable)
    assert proj.ranks[0].name == "m"


def test_rank_expression_dispatches_to_literal() -> None:
    """{kind: 'literal', value: 5} dispatches to RankConstantLiteral."""
    raw_json = json.dumps(
        {
            "tensor": "A",
            "ranks": [{"kind": "literal", "value": 5}],
        }
    )
    proj = TensorProjection.model_validate_json(raw_json)
    assert isinstance(proj.ranks[0], RankConstantLiteral)
    assert proj.ranks[0].value == 5


def test_rank_expression_dispatches_to_shape_sym() -> None:
    """{kind: 'shape_sym', name: '|V|'} dispatches to RankConstantShapeSym."""
    raw_json = json.dumps(
        {
            "tensor": "A",
            "ranks": [{"kind": "shape_sym", "name": "|V|"}],
        }
    )
    proj = TensorProjection.model_validate_json(raw_json)
    assert isinstance(proj.ranks[0], RankConstantShapeSym)
    assert proj.ranks[0].name == "|V|"


def test_rank_expression_dispatches_to_arith() -> None:
    raw_json = json.dumps(
        {
            "tensor": "A",
            "ranks": [
                {
                    "kind": "arith",
                    "op": "+",
                    "lhs": {"kind": "var", "name": "m"},
                    "rhs": {"kind": "literal", "value": 1},
                }
            ],
        }
    )
    proj = TensorProjection.model_validate_json(raw_json)
    assert isinstance(proj.ranks[0], RankArith)
    assert proj.ranks[0].op == "+"


def test_rank_expression_dispatches_to_func() -> None:
    raw_json = json.dumps(
        {
            "tensor": "A",
            "ranks": [
                {
                    "kind": "func",
                    "func_name": "min",
                    "args": [
                        {"kind": "var", "name": "a"},
                        {"kind": "var", "name": "w"},
                    ],
                }
            ],
        }
    )
    proj = TensorProjection.model_validate_json(raw_json)
    assert isinstance(proj.ranks[0], RankFunction)
    assert proj.ranks[0].func_name == "min"


def test_rank_expression_dispatches_with_nested_arith() -> None:
    """Nested RankArith should dispatch correctly through the union at
    each level of recursion."""
    raw_json = json.dumps(
        {
            "tensor": "A",
            "ranks": [
                {
                    "kind": "arith",
                    "op": "*",
                    "lhs": {
                        "kind": "arith",
                        "op": "+",
                        "lhs": {"kind": "var", "name": "m"},
                        "rhs": {"kind": "literal", "value": 1},
                    },
                    "rhs": {"kind": "literal", "value": 2},
                }
            ],
        }
    )
    proj = TensorProjection.model_validate_json(raw_json)
    outer = proj.ranks[0]
    assert isinstance(outer, RankArith)
    assert outer.op == "*"
    assert isinstance(outer.lhs, RankArith)
    assert outer.lhs.op == "+"


################################################################################
# TensorProjection
################################################################################


def test_tensor_projection_with_variables() -> None:
    """A[m, k]: the most common form."""
    proj = TensorProjection(
        tensor="A",
        ranks=[RankVariable(name="m"), RankVariable(name="k")],
    )
    assert proj.tensor == "A"
    assert len(proj.ranks) == 2


def test_tensor_projection_with_arith() -> None:
    """A[m + 1, k]: tensor projection with structured arithmetic."""
    proj = TensorProjection(
        tensor="A",
        ranks=[
            RankArith(
                op="+",
                lhs=RankVariable(name="m"),
                rhs=RankConstantLiteral(value=1),
            ),
            RankVariable(name="k"),
        ],
    )
    assert isinstance(proj.ranks[0], RankArith)
    assert isinstance(proj.ranks[1], RankVariable)


def test_tensor_projection_with_function() -> None:
    """A[min(a, w)]: tensor projection with a non-trivial RVE."""
    proj = TensorProjection(
        tensor="A",
        ranks=[
            RankFunction(
                func_name="min",
                args=[RankVariable(name="a"), RankVariable(name="w")],
            ),
        ],
    )
    assert isinstance(proj.ranks[0], RankFunction)


def test_tensor_projection_with_constant() -> None:
    """A[0, k]: a constant in a rank position (e.g., slicing)."""
    proj = TensorProjection(
        tensor="A",
        ranks=[RankConstantLiteral(value=0), RankVariable(name="k")],
    )
    assert isinstance(proj.ranks[0], RankConstantLiteral)


def test_tensor_projection_scalar() -> None:
    """Scalars: tensor projection with no rank expressions."""
    proj = TensorProjection(tensor="X", ranks=[])
    assert proj.ranks == []


def test_tensor_projection_rejects_lowercase_tensor() -> None:
    with pytest.raises(ValidationError):
        TensorProjection(tensor="lowercase", ranks=[])


def test_tensor_projection_round_trips() -> None:
    proj = TensorProjection(
        tensor="A",
        ranks=[RankVariable(name="m"), RankVariable(name="k")],
    )
    serialized = proj.model_dump_json()
    deserialized = TensorProjection.model_validate_json(serialized)
    assert proj == deserialized


def test_tensor_projection_round_trips_with_mixed_rank_kinds() -> None:
    """A[m + 1, min(a, w), 0, k]: all four RankExpression variants."""
    proj = TensorProjection(
        tensor="A",
        ranks=[
            RankArith(
                op="+",
                lhs=RankVariable(name="m"),
                rhs=RankConstantLiteral(value=1),
            ),
            RankFunction(
                func_name="min",
                args=[RankVariable(name="a"), RankVariable(name="w")],
            ),
            RankConstantLiteral(value=0),
            RankVariable(name="k"),
        ],
    )
    serialized = proj.model_dump_json()
    deserialized = TensorProjection.model_validate_json(serialized)
    assert proj == deserialized


################################################################################
# InputTensor
################################################################################


def test_input_tensor_constructs_with_simple_projection() -> None:
    """An InputTensor wraps a TensorProjection and tags itself as 'input'."""
    proj = TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    inp = InputTensor(proj=proj)
    assert inp.kind == "input"
    assert inp.proj == proj


def test_input_tensor_kind_defaults_to_input() -> None:
    """The 'kind' discriminator defaults to 'input' without explicit kwarg."""
    inp = InputTensor(proj=TensorProjection(tensor="X", ranks=[]))
    assert inp.kind == "input"


def test_input_tensor_is_frozen() -> None:
    """IRBase nodes are frozen; setattr after construction must raise."""
    inp = InputTensor(proj=TensorProjection(tensor="X", ranks=[]))
    with pytest.raises(ValidationError):
        inp.proj = TensorProjection(tensor="Y", ranks=[])  # type: ignore[misc]


def test_input_tensor_round_trips() -> None:
    """JSON serialization round-trips preserve InputTensor structure."""
    inp = InputTensor(
        proj=TensorProjection(
            tensor="A",
            ranks=[RankVariable(name="m"), RankVariable(name="k")],
        )
    )
    serialized = inp.model_dump_json()
    deserialized = InputTensor.model_validate_json(serialized)
    assert inp == deserialized


################################################################################
# UnaryApp
################################################################################


def test_unary_app_with_builtin_not() -> None:
    """The builtin negation 'not' wrapping an InputTensor is valid."""
    operand = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    app = UnaryApp(op=BuiltinUnaryOp(symbol="not"), operand=operand)
    assert app.kind == "unary"
    assert app.operand == operand
    assert isinstance(app.op, BuiltinUnaryOp)
    assert app.op.symbol == "not"


def test_unary_app_with_user_defined_unary() -> None:
    """User-defined unary functions wrap an InputTensor."""
    operand = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    app = UnaryApp(op=UserDefinedUnaryOp(name="sqrt"), operand=operand)
    assert isinstance(app.op, UserDefinedUnaryOp)
    assert app.op.name == "sqrt"


def test_unary_app_rejects_binary_app_as_operand() -> None:
    """Per grammar, the unary's operand must be an InputTensor, not a
    sub-expression. Passing a BinaryApp must be rejected."""
    leaf = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    inner = BinaryApp(label=0, lhs=leaf, rhs=leaf)
    with pytest.raises(ValidationError):
        UnaryApp(
            op=BuiltinUnaryOp(symbol="not"),
            operand=inner,  # type: ignore[arg-type]
        )


def test_unary_app_rejects_raw_tensor_projection_as_operand() -> None:
    """A raw TensorProjection (no 'kind' field, not an InputTensor) must
    not satisfy the operand constraint."""
    proj = TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    with pytest.raises(ValidationError):
        UnaryApp(
            op=BuiltinUnaryOp(symbol="not"),
            operand=proj,  # type: ignore[arg-type]
        )


def test_unary_app_round_trips() -> None:
    """JSON serialization round-trips preserve UnaryApp structure."""
    app = UnaryApp(
        op=BuiltinUnaryOp(symbol="not"),
        operand=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
    )
    serialized = app.model_dump_json()
    deserialized = UnaryApp.model_validate_json(serialized)
    assert app == deserialized


################################################################################
# BinaryApp
################################################################################


def test_binary_app_with_two_input_tensors() -> None:
    """BinaryApp combines two InputTensors and tags itself with a label."""
    lhs = InputTensor(proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")]))
    rhs = InputTensor(proj=TensorProjection(tensor="B", ranks=[RankVariable(name="k")]))
    app = BinaryApp(label=3, lhs=lhs, rhs=rhs)
    assert app.kind == "binary"
    assert app.label == 3
    assert app.lhs == lhs
    assert app.rhs == rhs


def test_binary_app_accepts_negative_label() -> None:
    """Label is an unconstrained int, so negative values are allowed."""
    leaf = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    app = BinaryApp(label=-1, lhs=leaf, rhs=leaf)
    assert app.label == -1


def test_binary_app_accepts_zero_label() -> None:
    """Label zero is allowed (used as the default in many simple Einsums)."""
    leaf = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    app = BinaryApp(label=0, lhs=leaf, rhs=leaf)
    assert app.label == 0


def test_binary_app_nested_on_lhs_round_trips() -> None:
    """BinaryApp containing BinaryApp on the lhs preserves nesting through
    JSON round-trip."""
    leaf_a = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    leaf_b = InputTensor(
        proj=TensorProjection(tensor="B", ranks=[RankVariable(name="k")])
    )
    leaf_c = InputTensor(
        proj=TensorProjection(tensor="C", ranks=[RankVariable(name="n")])
    )
    inner = BinaryApp(label=1, lhs=leaf_a, rhs=leaf_b)
    outer = BinaryApp(label=2, lhs=inner, rhs=leaf_c)
    serialized = outer.model_dump_json()
    deserialized = BinaryApp.model_validate_json(serialized)
    assert outer == deserialized
    assert isinstance(deserialized.lhs, BinaryApp)
    assert deserialized.lhs.label == 1


def test_binary_app_round_trips() -> None:
    """JSON serialization round-trips preserve BinaryApp structure."""
    leaf = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    app = BinaryApp(label=7, lhs=leaf, rhs=leaf)
    serialized = app.model_dump_json()
    deserialized = BinaryApp.model_validate_json(serialized)
    assert app == deserialized


def test_binary_app_rejects_missing_label() -> None:
    """The label kwarg is required; omitting it must raise."""
    leaf = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    with pytest.raises(ValidationError):
        BinaryApp(lhs=leaf, rhs=leaf)  # type: ignore[call-arg]


################################################################################
# AnonymousTensor
################################################################################


def test_anonymous_tensor_with_simple_expression() -> None:
    """An AnonymousTensor wraps an inner expression with output ranks."""
    inner = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    anon = AnonymousTensor(
        expression=inner,
        ranks=[RankVariable(name="m")],
    )
    assert anon.kind == "anonymous"
    assert anon.expression == inner
    assert anon.ranks == [RankVariable(name="m")]


def test_anonymous_tensor_rejects_specs_field() -> None:
    """AnonymousTensor no longer carries a specs list; passing one is
    rejected by IRBase's extra='forbid'."""
    with pytest.raises(ValidationError):
        AnonymousTensor.model_validate(
            {
                "kind": "anonymous",
                "expression": {
                    "kind": "input",
                    "proj": {"tensor": "A", "ranks": [{"kind": "var", "name": "m"}]},
                },
                "ranks": [{"kind": "var", "name": "m"}],
                "specs": [],
            }
        )


def test_anonymous_tensor_allows_empty_ranks() -> None:
    """AnonymousTensor does not constrain the rank list size; empty is
    permitted at this layer."""
    inner = InputTensor(proj=TensorProjection(tensor="X", ranks=[]))
    anon = AnonymousTensor(expression=inner, ranks=[])
    assert anon.ranks == []


def test_anonymous_tensor_recursive_round_trip() -> None:
    """AnonymousTensor containing BinaryApp containing AnonymousTensor
    preserves recursive structure through JSON round-trip."""
    leaf = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    inner_anon = AnonymousTensor(
        expression=leaf,
        ranks=[RankVariable(name="m")],
    )
    binary = BinaryApp(label=1, lhs=leaf, rhs=inner_anon)
    outer_anon = AnonymousTensor(
        expression=binary,
        ranks=[RankVariable(name="m")],
    )
    serialized = outer_anon.model_dump_json()
    deserialized = AnonymousTensor.model_validate_json(serialized)
    assert outer_anon == deserialized
    # Confirm structural shape after round-trip.
    assert isinstance(deserialized.expression, BinaryApp)
    assert isinstance(deserialized.expression.rhs, AnonymousTensor)


def test_anonymous_tensor_kind_defaults_to_anonymous() -> None:
    """The 'kind' discriminator defaults to 'anonymous' without explicit kwarg."""
    anon = AnonymousTensor(
        expression=InputTensor(proj=TensorProjection(tensor="X", ranks=[])),
        ranks=[],
    )
    assert anon.kind == "anonymous"


################################################################################
# Expression union (recursive)
################################################################################


def test_expression_union_dispatches_to_input_tensor() -> None:
    """{kind: 'input', ...} dispatches through the union to InputTensor."""
    adapter: TypeAdapter[Expression] = TypeAdapter(Expression)
    raw = {
        "kind": "input",
        "proj": {"tensor": "A", "ranks": [{"kind": "var", "name": "m"}]},
    }
    obj = adapter.validate_python(raw)
    assert isinstance(obj, InputTensor)
    assert obj.proj.tensor == "A"


def test_expression_union_dispatches_to_unary_app() -> None:
    """{kind: 'unary', ...} dispatches through the union to UnaryApp."""
    adapter: TypeAdapter[Expression] = TypeAdapter(Expression)
    raw = {
        "kind": "unary",
        "op": {"kind": "builtin", "symbol": "not"},
        "operand": {
            "kind": "input",
            "proj": {"tensor": "A", "ranks": [{"kind": "var", "name": "m"}]},
        },
    }
    obj = adapter.validate_python(raw)
    assert isinstance(obj, UnaryApp)
    assert isinstance(obj.operand, InputTensor)


def test_expression_union_dispatches_to_binary_app() -> None:
    """{kind: 'binary', ...} dispatches through the union to BinaryApp."""
    adapter: TypeAdapter[Expression] = TypeAdapter(Expression)
    leaf = {
        "kind": "input",
        "proj": {"tensor": "A", "ranks": [{"kind": "var", "name": "m"}]},
    }
    raw = {"kind": "binary", "label": 5, "lhs": leaf, "rhs": leaf}
    obj = adapter.validate_python(raw)
    assert isinstance(obj, BinaryApp)
    assert obj.label == 5


def test_expression_union_dispatches_to_anonymous_tensor() -> None:
    """{kind: 'anonymous', ...} dispatches through the union to AnonymousTensor."""
    adapter: TypeAdapter[Expression] = TypeAdapter(Expression)
    raw = {
        "kind": "anonymous",
        "expression": {
            "kind": "input",
            "proj": {"tensor": "A", "ranks": [{"kind": "var", "name": "m"}]},
        },
        "ranks": [{"kind": "var", "name": "m"}],
    }
    obj = adapter.validate_python(raw)
    assert isinstance(obj, AnonymousTensor)
    assert obj.ranks == [RankVariable(name="m")]


def test_expression_union_rejects_unknown_kind() -> None:
    """A discriminator value outside the four known kinds must raise."""
    adapter: TypeAdapter[Expression] = TypeAdapter(Expression)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "weird"})


################################################################################
# Recursive round-trip integration
################################################################################


def test_deep_expression_tree_round_trips_through_union() -> None:
    """A heterogeneous tree mixing all four Expression variants survives
    a JSON round-trip through TypeAdapter[Expression]."""
    adapter: TypeAdapter[Expression] = TypeAdapter(Expression)
    leaf_a = InputTensor(
        proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
    )
    leaf_b = InputTensor(
        proj=TensorProjection(tensor="B", ranks=[RankVariable(name="k")])
    )
    inner_unary = UnaryApp(op=BuiltinUnaryOp(symbol="not"), operand=leaf_b)
    inner_binary = BinaryApp(label=2, lhs=leaf_a, rhs=inner_unary)
    inner_anon_binary = BinaryApp(label=3, lhs=leaf_a, rhs=leaf_b)
    anon = AnonymousTensor(
        expression=inner_anon_binary,
        ranks=[RankVariable(name="m")],
    )
    root = BinaryApp(label=1, lhs=inner_binary, rhs=anon)

    raw_json = root.model_dump_json()
    parsed = adapter.validate_python(json.loads(raw_json))
    assert parsed == root
