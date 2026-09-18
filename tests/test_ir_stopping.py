"""Tests for the structured stopping-condition IR nodes.

Covers the predicate tree in `edge_ir.ir.stopping`: value expressions
(TensorProjectionValue, RankExpressionValue, PropertyApp), the two
predicate variants (Comparison, DiamondBooleanApp), and the
StoppingCondition wrapper. Emphasis is on the real IR contract:
discriminated-union dispatch from raw JSON, strict-mode no-coercion,
min_length / name-pattern rejections, and lossless JSON round-trip for the
four catalog stopping conditions.

Scalars are NOT a ValueExpression variant: per the IR contract,
``ScalarValue`` was removed and literals are represented as
``TensorProjectionValue`` references to synthesized zero-rank
``TensorDeclaration`` nodes that carry the literal on their ``value`` field.
"""

from __future__ import annotations

import json

import pytest
from _stopping_helpers import (
    bfs_stopping_condition,
    fusemax_stopping_condition,
    skip_by_three_stopping_condition,
    sssp_stopping_condition,
)
from pydantic import TypeAdapter, ValidationError

from edge_ir.ir.expr import (
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankFunction,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.stopping import (
    Comparison,
    DiamondBooleanApp,
    Predicate,
    PropertyApp,
    RankExpressionValue,
    StoppingCondition,
    TensorProjectionValue,
    ValueExpression,
)

################################################################################
# Helpers
################################################################################


def _proj(tensor: str = "A", var: str = "m") -> TensorProjection:
    return TensorProjection(tensor=tensor, ranks=[RankVariable(name=var)])


def _zero_rank_proj(name: str = "Lit_0") -> TensorProjection:
    """A reference to a synthesized zero-rank tensor literal."""
    return TensorProjection(tensor=name, ranks=[])


def _lit_value(name: str = "Lit_0") -> TensorProjectionValue:
    """The ValueExpression form of a zero-rank literal reference."""
    return TensorProjectionValue(proj=_zero_rank_proj(name))


def _value_expression_adapter() -> TypeAdapter[ValueExpression]:
    return TypeAdapter(ValueExpression)


def _predicate_adapter() -> TypeAdapter[Predicate]:
    return TypeAdapter(Predicate)


################################################################################
# TensorProjectionValue
################################################################################


def test_tensor_projection_value_has_kind_and_wraps_projection() -> None:
    tv = TensorProjectionValue(proj=_proj("D", "i"))
    assert tv.kind == "tensor_value"
    assert isinstance(tv.proj, TensorProjection)
    assert tv.proj.tensor == "D"


def test_tensor_projection_value_round_trips_via_json() -> None:
    tv = TensorProjectionValue(
        proj=TensorProjection(
            tensor="F",
            ranks=[
                RankArith(
                    op="+",
                    lhs=RankVariable(name="i"),
                    rhs=RankConstantLiteral(value=1),
                )
            ],
        )
    )
    again = TensorProjectionValue.model_validate_json(tv.model_dump_json())
    assert tv == again
    assert isinstance(again.proj.ranks[0], RankArith)


def test_tensor_projection_value_rejects_non_projection() -> None:
    with pytest.raises(ValidationError):
        TensorProjectionValue(proj="F")  # type: ignore[arg-type]


def test_tensor_projection_value_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        TensorProjectionValue(proj=_proj(), bogus=1)  # type: ignore[call-arg]


def test_tensor_projection_value_accepts_zero_rank_projection() -> None:
    """A zero-rank tensor projection (used to refer to scalar literals
    synthesized as zero-rank TensorDeclarations) is a valid wrap target."""
    tv = TensorProjectionValue(proj=_zero_rank_proj("Lit_0"))
    assert tv.proj.tensor == "Lit_0"
    assert tv.proj.ranks == []


################################################################################
# ScalarValue removal
################################################################################


def test_scalar_value_is_no_longer_importable_from_stopping() -> None:
    """ScalarValue has been deleted; importing it must fail.

    This is a sanity check that the symbol is gone, not just deprecated.
    """
    with pytest.raises(ImportError):
        from edge_ir.ir.stopping import ScalarValue  # noqa: F401


def test_value_expression_union_has_exactly_three_variants() -> None:
    """The ValueExpression union now has three variants. A JSON object
    whose ``kind`` is the removed ``scalar_value`` no longer dispatches."""
    with pytest.raises(ValidationError):
        _value_expression_adapter().validate_python(
            {
                "kind": "scalar_value",
                "value": 0,
            }
        )


@pytest.mark.parametrize(
    ("raw", "expected_cls"),
    [
        # Case 1: TensorProjectionValue wrapping a zero-rank projection of F.
        # Discriminator "tensor_value" picks TensorProjectionValue;
        # `proj.tensor` names the referenced tensor and `ranks=[]` makes
        # this a scalar (zero-rank) projection.
        (
            {"kind": "tensor_value", "proj": {"tensor": "F", "ranks": []}},
            TensorProjectionValue,
        ),
        # Case 2: RankExpressionValue wrapping a bare rank variable `i`.
        # Discriminator "rank_value" picks RankExpressionValue; `expr` is
        # itself a discriminated union (here, `kind=var` picks RankVariable
        # with `name=i`).
        (
            {"kind": "rank_value", "expr": {"kind": "var", "name": "i"}},
            RankExpressionValue,
        ),
        # Case 3: PropertyApp invoking the built-in `occupancy` property on F.
        # Discriminator "property_app" picks PropertyApp; `name` is the
        # closed-Literal property name; `pinned_rank_variables` holds the
        # enclosing generational rank plus any extras the user pinned;
        # `operands` is the list of tensor projections the property is
        # applied to (one operand for occupancy).
        (
            {
                "kind": "property_app",
                "name": "occupancy",
                "pinned_rank_variables": ["i"],
                "operands": [{"tensor": "F", "ranks": []}],
            },
            PropertyApp,
        ),
    ],
)
def test_value_expression_union_three_variants_each_parse(
    raw: dict[str, object], expected_cls: type
) -> None:
    """Each of the three remaining ValueExpression variants parses to its
    concrete class via the discriminated union — AND the field values
    survive parsing. The isinstance check alone would catch a discriminator
    regression but not a "fields got dropped or coerced" regression; the
    per-variant assertions below do."""
    parsed = _value_expression_adapter().validate_python(raw)
    assert isinstance(parsed, expected_cls)
    if isinstance(parsed, TensorProjectionValue):
        assert parsed.proj.tensor == "F"
        assert parsed.proj.ranks == []
    elif isinstance(parsed, RankExpressionValue):
        assert isinstance(parsed.expr, RankVariable)
        assert parsed.expr.name == "i"
    elif isinstance(parsed, PropertyApp):
        assert parsed.name == "occupancy"
        assert parsed.pinned_rank_variables == ["i"]
        assert len(parsed.operands) == 1
        assert parsed.operands[0].tensor == "F"
        assert parsed.operands[0].ranks == []


def test_comparison_rhs_with_scalar_value_kind_json_raises() -> None:
    """A Comparison JSON whose rhs uses the removed ``scalar_value`` kind
    must fail validation end-to-end (not just at the adapter level)."""
    raw = {
        "kind": "comparison",
        "lhs": {"kind": "tensor_value", "proj": {"tensor": "F", "ranks": []}},
        "op": "==",
        "rhs": {"kind": "scalar_value", "value": 0},
    }
    with pytest.raises(ValidationError):
        Comparison.model_validate(raw)


################################################################################
# RankExpressionValue
################################################################################


def test_rank_expression_value_wraps_rank_variable() -> None:
    rv = RankExpressionValue(expr=RankVariable(name="i"))
    assert rv.kind == "rank_value"
    assert isinstance(rv.expr, RankVariable)
    assert rv.expr.name == "i"


def test_rank_expression_value_wraps_rank_arith() -> None:
    rv = RankExpressionValue(
        expr=RankArith(
            op="+",
            lhs=RankVariable(name="i"),
            rhs=RankConstantLiteral(value=1),
        )
    )
    assert isinstance(rv.expr, RankArith)
    assert rv.expr.op == "+"


def test_rank_expression_value_wraps_rank_function() -> None:
    rv = RankExpressionValue(
        expr=RankFunction(func_name="min", args=[RankVariable(name="a")])
    )
    assert isinstance(rv.expr, RankFunction)


def test_rank_expression_value_round_trips_via_json() -> None:
    rv = RankExpressionValue(
        expr=RankArith(
            op="*", lhs=RankVariable(name="k"), rhs=RankConstantShapeSym(name="|V|")
        )
    )
    again = RankExpressionValue.model_validate_json(rv.model_dump_json())
    assert rv == again
    assert isinstance(again.expr, RankArith)
    assert again.expr.rhs == RankConstantShapeSym(name="|V|")


def test_rank_expression_value_rejects_non_rank_expression() -> None:
    """A TensorProjectionValue is not a RankExpression, so it must be
    rejected when supplied as the wrapped ``expr``."""
    with pytest.raises(ValidationError):
        RankExpressionValue(expr=_lit_value())  # type: ignore[arg-type]


################################################################################
# PropertyApp
################################################################################


def test_property_app_occupancy_construction() -> None:
    pa = PropertyApp(
        name="occupancy", pinned_rank_variables=["i"], operands=[_proj("F", "i")]
    )
    assert pa.kind == "property_app"
    assert pa.name == "occupancy"
    assert pa.pinned_rank_variables == ["i"]
    assert len(pa.operands) == 1


def test_property_app_rejects_unknown_name() -> None:
    with pytest.raises(ValidationError):
        PropertyApp(
            name="cardinality",  # type: ignore[arg-type]
            pinned_rank_variables=["i"],
            operands=[_proj("F", "i")],
        )


def test_property_app_rejects_empty_pinned_rank_variables() -> None:
    with pytest.raises(ValidationError):
        PropertyApp(name="occupancy", pinned_rank_variables=[], operands=[_proj("F")])


def test_property_app_rejects_empty_operands() -> None:
    with pytest.raises(ValidationError):
        PropertyApp(name="occupancy", pinned_rank_variables=["i"], operands=[])


def test_property_app_allows_multiple_pinned_ranks_and_operands() -> None:
    pa = PropertyApp(
        name="occupancy",
        pinned_rank_variables=["i", "j"],
        operands=[_proj("F", "i"), _proj("G", "j")],
    )
    assert pa.pinned_rank_variables == ["i", "j"]
    assert len(pa.operands) == 2


def test_property_app_round_trips_via_json() -> None:
    pa = PropertyApp(
        name="occupancy",
        pinned_rank_variables=["i"],
        operands=[
            TensorProjection(
                tensor="F",
                ranks=[
                    RankArith(
                        op="+",
                        lhs=RankVariable(name="i"),
                        rhs=RankConstantLiteral(value=1),
                    )
                ],
            )
        ],
    )
    again = PropertyApp.model_validate_json(pa.model_dump_json())
    assert pa == again


################################################################################
# ValueExpression union dispatch
################################################################################


def test_value_expression_union_via_comparison_resolves_both_sides() -> None:
    raw = {
        "kind": "comparison",
        "lhs": {
            "kind": "property_app",
            "name": "occupancy",
            "pinned_rank_variables": ["i"],
            "operands": [{"tensor": "F", "ranks": []}],
        },
        "op": "==",
        "rhs": {"kind": "tensor_value", "proj": {"tensor": "Lit_0", "ranks": []}},
    }
    cmp = Comparison.model_validate(raw)
    assert isinstance(cmp.lhs, PropertyApp)
    assert isinstance(cmp.rhs, TensorProjectionValue)
    assert cmp.rhs.proj.tensor == "Lit_0"


def test_value_expression_union_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        _value_expression_adapter().validate_python({"kind": "mystery", "value": 0})


def test_value_expression_union_rejects_missing_kind() -> None:
    with pytest.raises(ValidationError):
        _value_expression_adapter().validate_python({"value": 0})


################################################################################
# Comparison
################################################################################


@pytest.mark.parametrize("op", ["==", "!=", ">", "<", ">=", "<="])
def test_comparison_accepts_each_valid_op(op: str) -> None:
    cmp = Comparison(lhs=_lit_value(), op=op, rhs=_lit_value())  # type: ignore[arg-type]
    assert cmp.kind == "comparison"
    assert cmp.op == op


@pytest.mark.parametrize("bad_op", ["≡", "===", "=", "<>", "≠", "≥", "is"])
def test_comparison_rejects_unknown_op(bad_op: str) -> None:
    """The IR uses ASCII operators only; Unicode glyphs like ≡ are
    rewritten to ASCII by the lowering pass, never stored."""
    with pytest.raises(ValidationError):
        Comparison(lhs=_lit_value(), op=bad_op, rhs=_lit_value())  # type: ignore[arg-type]


def test_comparison_round_trips_via_json() -> None:
    cmp = Comparison(
        lhs=TensorProjectionValue(proj=_proj("D", "i")),
        op="!=",
        rhs=RankExpressionValue(expr=RankConstantLiteral(value=0)),
    )
    again = Comparison.model_validate_json(cmp.model_dump_json())
    assert cmp == again
    assert isinstance(again.lhs, TensorProjectionValue)
    assert isinstance(again.rhs, RankExpressionValue)


################################################################################
# DiamondBooleanApp
################################################################################


def test_diamond_boolean_app_construction() -> None:
    app = DiamondBooleanApp(
        name="converged",
        pinned_rank_variables=["i"],
        operands=[_proj("D", "i")],
    )
    assert app.kind == "diamond_boolean_app"
    assert app.name == "converged"


def test_diamond_boolean_app_rejects_invalid_name() -> None:
    with pytest.raises(ValidationError):
        DiamondBooleanApp(
            name="1bad", pinned_rank_variables=["i"], operands=[_proj("D", "i")]
        )


def test_diamond_boolean_app_accepts_hyphenated_body() -> None:
    """Per REGEX_USER_DEFINED_NAME, hyphens are allowed in the body
    (matching the paper's UDF names like `is-zero`)."""
    app = DiamondBooleanApp(
        name="is-zero", pinned_rank_variables=["i"], operands=[_proj("F", "i")]
    )
    assert app.name == "is-zero"


def test_diamond_boolean_app_rejects_empty_operands() -> None:
    with pytest.raises(ValidationError):
        DiamondBooleanApp(name="converged", pinned_rank_variables=["i"], operands=[])


def test_diamond_boolean_app_rejects_empty_pinned_rank_variables() -> None:
    with pytest.raises(ValidationError):
        DiamondBooleanApp(
            name="converged", pinned_rank_variables=[], operands=[_proj("D", "i")]
        )


def test_diamond_boolean_app_allows_multiple_operands() -> None:
    app = DiamondBooleanApp(
        name="agree",
        pinned_rank_variables=["i"],
        operands=[_proj("D", "i"), _proj("P", "i")],
    )
    assert len(app.operands) == 2


def test_diamond_boolean_app_round_trips_via_json() -> None:
    app = DiamondBooleanApp(
        name="is-zero",
        pinned_rank_variables=["i", "j"],
        operands=[
            TensorProjection(
                tensor="F",
                ranks=[
                    RankArith(
                        op="+",
                        lhs=RankVariable(name="i"),
                        rhs=RankConstantLiteral(value=1),
                    )
                ],
            )
        ],
    )
    again = DiamondBooleanApp.model_validate_json(app.model_dump_json())
    assert app == again


################################################################################
# Predicate union dispatch
################################################################################


@pytest.mark.parametrize(
    ("raw", "expected_cls"),
    [
        (
            {
                "kind": "comparison",
                "lhs": {
                    "kind": "tensor_value",
                    "proj": {"tensor": "Lit_0", "ranks": []},
                },
                "op": "==",
                "rhs": {
                    "kind": "tensor_value",
                    "proj": {"tensor": "Lit_0", "ranks": []},
                },
            },
            Comparison,
        ),
        (
            {
                "kind": "diamond_boolean_app",
                "name": "converged",
                "pinned_rank_variables": ["i"],
                "operands": [{"tensor": "D", "ranks": []}],
            },
            DiamondBooleanApp,
        ),
    ],
)
def test_predicate_union_dispatches_to_concrete_subclass(
    raw: dict[str, object], expected_cls: type
) -> None:
    parsed = _predicate_adapter().validate_python(raw)
    assert isinstance(parsed, expected_cls)


def test_predicate_dispatch_through_stopping_condition_model_validate() -> None:
    sc = StoppingCondition.model_validate(
        {
            "rank_variable": "i",
            "predicate": {
                "kind": "diamond_boolean_app",
                "name": "converged",
                "pinned_rank_variables": ["i"],
                "operands": [{"tensor": "D", "ranks": []}],
            },
        }
    )
    assert isinstance(sc.predicate, DiamondBooleanApp)
    assert sc.predicate.name == "converged"


def test_predicate_union_rejects_dict_matching_neither_variant() -> None:
    with pytest.raises(ValidationError):
        _predicate_adapter().validate_python({"kind": "occupancy_zero"})


################################################################################
# StoppingCondition: rank_variable pattern
################################################################################


def _trivial_predicate() -> Comparison:
    return Comparison(lhs=_lit_value(), op="==", rhs=_lit_value())


@pytest.mark.parametrize("rank_variable", ["I", "Bad", "1i", "9", "", "i j", "_i"])
def test_stopping_condition_rejects_invalid_rank_variable(rank_variable: str) -> None:
    """rank_variable must match REGEX_RANK_VARIABLE: lowercase start,
    then alphanumerics."""
    with pytest.raises(ValidationError):
        StoppingCondition(rank_variable=rank_variable, predicate=_trivial_predicate())


@pytest.mark.parametrize("rank_variable", ["i", "j", "sm", "k0", "iter"])
def test_stopping_condition_accepts_valid_rank_variable(rank_variable: str) -> None:
    sc = StoppingCondition(rank_variable=rank_variable, predicate=_trivial_predicate())
    assert sc.rank_variable == rank_variable


def test_stopping_condition_predicate_rejects_dict_matching_neither_variant() -> None:
    with pytest.raises(ValidationError):
        StoppingCondition.model_validate(
            {
                "rank_variable": "i",
                "predicate": {"kind": "not_a_predicate"},
            }
        )


def test_stopping_condition_is_reexported_from_einsum_module() -> None:
    from edge_ir.ir import einsum, stopping

    assert einsum.StoppingCondition is stopping.StoppingCondition


################################################################################
# Catalog stopping conditions: end-to-end build + round-trip + structure
################################################################################


def test_bfs_stopping_condition_structure() -> None:
    sc = bfs_stopping_condition()
    assert sc.rank_variable == "i"
    assert isinstance(sc.predicate, Comparison)
    assert sc.predicate.op == "=="
    assert isinstance(sc.predicate.lhs, PropertyApp)
    assert sc.predicate.lhs.name == "occupancy"
    assert sc.predicate.lhs.operands[0].tensor == "F"
    assert isinstance(sc.predicate.lhs.operands[0].ranks[0], RankArith)
    # rhs is now a TensorProjectionValue referencing a synthesized
    # zero-rank tensor literal, not a ScalarValue.
    assert isinstance(sc.predicate.rhs, TensorProjectionValue)
    assert sc.predicate.rhs.proj.ranks == []


def test_sssp_stopping_condition_structure() -> None:
    sc = sssp_stopping_condition()
    assert isinstance(sc.predicate, Comparison)
    assert sc.predicate.op == "=="
    assert isinstance(sc.predicate.lhs, TensorProjectionValue)
    assert isinstance(sc.predicate.rhs, TensorProjectionValue)
    assert sc.predicate.lhs.proj.tensor == "D"
    assert sc.predicate.rhs.proj.tensor == "D"
    # lhs subscripts are [i+1, s]; rhs subscripts are [i, s].
    assert isinstance(sc.predicate.lhs.proj.ranks[0], RankArith)
    assert sc.predicate.rhs.proj.ranks == [
        RankVariable(name="i"),
        RankVariable(name="s"),
    ]


def test_fusemax_stopping_condition_structure() -> None:
    sc = fusemax_stopping_condition()
    assert isinstance(sc.predicate, Comparison)
    assert sc.predicate.op == ">="
    assert isinstance(sc.predicate.lhs, RankExpressionValue)
    assert isinstance(sc.predicate.rhs, RankExpressionValue)
    assert sc.predicate.lhs.expr == RankVariable(name="i")
    assert sc.predicate.rhs.expr == RankConstantShapeSym(name="K")


def test_skip_by_three_stopping_condition_structure() -> None:
    sc = skip_by_three_stopping_condition()
    assert sc.rank_variable == "i"
    assert isinstance(sc.predicate, Comparison)
    assert isinstance(sc.predicate.lhs, TensorProjectionValue)
    assert isinstance(sc.predicate.rhs, TensorProjectionValue)
    rhs_ranks = sc.predicate.rhs.proj.ranks
    assert rhs_ranks == [RankVariable(name="i")]
    lhs_rank = sc.predicate.lhs.proj.ranks[0]
    assert isinstance(lhs_rank, RankArith)
    assert lhs_rank.rhs == RankConstantLiteral(value=3)


@pytest.mark.parametrize(
    "builder",
    [
        bfs_stopping_condition,
        sssp_stopping_condition,
        fusemax_stopping_condition,
        skip_by_three_stopping_condition,
    ],
)
def test_catalog_stopping_condition_round_trips_via_json(
    builder: object,
) -> None:
    sc = builder()  # type: ignore[operator]
    again = StoppingCondition.model_validate_json(sc.model_dump_json())
    assert sc == again


def test_bfs_stopping_condition_rhs_serializes_as_tensor_value() -> None:
    """The bfs stopping-condition rhs serializes as a ``tensor_value``
    kind whose projection refers to a zero-rank synthesized literal,
    not as the removed ``scalar_value`` kind."""
    sc = bfs_stopping_condition()
    raw = json.loads(sc.model_dump_json())
    rhs = raw["predicate"]["rhs"]
    assert rhs["kind"] == "tensor_value"
    assert rhs["proj"]["ranks"] == []
    # The removed "scalar_value" kind must not appear anywhere in the rhs.
    assert "value" not in rhs
