"""Tests for restricted-iteration predicates.

The five predicate node types (SetMembership, PredicateComparison,
LogicalAnd, LogicalOr, LogicalNot) and the discriminated
IterationPredicate union. See ``edge_ir/ir/predicate.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from edge_ir.ir.expr import (
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankFunction,
    RankVariable,
)
from edge_ir.ir.predicate import (
    IterationPredicate,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    PredicateComparison,
    SetMembership,
)
from edge_ir.ir.tensor import (
    CoordSetAlias,
    CoordSetEnum,
    CoordSetInterval,
    CoordSetName,
)

################################################################################
# Helpers
################################################################################


def _basic_set_membership() -> SetMembership:
    return SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetName(name="id"),
    )


def _basic_comparison() -> PredicateComparison:
    return PredicateComparison(
        lhs=RankVariable(name="s"),
        op="<",
        rhs=RankVariable(name="d"),
    )


################################################################################
# SetMembership
################################################################################


def test_set_membership_with_rank_variable_member_round_trips() -> None:
    """Simple member: RankVariable. Round-trip preserves all fields."""
    p = SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetName(name="id"),
    )
    assert p.kind == "set_membership"
    assert isinstance(p.member, RankVariable)
    assert p.member.name == "s"
    assert isinstance(p.coord_set, CoordSetName)
    assert p.coord_set.name == "id"
    assert p.negated is False
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized == p


def test_set_membership_with_rank_arith_member_round_trips() -> None:
    """RankArith member (e.g. s + 1) survives a JSON round-trip."""
    p = SetMembership(
        member=RankArith(
            op="+",
            lhs=RankVariable(name="s"),
            rhs=RankConstantLiteral(value=1),
        ),
        coord_set=CoordSetName(name="id"),
    )
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.member, RankArith)
    assert deserialized.member.op == "+"


def test_set_membership_with_rank_function_member_round_trips() -> None:
    """RankFunction member (e.g. f(s)) survives a JSON round-trip."""
    p = SetMembership(
        member=RankFunction(
            func_name="f",
            args=[RankVariable(name="s")],
        ),
        coord_set=CoordSetName(name="id"),
    )
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.member, RankFunction)
    assert deserialized.member.func_name == "f"


def test_set_membership_with_coord_set_enum_round_trips() -> None:
    """CoordSetEnum is admissible as the coord_set field."""
    p = SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetEnum(coords=["a", "b", "c"]),
    )
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.coord_set, CoordSetEnum)
    assert deserialized.coord_set.coords == ["a", "b", "c"]


def test_set_membership_with_coord_set_interval_round_trips() -> None:
    """CoordSetInterval is admissible as the coord_set field."""
    p = SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetInterval(lo=0, hi="|V|"),
    )
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.coord_set, CoordSetInterval)
    assert deserialized.coord_set.hi == "|V|"


def test_set_membership_with_coord_set_alias_round_trips() -> None:
    """CoordSetAlias is admissible as the coord_set field."""
    p = SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetAlias(tensor="G", rank="S"),
    )
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.coord_set, CoordSetAlias)
    assert deserialized.coord_set.tensor == "G"


def test_set_membership_with_coord_set_name_round_trips() -> None:
    """CoordSetName is admissible as the coord_set field."""
    p = SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetName(name="id"),
    )
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.coord_set, CoordSetName)
    assert deserialized.coord_set.name == "id"


def test_set_membership_negated_defaults_to_false() -> None:
    """The negated field defaults to False when not supplied."""
    p = SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetName(name="id"),
    )
    assert p.negated is False


def test_set_membership_negated_true_round_trips() -> None:
    """negated=True is preserved through a JSON round-trip."""
    p = SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetName(name="id"),
        negated=True,
    )
    assert p.negated is True
    deserialized = SetMembership.model_validate_json(p.model_dump_json())
    assert deserialized.negated is True
    assert deserialized == p


################################################################################
# PredicateComparison
################################################################################


def test_predicate_comparison_var_lt_var_round_trips() -> None:
    """The L-matrix case: s < d. Round-trip preserves all fields."""
    p = PredicateComparison(
        lhs=RankVariable(name="s"),
        op="<",
        rhs=RankVariable(name="d"),
    )
    assert p.kind == "comparison"
    assert p.op == "<"
    assert isinstance(p.lhs, RankVariable)
    assert isinstance(p.rhs, RankVariable)
    assert p.lhs.name == "s"
    assert p.rhs.name == "d"
    deserialized = PredicateComparison.model_validate_json(p.model_dump_json())
    assert deserialized == p


def test_predicate_comparison_var_eq_literal_round_trips() -> None:
    """RankVariable vs RankConstantLiteral round-trip."""
    p = PredicateComparison(
        lhs=RankVariable(name="s"),
        op="==",
        rhs=RankConstantLiteral(value=0),
    )
    deserialized = PredicateComparison.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.rhs, RankConstantLiteral)
    assert deserialized.rhs.value == 0


def test_predicate_comparison_var_ge_shape_sym_round_trips() -> None:
    """RankVariable vs RankConstantShapeSym round-trip."""
    p = PredicateComparison(
        lhs=RankVariable(name="s"),
        op=">=",
        rhs=RankConstantShapeSym(name="|V|"),
    )
    deserialized = PredicateComparison.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert isinstance(deserialized.rhs, RankConstantShapeSym)
    assert deserialized.rhs.name == "|V|"


@pytest.mark.parametrize("op", ["==", "!=", ">", "<", ">=", "<="])
def test_predicate_comparison_all_ops_round_trip(op: str) -> None:
    """All six ComparisonOp values survive a JSON round-trip."""
    p = PredicateComparison(
        lhs=RankVariable(name="s"),
        op=op,  # type: ignore[arg-type]
        rhs=RankVariable(name="d"),
    )
    deserialized = PredicateComparison.model_validate_json(p.model_dump_json())
    assert deserialized == p
    assert deserialized.op == op


def test_predicate_comparison_rejects_unknown_op() -> None:
    """An op outside the six known values must be rejected."""
    with pytest.raises(
        ValidationError,
        match=r"Input should be '==', '!=', '>', '<', '>=' or '<='",
    ):
        PredicateComparison(
            lhs=RankVariable(name="s"),
            op="<<",  # type: ignore[arg-type]
            rhs=RankVariable(name="d"),
        )


################################################################################
# LogicalAnd / LogicalOr / LogicalNot
################################################################################


def test_logical_and_with_two_operands_round_trips() -> None:
    """A LogicalAnd of two heterogeneous predicates survives round-trip."""
    p = _basic_set_membership()
    q = _basic_comparison()
    conj = LogicalAnd(operands=[p, q])
    assert conj.kind == "and"
    assert len(conj.operands) == 2
    deserialized = LogicalAnd.model_validate_json(conj.model_dump_json())
    assert deserialized == conj
    assert isinstance(deserialized.operands[0], SetMembership)
    assert isinstance(deserialized.operands[1], PredicateComparison)


def test_logical_and_with_one_operand_is_rejected() -> None:
    """LogicalAnd requires at least 2 operands (min_length=2)."""
    p = _basic_set_membership()
    with pytest.raises(ValidationError):
        LogicalAnd(operands=[p])


def test_logical_and_with_zero_operands_is_rejected() -> None:
    """LogicalAnd requires at least 2 operands (min_length=2)."""
    with pytest.raises(ValidationError):
        LogicalAnd(operands=[])


def test_logical_or_with_three_operands_round_trips() -> None:
    """A LogicalOr can carry 3+ operands. Round-trip preserves all of them."""
    p = _basic_set_membership()
    q = _basic_comparison()
    r = SetMembership(
        member=RankVariable(name="d"),
        coord_set=CoordSetName(name="id"),
        negated=True,
    )
    disj = LogicalOr(operands=[p, q, r])
    assert disj.kind == "or"
    assert len(disj.operands) == 3
    deserialized = LogicalOr.model_validate_json(disj.model_dump_json())
    assert deserialized == disj
    assert len(deserialized.operands) == 3


def test_logical_or_with_one_operand_is_rejected() -> None:
    """LogicalOr requires at least 2 operands (min_length=2)."""
    p = _basic_set_membership()
    with pytest.raises(ValidationError):
        LogicalOr(operands=[p])


def test_logical_not_round_trips() -> None:
    """LogicalNot wraps a single predicate. Round-trip preserves it."""
    inner = _basic_set_membership()
    neg = LogicalNot(operand=inner)
    assert neg.kind == "not"
    deserialized = LogicalNot.model_validate_json(neg.model_dump_json())
    assert deserialized == neg
    assert isinstance(deserialized.operand, SetMembership)


def test_nested_logical_not_of_and_round_trips() -> None:
    """Nested composition: NOT(AND(SetMembership, PredicateComparison))
    survives a JSON round-trip and the inner kinds dispatch correctly."""
    p = _basic_set_membership()
    q = _basic_comparison()
    inner = LogicalAnd(operands=[p, q])
    outer = LogicalNot(operand=inner)
    raw = outer.model_dump_json()
    deserialized = LogicalNot.model_validate_json(raw)
    assert deserialized == outer
    assert isinstance(deserialized.operand, LogicalAnd)
    assert isinstance(deserialized.operand.operands[0], SetMembership)
    assert isinstance(deserialized.operand.operands[1], PredicateComparison)


################################################################################
# Discriminator dispatch via TypeAdapter[IterationPredicate]
################################################################################


def test_iteration_predicate_dispatches_to_set_membership() -> None:
    """A JSON dict with kind='set_membership' parses to SetMembership."""
    adapter: TypeAdapter[IterationPredicate] = TypeAdapter(IterationPredicate)
    raw: dict[str, Any] = {
        "kind": "set_membership",
        "member": {"kind": "var", "name": "s"},
        "coord_set": {"kind": "named", "name": "id"},
    }
    obj = adapter.validate_python(raw)
    assert isinstance(obj, SetMembership)
    assert obj.coord_set.kind == "named"
    assert isinstance(obj.coord_set, CoordSetName)
    assert obj.coord_set.name == "id"


def test_iteration_predicate_dispatches_to_comparison() -> None:
    """A JSON dict with kind='comparison' parses to PredicateComparison."""
    adapter: TypeAdapter[IterationPredicate] = TypeAdapter(IterationPredicate)
    raw: dict[str, Any] = {
        "kind": "comparison",
        "lhs": {"kind": "var", "name": "s"},
        "op": "<",
        "rhs": {"kind": "var", "name": "d"},
    }
    obj = adapter.validate_python(raw)
    assert isinstance(obj, PredicateComparison)
    assert obj.op == "<"


def test_iteration_predicate_dispatches_to_and() -> None:
    """kind='and' parses to LogicalAnd."""
    adapter: TypeAdapter[IterationPredicate] = TypeAdapter(IterationPredicate)
    raw: dict[str, Any] = {
        "kind": "and",
        "operands": [
            {
                "kind": "set_membership",
                "member": {"kind": "var", "name": "s"},
                "coord_set": {"kind": "named", "name": "id"},
            },
            {
                "kind": "comparison",
                "lhs": {"kind": "var", "name": "s"},
                "op": "<",
                "rhs": {"kind": "var", "name": "d"},
            },
        ],
    }
    obj = adapter.validate_python(raw)
    assert isinstance(obj, LogicalAnd)
    assert len(obj.operands) == 2


def test_iteration_predicate_rejects_unknown_kind() -> None:
    """A discriminator value outside the five known kinds raises."""
    adapter: TypeAdapter[IterationPredicate] = TypeAdapter(IterationPredicate)
    with pytest.raises(
        ValidationError,
        match=(
            r"Input tag 'weird' found using 'kind' "
            r"does not match any of the expected tags"
        ),
    ):
        adapter.validate_python({"kind": "weird"})


def test_iteration_predicate_list_with_mixed_kinds_round_trips() -> None:
    """A list mixing all five predicate kinds round-trips through
    TypeAdapter[list[IterationPredicate]] with each variant dispatched
    correctly on the way back."""
    sm = _basic_set_membership()
    cmp_ = _basic_comparison()
    conj = LogicalAnd(operands=[sm, cmp_])
    disj = LogicalOr(operands=[sm, cmp_])
    neg = LogicalNot(operand=sm)

    adapter: TypeAdapter[list[IterationPredicate]] = TypeAdapter(
        list[IterationPredicate]
    )
    dumped = adapter.dump_json([sm, cmp_, conj, disj, neg])
    parsed = adapter.validate_python(json.loads(dumped))
    assert len(parsed) == 5
    assert isinstance(parsed[0], SetMembership)
    assert isinstance(parsed[1], PredicateComparison)
    assert isinstance(parsed[2], LogicalAnd)
    assert isinstance(parsed[3], LogicalOr)
    assert isinstance(parsed[4], LogicalNot)
    assert parsed == [sm, cmp_, conj, disj, neg]
