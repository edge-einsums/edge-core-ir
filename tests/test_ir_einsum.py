"""Tests for Einsum, StoppingCondition, and NestedCascade IR nodes."""

from __future__ import annotations

import pytest
from _stopping_helpers import bfs_stopping_condition
from pydantic import ValidationError

from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade, StoppingCondition
from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    InputTensor,
    RankConstantLiteral,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
)
from edge_ir.ir.predicate import (
    LogicalAnd,
    PredicateComparison,
    SetMembership,
)
from edge_ir.ir.stopping import (
    Comparison,
    DiamondBooleanApp,
    TensorProjectionValue,
)
from edge_ir.ir.tensor import CoordSetName

################################################################################
# Helpers
################################################################################


def _simple_input_tensor(name: str = "A", var: str = "m") -> InputTensor:
    return InputTensor(
        proj=TensorProjection(tensor=name, ranks=[RankVariable(name=var)])
    )


def _simple_binary(label: int = 1) -> BinaryApp:
    return BinaryApp(
        label=label,
        lhs=_simple_input_tensor("A", "m"),
        rhs=_simple_input_tensor("B", "m"),
    )


def _simple_map_spec(label: int = 1) -> MapSpec:
    return MapSpec(
        label=label,
        rank_list=["m"],
        compute_op=BuiltinComputeOp(symbol="+"),
        merge_op=BuiltinMergeOp(symbol="intersect"),
    )


def _simple_reduce_spec(label: int = 1) -> ReduceSpec:
    return ReduceSpec(
        label=label,
        rank_list=["m"],
        compute_op=BuiltinComputeOp(symbol="+"),
        merge_op=BuiltinMergeOp(symbol="union"),
    )


def _simple_einsum() -> Einsum:
    return Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_binary(label=1),
        specs=[_simple_map_spec(label=1)],
    )


################################################################################
# Einsum
################################################################################


def test_einsum_with_simple_input_tensor_and_no_specs() -> None:
    """An Einsum whose RHS is a bare InputTensor has no binary ops, so
    its specs list is legitimately empty."""
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("A", "m"),
        specs=[],
    )
    assert einsum.output_tensor == "C"
    assert einsum.output_ranks == [RankVariable(name="m")]
    assert isinstance(einsum.expression, InputTensor)
    assert einsum.specs == []


def test_einsum_with_binary_expression_and_map_spec() -> None:
    """A binary expression labeled k pairs with a MapSpec(label=k)."""
    binary = _simple_binary(label=1)
    spec = _simple_map_spec(label=1)
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=binary,
        specs=[spec],
    )
    assert isinstance(einsum.expression, BinaryApp)
    assert einsum.expression.label == 1
    assert len(einsum.specs) == 1
    assert isinstance(einsum.specs[0], MapSpec)
    assert einsum.specs[0].label == 1


def test_einsum_with_binary_expression_and_reduce_spec() -> None:
    """A binary expression can be paired with a ReduceSpec by label."""
    binary = _simple_binary(label=2)
    spec = _simple_reduce_spec(label=2)
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[],
        expression=binary,
        specs=[spec],
    )
    assert isinstance(einsum.specs[0], ReduceSpec)
    assert einsum.specs[0].label == 2
    assert einsum.specs[0].rank_list == ["m"]


def test_einsum_with_anonymous_tensor_in_expression_round_trips() -> None:
    """An Einsum whose expression tree contains an AnonymousTensor is
    valid and survives JSON round-trip."""
    inner = _simple_binary(label=10)
    inner_spec = _simple_map_spec(label=10)
    anon = AnonymousTensor(
        expression=inner,
        ranks=[RankVariable(name="m")],
    )
    outer = BinaryApp(label=1, lhs=_simple_input_tensor("A", "m"), rhs=anon)
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=outer,
        specs=[_simple_map_spec(label=1), inner_spec],
    )
    serialized = einsum.model_dump_json()
    deserialized = Einsum.model_validate_json(serialized)
    assert einsum == deserialized
    assert isinstance(deserialized.expression, BinaryApp)
    assert isinstance(deserialized.expression.rhs, AnonymousTensor)
    assert isinstance(deserialized.expression.rhs.expression, BinaryApp)
    assert deserialized.expression.rhs.expression.label == 10
    assert {s.label for s in deserialized.specs} == {1, 10}


def test_einsum_rejects_lowercase_output_tensor() -> None:
    """Tensor names must start with an uppercase letter (REGEX_TENSOR_NAME)."""
    with pytest.raises(ValidationError):
        Einsum(
            output_tensor="a",
            output_ranks=[RankVariable(name="m")],
            expression=_simple_input_tensor("A", "m"),
            specs=[],
        )


def test_einsum_rejects_empty_output_tensor_name() -> None:
    """Empty string violates REGEX_TENSOR_NAME (requires >=1 uppercase
    start)."""
    with pytest.raises(ValidationError):
        Einsum(
            output_tensor="",
            output_ranks=[RankVariable(name="m")],
            expression=_simple_input_tensor("A", "m"),
            specs=[],
        )


def test_einsum_rejects_digit_start_output_tensor_name() -> None:
    """Tensor names cannot start with a digit."""
    with pytest.raises(ValidationError):
        Einsum(
            output_tensor="1foo",
            output_ranks=[RankVariable(name="m")],
            expression=_simple_input_tensor("A", "m"),
            specs=[],
        )


def test_einsum_round_trips_via_json() -> None:
    """JSON round-trip preserves the full Einsum structure including
    ranks, expression, and specs."""
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m"), RankConstantLiteral(value=0)],
        expression=_simple_binary(label=1),
        specs=[_simple_map_spec(label=1)],
    )
    serialized = einsum.model_dump_json()
    deserialized = Einsum.model_validate_json(serialized)
    assert einsum == deserialized
    assert deserialized.output_tensor == "C"
    assert len(deserialized.output_ranks) == 2
    assert isinstance(deserialized.expression, BinaryApp)


################################################################################
# StoppingCondition (re-export smoke tests; full coverage in test_ir_stopping.py)
################################################################################


def test_stopping_condition_is_reexported_from_einsum_module() -> None:
    """StoppingCondition moved to edge_ir.ir.stopping but stays
    importable from edge_ir.ir.einsum for backwards compatibility."""
    from edge_ir.ir import stopping

    assert StoppingCondition is stopping.StoppingCondition


def test_stopping_condition_basic_construction_via_einsum_import() -> None:
    sc = bfs_stopping_condition()
    assert sc.rank_variable == "i"
    assert isinstance(sc.predicate, Comparison)


def test_stopping_condition_rejects_uppercase_rank_variable() -> None:
    """Rank variables follow REGEX_RANK_VARIABLE: lowercase start."""
    _lit = TensorProjectionValue(proj=TensorProjection(tensor="Lit_0", ranks=[]))
    with pytest.raises(ValidationError):
        StoppingCondition(
            rank_variable="I",
            predicate=Comparison(lhs=_lit, op="==", rhs=_lit),
        )


################################################################################
# NestedCascade
################################################################################


def test_nested_cascade_with_one_einsum_and_no_stopping_conditions() -> None:
    """The minimal NestedCascade: one einsum, no stopping conditions
    (default empty list)."""
    cascade = NestedCascade(einsums=[_simple_einsum()])
    assert len(cascade.einsums) == 1
    assert cascade.stopping_conditions == []


def test_nested_cascade_with_multiple_einsums_and_no_stopping_conditions() -> None:
    """A cascade with several einsums, no stopping conditions, models a
    plain einsum cascade (no iteration)."""
    e1 = _simple_einsum()
    e2 = Einsum(
        output_tensor="D",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("C", "m"),
        specs=[],
    )
    cascade = NestedCascade(einsums=[e1, e2])
    assert len(cascade.einsums) == 2
    assert cascade.stopping_conditions == []


def test_nested_cascade_bfs_shape_with_one_einsum_and_one_stopping_condition() -> None:
    """The BFS shape: one einsum and one stopping condition controlling
    iteration."""
    cascade = NestedCascade(
        einsums=[_simple_einsum()],
        stopping_conditions=[bfs_stopping_condition()],
    )
    assert len(cascade.einsums) == 1
    assert len(cascade.stopping_conditions) == 1
    assert cascade.stopping_conditions[0].rank_variable == "i"


def test_nested_cascade_with_multiple_einsums_and_multiple_stopping_conditions() -> (
    None
):
    """A cascade can carry multiple einsums and multiple stopping
    conditions simultaneously, including both predicate variants."""
    e1 = _simple_einsum()
    e2 = Einsum(
        output_tensor="D",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("C", "m"),
        specs=[],
    )
    sc1 = bfs_stopping_condition(rank_variable="i")
    sc2 = StoppingCondition(
        rank_variable="j",
        predicate=DiamondBooleanApp(
            name="converged",
            pinned_rank_variables=["j"],
            operands=[
                TensorProjection(tensor="D", ranks=[RankVariable(name="j")]),
            ],
        ),
    )
    cascade = NestedCascade(einsums=[e1, e2], stopping_conditions=[sc1, sc2])
    assert len(cascade.einsums) == 2
    assert len(cascade.stopping_conditions) == 2
    assert isinstance(cascade.stopping_conditions[1].predicate, DiamondBooleanApp)
    assert cascade.stopping_conditions[1].predicate.name == "converged"


def test_nested_cascade_rejects_empty_einsums_list() -> None:
    """The einsums list has min_length=1; an empty list must be rejected."""
    with pytest.raises(ValidationError):
        NestedCascade(einsums=[])


def test_nested_cascade_rejects_missing_einsums_kwarg() -> None:
    """The einsums field is mandatory; omitting it must raise."""
    with pytest.raises(ValidationError):
        NestedCascade()  # type: ignore[call-arg]


def test_nested_cascade_round_trips_via_json() -> None:
    """JSON round-trip preserves both einsums and stopping conditions."""
    cascade = NestedCascade(
        einsums=[_simple_einsum()],
        stopping_conditions=[bfs_stopping_condition()],
    )
    serialized = cascade.model_dump_json()
    deserialized = NestedCascade.model_validate_json(serialized)
    assert cascade == deserialized
    assert len(deserialized.einsums) == 1
    assert deserialized.einsums[0].output_tensor == "C"
    assert len(deserialized.stopping_conditions) == 1
    assert deserialized.stopping_conditions[0].rank_variable == "i"


################################################################################
# Einsum.predicates (restricted-iteration predicates)
################################################################################


def _membership_s_in_id() -> SetMembership:
    return SetMembership(
        member=RankVariable(name="s"),
        coord_set=CoordSetName(name="id"),
    )


def _comparison_s_lt_d() -> PredicateComparison:
    return PredicateComparison(
        lhs=RankVariable(name="s"),
        op="<",
        rhs=RankVariable(name="d"),
    )


def test_einsum_predicates_default_is_empty_list() -> None:
    """When no predicates kwarg is supplied, einsum.predicates == []."""
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("A", "m"),
        specs=[],
    )
    assert einsum.predicates == []


def test_einsum_with_empty_predicates_round_trips() -> None:
    """An Einsum with predicates=[] survives JSON round-trip; the field
    is preserved as an empty list."""
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("A", "m"),
        specs=[],
        predicates=[],
    )
    deserialized = Einsum.model_validate_json(einsum.model_dump_json())
    assert deserialized == einsum
    assert deserialized.predicates == []


def test_einsum_with_one_set_membership_predicate_round_trips() -> None:
    """An Einsum with a single SetMembership predicate round-trips
    byte-equivalently through JSON; predicates length stays 1 and
    fields are preserved."""
    pred = _membership_s_in_id()
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("A", "m"),
        specs=[],
        predicates=[pred],
    )
    assert len(einsum.predicates) == 1
    raw = einsum.model_dump_json()
    deserialized = Einsum.model_validate_json(raw)
    assert deserialized == einsum
    assert len(deserialized.predicates) == 1
    assert isinstance(deserialized.predicates[0], SetMembership)
    # And serializing again is byte-equal (no information loss).
    assert deserialized.model_dump_json() == raw


def test_einsum_with_two_predicates_is_rejected() -> None:
    """An Einsum can hold at most one predicate. To combine multiple
    conditions, the user must wrap them in LogicalAnd / LogicalOr /
    LogicalNot. The error message says "at most 1" and names the
    composition wrappers so the user knows how to fix it."""
    p1 = _membership_s_in_id()
    p2 = _comparison_s_lt_d()
    with pytest.raises(ValidationError, match=r"at most 1"):
        Einsum(
            output_tensor="C",
            output_ranks=[RankVariable(name="m")],
            expression=_simple_input_tensor("A", "m"),
            specs=[],
            predicates=[p1, p2],
        )
    # Also confirm the error names the composition wrappers so the user
    # knows what to do — no external doc reference needed.
    with pytest.raises(ValidationError, match=r"LogicalAnd"):
        Einsum(
            output_tensor="C",
            output_ranks=[RankVariable(name="m")],
            expression=_simple_input_tensor("A", "m"),
            specs=[],
            predicates=[p1, p2],
        )


def test_einsum_with_logical_and_as_single_predicate_is_accepted() -> None:
    """The canonical multi-clause form: a single LogicalAnd containing
    multiple sub-predicates is one element in the predicates list, which
    satisfies the at-most-one rule."""
    conj = LogicalAnd(operands=[_membership_s_in_id(), _comparison_s_lt_d()])
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("A", "m"),
        specs=[],
        predicates=[conj],
    )
    assert len(einsum.predicates) == 1
    assert isinstance(einsum.predicates[0], LogicalAnd)
    assert len(einsum.predicates[0].operands) == 2
    deserialized = Einsum.model_validate_json(einsum.model_dump_json())
    assert deserialized == einsum


def test_einsum_with_predicates_json_byte_round_trip() -> None:
    """An Einsum with a non-empty predicates field survives
    model_dump_json -> model_validate_json -> model_dump_json
    byte-for-byte."""
    pred = _membership_s_in_id()
    einsum = Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=_simple_input_tensor("A", "m"),
        specs=[],
        predicates=[pred],
    )
    raw_1 = einsum.model_dump_json()
    re_parsed = Einsum.model_validate_json(raw_1)
    raw_2 = re_parsed.model_dump_json()
    assert raw_1 == raw_2
    assert re_parsed == einsum
