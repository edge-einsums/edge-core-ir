"""Populate, round by round (Inset 4), and the optional recording of it.

``evaluate_einsum(record_populate=True)`` keeps what each round did: the
mutable fiber's pre-state ``S``, the Coordinate operator's write-del set, the
write set ``W``, the delete set ``D`` and the fiber afterwards. These tests
pin the recording and, through it, the three things Inset 4 asks of a round:

(a) writes only target locations empty at the start of the round;
(b) a location deleted in this round cannot also be written in this round;
(c) future rounds may write to locations deleted in this round;

together ``W and (dom(S) or D) = {}``, checked against the pre-state.
"""

from __future__ import annotations

import pytest

from edge_ir.evaluator import (
    PopulateConstraintViolation,
    build_environment,
    evaluate_einsum,
)
from edge_ir.evaluator.context import PopulateAction
from edge_ir.evaluator.order import ShuffledSchedule
from edge_ir.ir.actions import PopulateSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import InputTensor, RankVariable, TensorProjection
from edge_ir.ir.op import CoordinateOp, UserDefinedComputeOp
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration
from edge_ir.udf.categories import UdfCategory

SHAPE = 5


def _decl(name: str) -> TensorDeclaration:
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name="M", shape=SHAPE)],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )


def _setup(coord_op: str, a: dict, z: dict | None = None):
    """``Z_{m*} = A_m :: populate_{m*} 1(coord_op)`` over one fiber."""
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
        specs=[
            PopulateSpec(
                rank_list=["m"],
                compute_op=UserDefinedComputeOp(name="identity"),
                coord_op=CoordinateOp(name=coord_op),
            )
        ],
    )
    program = Program(
        declarations=[_decl("Z"), _decl("A")],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )
    env = build_environment(program, inputs={"A": a}, shape_env={})
    for coord, value in (z or {}).items():
        env.tensors["Z"].write(coord, value)
    return einsum, env


def _register(env, name, fn) -> None:
    env.functions.register_python(
        name, UdfCategory.COORDINATE, fn, arity=3, wants_context=True
    )


def _holds(r) -> bool:
    """Inset 4, recomputed from the recorded sets."""
    w, d, dom = set(r.writes), set(r.deletes), set(r.pre_state)
    return not (w & (dom | d))


A = {(0,): 7, (1,): 3, (2,): 9, (3,): 3}


def test_recording_is_off_by_default_and_changes_nothing() -> None:
    einsum, env = _setup("select-min-val", A)
    plain = evaluate_einsum(einsum, env=env)
    assert plain.populate_rounds is None

    einsum2, env2 = _setup("select-min-val", A)
    recorded = evaluate_einsum(einsum2, env=env2, record_populate=True)
    assert recorded.populate_rounds is not None
    assert dict(env.tensors["Z"].items()) == dict(env2.tensors["Z"].items())


def test_assignment_records_no_rounds() -> None:
    """Default Populate (no ``*``) is Inset 5's assignment, not a round."""
    einsum, env = _setup("select-min-val", A)
    plain = Einsum(
        output_tensor="Z",
        output_ranks=einsum.output_ranks,
        expression=einsum.expression,
        specs=[],
    )
    assert evaluate_einsum(plain, env=env, record_populate=True).populate_rounds == []


def test_select_min_val_round_by_round() -> None:
    einsum, env = _setup("select-min-val", A)
    rounds = evaluate_einsum(einsum, env=env, record_populate=True).populate_rounds
    # One round per Reduce Temporary element, in schedule (here sorted) order.
    assert [r.output_coord for r in rounds] == [(0,), (1,), (2,), (3,)]
    assert [r.value for r in rounds] == [7, 3, 9, 3]
    assert all(r.fiber_key == () and len(r.locations) == SHAPE for r in rounds)

    first, second, third, fourth = rounds
    # Round 1: the fiber is empty, the arrival is written.
    assert (first.pre_state, first.writes, first.deletes) == ({}, {(0,): 7}, [])
    assert first.post_state == {(0,): 7}
    # Round 2: 3 < 7, so the incumbent is deleted and the arrival written.
    assert second.pre_state == {(0,): 7}
    assert (second.writes, second.deletes) == ({(1,): 3}, [(0,)])
    assert second.post_state == {(1,): 3}
    # Round 3: 9 loses; nothing is written or deleted.
    assert (third.writes, third.deletes, third.post_state) == ({}, [], {(1,): 3})
    # Round 4: a tie on the value is broken by the smaller coordinate.
    assert (fourth.writes, fourth.deletes, fourth.post_state) == ({}, [], {(1,): 3})

    # Each round starts from the state the previous one left.
    for before, after in zip(rounds, rounds[1:], strict=False):
        assert after.pre_state == before.post_state
    assert all(r.constraint_checked and _holds(r) for r in rounds)
    assert dict(env.tensors["Z"].items()) == rounds[-1].post_state
    # The write-del set has one element per location of the fiber.
    assert all(len(r.write_del) == SHAPE for r in rounds)


def test_select_any_keeps_the_first_arrival_of_the_schedule() -> None:
    """The choice is free: the schedule's order decides which one is kept."""
    kept = set()
    for seed in range(8):
        einsum, env = _setup("select-any", A)
        schedule = ShuffledSchedule(seed=seed)
        rounds = evaluate_einsum(
            einsum, env=env, schedule=schedule, record_populate=True
        ).populate_rounds
        assert sorted(r.output_coord for r in rounds) == sorted(A)
        assert rounds[0].writes == {rounds[0].output_coord: rounds[0].value}
        for r in rounds[1:]:
            assert (r.writes, r.deletes) == ({}, [])
            assert r.pre_state == rounds[0].post_state == r.post_state
        assert all(r.constraint_checked and _holds(r) for r in rounds)
        kept.add(rounds[0].output_coord)
    assert len(kept) > 1


def test_a_later_round_may_write_a_location_deleted_earlier() -> None:
    """Inset 4 (c)."""

    def bounce(fiber, coord, value, *, ctx):
        present = [e.coord for e in fiber if e.present]
        if not present:
            return [(coord, PopulateAction.WRITE)]
        target = (0,) if present[0] != (0,) else (1,)
        return [(present[0], PopulateAction.DELETE), (target, PopulateAction.WRITE)]

    einsum, env = _setup("bounce", {(0,): 1, (1,): 2, (2,): 3})
    _register(env, "bounce", bounce)
    rounds = evaluate_einsum(einsum, env=env, record_populate=True).populate_rounds
    assert [(sorted(r.writes), r.deletes) for r in rounds] == [
        ([(0,)], []),
        ([(1,)], [(0,)]),  # (0,) deleted in round 2 ...
        ([(0,)], [(1,)]),  # ... and written again in round 3
    ]
    assert all(r.constraint_checked and _holds(r) for r in rounds)
    assert dict(env.tensors["Z"].items()) == {(0,): 3}


def test_a_write_is_checked_against_the_pre_state_not_after_the_deletes() -> None:
    """Inset 4 (a): deleting a location does not free it in the same round."""

    def replace(fiber, coord, value, *, ctx):
        return [((0,), PopulateAction.DELETE), ((0,), PopulateAction.WRITE)]

    einsum, env = _setup("replace", {(1,): 5}, z={(0,): 99})
    _register(env, "replace", replace)
    with pytest.raises(PopulateConstraintViolation, match="already hold a value"):
        evaluate_einsum(einsum, env=env, record_populate=True)
    assert dict(env.tensors["Z"].items()) == {(0,): 99}  # nothing was applied


def test_a_location_cannot_be_deleted_and_written_in_one_round() -> None:
    """Inset 4 (b), on a location that is empty in the pre-state."""

    def both(fiber, coord, value, *, ctx):
        return [((2,), PopulateAction.DELETE), ((2,), PopulateAction.WRITE)]

    einsum, env = _setup("both", {(1,): 5})
    _register(env, "both", both)
    with pytest.raises(PopulateConstraintViolation, match="both Write and"):
        evaluate_einsum(einsum, env=env)
