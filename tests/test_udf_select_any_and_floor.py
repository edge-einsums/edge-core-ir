"""``select-any`` and ``floordiv``: the two operators stamped-stack
backtracking needs and the registry did not have.

Both exist because of Knuth's Algorithm X, expressed as an EDGE cascade:

``floor``
    The search encodes a depth and a candidate row in one integer,
    ``sigma(d, r) = d * |R| + r``, so that a single max-reduction over the
    stack pops the deepest choice. Recovering the depth is written

        d* = floor( F_{i,r} / |R| )

    which is a UNARY floor applied to a division. The division is already a
    builtin, so the only missing piece is the floor -- and the UNARY category
    had no entries at all, because ``not`` is a BuiltinUnaryOp and never
    reaches the registry.

    It is deliberately NOT a fused binary ``floordiv``. That would put an
    operator in the program that the specification does not contain, so a
    viewer could not render what the author actually wrote, and it would
    compose with nothing but division.

``select-any``
    The search branches on *any* row covering the chosen column, exactly as
    max-flow pushes along *any* admissible edge. Writing that with
    ``select-min-coord`` would make the specification claim a tie-break the
    algorithm does not have, and would sample the legal executions with a
    bias -- so a bug that shows up only on some choices would never show up.

The interesting one is ``select-any``, because "any" has to be reconciled
with the evaluator's order-independence invariant. It is, by keeping the
operator a pure function of the fiber it is handed -- first arrival wins --
and leaving the choice to the order ``Schedule`` hands elements out in. That
is the same split ``ANY`` uses for reductions. The last test here is the one
that matters: it shows the schedule really is what decides.
"""

from __future__ import annotations

import pytest

from edge_ir.evaluator import build_environment, evaluate_einsum
from edge_ir.evaluator.context import PopulateAction
from edge_ir.evaluator.order import SequentialSchedule, ShuffledSchedule
from edge_ir.ir.actions import MapSpec, PopulateSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import InputTensor, RankVariable, TensorProjection
from edge_ir.ir.op import BuiltinMergeOp, CoordinateOp, UserDefinedComputeOp
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration
from edge_ir.udf.builtins import default_registry, floor, select_any
from edge_ir.udf.registry import UdfCategory


def _decl(name: str, ranks: list[str], empty: object, shape: int = 4):
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name=r, shape=shape) for r in ranks],
        data_type=BuiltinDataType(name="int"),
        empty_value=empty,
    )


################################################################################
# Registration
################################################################################


def test_floor_is_registered_as_a_unary() -> None:
    """A UNARY, not a compute operator: the tutorial applies it to a
    division, it does not fuse with one."""
    reg = default_registry()
    assert reg.impl(UdfCategory.UNARY, "floor") is not None


def test_floor_is_not_registered_as_a_compute_operator() -> None:
    """Guards against quietly reintroducing the fused form."""
    reg = default_registry()
    for cat in (UdfCategory.MAP_COMPUTE, UdfCategory.REDUCE_COMPUTE):
        for name in ("floor", "floordiv", "floor_div"):
            with pytest.raises(Exception):
                reg.impl(cat, name)


def test_floor_maps_empty_to_empty() -> None:
    """Not boilerplate: this is what makes the derived merge `take_left`,
    so floor visits only the operand's non-empty points. It holds because
    floor fixes every integer, so it cannot manufacture a present value
    where the operand had none."""
    assert default_registry().decl(UdfCategory.UNARY, "floor").maps_empty_to_empty


@pytest.mark.parametrize("name", ["select-any", "select_any", "any-1"])
def test_select_any_is_registered(name: str) -> None:
    """Hyphen, underscore and the ``-1`` glyph spelling, as select-* ships."""
    assert default_registry().impl(UdfCategory.COORDINATE, name) is not None


################################################################################
# floordiv
################################################################################


@pytest.mark.parametrize(
    ("stamp", "n_rows", "depth"),
    [
        (7, 4, 1),  # the tutorial's own example: sigma(1, 3) = 4*1 + 3
        (3, 4, 0),
        (4, 4, 1),
        (23, 6, 3),
        (0, 4, 0),
    ],
)
def test_floor_of_a_division_inverts_the_depth_stamp(stamp, n_rows, depth) -> None:
    """``d* = floor(sigma / |R|)`` recovers the depth the stamp encodes.

    Written as the two operators the specification uses -- the builtin ``/``
    produces a real quotient, and the unary floor rounds it down -- rather
    than as one fused operation.
    """
    assert floor(stamp / n_rows) == depth


def test_floor_rounds_toward_minus_infinity_not_toward_zero() -> None:
    """The distinction C-style truncation gets wrong.

    Stamps are non-negative in Algorithm X, so nothing here depends on it
    today -- which is exactly why it is worth pinning rather than leaving to
    whichever division the host language happens to use.
    """
    assert floor(-1 / 4) == -1  # truncation would give 0
    assert floor(-5 / 4) == -2  # truncation would give -1


################################################################################
# select-any, as a function
################################################################################


class _Entry:
    def __init__(self, coord, present, value=None):
        self.coord, self.present, self.value = coord, present, value


def _actions(fiber, red_coord):
    return dict(select_any(tuple(fiber), red_coord, 1, ctx=None))


def test_select_any_writes_the_arrival_into_an_empty_fiber() -> None:
    fiber = [_Entry((0,), False), _Entry((1,), False), _Entry((2,), False)]
    actions = _actions(fiber, (1,))
    assert actions[(1,)] == PopulateAction.WRITE
    assert actions[(0,)] == PopulateAction.NONE
    assert actions[(2,)] == PopulateAction.NONE


def test_select_any_keeps_the_incumbent_and_drops_later_arrivals() -> None:
    """First arrival wins. This is what makes the operator a pure function."""
    fiber = [_Entry((0,), False), _Entry((1,), True, 9), _Entry((2,), False)]
    actions = _actions(fiber, (2,))
    assert set(actions.values()) == {PopulateAction.NONE}, (
        "an occupied fiber must be left exactly as it is"
    )


def test_select_any_never_deletes() -> None:
    """Unlike select-min/max, which delete the losers.

    There is nothing to delete: at most one coordinate is ever present, so
    the incumbent is either kept or the slot is empty and gets written.
    """
    for fiber, arrival in (
        ([_Entry((0,), True, 5), _Entry((1,), False)], (1,)),
        ([_Entry((0,), False), _Entry((1,), False)], (0,)),
    ):
        assert PopulateAction.DELETE not in _actions(fiber, arrival).values()


def test_select_any_keeps_exactly_one_coordinate() -> None:
    """The invariant shared with the whole select-* family."""
    fiber = [_Entry((c,), False) for c in range(4)]
    actions = _actions(fiber, (2,))
    writes = [c for c, a in actions.items() if a == PopulateAction.WRITE]
    assert len(writes) == 1


################################################################################
# select-any, in a Populate -- and who actually chooses
################################################################################


def _pick_one_program():
    """``Z_{m*} = A_m <<<_{m*} 1(select-any)`` -- keep one of A's entries."""
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
                coord_op=CoordinateOp(name="select-any"),
            ),
        ],
    )
    return Program(
        declarations=[_decl("Z", ["M"], 0), _decl("A", ["M"], 0)],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )


def _run(schedule):
    program = _pick_one_program()
    env = build_environment(
        program,
        inputs={"A": {(0,): 10, (1,): 20, (2,): 30, (3,): 40}},
        shape_env={},
    )
    evaluate_einsum(
        program.main_edge.cascade.einsums[0], env=env, schedule=schedule
    )
    return dict(env.tensors["Z"].items())


def test_select_any_keeps_exactly_one_of_four_candidates() -> None:
    """Whatever is chosen, the result is ONE entry, and a real candidate."""
    result = _run(SequentialSchedule())
    assert len(result) == 1
    (coord, value), = result.items()
    assert {(0,): 10, (1,): 20, (2,): 30, (3,): 40}[coord] == value


def test_the_same_schedule_gives_the_same_answer() -> None:
    """Pure function of the order: reproducible, so a run can be pinned."""
    assert _run(SequentialSchedule()) == _run(SequentialSchedule())
    assert _run(ShuffledSchedule(seed=7)) == _run(ShuffledSchedule(seed=7))


def test_the_schedule_is_what_chooses() -> None:
    """The claim the whole design rests on.

    The operator contains no randomness and never inspects a value; the
    freedom lives entirely in the order Populate's fold sees its elements.
    So varying ONLY the schedule must vary which candidate survives -- if it
    did not, `select-any` would be a deterministic rule wearing the name of a
    free choice, which is the thing it exists to avoid.
    """
    chosen = {
        tuple(_run(ShuffledSchedule(seed=s))) for s in range(25)
    }
    assert len(chosen) > 1, (
        "every seed picked the same candidate; the schedule is not reaching "
        "the coordinate operator, so `select-any` is not actually free"
    )


################################################################################
# floor, in an Einsum, written the way the tutorial writes it
################################################################################


def test_depth_extraction_needs_two_einsums_and_evaluates_as_written() -> None:
    """``d* = floor( F_{i,r} / |R| )`` -- and what it actually lowers to.

    The tutorial writes depth extraction as one line. It is not one Einsum.
    ``UnaryApp.operand`` is typed ``InputTensor``, because the grammar
    restricts a unary's operand to an input tensor rather than an arbitrary
    expression (a deliberate constraint: unary functions are sugar for a
    degenerate Map). An AnonymousTensor is not a door out of this either.

    So the paper-legal form names the quotient first:

        Q_{i,r}  = F_{i,r} / N      :: map /(intersect)
        d*_{i,r} = floor( Q_{i,r} )

    This test builds exactly that and requires both halves to evaluate. It is
    the shape that matters, not the arithmetic -- if this passes, a program
    transcribed from the tutorial runs with no fused operator anywhere.

    For the viewer this is the same situation as max-flow's E04, where one
    written `cases` lowers to three IR nodes: cascade.json maps one label to a
    LIST of node indices, so the reader still sees the one Einsum they wrote.

    Stamps are sigma(d, r) = 4d + r over |R| = 4: 7 -> 1, 3 -> 0, 23 -> 5.
    """
    from edge_ir.ir.expr import BinaryApp, UnaryApp
    from edge_ir.ir.op import BuiltinComputeOp, UserDefinedUnaryOp

    m = [RankVariable(name="m")]
    quotient = Einsum(
        output_tensor="Q",
        output_ranks=m,
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(proj=TensorProjection(tensor="F", ranks=m)),
            rhs=InputTensor(proj=TensorProjection(tensor="N", ranks=[])),
        ),
        specs=[
            MapSpec(
                label=1,
                rank_list=["m"],
                compute_op=BuiltinComputeOp(symbol="/"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )
    depth = Einsum(
        output_tensor="Z",
        output_ranks=m,
        expression=UnaryApp(
            op=UserDefinedUnaryOp(name="floor"),
            operand=InputTensor(proj=TensorProjection(tensor="Q", ranks=m)),
        ),
        specs=[],
    )

    program = Program(
        declarations=[
            _decl("Z", ["M"], -1),
            _decl("Q", ["M"], -1),
            _decl("F", ["M"], -1),
            TensorDeclaration(
                name="N",
                ranks=[],
                data_type=BuiltinDataType(name="int"),
                empty_value=0,
                value=4,
            ),
        ],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[quotient, depth])),
    )

    env = build_environment(
        program, inputs={"F": {(0,): 7, (1,): 3, (2,): 23}}, shape_env={}
    )
    evaluate_einsum(quotient, env=env)
    assert dict(env.tensors["Q"].items()) == {(0,): 1.75, (1,): 0.75, (2,): 5.75}

    evaluate_einsum(depth, env=env)
    assert dict(env.tensors["Z"].items()) == {(0,): 1, (1,): 0, (2,): 5}


def test_a_unary_may_not_wrap_an_expression() -> None:
    """The constraint that forces the split above, pinned directly.

    Worth its own test because it is the thing a spec author will get wrong:
    the tutorial's notation reads as one operation, and the IR will not take
    it as one.
    """
    import pydantic

    from edge_ir.ir.expr import BinaryApp, UnaryApp
    from edge_ir.ir.op import BuiltinComputeOp, UserDefinedUnaryOp

    m = [RankVariable(name="m")]
    with pytest.raises(pydantic.ValidationError):
        UnaryApp(
            op=UserDefinedUnaryOp(name="floor"),
            operand=BinaryApp(
                label=1,
                lhs=InputTensor(proj=TensorProjection(tensor="F", ranks=m)),
                rhs=InputTensor(proj=TensorProjection(tensor="N", ranks=[])),
            ),
        )
