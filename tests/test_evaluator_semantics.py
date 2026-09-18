"""The evaluator against the set-theoretic semantics of EDGE.

Each test names the passage it pins. These exercise EDGE semantics -- merge
deciding existence separately from compute deciding value, Reduce observing
the whole iteration space, Populate's write constraint -- not the shape of
the Python API.
"""

from __future__ import annotations

import math

import pytest

from edge_ir.evaluator import (
    PopulateConstraintViolation,
    Tensor,
    UnderSpecifiedProgramError,
    build_environment,
    evaluate_einsum,
)
from edge_ir.evaluator.spaces import (
    CoordinateSpace,
    RankCoordinateSet,
    is_empty_value,
)
from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankVariable,
    TensorProjection,
    UnaryApp,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    BuiltinUnaryOp,
    CoordinateOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration
from edge_ir.runtime.op_properties import MERGE_OP_PROPERTIES

################################################################################
# Builders
################################################################################


def _decl(name: str, ranks: list[str], empty: object, shape: int | str = 4):
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name=r, shape=shape) for r in ranks],
        data_type=BuiltinDataType(name="int"),
        empty_value=empty,
    )


def _binary_einsum(merge: str, compute: str = "+") -> Einsum:
    """``Z_m = A_m . B_m :: map compute(merge)``."""
    return Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
            ),
            rhs=InputTensor(
                proj=TensorProjection(tensor="B", ranks=[RankVariable(name="m")])
            ),
        ),
        specs=[
            MapSpec(
                label=1,
                compute_op=BuiltinComputeOp(symbol=compute)
                if compute in "+-*/"
                else UserDefinedComputeOp(name=compute),
                merge_op=BuiltinMergeOp(symbol=merge),  # type: ignore[arg-type]
            )
        ],
    )


def _program(einsum: Einsum, decls: list[TensorDeclaration]) -> Program:
    return Program(
        declarations=decls,
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )


################################################################################
# Tensors are partial functions into the NON-EMPTY data space
################################################################################


def test_a_tensor_cannot_hold_its_own_empty_value() -> None:
    """Tensor: ``T : CS^T -> DS^T_ne`` (partial).

    The codomain excludes ``E^T``, so "present with the empty value" is not
    a representable state -- writing the empty value removes the coordinate.
    This is what makes existence and non-emptiness the same predicate.
    """
    space = CoordinateSpace(tensor="Z", ranks=(RankCoordinateSet("M", (0, 1, 2)),))
    z = Tensor("Z", space, empty_value=0)
    z.write((0,), 5)
    z.write((1,), 0)
    assert z.exists((0,))
    assert not z.exists((1,))
    z.write((0,), 0)
    assert not z.exists((0,))
    assert z.occupancy() == 0


def test_existence_is_false_outside_the_coordinate_space() -> None:
    """Existence check: ``dom(T) subset-of CS^T``."""
    space = CoordinateSpace(tensor="Z", ranks=(RankCoordinateSet("M", (0, 1)),))
    z = Tensor("Z", space, empty_value=0)
    assert not z.exists((7,))
    assert z.value_or_empty((7,)) == 0


def test_empty_value_never_confuses_bool_with_int() -> None:
    """A Boolean tensor's ``empty=False`` must not swallow another tensor's 0."""
    assert is_empty_value(False, False)
    assert not is_empty_value(0, False)
    assert not is_empty_value(False, 0)
    assert is_empty_value(math.nan, math.nan)


################################################################################
# Map: merge decides existence, compute decides value
################################################################################


@pytest.mark.parametrize("symbol", sorted(MERGE_OP_PROPERTIES))
def test_map_presence_follows_the_merge_truth_table(symbol: str) -> None:
    """Map: the merge operator is the guard.

    All 16 merges are exercised: for each of the four presence combinations
    the output exists iff the truth table says so. The compute operator has
    no say in existence.
    """
    table = MERGE_OP_PROPERTIES[symbol].truth_table
    decls = [_decl("Z", ["M"], 0), _decl("A", ["M"], 0), _decl("B", ["M"], 0)]
    program = _program(_binary_einsum(symbol), decls)
    for left_present, right_present in table:
        inputs = {}
        if left_present:
            inputs["A"] = {(0,): 3}
        if right_present:
            inputs["B"] = {(0,): 4}
        env = build_environment(program, inputs=inputs, shape_env={})
        evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
        expected = table[(left_present, right_present)]
        # 3+4=7, 3+0=3, 0+4=4 are all non-empty; 0+0=0 is Z's empty value and
        # so is dropped by the "dv not in E^Z" rule even when merge admits it.
        both_absent = not left_present and not right_present
        assert ((0,) in env.tensors["Z"]) is (expected and not both_absent), (
            symbol,
            left_present,
            right_present,
        )


def test_map_substitutes_the_empty_value_for_a_missing_operand() -> None:
    """Map semantics: "if the evaluated value of either A or B is undefined ...
    compute substitutes the corresponding empty value as the operand"."""
    decls = [_decl("Z", ["M"], 0), _decl("A", ["M"], 0), _decl("B", ["M"], 9)]
    program = _program(_binary_einsum("union"), decls)
    env = build_environment(program, inputs={"A": {(0,): 3}}, shape_env={})
    evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
    # B is absent, so compute sees B's empty value 9: 3 + 9 = 12.
    assert env.tensors["Z"].value_or_empty((0,)) == 12


def test_map_drops_a_computed_value_that_lands_in_the_output_empty() -> None:
    """Map semantics: "If the computed value lies in the empty space of the
    output, the MapTmp element is simply omitted at this point"."""
    decls = [_decl("Z", ["M"], 7), _decl("A", ["M"], 0), _decl("B", ["M"], 0)]
    program = _program(_binary_einsum("intersect"), decls)
    env = build_environment(
        program, inputs={"A": {(0,): 3}, "B": {(0,): 4}}, shape_env={}
    )
    evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
    assert (0,) not in env.tensors["Z"]  # 3 + 4 == 7 == Z's empty value


def test_a_binary_without_a_map_spec_fails_loudly() -> None:
    """An action with no operator label is under-specification, not a default."""
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
            ),
            rhs=InputTensor(
                proj=TensorProjection(tensor="B", ranks=[RankVariable(name="m")])
            ),
        ),
        specs=[],
    )
    decls = [_decl("Z", ["M"], 0), _decl("A", ["M"], 0), _decl("B", ["M"], 0)]
    env = build_environment(_program(einsum, decls), inputs={"A": {(0,): 1}})
    with pytest.raises(UnderSpecifiedProgramError, match="no Map spec"):
        evaluate_einsum(einsum, env=env)


################################################################################
# Unary as a degenerate Map (the merge matters)
################################################################################


def _unary_einsum(merge: str | None) -> Einsum:
    """``Z_m = A_m . (not B_m) :: map take_left(intersect)``."""
    unary = UnaryApp(
        op=BuiltinUnaryOp(symbol="not"),
        operand=InputTensor(
            proj=TensorProjection(tensor="B", ranks=[RankVariable(name="m")])
        ),
        merge_op=BuiltinMergeOp(symbol=merge) if merge else None,  # type: ignore[arg-type]
    )
    return Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
            ),
            rhs=unary,
        ),
        specs=[
            MapSpec(
                label=1,
                compute_op=UserDefinedComputeOp(name="take_left"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )


@pytest.mark.parametrize("merge", [None, "not_left"])
def test_negation_masks_by_complement(merge: str | None) -> None:
    """EDGE's decomposition of unary functions.

    "For unary operators that map empty to non-empty, such as logical
    complement, the map action must instead visit the points where A is
    empty, using the not-take-left merge operator."

    So ``A . not B :: map take_left(intersect)`` is ``A minus B`` -- the
    masking idiom every BFS-family cascade uses. Under a take_left merge it
    would compute ``A and B`` instead, the exact complement.
    """
    decls = [
        _decl("Z", ["M"], 0),
        _decl("A", ["M"], 0),
        TensorDeclaration(
            name="B",
            ranks=[RankDeclaration(name="M", shape=4)],
            data_type=BuiltinDataType(name="bool"),
            empty_value=False,
        ),
    ]
    program = _program(_unary_einsum(merge), decls)
    env = build_environment(
        program,
        inputs={"A": {(0,): 1, (1,): 2, (2,): 3}, "B": {(1,): True}},
        shape_env={},
    )
    evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
    # A is present at 0,1,2; B masks out 1.
    assert sorted(c for c, _ in env.tensors["Z"].items()) == [(0,), (2,)]


def test_take_left_merge_on_a_unary_gives_the_other_reading() -> None:
    """The same expression under ``take_left`` computes A-and-B, not A-minus-B.

    This is why the merge cannot be left implicit: the two readings are
    exact complements of each other.
    """
    decls = [
        _decl("Z", ["M"], 0),
        _decl("A", ["M"], 0),
        TensorDeclaration(
            name="B",
            ranks=[RankDeclaration(name="M", shape=4)],
            data_type=BuiltinDataType(name="bool"),
            empty_value=False,
        ),
    ]
    program = _program(_unary_einsum("take_left"), decls)
    env = build_environment(
        program,
        inputs={"A": {(0,): 1, (1,): 2, (2,): 3}, "B": {(1,): True}},
        shape_env={},
    )
    evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
    assert sorted(c for c, _ in env.tensors["Z"].items()) == [(1,)]


################################################################################
# Reduce
################################################################################


def _reduce_einsum(merge: str, compute: str) -> Einsum:
    """``Z_m = A_{m,n} :: reduce compute(merge)`` -- a single-operand reduce."""
    return Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(
                tensor="A", ranks=[RankVariable(name="m"), RankVariable(name="n")]
            )
        ),
        specs=[
            ReduceSpec(
                compute_op=BuiltinComputeOp(symbol=compute)
                if compute in "+-*/"
                else UserDefinedComputeOp(name=compute),
                merge_op=BuiltinMergeOp(symbol=merge),  # type: ignore[arg-type]
            )
        ],
    )


def test_a_reduce_needs_no_binary_to_label() -> None:
    """Reduce's operands are the reduction state and the map temporary
    (Reduce semantics, "Inputs and Outputs"); it never refers to a binary. So a
    single-operand Einsum reduces with ``label=None``."""
    decls = [
        _decl("Z", ["M"], 0),
        TensorDeclaration(
            name="A",
            ranks=[
                RankDeclaration(name="M", shape=3),
                RankDeclaration(name="N", shape=3),
            ],
            data_type=BuiltinDataType(name="int"),
            empty_value=0,
        ),
    ]
    program = _program(_reduce_einsum("union", "+"), decls)
    env = build_environment(
        program, inputs={"A": {(0, 0): 1, (0, 1): 2, (1, 2): 5}}, shape_env={}
    )
    evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
    assert dict(env.tensors["Z"].items()) == {(0,): 3, (1,): 5}


def test_reduce_observes_points_where_the_map_temporary_is_absent() -> None:
    """Reduce semantics: Reduce "iterates over the same iteration space as Map,
    queries whether Map produced a non-empty value at each point".

    A point with no MapTmp still reaches the merge as ``b_m = False``, and an
    intersect merge then invalidates the accumulated state -- "a missing
    MapTmp value invalidates the accumulated reduction state for that output
    coordinate". A reduce that walked only MapTmp's support could not
    produce this.
    """
    decls = [
        _decl("Z", ["M"], 0),
        TensorDeclaration(
            name="A",
            ranks=[
                RankDeclaration(name="M", shape=2),
                RankDeclaration(name="N", shape=3),
            ],
            data_type=BuiltinDataType(name="int"),
            empty_value=0,
        ),
    ]
    program = _program(_reduce_einsum("intersect", "+"), decls)
    # Row 0 is dense across N; row 1 has a hole at n=1.
    env = build_environment(
        program,
        inputs={"A": {(0, 0): 1, (0, 1): 1, (0, 2): 1, (1, 0): 5, (1, 2): 5}},
        shape_env={},
    )
    evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
    # Neither row survives: an intersect reduce also fails on the FIRST touch,
    # because a fresh state holds the identity 0, which is Z's empty value, so
    # b_s is False and intersect(False, True) is False.
    assert dict(env.tensors["Z"].items()) == {}


def test_reduce_with_no_identity_fails_loudly() -> None:
    """The Reduce semantics seed a fresh state with the compute operator's identity.
    ``-`` has none, so the program is under-specified rather than defaulted."""
    decls = [
        _decl("Z", ["M"], 0),
        TensorDeclaration(
            name="A",
            ranks=[
                RankDeclaration(name="M", shape=2),
                RankDeclaration(name="N", shape=2),
            ],
            data_type=BuiltinDataType(name="int"),
            empty_value=0,
        ),
    ]
    program = _program(_reduce_einsum("union", "-"), decls)
    env = build_environment(program, inputs={"A": {(0, 0): 1}}, shape_env={})
    with pytest.raises(UnderSpecifiedProgramError, match="no two-sided identity"):
        evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)


def test_colliding_output_coordinates_without_a_reduce_fail_loudly() -> None:
    """With no Reduce spec there is no rule for combining two contributions
    to one output coordinate, so the evaluator refuses to pick a winner."""
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(
                tensor="A", ranks=[RankVariable(name="m"), RankVariable(name="n")]
            )
        ),
        specs=[],
    )
    decls = [
        _decl("Z", ["M"], 0),
        TensorDeclaration(
            name="A",
            ranks=[
                RankDeclaration(name="M", shape=2),
                RankDeclaration(name="N", shape=2),
            ],
            data_type=BuiltinDataType(name="int"),
            empty_value=0,
        ),
    ]
    env = build_environment(
        _program(einsum, decls), inputs={"A": {(0, 0): 1, (0, 1): 2}}
    )
    with pytest.raises(UnderSpecifiedProgramError, match="no Reduce spec"):
        evaluate_einsum(einsum, env=env)


################################################################################
# Populate
################################################################################


def test_populate_refuses_to_write_over_an_occupied_coordinate() -> None:
    """Populate temporal constraint: ``W and dom(S) = {}``.

    "an implementation of EDGE should raise an error (at compile or runtime)
    if a violation is detected".
    """
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
                coord_op=CoordinateOp(name="write-everything"),
            )
        ],
    )
    decls = [_decl("Z", ["M"], 0, shape=3), _decl("A", ["M"], 0, shape=3)]
    program = _program(einsum, decls)
    env = build_environment(program, inputs={"A": {(0,): 4}}, shape_env={})
    env.tensors["Z"].write((0,), 99)  # pre-state: occupied

    from edge_ir.evaluator.context import PopulateAction
    from edge_ir.udf.categories import UdfCategory

    env.functions.register_python(
        "write-everything",
        UdfCategory.COORDINATE,
        lambda fiber, coord, value, *, ctx: [
            (e.coord, PopulateAction.WRITE) for e in fiber
        ],
        arity=3,
        wants_context=True,
    )
    with pytest.raises(PopulateConstraintViolation, match="already hold a value"):
        evaluate_einsum(einsum, env=env)


def test_default_assignment_needs_no_populate_spec() -> None:
    """Populate default: assignment is the no-``*`` case."""
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
        specs=[],
    )
    decls = [_decl("Z", ["M"], 0, shape=3), _decl("A", ["M"], 0, shape=3)]
    program = _program(einsum, decls)
    env = build_environment(program, inputs={"A": {(0,): 4, (2,): 6}}, shape_env={})
    evaluate_einsum(einsum, env=env)
    assert dict(env.tensors["Z"].items()) == {(0,): 4, (2,): 6}


################################################################################
# Fail-open regressions
#
# Found by an external consumer executing ~28 einsums from an external
# workload against this evaluator. Each case below used to return a plausible
# answer -- an empty tensor, or a broadcast in place of a contraction -- with
# no diagnostic. The evaluator's contract is to fail loudly, so each is pinned.
################################################################################


def test_projecting_a_tensor_with_the_wrong_number_of_subscripts_raises() -> None:
    """``Exists_T(c)`` is defined over ``CS^T``, whose elements are N-tuples
    for an N-rank tensor (EDGE semantics, "Tensor Coordinate Space Sets").

    A 2-tuple is not a point of a 3-rank tensor's coordinate space at all --
    it is a projection written with the wrong number of subscripts. Answering
    "absent" made an entire cascade evaluate to empty from its first line, with
    every downstream stage inheriting the emptiness.
    """
    space = CoordinateSpace(
        tensor="A",
        ranks=tuple(RankCoordinateSet(r, (0, 1, 2, 3)) for r in ("J", "M", "N")),
    )
    a = Tensor("A", space, empty_value=0, graph={(0, 1, 2): 9})
    for probe in (a.exists, a.value_or_empty):
        with pytest.raises(UnderSpecifiedProgramError, match="rank"):
            probe((1, 2))


def test_writing_outside_the_coordinate_space_raises() -> None:
    """``dom(T) subset-of CS^T`` (EDGE semantics, Existence Check).

    Storing host data at a coordinate the declaration does not admit makes it
    read back as PRESENT, which silently rescues a rank declared with the wrong
    shape -- the error rule 6 of the Layer 2 validator plan exists to catch.
    """
    space = CoordinateSpace(
        tensor="MZ",
        ranks=(
            RankCoordinateSet("V", (0, 1, 2, 3)),
            RankCoordinateSet("C", (0, 1, 2, 3)),
        ),
    )
    mz = Tensor("MZ", space, empty_value=0)
    with pytest.raises(
        UnderSpecifiedProgramError, match="outside its coordinate space"
    ):
        mz.write((0, 6), 5)
    assert mz.occupancy() == 0


def test_an_anonymous_tensor_that_contracts_an_axis_raises() -> None:
    """``(A_{m,k} . X_k)_m`` reduces ``k`` INSIDE the parentheses.

    This evaluator applies an Einsum's Reduce spec at the output coordinate,
    not at a nested level. Evaluating the inner expression pointwise and
    leaving the rank list unread would broadcast instead of contracting, which
    computes a different quantity and says nothing about it.
    """
    from edge_ir.evaluator import UnsupportedConstructError
    from edge_ir.ir.expr import AnonymousTensor

    inner = BinaryApp(
        label=1,
        lhs=InputTensor(
            proj=TensorProjection(
                tensor="A", ranks=[RankVariable(name="m"), RankVariable(name="k")]
            )
        ),
        rhs=InputTensor(
            proj=TensorProjection(tensor="X", ranks=[RankVariable(name="k")])
        ),
    )
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=AnonymousTensor(expression=inner, ranks=[RankVariable(name="m")]),
        specs=[
            MapSpec(
                label=1,
                compute_op=BuiltinComputeOp(symbol="*"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )
    decls = [
        _decl("Z", ["M"], 0),
        TensorDeclaration(
            name="A",
            ranks=[
                RankDeclaration(name="M", shape=4),
                RankDeclaration(name="K", shape=4),
            ],
            data_type=BuiltinDataType(name="int"),
            empty_value=0,
        ),
        _decl("X", ["K"], 0),
    ]
    env = build_environment(
        _program(einsum, decls),
        inputs={"A": {(0, 0): 1, (0, 1): 2}, "X": {(0,): 10, (1,): 20}},
    )
    with pytest.raises(UnsupportedConstructError, match="reduced away inside"):
        evaluate_einsum(einsum, env=env)


def test_a_unary_cannot_wrap_an_anonymous_tensor() -> None:
    """The grammar restricts a unary's operand to ``<input-tensor>``, and this
    field is typed to match. An anonymous tensor is NOT an escape hatch -- an
    earlier docstring said it was, and it never worked. To apply a unary to a
    sub-expression, name the sub-expression with its own Einsum."""
    from pydantic import ValidationError

    from edge_ir.ir.expr import AnonymousTensor

    anon = AnonymousTensor(
        expression=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
        ranks=[RankVariable(name="m")],
    )
    with pytest.raises(ValidationError):
        UnaryApp(op=BuiltinUnaryOp(symbol="not"), operand=anon)  # type: ignore[arg-type]
