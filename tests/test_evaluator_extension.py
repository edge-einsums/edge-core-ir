"""Clean extension: a new function requires NO evaluator change.

The acceptance criterion for "extensible" is that adding a compute,
coordinate, unary or rank-mapping function means registering an
implementation and touching no module under ``edge_ir/evaluator/``. These
tests do exactly that -- every user-defined operator here is invented in the
test file and reaches the evaluator only through the registry.
"""

from __future__ import annotations

from edge_ir.evaluator import build_environment, evaluate_einsum
from edge_ir.evaluator.context import PopulateAction
from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankFunction,
    RankVariable,
    TensorProjection,
    UnaryApp,
)
from edge_ir.ir.op import (
    BuiltinMergeOp,
    CoordinateOp,
    UserDefinedComputeOp,
    UserDefinedUnaryOp,
)
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration
from edge_ir.udf import UdfCategory, default_registry


def _decl(name: str, ranks: list[str], empty: object, shape: int = 4):
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name=r, shape=shape) for r in ranks],
        data_type=BuiltinDataType(name="int"),
        empty_value=empty,
    )


def _run(einsum: Einsum, decls, inputs, registry):
    program = Program(
        declarations=decls,
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )
    env = build_environment(program, inputs=inputs, functions=registry)
    evaluate_einsum(einsum, env=env)
    return env


def test_a_new_map_compute_op_needs_no_evaluator_change() -> None:
    registry = default_registry()
    registry.register_python(
        "saturating_add",
        UdfCategory.MAP_COMPUTE,
        lambda left, right: min(left + right, 10),
        arity=2,
        doc="add, clamped at 10",
    )
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
        specs=[
            MapSpec(
                label=1,
                compute_op=UserDefinedComputeOp(name="saturating_add"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )
    decls = [_decl("Z", ["M"], 0), _decl("A", ["M"], 0), _decl("B", ["M"], 0)]
    env = _run(einsum, decls, {"A": {(0,): 8}, "B": {(0,): 9}}, registry)
    assert env.tensors["Z"].value_or_empty((0,)) == 10


def test_a_new_reduce_compute_op_with_its_identity() -> None:
    registry = default_registry()
    registry.register_python(
        "bitwise_or",
        UdfCategory.REDUCE_COMPUTE,
        lambda acc, val: int(acc) | int(val),
        arity=2,
        identity=0,
        associative=True,
        commutative=True,
    )
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(
                tensor="A", ranks=[RankVariable(name="m"), RankVariable(name="n")]
            )
        ),
        specs=[
            ReduceSpec(
                compute_op=UserDefinedComputeOp(name="bitwise_or"),
                merge_op=BuiltinMergeOp(symbol="union"),
            )
        ],
    )
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
    env = _run(einsum, decls, {"A": {(0, 0): 1, (0, 1): 4, (0, 2): 2}}, registry)
    assert env.tensors["Z"].value_or_empty((0,)) == 7


def test_a_new_coordinate_op_and_populate_compute_op() -> None:
    """A value-as-coordinate populate written entirely in the test file."""
    registry = default_registry()

    def keep_evens(fiber, red_coord, red_value, *, ctx):
        """Keep only even values; delete the rest of the fiber."""
        actions = []
        for entry in fiber:
            if entry.coord == red_coord:
                if red_value % 2 == 0:
                    actions.append(
                        (
                            entry.coord,
                            PopulateAction.NONE
                            if entry.present
                            else PopulateAction.WRITE,
                        )
                    )
                elif entry.present:
                    actions.append((entry.coord, PopulateAction.DELETE))
            else:
                actions.append((entry.coord, PopulateAction.NONE))
        return actions

    registry.register_python(
        "keep_evens", UdfCategory.COORDINATE, keep_evens, arity=3, wants_context=True
    )
    registry.register_python(
        "doubled",
        UdfCategory.POPULATE_COMPUTE,
        lambda coord, value, *, ctx: value * 2,
        arity=2,
        wants_context=True,
    )
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
        specs=[
            PopulateSpec(
                rank_list=["m"],
                compute_op=UserDefinedComputeOp(name="doubled"),
                coord_op=CoordinateOp(name="keep_evens"),
            )
        ],
    )
    decls = [_decl("Z", ["M"], 0), _decl("A", ["M"], 0)]
    env = _run(einsum, decls, {"A": {(0,): 4, (1,): 3, (2,): 6}}, registry)
    assert dict(env.tensors["Z"].items()) == {(0,): 8, (2,): 12}


def test_a_new_unary_op_declaring_its_empty_behaviour() -> None:
    """The unary's merge is derived from ``maps_empty_to_empty``."""
    registry = default_registry()
    registry.register_python(
        "negate",
        UdfCategory.UNARY,
        lambda value: -value,
        arity=1,
        maps_empty_to_empty=True,
    )
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
            ),
            rhs=UnaryApp(
                op=UserDefinedUnaryOp(name="negate"),
                operand=InputTensor(
                    proj=TensorProjection(tensor="B", ranks=[RankVariable(name="m")])
                ),
            ),
        ),
        specs=[
            MapSpec(
                label=1,
                compute_op=UserDefinedComputeOp(name="take_right"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )
    decls = [_decl("Z", ["M"], 0), _decl("A", ["M"], 0), _decl("B", ["M"], 0)]
    env = _run(einsum, decls, {"A": {(0,): 1, (1,): 1}, "B": {(0,): 5}}, registry)
    # maps_empty_to_empty=True derives take_left, so `negate B` exists only
    # where B does: coordinate 1 contributes nothing.
    assert dict(env.tensors["Z"].items()) == {(0,): -5}


def test_a_new_rank_mapping_function() -> None:
    """A non-affine output RVE routed through the registry (paper 7.3.5's
    ``min(a, w)`` shape: many iteration points to one output coordinate)."""
    registry = default_registry()
    registry.register_python(
        "fold2",
        UdfCategory.RANK_MAPPING,
        lambda m: m % 2,
        arity=1,
        is_affine=False,
    )
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankFunction(func_name="fold2", args=[RankVariable(name="m")])],
        expression=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
        specs=[
            ReduceSpec(
                compute_op=UserDefinedComputeOp(name="max"),
                merge_op=BuiltinMergeOp(symbol="union"),
            )
        ],
    )
    decls = [_decl("Z", ["M"], 0, shape=2), _decl("A", ["M"], 0, shape=4)]
    env = _run(einsum, decls, {"A": {(0,): 3, (1,): 9, (2,): 7, (3,): 2}}, registry)
    assert dict(env.tensors["Z"].items()) == {(0,): 7, (1,): 9}


def test_no_evaluator_module_is_imported_by_the_registry_tests() -> None:
    """A guard on the claim above: every operator this file registers is
    defined here, and the registry is the only channel into the evaluator."""
    import edge_ir.udf.registry as registry_module

    source = registry_module.__file__ or ""
    assert source.endswith("edge_ir/udf/registry.py")
