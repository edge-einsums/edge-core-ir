"""``||T||`` inside an Einsum, not only in a stopping condition.

Exact cover writes ``Success_{i+1} = ||ActiveCol_{i+1}|| == 0``. Without an
expression node for occupancy that one line had to become an OR-reduction, a
named intermediate and a negation -- a different Einsum from the one the
author wrote. ``Occupancy`` lets it stay one Einsum.
"""

from __future__ import annotations

from edge_ir.evaluator import build_environment, evaluate_einsum
from edge_ir.ir.actions import MapSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import BinaryApp, InputTensor, Occupancy, TensorProjection
from edge_ir.ir.op import BuiltinMergeOp, UserDefinedComputeOp
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration


def _decl(name, ranks, dtype, empty, value=None):
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name=r, shape=4) for r in ranks],
        data_type=BuiltinDataType(name=dtype),
        empty_value=empty,
        value=value,
    )


def _success_is_empty_check() -> tuple[Program, Einsum]:
    """Success = ||Active|| == 0 :: /\\ ==(^), over a rank-1 Active."""
    einsum = Einsum(
        output_tensor="Success",
        output_ranks=[],
        expression=BinaryApp(
            label=1,
            lhs=Occupancy(operand=TensorProjection(tensor="Active", ranks=[])),
            rhs=InputTensor(proj=TensorProjection(tensor="Zero", ranks=[])),
        ),
        specs=[
            MapSpec(
                label=1,
                compute_op=UserDefinedComputeOp(name="eq"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )
    program = Program(
        declarations=[
            _decl("Active", ["C"], "bool", False),
            _decl("Success", [], "bool", False),
            _decl("Zero", [], "int", None, value=0),
        ],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )
    return program, einsum


def _success(active: dict) -> dict:
    program, einsum = _success_is_empty_check()
    env = build_environment(program, inputs={"Active": active}, shape_env={})
    evaluate_einsum(einsum, env=env)
    return dict(env.tensors["Success"].items())


def test_occupancy_validates_as_an_expression() -> None:
    program, _ = _success_is_empty_check()
    again = Program.model_validate(program.model_dump())
    assert again.main_edge.cascade.einsums[0].expression.lhs.kind == "occupancy"


def test_occupancy_of_an_empty_tensor_is_zero() -> None:
    """Nothing left: the count is 0, so the comparison is True."""
    assert _success({}) == {(): True}


def test_occupancy_counts_present_entries() -> None:
    """Something left: the count is 2, the comparison is False -- which is
    the Boolean empty value, so Success is absent."""
    assert _success({(1,): True, (3,): True}) == {}
