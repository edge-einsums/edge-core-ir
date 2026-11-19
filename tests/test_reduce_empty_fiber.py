"""A Reduce over a fiber with no present Map value leaves the output empty.

The compute operator's identity only seeds the fold. ``max``'s identity is
``-inf``, which is not an int tensor's empty value, so writing the seed would
invent an entry: exact cover's ``d*_i = floor(F_{i,r} / |R|) :: \\/ max(u)``
wrote ``d*_0 = -inf`` on an instance whose stack starts empty.
"""

from __future__ import annotations

from edge_ir.evaluator import build_environment, evaluate_einsum
from edge_ir.ir.actions import ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import InputTensor, RankVariable, TensorProjection
from edge_ir.ir.op import BuiltinMergeOp, UserDefinedComputeOp
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration


def _decl(name, ranks):
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name=r, shape=3) for r in ranks],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )


def _max_over_k(x: dict) -> dict:
    """Z_m = X_{m,k} :: \\/_k max(u)"""
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(
                tensor="X", ranks=[RankVariable(name="m"), RankVariable(name="k")]
            )
        ),
        specs=[
            ReduceSpec(
                label=0,
                rank_list=["k"],
                compute_op=UserDefinedComputeOp(name="max"),
                merge_op=BuiltinMergeOp(symbol="union"),
            )
        ],
    )
    program = Program(
        declarations=[_decl("Z", ["M"]), _decl("X", ["M", "K"])],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )
    env = build_environment(program, inputs={"X": x}, shape_env={})
    evaluate_einsum(einsum, env=env)
    return dict(env.tensors["Z"].items())


def test_an_empty_fiber_reduces_to_empty_not_the_identity() -> None:
    assert _max_over_k({}) == {}


def test_only_the_empty_fibers_stay_empty() -> None:
    assert _max_over_k({(0, 1): 4, (0, 2): 9, (2, 0): 3}) == {(0,): 9, (2,): 3}
