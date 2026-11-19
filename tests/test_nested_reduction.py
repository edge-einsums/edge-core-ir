"""A reduction written inside parentheses is evaluated there.

The EDGE paper's own example of an anonymous tensor (Section 7.3.2, Eq. 143)
folds BFS's gather and filter into one Einsum:

    F_{i+1,d} = (G_{s,d} .^1 F_{i,s})_{i,d} .^2 not P_{i,d}
                :: /\\^1 +(^)  \\/^1 ANY(u)  /\\^2 <-(^)

The Reduce on label 1 contracts ``s`` inside the parentheses, before label 2
uses the result. Exact cover's E18 has the same shape. These tests pin that
the contraction happens at that level and not at the output.
"""

from __future__ import annotations

import sys
from pathlib import Path

from edge_ir.evaluator import build_environment, evaluate, evaluate_einsum
from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import (
    AnonymousTensor,
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
    UserDefinedComputeOp,
)
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration

ROOT = Path(__file__).resolve().parent.parent


def _decl(name, ranks, empty=0, dtype="int"):
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name=r, shape=3) for r in ranks],
        data_type=BuiltinDataType(name=dtype),
        empty_value=empty,
    )


def _inp(tensor, *ranks):
    return InputTensor(
        proj=TensorProjection(
            tensor=tensor, ranks=[RankVariable(name=r) for r in ranks]
        )
    )


def _map(label, compute, merge="intersect"):
    op = (
        BuiltinComputeOp(symbol=compute)
        if compute in "+-*/"
        else UserDefinedComputeOp(name=compute)
    )
    return MapSpec(label=label, compute_op=op, merge_op=BuiltinMergeOp(symbol=merge))


def _reduce(label, compute, ranks, merge="union"):
    op = (
        BuiltinComputeOp(symbol=compute)
        if compute in "+-*/"
        else UserDefinedComputeOp(name=compute)
    )
    return ReduceSpec(
        label=label,
        rank_list=ranks,
        compute_op=op,
        merge_op=BuiltinMergeOp(symbol=merge),
    )


def _run(einsum, decls, inputs, out):
    program = Program(
        declarations=decls,
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )
    env = build_environment(program, inputs=inputs, shape_env={})
    evaluate_einsum(einsum, env=env)
    return dict(env.tensors[out].items())


def test_the_contraction_happens_inside_the_parentheses() -> None:
    """Z_m = (A_{m,k} .^1 X_k)_m .^2 W_m :: /\\^1 *(^) \\/^1_k +(u) /\\^2 +(^)

    Reduced inside: sum_k(A*X) + W = 1*10 + 2*20 + 5 = 55.
    Reduced at the output instead: sum_k(A*X + W) = 60 -- W added once per k.
    """
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=BinaryApp(
            label=2,
            lhs=AnonymousTensor(
                expression=BinaryApp(
                    label=1, lhs=_inp("A", "m", "k"), rhs=_inp("X", "k")
                ),
                ranks=[RankVariable(name="m")],
            ),
            rhs=_inp("W", "m"),
        ),
        specs=[_map(1, "*"), _reduce(1, "+", ["k"]), _map(2, "+")],
    )
    decls = [
        _decl("Z", ["M"]),
        _decl("A", ["M", "K"]),
        _decl("X", ["K"]),
        _decl("W", ["M"]),
    ]
    got = _run(
        einsum,
        decls,
        {"A": {(0, 0): 1, (0, 1): 2}, "X": {(0,): 10, (1,): 20}, "W": {(0,): 5}},
        "Z",
    )
    assert got == {(0,): 55}


def test_paper_eq_143_matches_the_two_einsum_form() -> None:
    """BFS gather + filter as one Einsum equals gather, then filter."""
    G = {(0, 1): 1, (0, 2): 1, (1, 2): 1}
    F = {(0,): 1}
    P = {(0,): True, (1,): True}

    fused = Einsum(
        output_tensor="Out",
        output_ranks=[RankVariable(name="d")],
        expression=BinaryApp(
            label=2,
            lhs=AnonymousTensor(
                expression=BinaryApp(
                    label=1, lhs=_inp("G", "s", "d"), rhs=_inp("F", "s")
                ),
                ranks=[RankVariable(name="d")],
            ),
            rhs=UnaryApp(op=BuiltinUnaryOp(symbol="not"), operand=_inp("P", "d")),
        ),
        specs=[_map(1, "+"), _reduce(1, "ANY", ["s"]), _map(2, "take_left")],
    )
    decls = [
        _decl("Out", ["D"]),
        _decl("G", ["S", "D"]),
        _decl("F", ["S"]),
        _decl("P", ["D"], empty=False, dtype="bool"),
        _decl("T", ["D"]),
    ]
    one = _run(fused, decls, {"G": G, "F": F, "P": P}, "Out")

    gather = Einsum(
        output_tensor="T",
        output_ranks=[RankVariable(name="d")],
        expression=BinaryApp(label=1, lhs=_inp("G", "s", "d"), rhs=_inp("F", "s")),
        specs=[_map(1, "+"), _reduce(1, "ANY", ["s"])],
    )
    t = _run(gather, decls, {"G": G, "F": F, "P": P}, "T")
    # Filter by not P by hand: keep d where P is absent.
    two = {d: v for d, v in t.items() if d not in P}
    assert one == two == {(2,): 2}


def test_exact_cover_with_e18_as_one_einsum_finds_both_solutions() -> None:
    """The whole exact-cover program, E18 written as the paper's labelled
    parenthesized form, on Working Example 1."""
    alg = ROOT / "examples" / "algorithms" / "exact-cover"
    sys.path.insert(0, str(alg))
    from build_exact_cover_program import build_exact_cover
    from exact_cover_functions import exact_cover_registry

    program = build_exact_cover()
    a = {(0, 0): True, (1, 1): True, (2, 0): True, (2, 1): True, (3, 2): True}
    result = evaluate(
        program,
        inputs={
            "A": a,
            "Col": {(c,): True for c in range(3)},
            "Row": {(r,): True for r in range(4)},
            "InitS": {(3,): 7},
            "RowCount": {(): 4},
        },
        shape_env={"|R|": 4, "|C|": 3},
        functions=exact_cover_registry(),
        max_iterations=60,
    )
    solutions: dict[int, list[int]] = {}
    for coord in result.tensors["Sol"]:
        solutions.setdefault(coord[0], []).append(coord[1])
    assert sorted(sorted(v) for v in solutions.values()) == [[0, 1, 3], [2, 3]]
