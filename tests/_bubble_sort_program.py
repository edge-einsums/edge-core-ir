"""Bubble sort as the EDGE zoo writes it: a nested cascade of case statements.

Shared by the local-patch tests. The outer cascade iterates the pass rank
``i``, the inner one the step rank ``j``; every Einsum is a case statement on
the position ``n``.

    Z_{0,0,n} = { min(A_0, A_1)  n = 0 ; max(A_0, A_1)  n = 1 ; empty  o/w }
    -- outer cascade (passes), i = 0
      -- inner cascade (steps)
      Z_{0,j,n} = { Z_{0,j-1,n}                 n < j
                  { min(Z_{0,j-1,j}, A_{j+1})   n = j
                  { max(Z_{0,j-1,j}, A_{j+1})   n = j+1
                  { empty                       o/w
      Z_{i,0,n} = { min(Z_{i-1,J-1,0}, Z_{i-1,J-1,1})  n = 0
                  { max(Z_{i-1,J-1,0}, Z_{i-1,J-1,1})  n = 1
                  { Z_{i-1,J-1,n}                      n >= 2
      Z_{i,j,n} = { Z_{i,j-1,n}                        n < j
                  { min(Z_{i,j-1,j}, Z_{i-1,J-1,j+1})  n = j
                  { max(Z_{i,j-1,j}, Z_{i-1,J-1,j+1})  n = j+1
                  { Z_{i-1,J-1,n}                      n > j+1
      <><> : j == J
    <> : Z_{i+1,J-1} == Z_{i,J-1}

``min(x, y)`` and ``max(x, y)`` are lowered with the intersect merge: a read
that falls outside the tensor (pass 0 has no previous pass, step 0 no
previous step) is absent, so the arm writes nothing there.
"""

from __future__ import annotations

from edge_ir.ir.actions import MapSpec
from edge_ir.ir.einsum import (
    CaseArm,
    Cases,
    Einsum,
    IterativeRankSpec,
    NestedCascade,
)
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.op import BuiltinMergeOp, UserDefinedComputeOp
from edge_ir.ir.predicate import PredicateComparison
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import (
    Comparison,
    RankExpressionValue,
    StoppingCondition,
    TensorProjectionValue,
)
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration

STEPS = "N-1"  # the shape of both the pass rank I and the step rank J


def rv(name):
    return RankVariable(name=name)


def lit(value):
    return RankConstantLiteral(value=value)


def plus(rank, k):
    return RankArith(op="+" if k > 0 else "-", lhs=rank, rhs=lit(abs(k)))


LAST = plus(RankConstantShapeSym(name=STEPS), -1)  # J-1


def inp(tensor, *ranks):
    return InputTensor(proj=TensorProjection(tensor=tensor, ranks=list(ranks)))


def pick(op, left, right):
    """`min(x, y)` / `max(x, y)`: one Map, intersect merge."""
    return {
        "expression": BinaryApp(label=1, lhs=left, rhs=right),
        "specs": [
            MapSpec(
                label=1,
                compute_op=UserDefinedComputeOp(name=op),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    }


def where(op, rhs):
    return PredicateComparison(lhs=rv("n"), op=op, rhs=rhs)


def z(out, arms):
    return Einsum(
        output_tensor="Z",
        output_ranks=list(out),
        expression=Cases(arms=arms),
        specs=[],
    )


def bubble_sort_program() -> Program:
    i, j, n = rv("i"), rv("j"), rv("n")
    init = z(
        [lit(0), lit(0), n],
        [
            CaseArm(
                **pick("min", inp("A", lit(0)), inp("A", lit(1))),
                where=where("==", lit(0)),
            ),
            CaseArm(
                **pick("max", inp("A", lit(0)), inp("A", lit(1))),
                where=where("==", lit(1)),
            ),
            CaseArm(expression=None),
        ],
    )
    running0 = inp("Z", lit(0), plus(j, -1), j)
    incoming0 = inp("A", plus(j, 1))
    pass0 = z(
        [lit(0), j, n],
        [
            CaseArm(expression=inp("Z", lit(0), plus(j, -1), n), where=where("<", j)),
            CaseArm(**pick("min", running0, incoming0), where=where("==", j)),
            CaseArm(**pick("max", running0, incoming0), where=where("==", plus(j, 1))),
            CaseArm(expression=None),
        ],
    )
    prev0 = inp("Z", plus(i, -1), LAST, lit(0))
    prev1 = inp("Z", plus(i, -1), LAST, lit(1))
    step0 = z(
        [i, lit(0), n],
        [
            CaseArm(**pick("min", prev0, prev1), where=where("==", lit(0))),
            CaseArm(**pick("max", prev0, prev1), where=where("==", lit(1))),
            CaseArm(
                expression=inp("Z", plus(i, -1), LAST, n), where=where(">=", lit(2))
            ),
        ],
    )
    running = inp("Z", i, plus(j, -1), j)
    incoming = inp("Z", plus(i, -1), LAST, plus(j, 1))
    sweep = z(
        [i, j, n],
        [
            CaseArm(expression=inp("Z", i, plus(j, -1), n), where=where("<", j)),
            CaseArm(**pick("min", running, incoming), where=where("==", j)),
            CaseArm(**pick("max", running, incoming), where=where("==", plus(j, 1))),
            CaseArm(
                expression=inp("Z", plus(i, -1), LAST, n), where=where(">", plus(j, 1))
            ),
        ],
    )
    inner = NestedCascade(
        einsums=[pass0, step0, sweep],
        stopping_conditions=[
            StoppingCondition(
                rank_variable="j",
                predicate=Comparison(
                    lhs=RankExpressionValue(expr=j),
                    op="==",
                    rhs=RankExpressionValue(expr=RankConstantShapeSym(name=STEPS)),
                ),
            )
        ],
        iterative_rank=IterativeRankSpec(rank_variable="j"),
    )
    outer = NestedCascade(
        einsums=[inner],
        stopping_conditions=[
            StoppingCondition(
                rank_variable="i",
                predicate=Comparison(
                    lhs=TensorProjectionValue(
                        proj=TensorProjection(tensor="Z", ranks=[plus(i, 1), LAST])
                    ),
                    op="==",
                    rhs=TensorProjectionValue(
                        proj=TensorProjection(tensor="Z", ranks=[i, LAST])
                    ),
                ),
            )
        ],
        iterative_rank=IterativeRankSpec(rank_variable="i"),
    )
    integer = BuiltinDataType(name="int")
    return Program(
        declarations=[
            TensorDeclaration(
                name="A",
                ranks=[RankDeclaration(name="N", shape="|A|")],
                data_type=integer,
                empty_value=0,
            ),
            TensorDeclaration(
                name="Z",
                ranks=[
                    RankDeclaration(name="I", shape=STEPS),
                    RankDeclaration(name="J", shape=STEPS),
                    RankDeclaration(name="M", shape="N"),
                ],
                data_type=integer,
                empty_value=0,
            ),
        ],
        initialization=Initialization(einsums=[init]),
        main_edge=MainEdge(cascade=outer),
    )


def run_bubble_sort(values, **kwargs):
    from edge_ir.evaluator import evaluate

    size = len(values)
    return evaluate(
        bubble_sort_program(),
        inputs={"A": {(k,): v for k, v in enumerate(values)}},
        shape_env={"|A|": size, "N": size, STEPS: size - 1},
        **kwargs,
    )
