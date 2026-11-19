"""Case statements: one Einsum, different expressions per region.

LOCAL PATCH (edge-workbench issue 8). The paper (selective computations):
"EDGE supports the assignment of different expressions in different regions
of the iteration space, using case statements ... Each branch of a case
statement is itself a valid EDGE expression, with its own modifier if
needed", and "EDGE allows any Boolean predicate as conditions for each case."
"""

from __future__ import annotations

import pytest

from edge_ir.evaluator import build_environment, evaluate, evaluate_einsum
from edge_ir.evaluator.errors import UnderSpecifiedProgramError
from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import CaseArm, Cases, Einsum, NestedCascade
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    CoordinateOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.predicate import PredicateComparison
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import Comparison, TensorProjectionValue
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration
from edge_ir.validator.single_assignment import check_single_assignment
from edge_ir.validator.udf_names import udf_uses


def rv(name):
    return RankVariable(name=name)


def inp(tensor, *ranks):
    return InputTensor(proj=TensorProjection(tensor=tensor, ranks=list(ranks)))


def decl(name, ranks, dtype="int", empty=0, value=None):
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name=r, shape=3) for r in ranks],
        data_type=BuiltinDataType(name=dtype),
        empty_value=empty,
        value=value,
    )


def mul_reduce():
    return [
        MapSpec(
            label=1,
            compute_op=BuiltinComputeOp(symbol="*"),
            merge_op=BuiltinMergeOp(symbol="intersect"),
        ),
        ReduceSpec(
            label=1,
            rank_list=["k"],
            compute_op=BuiltinComputeOp(symbol="+"),
            merge_op=BuiltinMergeOp(symbol="union"),
        ),
    ]


def paper_equation_61() -> Einsum:
    """Z_{m,n} = { A_{m,n}                              if n != m
    { B_{k,m} . C_{k,n} :: /\\_k *(^) \\/_k +(u)   otherwise"""
    return Einsum(
        output_tensor="Z",
        output_ranks=[rv("m"), rv("n")],
        expression=Cases(
            arms=[
                CaseArm(
                    expression=inp("A", rv("m"), rv("n")),
                    where=PredicateComparison(lhs=rv("n"), op="!=", rhs=rv("m")),
                ),
                CaseArm(
                    expression=BinaryApp(
                        label=1,
                        lhs=inp("B", rv("k"), rv("m")),
                        rhs=inp("C", rv("k"), rv("n")),
                    ),
                    specs=mul_reduce(),
                ),
            ]
        ),
        specs=[],
    )


def program_of(einsum, extra=()):
    return Program(
        declarations=[
            decl("Z", ["M", "N"]),
            decl("A", ["M", "N"]),
            decl("B", ["K", "M"]),
            decl("C", ["K", "N"]),
            *extra,
        ],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )


A = {(m, n): 10 * m + n + 1 for m in range(3) for n in range(3)}
B = {(k, m): k + m + 1 for k in range(3) for m in range(3)}
C = {(k, n): 2 * k + n + 1 for k in range(3) for n in range(3)}


def test_equation_61_matches_its_own_desugaring() -> None:
    """Eq. 155: T1 = B.C reduced; T2_{m,n : n != m} = A; Z = T1 << T2."""
    result = evaluate(program_of(paper_equation_61()), inputs={"A": A, "B": B, "C": C})
    want = {}
    for m in range(3):
        for n in range(3):
            if n != m:
                want[(m, n)] = A[(m, n)]
            else:
                want[(m, n)] = sum(B[(k, m)] * C[(k, n)] for k in range(3))
    assert dict(result["Z"].items()) == want


def test_each_arm_runs_only_over_its_region() -> None:
    program = program_of(paper_equation_61())
    env = build_environment(program, inputs={"A": A, "B": B, "C": C})
    result = evaluate_einsum(program.main_edge.cascade.einsums[0], env=env)
    off_diagonal, diagonal = result.arms
    assert len(off_diagonal.iteration_space) == 6
    assert off_diagonal.iteration_space.axes == ("m", "n")
    # The reduced rank k is an axis of the second arm alone.
    assert diagonal.iteration_space.axes == ("k", "m", "n")
    assert len(diagonal.iteration_space) == 9


def test_a_case_statement_round_trips_and_is_one_write() -> None:
    program = program_of(paper_equation_61())
    again = Program.model_validate_json(program.model_dump_json())
    assert again == program
    assert again.main_edge.cascade.einsums[0].expression.kind == "cases"
    assert check_single_assignment(program) == []


def test_first_matching_arm_wins_and_otherwise_takes_the_rest() -> None:
    """Overlapping conditions: the order written decides."""
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[rv("m"), rv("n")],
        expression=Cases(
            arms=[
                CaseArm(
                    expression=inp("A", rv("m"), rv("n")),
                    where=PredicateComparison(lhs=rv("m"), op="==", rhs=rv("n")),
                ),
                CaseArm(
                    expression=inp("B", rv("m"), rv("n")),
                    where=PredicateComparison(lhs=rv("m"), op="<=", rhs=rv("n")),
                ),
                CaseArm(expression=inp("C", rv("m"), rv("n"))),
            ]
        ),
        specs=[],
    )
    result = evaluate(program_of(einsum), inputs={"A": A, "B": B, "C": C})
    got = dict(result["Z"].items())
    for m in range(3):
        for n in range(3):
            source = A if m == n else B if m < n else C
            assert got[(m, n)] == source[(m, n)]


def test_an_empty_arm_writes_nothing_in_its_region() -> None:
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[rv("m"), rv("n")],
        expression=Cases(
            arms=[
                CaseArm(
                    expression=inp("A", rv("m"), rv("n")),
                    where=PredicateComparison(lhs=rv("n"), op="<", rhs=rv("m")),
                ),
                CaseArm(expression=None),
            ]
        ),
        specs=[],
    )
    result = evaluate(program_of(einsum), inputs={"A": A})
    assert dict(result["Z"].items()) == {
        (m, n): A[(m, n)] for m in range(3) for n in range(3) if n < m
    }


def test_a_condition_may_read_a_tensor() -> None:
    """docs/cases.md Pattern 2: `if X_{m,n} > 0`, decided point by point."""
    zero = decl("Lit_0", [], empty=None, value=0)
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[rv("m"), rv("n")],
        expression=Cases(
            arms=[
                CaseArm(
                    expression=inp("A", rv("m"), rv("n")),
                    when=Comparison(
                        lhs=TensorProjectionValue(
                            proj=TensorProjection(tensor="X", ranks=[rv("m"), rv("n")])
                        ),
                        op=">",
                        rhs=TensorProjectionValue(
                            proj=TensorProjection(tensor="Lit_0", ranks=[])
                        ),
                    ),
                ),
                CaseArm(expression=inp("B", rv("m"), rv("n"))),
            ]
        ),
        specs=[],
    )
    x = {(0, 0): 5, (1, 2): 7, (2, 2): -1}
    result = evaluate(
        program_of(einsum, extra=[decl("X", ["M", "N"]), zero]),
        inputs={"A": A, "B": B, "X": x},
    )
    got = dict(result["Z"].items())
    for m in range(3):
        for n in range(3):
            source = A if x.get((m, n), 0) > 0 else B
            assert got[(m, n)] == source[(m, n)]


def test_an_arms_operators_are_seen_by_the_name_check() -> None:
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[rv("m"), rv("n")],
        expression=Cases(
            arms=[
                CaseArm(
                    expression=BinaryApp(
                        label=1,
                        lhs=inp("A", rv("m"), rv("n")),
                        rhs=inp("B", rv("m"), rv("n")),
                    ),
                    specs=[
                        MapSpec(
                            label=1,
                            compute_op=UserDefinedComputeOp(name="min"),
                            merge_op=BuiltinMergeOp(symbol="intersect"),
                        )
                    ],
                )
            ]
        ),
        specs=[],
    )
    assert "min" in {use.name for use in udf_uses(program_of(einsum))}


def test_populate_belongs_to_the_einsum_not_to_an_arm() -> None:
    populate = PopulateSpec(
        rank_list=["n"],
        coord_op=CoordinateOp(name="select-min-val"),
        compute_op=UserDefinedComputeOp(name="identity"),
    )
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[rv("m"), rv("n")],
        expression=Cases(
            arms=[CaseArm(expression=inp("A", rv("m"), rv("n")), specs=[populate])]
        ),
        specs=[],
    )
    with pytest.raises(UnderSpecifiedProgramError, match="Populate"):
        evaluate(program_of(einsum), inputs={"A": A})
