"""Single assignment: no two Einsums may write the same tensor slice.

The shipped programs are the real cases. bfs, bellman-ford, dfs and
label-propagation are single assignment. max-flow is not, and the pass has to
name every place it is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_ir.ir.actions import MapSpec
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankExpression,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.op import BuiltinComputeOp, BuiltinMergeOp
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration
from edge_ir.validator import check_single_assignment

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "bfs": "bfs/bfs_program.json",
    "bellman-ford": "bellman-ford/bellman_ford_program.json",
    "dfs": "dfs/dfs_program.json",
    "label-propagation": "label-propagation/label_propagation_program.json",
    "max-flow": "max-flow/max_flow_program.json",
}


def _artifact(name: str) -> Program:
    for folder in ("examples/algorithms",):
        path = ROOT / folder / ARTIFACTS[name]
        if path.exists():
            return Program.model_validate_json(path.read_text())
    raise AssertionError(f"no artifact for {name}")


################################################################################
# Helpers for synthetic programs
################################################################################


def _decl(name: str) -> TensorDeclaration:
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name="M", shape=10)],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )


def _einsum(output: str, ranks: list[RankExpression]) -> Einsum:
    return Einsum(
        output_tensor=output,
        output_ranks=ranks,
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
                rank_list=["m"],
                compute_op=BuiltinComputeOp(symbol="+"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )


def _program(cascade: list[Einsum], init: list[Einsum] | None = None) -> Program:
    return Program(
        declarations=[_decl("A"), _decl("B"), _decl("C")],
        initialization=Initialization(einsums=init or []),
        main_edge=MainEdge(cascade=NestedCascade(einsums=cascade)),
    )


def _var(name: str) -> RankVariable:
    return RankVariable(name=name)


def _plus_one(name: str) -> RankArith:
    return RankArith(op="+", lhs=_var(name), rhs=RankConstantLiteral(value=1))


def _const(value: int) -> RankConstantLiteral:
    return RankConstantLiteral(value=value)


################################################################################
# Shipped programs
################################################################################


@pytest.mark.parametrize("name", ["bfs", "bellman-ford", "dfs", "label-propagation"])
def test_shipped_programs_are_single_assignment(name: str) -> None:
    assert check_single_assignment(_artifact(name)) == []


def test_max_flow_is_not_single_assignment_yet() -> None:
    """max-flow writes four tensors twice. Three are workarounds for missing
    IR features: initialization runs inside the cascade (F, E, R), and E04's
    case statement is lowered to three writes of the same slice (R). Act is
    written once by the push step and once by the relabel step.

    This pins the current state. Fixing the cascade should shrink this list,
    and the test should shrink with it.
    """
    violations = check_single_assignment(_artifact("max-flow"))
    assert {v.tensor for v in violations} == {"D", "F", "E", "R", "Act"}
    r_writes = [v for v in violations if v.tensor == "R"]
    assert len(r_writes) >= 6, "the three E04 arms and the update all write R"


################################################################################
# The rule itself
################################################################################


def test_two_writes_of_one_constant_slice_conflict() -> None:
    violations = check_single_assignment(
        _program(
            [_einsum("C", [_const(1), _var("u")]), _einsum("C", [_const(1), _var("u")])]
        )
    )
    assert len(violations) == 1
    assert violations[0].detail == "the same slice"


def test_distinct_constant_generations_do_not_conflict() -> None:
    assert (
        check_single_assignment(
            _program(
                [
                    _einsum("C", [_const(0), _var("u")]),
                    _einsum("C", [_const(1), _var("u")]),
                ]
            )
        )
        == []
    )


def test_i_and_i_plus_one_conflict_across_rounds() -> None:
    """The reason the rule is not obvious: round i writes C[i+1], and round
    i+1 writes C[i+1] again."""
    violations = check_single_assignment(
        _program(
            [
                _einsum("C", [_var("i"), _var("u")]),
                _einsum("C", [_plus_one("i"), _var("u")]),
            ]
        )
    )
    assert len(violations) == 1
    assert "advances" in violations[0].detail


def test_a_generation_the_rank_never_reaches_does_not_conflict() -> None:
    """The bfs shape: initialization writes generation 0 and the cascade
    writes i+1, which starts at 1."""
    assert (
        check_single_assignment(
            _program(
                cascade=[_einsum("C", [_plus_one("i"), _var("u")])],
                init=[_einsum("C", [_const(0), _var("u")])],
            )
        )
        == []
    )


def test_a_constant_generation_conflicts_with_the_advancing_rank() -> None:
    violations = check_single_assignment(
        _program(
            [
                _einsum("C", [_const(1), _var("u")]),
                _einsum("C", [_plus_one("i"), _var("u")]),
            ]
        )
    )
    assert len(violations) == 1
    assert "passes through" in violations[0].detail


def test_initialization_and_cascade_are_compared() -> None:
    violations = check_single_assignment(
        _program(
            cascade=[_einsum("C", [_const(1), _var("u")])],
            init=[_einsum("C", [_const(1), _var("u")])],
        )
    )
    assert len(violations) == 1
    assert violations[0].first.path == "initialization[0]"
    assert violations[0].second.path == "cascade[0]"


def test_different_tensors_never_conflict() -> None:
    assert (
        check_single_assignment(
            _program([_einsum("C", [_var("u")]), _einsum("A", [_var("u")])])
        )
        == []
    )


def test_violation_reads_as_a_sentence() -> None:
    violations = check_single_assignment(
        _program(
            [
                _einsum("C", [_var("i"), _var("u")]),
                _einsum("C", [_plus_one("i"), _var("u")]),
            ]
        )
    )
    text = str(violations[0])
    assert "cascade[0] writes C[i, u]" in text
    assert "cascade[1] writes C[i+1, u]" in text
