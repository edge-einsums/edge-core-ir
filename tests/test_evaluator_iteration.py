"""Iteration-space construction, the iterative rank, and failing loudly.

The EDGE semantics define ``IS = R_0 x ... x R_{D-1}`` with each ``R_d``
defaulting to the whole numbers, and notes that an implementation need not
enumerate all of it. These tests pin how this evaluator bounds the space,
how a predicate restricts it, and what it refuses to guess.
"""

from __future__ import annotations

import pytest

from edge_ir.evaluator import (
    IterationLimitExceeded,
    UnboundNameError,
    UnderSpecifiedProgramError,
    build_environment,
    detect_iterative_rank,
    evaluate,
)
from edge_ir.evaluator.iteration import build_iteration_space
from edge_ir.ir.actions import MapSpec
from edge_ir.ir.einsum import Einsum, IterativeRankSpec, NestedCascade
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.op import BuiltinComputeOp, BuiltinMergeOp
from edge_ir.ir.predicate import LogicalAnd, PredicateComparison, SetMembership
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import (
    BuiltinDataType,
    CoordSetName,
    RankDeclaration,
    TensorDeclaration,
)


def _int_decl(name: str, ranks: list[RankDeclaration], empty: object = 0):
    return TensorDeclaration(
        name=name,
        ranks=ranks,
        data_type=BuiltinDataType(name="int"),
        empty_value=empty,
    )


def _copy_einsum(predicates=None) -> Einsum:
    """``Z_m = A_m``, optionally restricted."""
    return Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m")],
        expression=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
        specs=[],
        predicates=predicates or [],
    )


def _space(einsum: Einsum, decls, shape_env=None, coord_sets=None, pinned=None):
    program = Program(
        declarations=decls,
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[einsum])),
    )
    env = build_environment(
        program, shape_env=shape_env or {}, coord_sets=coord_sets or {}
    )
    return build_iteration_space(
        einsum,
        declarations=list(env.declarations.values()),
        spaces=env.spaces,
        shape_env=env.shape_env,
        coord_sets=env.coord_sets,
        registry=env.functions,
        pinned=pinned or {},
    )


################################################################################
# Bounding a rank variable set
################################################################################


def test_a_shape_symbol_resolves_to_a_dense_coordinate_set() -> None:
    """Dense rank coordinate set: ``DenseCS_R = {r | r in [0, RS)}``.

    ``|V|`` is a D22 bucket-B name: symbolic until iteration-space
    construction, resolved once from the shape environment.
    """
    decls = [
        _int_decl("Z", [RankDeclaration(name="M", shape="|V|")]),
        _int_decl("A", [RankDeclaration(name="M", shape="|V|")]),
    ]
    space = _space(_copy_einsum(), decls, shape_env={"|V|": 5})
    assert space.axes == ("m",)
    assert space.axis_sets == ((0, 1, 2, 3, 4),)


def test_an_unresolvable_shape_symbol_fails_loudly() -> None:
    decls = [
        _int_decl("Z", [RankDeclaration(name="M", shape="|V|")]),
        _int_decl("A", [RankDeclaration(name="M", shape="|V|")]),
    ]
    with pytest.raises(UnboundNameError, match=r"\|V\|"):
        _space(_copy_einsum(), decls, shape_env={})


def test_an_unbounded_rank_variable_set_fails_loudly() -> None:
    """A rank with neither a shape nor a coordinate set keeps the default
    whole-number rank variable set, which cannot be enumerated. The
    evaluator says so rather than picking a bound."""
    decls = [
        _int_decl("Z", [RankDeclaration(name="M")]),
        _int_decl("A", [RankDeclaration(name="M")]),
    ]
    with pytest.raises(UnderSpecifiedProgramError, match="cannot bound the rank"):
        _space(_copy_einsum(), decls)


def test_an_integer_shift_is_inverted_against_the_rank_coordinate_set() -> None:
    """``Z_{m+1} = A_m``: the axis is still ``m``, and its set is the output
    coordinate set shifted back by one."""
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[
            RankArith(
                op="+", lhs=RankVariable(name="m"), rhs=RankConstantLiteral(value=1)
            )
        ],
        expression=InputTensor(
            proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
        ),
        specs=[],
    )
    decls = [
        _int_decl("Z", [RankDeclaration(name="M", shape=3)]),
        _int_decl("A", [RankDeclaration(name="M", shape=3)]),
    ]
    space = _space(einsum, decls)
    # A contributes {0,1,2}; the shifted output contributes {-1,0,1}. The
    # union over-approximates, and the points whose output coordinate falls
    # outside CS^Z are pruned later, where EP is undefined.
    assert space.axis_sets == ((-1, 0, 1, 2),)


################################################################################
# Predicates restrict the space
################################################################################


def test_a_membership_predicate_materializes_the_restricted_set() -> None:
    """D23's worked example: ``F_{0, s : s in id}`` with ``id = {2,5,7}`` and
    ``|V| = 100`` walks three points, not a hundred with skips."""
    decls = [
        _int_decl("Z", [RankDeclaration(name="M", shape="|V|")]),
        _int_decl("A", [RankDeclaration(name="M", shape="|V|")]),
    ]
    einsum = _copy_einsum(
        [
            SetMembership(
                member=RankVariable(name="m"), coord_set=CoordSetName(name="id")
            )
        ]
    )
    space = _space(einsum, decls, shape_env={"|V|": 100}, coord_sets={"id": [2, 5, 7]})
    assert space.axis_sets == ((2, 5, 7),)
    assert space.points == ((2,), (5,), (7,))


def test_a_comparison_predicate_filters_points() -> None:
    """RVE predicates: ``B_{n, k : k < n}``."""
    decls = [
        _int_decl(
            "Z",
            [RankDeclaration(name="M", shape=3), RankDeclaration(name="N", shape=3)],
        ),
        _int_decl(
            "A",
            [RankDeclaration(name="M", shape=3), RankDeclaration(name="N", shape=3)],
        ),
    ]
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[RankVariable(name="m"), RankVariable(name="n")],
        expression=InputTensor(
            proj=TensorProjection(
                tensor="A", ranks=[RankVariable(name="m"), RankVariable(name="n")]
            )
        ),
        specs=[],
        predicates=[
            PredicateComparison(
                lhs=RankVariable(name="n"), op="<", rhs=RankVariable(name="m")
            )
        ],
    )
    space = _space(einsum, decls)
    assert space.points == ((1, 0), (2, 0), (2, 1))


def test_a_compound_predicate_is_honoured() -> None:
    decls = [
        _int_decl("Z", [RankDeclaration(name="M", shape=6)]),
        _int_decl("A", [RankDeclaration(name="M", shape=6)]),
    ]
    einsum = _copy_einsum(
        [
            LogicalAnd(
                operands=[
                    PredicateComparison(
                        lhs=RankVariable(name="m"),
                        op=">=",
                        rhs=RankConstantLiteral(value=2),
                    ),
                    PredicateComparison(
                        lhs=RankVariable(name="m"),
                        op="<",
                        rhs=RankConstantLiteral(value=5),
                    ),
                ]
            )
        ]
    )
    assert _space(einsum, decls).points == ((2,), (3,), (4,))


################################################################################
# The iterative rank
################################################################################


def _advancing_cascade(stopping=(), iterative_rank=None) -> NestedCascade:
    einsum = Einsum(
        output_tensor="Z",
        output_ranks=[
            RankArith(
                op="+", lhs=RankVariable(name="i"), rhs=RankConstantLiteral(value=1)
            ),
            RankVariable(name="m"),
        ],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(
                    tensor="Z", ranks=[RankVariable(name="i"), RankVariable(name="m")]
                )
            ),
            rhs=InputTensor(
                proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
            ),
        ),
        specs=[
            MapSpec(
                label=1,
                compute_op=BuiltinComputeOp(symbol="+"),
                merge_op=BuiltinMergeOp(symbol="union"),
            )
        ],
    )
    return NestedCascade(
        einsums=[einsum],
        stopping_conditions=list(stopping),
        iterative_rank=iterative_rank,
    )


def test_the_iterative_rank_is_detected_structurally() -> None:
    """A rank variable written at a non-zero integer shift of itself is what
    makes a cascade iterative -- ``Z_{i+1,m} = Z_{i,m} . A_m`` advances i."""
    assert detect_iterative_rank(_advancing_cascade()) == "i"


def test_the_cascade_may_name_its_iterative_rank_and_initial_value() -> None:
    """EDGE syntax: an iterative cascade specifies the rank
    variable, its initial value and its rank variable set. Max-flow's preflow
    indexes its first computed generation 1, not 0."""
    cascade = _advancing_cascade(
        iterative_rank=IterativeRankSpec(rank_variable="i", initial_value=1)
    )
    assert detect_iterative_rank(cascade) == "i"
    assert cascade.iterative_rank is not None
    assert cascade.iterative_rank.initial_value == 1


def test_an_iterative_cascade_with_no_stopping_condition_fails_loudly() -> None:
    """Nothing says when it ends, so the evaluator refuses to spin to the
    iteration cap and call the result an answer."""
    program = Program(
        declarations=[
            _int_decl(
                "Z", [RankDeclaration(name="I"), RankDeclaration(name="M", shape=2)]
            ),
            _int_decl("A", [RankDeclaration(name="M", shape=2)]),
        ],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=_advancing_cascade()),
    )
    with pytest.raises(UnderSpecifiedProgramError, match="declares no"):
        evaluate(program, inputs={"A": {(0,): 1}})


def test_an_explicit_generation_count_runs_a_stoppingless_cascade() -> None:
    program = Program(
        declarations=[
            _int_decl(
                "Z", [RankDeclaration(name="I"), RankDeclaration(name="M", shape=2)]
            ),
            _int_decl("A", [RankDeclaration(name="M", shape=2)]),
        ],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=_advancing_cascade()),
    )
    result = evaluate(program, inputs={"A": {(0,): 1}}, generations=3)
    assert result.trace.generations == 3
    assert result.trace.stopped_by == "generation-count"
    # Z_{i+1,0} accumulates A each generation: 1, 2, 3.
    assert dict(result["Z"].items()) == {(1, 0): 1, (2, 0): 2, (3, 0): 3}


def test_the_iteration_cap_stops_a_non_converging_cascade() -> None:
    """A buggy stopping condition cannot hang the evaluator."""
    from edge_ir.ir.stopping import (
        Comparison,
        RankExpressionValue,
        StoppingCondition,
    )

    never = StoppingCondition(
        rank_variable="i",
        predicate=Comparison(
            lhs=RankExpressionValue(expr=RankConstantLiteral(value=0)),
            op="==",
            rhs=RankExpressionValue(expr=RankConstantLiteral(value=1)),
        ),
    )
    program = Program(
        declarations=[
            _int_decl(
                "Z", [RankDeclaration(name="I"), RankDeclaration(name="M", shape=2)]
            ),
            _int_decl("A", [RankDeclaration(name="M", shape=2)]),
        ],
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=_advancing_cascade(stopping=[never])),
    )
    with pytest.raises(IterationLimitExceeded, match="without its"):
        evaluate(program, inputs={"A": {(0,): 1}}, max_iterations=25)


def test_an_unbound_tensor_fails_loudly() -> None:
    decls = [
        _int_decl("Z", [RankDeclaration(name="M", shape=2)]),
        _int_decl("A", [RankDeclaration(name="M", shape=2)]),
    ]
    program = Program(
        declarations=decls,
        initialization=Initialization(einsums=[]),
        main_edge=MainEdge(cascade=NestedCascade(einsums=[_copy_einsum()])),
    )
    with pytest.raises(UnboundNameError, match="does not declare"):
        evaluate(program, inputs={"Nope": {}})
