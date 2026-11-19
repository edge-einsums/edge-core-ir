"""Macros and conditional cascades.

LOCAL PATCH (edge-workbench issue 9).

A macro (paper, appendix on macros) is "a named, parameterized Einsum
template. Invoking a macro expands it into one or more Einsums via lexical
substitution of its tensor and rank variable expression parameters."
``edge_ir.ir.macro.expand`` is that substitution.

A conditional cascade is a case statement whose arms are cascades -- in the
direction-optimizing BFS tutorial, cascade-form macro invocations:

    (F_{i+1,d}, P_{i+1,d}) =
        { TOP-DOWN(G_{s,d}, F_{i,s}, P_{i,d})   if TopDown_i == True
        { BOTTOM-UP(G_{s,d}, F_{i,s}, P_{i,d})  if TopDown_i == False
"""

from __future__ import annotations

import pytest

from edge_ir.evaluator import evaluate
from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.einsum import (
    ConditionalArm,
    ConditionalCascade,
    Einsum,
    NestedCascade,
    flatten,
)
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankVariable,
    TensorProjection,
    UnaryApp,
)
from edge_ir.ir.macro import MacroDefinition, MacroExpansionError, expand
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    BuiltinUnaryOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.predicate import SetMembership
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import (
    Comparison,
    PropertyApp,
    StoppingCondition,
    TensorProjectionValue,
)
from edge_ir.ir.tensor import (
    BuiltinDataType,
    CoordSetName,
    RankDeclaration,
    TensorDeclaration,
)
from edge_ir.validator.single_assignment import check_single_assignment


def rv(name):
    return RankVariable(name=name)


def lit(value):
    return RankConstantLiteral(value=value)


def nxt(name="i"):
    return RankArith(op="+", lhs=rv(name), rhs=lit(1))


def proj(tensor, *ranks):
    return TensorProjection(tensor=tensor, ranks=list(ranks))


def inp(tensor, *ranks):
    return InputTensor(proj=proj(tensor, *ranks))


def op(name):
    if name in "+-*/":
        return BuiltinComputeOp(symbol=name)
    return UserDefinedComputeOp(name=name)


def map_(compute, merge, label=1):
    return MapSpec(
        label=label, compute_op=op(compute), merge_op=BuiltinMergeOp(symbol=merge)
    )


def reduce_(compute, ranks, label=1):
    return ReduceSpec(
        label=label,
        rank_list=ranks,
        compute_op=op(compute),
        merge_op=BuiltinMergeOp(symbol="union"),
    )


def einsum(tensor, ranks, lhs, rhs, *specs):
    return Einsum(
        output_tensor=tensor,
        output_ranks=list(ranks),
        expression=BinaryApp(label=1, lhs=lhs, rhs=rhs),
        specs=list(specs),
    )


def negated(tensor, *ranks):
    return UnaryApp(op=BuiltinUnaryOp(symbol="not"), operand=inp(tensor, *ranks))


################################################################################
# Macro expansion
################################################################################


def gemm() -> MacroDefinition:
    """MACRO GEMM(TA_{a,c}, TB_{c,b}) -> T_{a,b} :  T_{a,b} = TA_{a,c} x TB_{c,b}"""
    a, b, c = rv("a"), rv("b"), rv("c")
    return MacroDefinition(
        name="GEMM",
        inputs=[proj("TA", a, c), proj("TB", c, b)],
        outputs=[proj("T", a, b)],
        body=[
            einsum(
                "T",
                [a, b],
                inp("TA", a, c),
                inp("TB", c, b),
                map_("*", "intersect"),
                reduce_("+", ["c"]),
            )
        ],
    )


def test_assignment_form_substitutes_tensors_and_rank_variables() -> None:
    """Z_{m,n} = GEMM(A_{m,k}, B_{k,n}): TA -> A, a,c -> m,k; TB -> B; T -> Z."""
    m, k, n = rv("m"), rv("k"), rv("n")
    (expanded,) = expand(gemm(), [proj("A", m, k), proj("B", k, n)], [proj("Z", m, n)])
    assert expanded == einsum(
        "Z",
        [m, n],
        inp("A", m, k),
        inp("B", k, n),
        map_("*", "intersect"),
        reduce_("+", ["k"]),
    )


def test_an_argument_may_be_a_rank_variable_expression() -> None:
    """The paper's own example, GEMM(A_{m+1,k}, B_{k,n})_{m,n}: "TA becomes A;
    rank variables a, c become m+1, k". Here the output is given as Z_{m+1,n},
    the consistent subscript for the rank variable the two parameters share."""
    k, n = rv("k"), rv("n")
    (expanded,) = expand(
        gemm(), [proj("A", nxt("m"), k), proj("B", k, n)], [proj("Z", nxt("m"), n)]
    )
    assert expanded.output_ranks == [nxt("m"), n]
    assert expanded.expression.lhs == inp("A", nxt("m"), k)


def test_inconsistent_subscripts_across_parameters_are_refused() -> None:
    m, k, n = rv("m"), rv("k"), rv("n")
    with pytest.raises(MacroExpansionError, match="inconsistently"):
        expand(gemm(), [proj("A", nxt("m"), k), proj("B", k, n)], [proj("Z", m, n)])
    with pytest.raises(MacroExpansionError, match="invoked with 1"):
        expand(gemm(), [proj("A", m, k)], [proj("Z", m, n)])


def test_a_shifted_formal_binds_its_variable_through_the_shift() -> None:
    """MACRO STEP(TX_{i,v}) -> TX_{i+1,v}, invoked as X_{g,v} = STEP(X_{g-1,v})."""
    i, v, g = rv("i"), rv("v"), rv("g")
    step = MacroDefinition(
        name="STEP",
        inputs=[proj("TX", i, v)],
        outputs=[proj("TX", nxt(), v)],
        body=[
            Einsum(
                output_tensor="TX",
                output_ranks=[nxt(), v],
                expression=inp("TX", i, v),
                specs=[],
            )
        ],
    )
    previous = RankArith(op="-", lhs=g, rhs=lit(1))
    (expanded,) = expand(step, [proj("X", previous, v)], [proj("X", g, v)])
    assert expanded.output_tensor == "X"
    assert expanded.output_ranks == [g, v]  # (g-1)+1
    assert expanded.expression == inp("X", previous, v)


################################################################################
# A conditional cascade over two macros: direction-switching BFS
################################################################################


def top_down() -> MacroDefinition:
    """MACRO TOP-DOWN(TG_{s,d}, TF_{i,s}, TP_{i,d}) -> TF_{i+1,d}, TP_{i+1,d}"""
    i, s, d = rv("i"), rv("s"), rv("d")
    return MacroDefinition(
        name="TOP-DOWN",
        inputs=[proj("TG", s, d), proj("TF", i, s), proj("TP", i, d)],
        outputs=[proj("TF", nxt(), d), proj("TP", nxt(), d)],
        body=[
            einsum(
                "T",
                [i, d],
                inp("TG", s, d),
                inp("TF", i, s),
                map_("AND", "intersect"),
                reduce_("OR", ["s"]),
            ),
            einsum(
                "TF",
                [nxt(), d],
                inp("T", i, d),
                negated("TP", i, d),
                map_("AND", "intersect"),
            ),
            einsum(
                "TP",
                [nxt(), d],
                inp("TP", i, d),
                inp("TF", nxt(), d),
                map_("OR", "union"),
            ),
        ],
    )


def bottom_up() -> MacroDefinition:
    """MACRO BOTTOM-UP(TG_{s,d}, TF_{i,s}, TP_{i,d}) -> TF_{i+1,d}, TP_{i+1,d}"""
    i, s, d = rv("i"), rv("s"), rv("d")
    return MacroDefinition(
        name="BOTTOM-UP",
        inputs=[proj("TG", s, d), proj("TF", i, s), proj("TP", i, d)],
        outputs=[proj("TF", nxt(), d), proj("TP", nxt(), d)],
        body=[
            einsum(
                "U",
                [i, s, d],
                inp("TG", s, d),
                negated("TP", i, d),
                map_("AND", "intersect"),
            ),
            einsum(
                "TF",
                [nxt(), d],
                inp("U", i, s, d),
                inp("TF", i, s),
                map_("AND", "intersect"),
                reduce_("OR", ["s"]),
            ),
            einsum(
                "TP",
                [nxt(), d],
                inp("TP", i, d),
                inp("TF", nxt(), d),
                map_("OR", "union"),
            ),
        ],
    )


def mode_is(value: bool) -> Comparison:
    return Comparison(
        lhs=TensorProjectionValue(proj=proj("TopDown", rv("i"))),
        op="==",
        rhs=TensorProjectionValue(proj=proj("Lit_True" if value else "Lit_False")),
    )


def switching_bfs() -> Program:
    """Top-down while the frontier has fewer than two vertices, else bottom-up."""
    i, s, d = rv("i"), rv("s"), rv("d")
    actual_in = [proj("G", s, d), proj("F", i, s), proj("P", i, d)]
    actual_out = [proj("F", nxt(), d), proj("P", nxt(), d)]

    def boolean(name, ranks):
        return TensorDeclaration(
            name=name,
            ranks=[
                RankDeclaration(name=r, shape=None if r == "I" else "|V|")
                for r in ranks
            ],
            data_type=BuiltinDataType(name="bool"),
            empty_value=False,
        )

    def literal(name, dtype, value):
        return TensorDeclaration(
            name=name,
            ranks=[],
            data_type=BuiltinDataType(name=dtype),
            empty_value=None,
            value=value,
        )

    root = [SetMembership(member=rv("n"), coord_set=CoordSetName(name="root"))]
    return Program(
        declarations=[
            boolean("G", ["S", "D"]),
            boolean("F", ["I", "S"]),
            boolean("P", ["I", "S"]),
            boolean("T", ["I", "D"]),
            boolean("U", ["I", "S", "D"]),
            boolean("TopDown", ["I"]),
            TensorDeclaration(
                name="NF",
                ranks=[RankDeclaration(name="I", shape=None)],
                data_type=BuiltinDataType(name="int"),
                empty_value=0,
            ),
            literal("Lit_True", "bool", True),
            literal("Lit_False", "bool", False),
            literal("Lit_0", "int", 0),
            literal("Lit_2", "int", 2),
        ],
        initialization=Initialization(
            einsums=[
                Einsum(
                    output_tensor="F",
                    output_ranks=[lit(0), rv("n")],
                    expression=inp("Lit_True"),
                    specs=[],
                    predicates=root,
                ),
                Einsum(
                    output_tensor="P",
                    output_ranks=[lit(0), rv("n")],
                    expression=inp("Lit_True"),
                    specs=[],
                    predicates=root,
                ),
                Einsum(
                    output_tensor="TopDown",
                    output_ranks=[lit(0)],
                    expression=inp("Lit_True"),
                    specs=[],
                ),
            ]
        ),
        main_edge=MainEdge(
            cascade=NestedCascade(
                einsums=[
                    # NF_i = F_{i,s} :: \/ +(u)
                    einsum(
                        "NF",
                        [i],
                        inp("F", i, s),
                        inp("Lit_True"),
                        map_("take_left", "take_left"),
                        reduce_("+", ["s"]),
                    ),
                    ConditionalCascade(
                        arms=[
                            ConditionalArm(
                                when=mode_is(True),
                                einsums=expand(top_down(), actual_in, actual_out),
                                macro="TOP-DOWN",
                            ),
                            ConditionalArm(
                                when=mode_is(False),
                                einsums=expand(bottom_up(), actual_in, actual_out),
                                macro="BOTTOM-UP",
                            ),
                        ]
                    ),
                    # TopDown_{i+1} = NF_i . 2 :: /\ <(^)
                    einsum(
                        "TopDown",
                        [nxt()],
                        inp("NF", i),
                        inp("Lit_2"),
                        map_("lt", "intersect"),
                    ),
                ],
                stopping_conditions=[
                    StoppingCondition(
                        rank_variable="i",
                        predicate=Comparison(
                            lhs=PropertyApp(
                                name="occupancy",
                                pinned_rank_variables=["i"],
                                operands=[proj("F", nxt())],
                            ),
                            op="==",
                            rhs=TensorProjectionValue(proj=proj("Lit_0")),
                        ),
                    )
                ],
            )
        ),
    )


#      0 -> 1, 2, 3 ;  1,2,3 -> 4 ;  4 -> 5 ;  5 -> 6 ;  7 is unreachable
EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 5), (5, 6)]


def run(edges=EDGES, size=8, root=0):
    return evaluate(
        switching_bfs(),
        inputs={"G": {e: True for e in edges}},
        shape_env={"|V|": size},
        coord_sets={"root": [root]},
    )


def generation(result, tensor, g):
    return sorted(c[1:] for c, _ in result[tensor].items() if c[0] == g)


def test_the_expanded_program_validates_and_round_trips() -> None:
    program = switching_bfs()
    assert Program.model_validate_json(program.model_dump_json()) == program
    conditional = program.main_edge.cascade.einsums[1]
    assert [arm.macro for arm in conditional.arms] == ["TOP-DOWN", "BOTTOM-UP"]
    # Formal names are gone; what the body names itself (T, U) stays.
    assert [e.output_tensor for e in flatten(program.main_edge.cascade)] == [
        "NF",
        "T",
        "F",
        "P",
        "U",
        "F",
        "P",
        "TopDown",
    ]


def test_exactly_one_arm_runs_each_generation() -> None:
    result = run()
    # Frontiers: {0}, {1,2,3}, {4}, {5}, {6}. Top-down while |F| < 2.
    modes = {c[0]: v for c, v in result["TopDown"].items()}
    assert [modes.get(g, False) for g in range(4)] == [True, True, False, True]
    # T is written only by TOP-DOWN, U only by BOTTOM-UP.
    assert [bool(generation(result, "T", g)) for g in range(4)] == [
        True,
        True,
        False,
        True,
    ]
    assert [bool(generation(result, "U", g)) for g in range(4)] == [
        False,
        False,
        True,
        False,
    ]
    assert [len(g) for g in result.trace.per_generation] == [5, 5, 5, 5, 5]


def test_both_directions_find_the_same_frontiers() -> None:
    result = run()
    assert [generation(result, "F", g) for g in range(5)] == [
        [(0,)],
        [(1,), (2,), (3,)],
        [(4,)],
        [(5,)],
        [(6,)],
    ]
    assert generation(result, "F", 5) == []
    assert generation(result, "P", 5) == [(v,) for v in range(7)]


def test_two_arms_writing_one_slice_is_a_single_assignment() -> None:
    """Only one arm runs, so F_{i+1} is written once per generation."""
    assert check_single_assignment(switching_bfs()) == []


def test_no_arm_runs_when_no_condition_holds() -> None:
    program = switching_bfs()
    cascade = program.main_edge.cascade
    only_bottom_up = ConditionalCascade(arms=[cascade.einsums[1].arms[1]])
    lonely = program.model_copy(
        update={
            "main_edge": MainEdge(
                cascade=cascade.model_copy(
                    update={
                        "einsums": [
                            cascade.einsums[0],
                            only_bottom_up,
                            cascade.einsums[2],
                        ]
                    }
                )
            )
        }
    )
    result = evaluate(
        lonely,
        inputs={"G": {e: True for e in EDGES}},
        shape_env={"|V|": 8},
        coord_sets={"root": [0]},
    )
    # Generation 0 is top-down mode, which has no arm: nothing is written,
    # F_1 is empty and the cascade stops.
    assert result.trace.generations == 1
    assert generation(result, "F", 1) == []
