"""Construct the BFS program in IR form, dump indented JSON to stdout.

Reference: paper Equation 41, Section 7.3.

Save the output to examples/algorithms/bfs/bfs_program.json.
"""

from __future__ import annotations

import sys

from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade, StoppingCondition
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
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
from edge_ir.ir.predicate import SetMembership
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import Comparison, PropertyApp, TensorProjectionValue
from edge_ir.ir.tensor import (
    BuiltinDataType,
    CoordSetName,
    RankDeclaration,
    TensorDeclaration,
)


def build_bfs() -> Program:
    # Tensor declarations
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    f = TensorDeclaration(
        name="F",
        ranks=[
            RankDeclaration(name="I"),  # generational, no shape
            RankDeclaration(name="S", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=float("inf"),
    )
    p = TensorDeclaration(
        name="P",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    t = TensorDeclaration(
        name="T",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=float("inf"),
    )

    # Lit_0: synthesized zero-rank tensor for the stopping-condition
    # literal `0`. The other side of the comparison is `occupancy(F)`, a
    # PropertyApp — inherit data_type from the property's RETURN type (int),
    # NOT from F itself. For empty_value, int is ambiguous (could be 0,
    # +inf, -inf depending on context), so default to None.
    lit_0 = TensorDeclaration(
        name="Lit_0",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=None,
        value=0,
    )

    # Lit_True: synthesized zero-rank tensor for the P_0 init's literal True.
    # Inherits data_type from P (the output tensor of the init einsum), bool.
    # empty_value=None per the synthesized-literal rule (docs/parsing.md).
    lit_true = TensorDeclaration(
        name="Lit_True",
        ranks=[],
        data_type=BuiltinDataType(name="bool"),
        empty_value=None,
        value=True,
    )

    # Initialization: BFS init block per Equation 41:
    #   F_{0, s : s ∈ id} = 0    (init front-frontier from the identity set)
    #   P_{0, d : d ∈ id} = True (init parent-mask from the identity set)
    # The `: s ∈ id` / `: d ∈ id` are restricted-iteration predicates
    # (see docs/parsing.md): the einsum walks only coordinates in the
    # host-supplied coordinate set named `id`.
    f_init = Einsum(
        output_tensor="F",
        output_ranks=[RankConstantLiteral(value=0), RankVariable(name="s")],
        expression=InputTensor(proj=TensorProjection(tensor="Lit_0", ranks=[])),
        specs=[],
        predicates=[
            SetMembership(
                member=RankVariable(name="s"),
                coord_set=CoordSetName(name="id"),
                negated=False,
            )
        ],
    )
    p_init = Einsum(
        output_tensor="P",
        output_ranks=[RankConstantLiteral(value=0), RankVariable(name="d")],
        expression=InputTensor(proj=TensorProjection(tensor="Lit_True", ranks=[])),
        specs=[],
        predicates=[
            SetMembership(
                member=RankVariable(name="d"),
                coord_set=CoordSetName(name="id"),
                negated=False,
            )
        ],
    )
    init = Initialization(einsums=[f_init, p_init])

    # Helper: i+1 as a rank expression
    def i_plus_one() -> RankArith:
        return RankArith(
            op="+",
            lhs=RankVariable(name="i"),
            rhs=RankConstantLiteral(value=1),
        )

    # Einsum (a): T_{i,d} = G_{s,d} . F_{i,s}
    #             :: Map ^1 _s + (intersect)
    #             :: Reduce ^1 _s ANY (union)
    einsum_a = Einsum(
        output_tensor="T",
        output_ranks=[
            RankVariable(name="i"),
            RankVariable(name="d"),
        ],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(
                    tensor="G",
                    ranks=[
                        RankVariable(name="s"),
                        RankVariable(name="d"),
                    ],
                ),
            ),
            rhs=InputTensor(
                proj=TensorProjection(
                    tensor="F",
                    ranks=[
                        RankVariable(name="i"),
                        RankVariable(name="s"),
                    ],
                ),
            ),
        ),
        specs=[
            MapSpec(
                label=1,
                rank_list=["s"],
                compute_op=BuiltinComputeOp(symbol="+"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            ),
            ReduceSpec(
                label=1,
                rank_list=["s"],
                compute_op=UserDefinedComputeOp(name="ANY"),
                merge_op=BuiltinMergeOp(symbol="union"),
            ),
        ],
    )

    # Einsum (b): F_{i+1,d} = T_{i,d} . not P_{i,d}
    #             :: Map ^1 _d take_left (intersect)
    # Note: take_left is a user-defined COMPUTE operator (selects the
    # left operand value). The merge operator is intersect.
    einsum_b = Einsum(
        output_tensor="F",
        output_ranks=[
            i_plus_one(),
            RankVariable(name="d"),
        ],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(
                    tensor="T",
                    ranks=[
                        RankVariable(name="i"),
                        RankVariable(name="d"),
                    ],
                ),
            ),
            # A unary is a degenerate Map against the all-ones tensor, and
            # its MERGE is what decides where the result exists
            # (EDGE's decomposition of unary functions). Logical
            # complement maps empty to NON-empty, so it must visit the points
            # where P is empty: the not_left merge. With take_left this line
            # would compute T AND P instead of T MINUS P -- the exact
            # complement of the intended frontier mask.
            rhs=UnaryApp(
                op=BuiltinUnaryOp(symbol="not"),
                merge_op=BuiltinMergeOp(symbol="not_left"),
                operand=InputTensor(
                    proj=TensorProjection(
                        tensor="P",
                        ranks=[
                            RankVariable(name="i"),
                            RankVariable(name="d"),
                        ],
                    ),
                ),
            ),
        ),
        specs=[
            MapSpec(
                label=1,
                rank_list=["d"],
                compute_op=UserDefinedComputeOp(name="take_left"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            ),
        ],
    )

    # Einsum (c): P_{i+1,d} = P_{i,d} . F_{i+1,d}
    #             :: Map ^1 _d OR (union)
    einsum_c = Einsum(
        output_tensor="P",
        output_ranks=[
            i_plus_one(),
            RankVariable(name="d"),
        ],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(
                    tensor="P",
                    ranks=[
                        RankVariable(name="i"),
                        RankVariable(name="d"),
                    ],
                ),
            ),
            rhs=InputTensor(
                proj=TensorProjection(
                    tensor="F",
                    ranks=[
                        i_plus_one(),
                        RankVariable(name="d"),
                    ],
                ),
            ),
        ),
        specs=[
            MapSpec(
                label=1,
                rank_list=["d"],
                compute_op=UserDefinedComputeOp(name="OR"),
                merge_op=BuiltinMergeOp(symbol="union"),
            ),
        ],
    )

    cascade = NestedCascade(
        einsums=[einsum_a, einsum_b, einsum_c],
        stopping_conditions=[
            StoppingCondition(
                rank_variable="i",
                predicate=Comparison(
                    lhs=PropertyApp(
                        name="occupancy",
                        pinned_rank_variables=["i"],
                        operands=[
                            TensorProjection(
                                tensor="F",
                                ranks=[
                                    RankArith(
                                        op="+",
                                        lhs=RankVariable(name="i"),
                                        rhs=RankConstantLiteral(value=1),
                                    )
                                ],
                            )
                        ],
                    ),
                    op="==",
                    rhs=TensorProjectionValue(
                        proj=TensorProjection(tensor="Lit_0", ranks=[])
                    ),
                ),
            ),
        ],
    )

    return Program(
        declarations=[g, f, p, t, lit_0, lit_true],
        initialization=init,
        main_edge=MainEdge(cascade=cascade),
    )


if __name__ == "__main__":
    program = build_bfs()
    sys.stdout.write(program.model_dump_json(indent=2))
