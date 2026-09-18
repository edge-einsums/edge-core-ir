"""Construct the canonical Bellman-Ford SSSP program in IR form, dump
indented JSON to stdout.

Reference: EDGE paper Cascade `cascade:bf`
(the `<<` merge
semantics are spelled out in the paper's Bellman-Ford appendix).

This is the CORRECT, canonical Bellman-Ford: it relaxes ALL edges every
iteration and re-relaxes a vertex whenever a shorter path is found, gated
only by the strict-improvement check ``N < D``. There is exactly ONE
distance tensor ``D`` (integer, empty = +inf). It is NOT the user-supplied
cascade that was reviewed and rejected: that version gated the frontier
with ``not-visited`` (``\neg V``), which blocks re-relaxation and degrades
to a single-visit weighted BFS (see metadata.md).

Save the output to examples/algorithms/bellman-ford/bellman_ford_program.json.

KNOWN IR GAPS (see metadata.md and the report):
  G1 -- the update merge `<<` ("take right where right exists, else take
        left") is not one of the 16 builtin merge symbols. Encoded as
        `take_right(union)`, which coincides with `<<` here because
        NewlyRelaxed is non-empty only where it improves D.
  G2 -- the improvement comparison `<` is not a builtin compute operator
        (only + - * / are). Encoded as a user-defined compute op
        `less_than`, the comparison-producing function the paper's
        `<(\\cup)` denotes.
"""

from __future__ import annotations

import sys
from typing import Any

from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade, StoppingCondition
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.predicate import SetMembership
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import Comparison, TensorProjectionValue
from edge_ir.ir.tensor import (
    BuiltinDataType,
    CoordSetName,
    RankDeclaration,
    TensorDeclaration,
)


def rv(name: str) -> RankVariable:
    return RankVariable(name=name)


def rc(value: int | str) -> RankConstantLiteral:
    return RankConstantLiteral(value=value)


def proj(tensor: str, *ranks: Any) -> TensorProjection:
    return TensorProjection(tensor=tensor, ranks=list(ranks))


def inp(tensor: str, *ranks: Any) -> InputTensor:
    return InputTensor(proj=proj(tensor, *ranks))


def compute(name: str) -> BuiltinComputeOp | UserDefinedComputeOp:
    if name in {"+", "-", "*", "/"}:
        return BuiltinComputeOp(symbol=name)  # type: ignore[arg-type]
    return UserDefinedComputeOp(name=name)


def merge(name: str) -> BuiltinMergeOp:
    return BuiltinMergeOp(symbol=name)  # type: ignore[arg-type]


def map_spec(
    label: int,
    compute_name: str,
    merge_name: str,
    rank_list: list[str] | None = None,
) -> MapSpec:
    return MapSpec(
        label=label,
        rank_list=rank_list,
        compute_op=compute(compute_name),
        merge_op=merge(merge_name),
    )


def reduce_spec(
    label: int,
    compute_name: str,
    merge_name: str,
    rank_list: list[str] | None = None,
) -> ReduceSpec:
    return ReduceSpec(
        label=label,
        rank_list=rank_list,
        compute_op=compute(compute_name),
        merge_op=merge(merge_name),
    )


def i_plus_one() -> RankArith:
    return RankArith(op="+", lhs=rv("i"), rhs=rc(1))


def build_bellman_ford() -> Program:
    # ----- Tensor declarations (paper cascade:bf "Tensors" block) -----
    # G: weighted adjacency, integer, empty 0. Static (no I rank).
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    # C: improvement mask (N < D), Boolean, empty False.
    c = TensorDeclaration(
        name="C",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="S", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    # N: relaxed candidate distances, integer, empty +inf.
    n = TensorDeclaration(
        name="N",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=float("inf"),
    )
    # NewlyRelaxed: candidate distances that strictly improve D, integer,
    # empty +inf.
    newly_relaxed = TensorDeclaration(
        name="NewlyRelaxed",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=float("inf"),
    )
    # D: THE single distance tensor, integer, empty +inf (= unreached).
    d = TensorDeclaration(
        name="D",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="S", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=float("inf"),
    )

    # Lit_0: synthesized zero-rank int for the source-distance literal `0`
    # in the D_0 initialization. empty_value None per the synthesized-literal
    # rule (the contextual source is the int tensor D).
    lit_0 = TensorDeclaration(
        name="Lit_0",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=None,
        value=0,
    )

    # ----- Initialization (paper: G user-specified; D_{0, s in root_id} = 0)
    # D_{0, s : s in root_id} = 0
    d_init = Einsum(
        output_tensor="D",
        output_ranks=[rc(0), rv("s")],
        expression=InputTensor(proj=proj("Lit_0")),
        specs=[],
        predicates=[
            SetMembership(
                member=rv("s"),
                coord_set=CoordSetName(name="root_id"),
                negated=False,
            )
        ],
    )
    init = Initialization(einsums=[d_init])

    # ----- Cascade (paper cascade:bf "Extended Einsum") -----

    # (1) RELAX: N_{i, d} = G_{s, d} . D_{i, s}
    #     :: Map _s +(intersect) ; Reduce _s min(union)
    #   Add the edge weight to the source's distance (intersect: only where
    #   both an edge and a finite source distance exist), then keep the
    #   minimum over all sources reaching d.
    relax = Einsum(
        output_tensor="N",
        output_ranks=[rv("i"), rv("d")],
        expression=BinaryApp(
            label=1,
            lhs=inp("G", rv("s"), rv("d")),
            rhs=inp("D", rv("i"), rv("s")),
        ),
        specs=[
            map_spec(1, "+", "intersect", rank_list=["s"]),
            reduce_spec(1, "min", "union", rank_list=["s"]),
        ],
    )

    # (2) IMPROVE: C_{i, d} = N_{i, d} . D_{i, d} :: Map _d less_than(union)
    #   Boolean: True where the new candidate is strictly less than the
    #   current distance. `<` is not a builtin compute op (only + - * /), so
    #   the comparison is the UDF compute op `less_than` (GAP G2).
    improve = Einsum(
        output_tensor="C",
        output_ranks=[rv("i"), rv("d")],
        expression=BinaryApp(
            label=1,
            lhs=inp("N", rv("i"), rv("d")),
            rhs=inp("D", rv("i"), rv("d")),
        ),
        specs=[map_spec(1, "less_than", "union", rank_list=["d"])],
    )

    # (3) RECORD: NewlyRelaxed_{i, d} = C_{i, d} . N_{i, d}
    #     :: Map _d take_right(intersect)
    #   Keep N's value (take_right) where both the improvement mask C and the
    #   candidate N are present -- i.e. the improved distances only.
    record = Einsum(
        output_tensor="NewlyRelaxed",
        output_ranks=[rv("i"), rv("d")],
        expression=BinaryApp(
            label=1,
            lhs=inp("C", rv("i"), rv("d")),
            rhs=inp("N", rv("i"), rv("d")),
        ),
        specs=[map_spec(1, "take_right", "intersect", rank_list=["d"])],
    )

    # (4) UPDATE: D_{i+1, d} = D_{i, d} . NewlyRelaxed_{i, d} :: Map _d <<(union)
    #   `<<` is a COMPUTE operator, not a merge: right operand where present,
    #   else carry the left forward (EDGE paper
    #   appendix). The IR cannot spell the glyph
    #   (REGEX_USER_DEFINED_NAME admits no `<`), so its ASCII name is
    #   `update`, shipped as a default compute op in the UDF registry.
    #   This previously read take_right(union), which returns the RIGHT
    #   operand's empty value where the right is absent and so DROPS the
    #   carried-forward distance -- the two diverge on (left present, right
    #   absent), i.e. "distance unchanged this round". See TODO.md Bugs.
    update = Einsum(
        output_tensor="D",
        output_ranks=[i_plus_one(), rv("d")],
        expression=BinaryApp(
            label=1,
            lhs=inp("D", rv("i"), rv("d")),
            rhs=inp("NewlyRelaxed", rv("i"), rv("d")),
        ),
        specs=[map_spec(1, "update", "union", rank_list=["d"])],
    )

    # (5) STOP: <> : D_{i+1} == D_i  (convergence; the "all points satisfy"
    #   lifting of Comparison is the paper's D_{i+1} == D_i, per stopping.py).
    cascade = NestedCascade(
        einsums=[relax, improve, record, update],
        stopping_conditions=[
            StoppingCondition(
                rank_variable="i",
                predicate=Comparison(
                    lhs=TensorProjectionValue(proj=proj("D", i_plus_one())),
                    op="==",
                    rhs=TensorProjectionValue(proj=proj("D", rv("i"))),
                ),
            ),
        ],
    )

    return Program(
        declarations=[g, c, n, newly_relaxed, d, lit_0],
        initialization=init,
        main_edge=MainEdge(cascade=cascade),
    )


if __name__ == "__main__":
    program = build_bellman_ford()
    sys.stdout.write(program.model_dump_json(indent=2))
