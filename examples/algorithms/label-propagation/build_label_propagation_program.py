"""Construct the label-propagation program in IR form, dump JSON to stdout.

Algorithm: connected components by max-label propagation (single-Einsum,
generational rank). Each vertex starts with a unique label (id + 1). Every
generation a vertex adopts the maximum label among its neighbours and itself
(``G`` is reflexive):

    L_{i+1, d} = G_{s, d} . L_{i, s} :: /\\_s *(intersect) \\/_s max(union)
    <> : L_{i+1} == L_i

``map(*, intersect)`` passes a neighbour's label through the 1.0 edge weight;
``reduce(max, union)`` keeps the largest. Labels only ever increase and are
bounded, so ``L`` converges: every vertex ends holding the maximum label in
its connected component.

Structurally this is the BFS advance einsum
(``T_{i,d} = G_{s,d}·F_{i,s} :: ⋀_s +(∩) ⋁_s ANY(∪)``) with ``+`` → ``*`` and
``ANY`` → ``max``. Verified admissible by the edge-expert (see metadata.md).

NO INIT EINSUM (and no synthesized literal tensors): both ``G`` and the
generation-0 labels ``L_0`` are host-provided input data, exactly like ``G``
in the other artifacts. ``L_0 = [1, 2, 3, ...]`` (label(v) = v + 1) is a
fully host-specified vector with no source restriction, so there is nothing
to compute in the init block -- this deliberately avoids the D21
rank-as-value gap that the DFS ``sigma`` init hit. See metadata.md.

Save the output to
examples/algorithms/label-propagation/label_propagation_program.json.
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
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import Comparison, TensorProjectionValue
from edge_ir.ir.tensor import (
    BuiltinDataType,
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


def i_plus_one() -> RankArith:
    return RankArith(op="+", lhs=rv("i"), rhs=rc(1))


def build_label_propagation() -> Program:
    # ----- Tensor declarations -----
    # G: symmetric, reflexive adjacency matrix. Float, empty 0.0 (the source
    # writes `float`; the labels happen to be whole numbers). Static (no I).
    # Reflexivity (diagonal = 1) is a PRECONDITION on the host-supplied G, not
    # something the einsum enforces -- see metadata.md.
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="R", shape="N"),
            RankDeclaration(name="C", shape="N"),
        ],
        data_type=BuiltinDataType(name="float"),
        empty_value=0.0,
    )
    # L: per-vertex label across generations. L_0 holds the unique starting
    # labels; each later generation holds the running max label per vertex.
    lbl = TensorDeclaration(
        name="L",
        ranks=[
            RankDeclaration(name="I"),  # generational, no shape
            RankDeclaration(name="R", shape="N"),
        ],
        data_type=BuiltinDataType(name="float"),
        empty_value=0.0,
    )

    # ----- Initialization -----
    # EMPTY. Both G and L_0 are host-provided data. L_0 = [1, 2, 3, ...]
    # (label(v) = v + 1) is fully host-specified with no source restriction,
    # so there is no init computation to encode (and no SetMembership). The
    # concrete L_0 vector lives in the host / test fixture, like the G matrix.
    init = Initialization(einsums=[])

    # ----- Main cascade -----
    # L_{i+1, d} = G_{s, d} . L_{i, s}
    #   :: Map ^1 _s * (intersect)   -- pass neighbour's label through weight 1.0
    #   :: Reduce ^1 _s max (union)  -- keep the largest incoming label
    # `s` is the contraction rank (mapped + reduced, absent from the output);
    # `d` is preserved; `i` is the generational rank. `*` is a builtin compute
    # op; `max` is a user-defined compute op (the dual of Bellman-Ford's UDF
    # `min`), since the builtin compute set is only {+, -, *, /}.
    propagate = Einsum(
        output_tensor="L",
        output_ranks=[i_plus_one(), rv("d")],
        expression=BinaryApp(
            label=1,
            lhs=inp("G", rv("s"), rv("d")),
            rhs=inp("L", rv("i"), rv("s")),
        ),
        specs=[
            MapSpec(
                label=1,
                rank_list=["s"],
                compute_op=BuiltinComputeOp(symbol="*"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            ),
            ReduceSpec(
                label=1,
                rank_list=["s"],
                compute_op=UserDefinedComputeOp(name="max"),
                merge_op=BuiltinMergeOp(symbol="union"),
            ),
        ],
    )

    # STOP: <> : L_{i+1} == L_i  (value convergence, NOT occupancy). Two
    # TensorProjectionValue of L at the i+1 and i generations; the "all points
    # satisfy" lifting of Comparison is the paper's L_{i+1} ≡ L_i. Same form as
    # Bellman-Ford's `D_{i+1} == D_i`.
    cascade = NestedCascade(
        einsums=[propagate],
        stopping_conditions=[
            StoppingCondition(
                rank_variable="i",
                predicate=Comparison(
                    lhs=TensorProjectionValue(proj=proj("L", i_plus_one())),
                    op="==",
                    rhs=TensorProjectionValue(proj=proj("L", rv("i"))),
                ),
            ),
        ],
    )

    return Program(
        declarations=[g, lbl],
        initialization=init,
        main_edge=MainEdge(cascade=cascade),
    )


if __name__ == "__main__":
    program = build_label_propagation()
    sys.stdout.write(program.model_dump_json(indent=2))
