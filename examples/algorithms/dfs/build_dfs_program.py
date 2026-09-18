"""Construct the DFS program in IR form, dump indented JSON to stdout.

Reference: the order-stamped-value DFS.

The container `S` is a stack whose payload is the stamp
``sigma(i, v) = i * |V| + v`` (stateless, lexicographic). PEEK is a
populate argmax over the stamp; pop is masked removal; push is an
update merge.

Save the output to examples/algorithms/dfs/dfs_program.json.

KNOWN IR GAP (see metadata.md and the report): the stamp operand
``sigma(i+1, v) = (i+1) * |V| + v`` is a per-iteration-point
rank-variable-as-value. The current IR `Expression` union has no
rank-as-value leaf (D21 keeps scalars as zero-rank broadcast tensors,
which are constants and cannot depend on (i, v)). Steps (4b) STAMP and
the initialization of `S` therefore use a zero-rank placeholder tensor
`Stamp` in the operand slot to keep the cascade structurally complete;
the placeholder is NOT the real semantics and is flagged for the
edge-expert.
"""

from __future__ import annotations

import sys
from typing import Any

from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade, StoppingCondition
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankValue,
    RankVariable,
    TensorProjection,
    UnaryApp,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    BuiltinUnaryOp,
    CoordinateOp,
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


def rv(name: str) -> RankVariable:
    return RankVariable(name=name)


def rc(value: int | str) -> RankConstantLiteral:
    return RankConstantLiteral(value=value)


def proj(tensor: str, *ranks: Any) -> TensorProjection:
    return TensorProjection(tensor=tensor, ranks=list(ranks))


def inp(tensor: str, *ranks: Any) -> InputTensor:
    return InputTensor(proj=proj(tensor, *ranks))


def sigma(generation: Any) -> RankValue:
    """The order stamp as a rank-as-value operand: (generation * |V|) + v.

    einsum.md: `sigma(i, v) = i * |V| + v  (a function of the iteration
    point)`. This is the paper's "Rank Variables as Tensors":
    at each point the coordinate is cast
    into the data space. It replaces the zero-rank `Stamp` placeholder the
    artifact used to carry, which was a CONSTANT and so could not vary with
    (i, v) at all.
    """
    return RankValue(
        expr=RankArith(
            op="+",
            lhs=RankArith(
                op="*",
                lhs=generation,
                rhs=RankConstantShapeSym(name="|V|"),
            ),
            rhs=rv("v"),
        )
    )


def unary_not(tensor: str, *ranks: Any) -> UnaryApp:
    """`not T`, with the merge that makes complement-masking correct.

    A unary is a degenerate Map against the all-ones tensor, and its MERGE
    decides where the result exists (EDGE's decomposition of unary functions).
    Logical complement maps empty to NON-empty, so it must
    visit the points where the operand is empty -- the not_left merge. Under
    take_left, `S . not F :: map take_left(intersect)` would keep exactly the
    popped vertex instead of everything but it.
    """
    return UnaryApp(
        op=BuiltinUnaryOp(symbol="not"),
        merge_op=BuiltinMergeOp(symbol="not_left"),
        operand=inp(tensor, *ranks),
    )


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


def build_dfs() -> Program:
    # ----- Tensor declarations -----
    # G: adjacency, Boolean, empty False. Static (no I rank).
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    # S: stack; payload = stamp sigma. Empty value -1 (outside stamp range
    # since stamps start at 0).
    s = TensorDeclaration(
        name="S",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="V", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=-1,
    )
    # P: visited/discovered set.
    p = TensorDeclaration(
        name="P",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="V", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    # F: one-hot top-of-stack (the peeked vertex), payload = its stamp.
    f = TensorDeclaration(
        name="F",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="V", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=-1,
    )
    # N: neighbors of the popped vertex.
    n = TensorDeclaration(
        name="N",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    # U: undiscovered neighbors (masked by visited).
    u = TensorDeclaration(
        name="U",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    # T: stack after the top is masked out (the pop result).
    t = TensorDeclaration(
        name="T",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="V", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=-1,
    )
    # Sprime (S'): newly discovered vertices carrying their fresh stamp.
    sprime = TensorDeclaration(
        name="Sprime",
        ranks=[
            RankDeclaration(name="I"),
            RankDeclaration(name="V", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=-1,
    )

    # One: zero-rank int literal 1, the right operand of the PEEK populate
    # (mirrors max-flow's PushCand = Adm . One :: populate). The populate's
    # value is produced by the compute op; `One` just supplies a binary so a
    # PopulateSpec (which attaches to a BinaryApp label) has a label to bind.
    one = TensorDeclaration(
        name="One",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=None,
        value=1,
    )

    # Stamp: the order stamp sigma(i, v) = i * |V| + v is now written
    # DIRECTLY as a rank-as-value operand -- see `sigma()` below -- so it no
    # longer needs a placeholder tensor. This zero-rank declaration is kept
    # only so the artifact's declaration list is unchanged for readers
    # comparing against earlier versions; nothing projects it.
    stamp = TensorDeclaration(
        name="Stamp",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=None,
        value=None,
    )

    # Lit_0: synthesized zero-rank int for the stopping condition literal `0`.
    # The other side is occupancy(S), a PropertyApp whose return type is int;
    # empty_value None per the synthesized-literal rule.
    lit_0 = TensorDeclaration(
        name="Lit_0",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=None,
        value=0,
    )

    # Lit_True: zero-rank bool literal True, the RHS of the P_0 init (mirrors
    # BFS's Lit_True). empty_value None per the synthesized-literal rule.
    lit_true = TensorDeclaration(
        name="Lit_True",
        ranks=[],
        data_type=BuiltinDataType(name="bool"),
        empty_value=None,
        value=True,
    )

    # ----- Initialization (root pushed via host coord set root_id) -----
    # S_{0, v : v in root_id} = sigma(0, v) = (0 * |V|) + v
    s_init = Einsum(
        output_tensor="S",
        output_ranks=[rc(0), rv("v")],
        expression=sigma(rc(0)),
        specs=[],
        predicates=[
            SetMembership(
                member=rv("v"),
                coord_set=CoordSetName(name="root_id"),
                negated=False,
            )
        ],
    )
    # P_{0, v : v in root_id} = True
    p_init = Einsum(
        output_tensor="P",
        output_ranks=[rc(0), rv("v")],
        expression=InputTensor(proj=proj("Lit_True")),
        specs=[],
        predicates=[
            SetMembership(
                member=rv("v"),
                coord_set=CoordSetName(name="root_id"),
                negated=False,
            )
        ],
    )
    init = Initialization(einsums=[s_init, p_init])

    # ----- Cascade -----

    # (1) PEEK: F_{i, v*} = S_{i, v} :: populate _v 1(select-max-val)
    # A SINGLE-OPERAND einsum, written as one. Populate's own operands are
    # the reduction temporary and the current output tensor (Populate
    # semantics, "Inputs and Outputs") -- it never needs a binary -- so
    # `PopulateSpec.label` is None and there is no synthetic `S . One`
    # operand inventing a Map action the einsum does not have.
    # The populate compute op is the paper's `1` glyph, whose ASCII name is
    # `identity` (pass the redtmp value through); the coordinate op
    # `select-max-val` keeps the single argmax-stamp coordinate (one-hot top).
    peek = Einsum(
        output_tensor="F",
        output_ranks=[rv("i"), rv("v")],
        expression=inp("S", rv("i"), rv("v")),
        specs=[
            PopulateSpec(
                rank_list=["v"],
                compute_op=compute("identity"),
                coord_op=CoordinateOp(name="select-max-val"),
            )
        ],
    )

    # (2) ADVANCE: N_{i, d} = G_{s, d} . F_{i, s}
    #     :: Map _s take_left(intersect) ; Reduce _s ANY(union)
    advance = Einsum(
        output_tensor="N",
        output_ranks=[rv("i"), rv("d")],
        expression=BinaryApp(
            label=1,
            lhs=inp("G", rv("s"), rv("d")),
            rhs=inp("F", rv("i"), rv("s")),
        ),
        specs=[
            map_spec(1, "take_left", "intersect", rank_list=["s"]),
            reduce_spec(1, "ANY", "union", rank_list=["s"]),
        ],
    )

    # (3) MASK-VISITED: U_{i, d} = N_{i, d} . not P_{i, d}
    #     :: Map _d take_left(intersect)
    mask = Einsum(
        output_tensor="U",
        output_ranks=[rv("i"), rv("d")],
        expression=BinaryApp(
            label=1,
            lhs=inp("N", rv("i"), rv("d")),
            rhs=unary_not("P", rv("i"), rv("d")),
        ),
        specs=[map_spec(1, "take_left", "intersect", rank_list=["d"])],
    )

    # (4a) POP: T_{i, v} = S_{i, v} . not F_{i, v} :: Map _v take_left(intersect)
    pop = Einsum(
        output_tensor="T",
        output_ranks=[rv("i"), rv("v")],
        expression=BinaryApp(
            label=1,
            lhs=inp("S", rv("i"), rv("v")),
            rhs=unary_not("F", rv("i"), rv("v")),
        ),
        specs=[map_spec(1, "take_left", "intersect", rank_list=["v"])],
    )

    # (4b) STAMP: Sprime_{i, v} = U_{i, v} . sigma(i+1, v)
    #     :: Map _v take_right(intersect)
    #   The right operand is the per-point stamp sigma(i+1, v) = (i+1)*|V|+v,
    #   written as a rank-as-value operand (see `sigma()`).
    #   take_right is the EDGE `->(intersect)` compute op: keep the right
    #   (the fresh stamp) where both U and the stamp are present.
    stamp_step = Einsum(
        output_tensor="Sprime",
        output_ranks=[rv("i"), rv("v")],
        expression=BinaryApp(
            label=1,
            lhs=inp("U", rv("i"), rv("v")),
            rhs=sigma(RankArith(op="+", lhs=rv("i"), rhs=rc(1))),
        ),
        specs=[map_spec(1, "take_right", "intersect", rank_list=["v"])],
    )

    # (4c) PUSH: S_{i+1, v} = T_{i, v} . Sprime_{i, v} :: Map _v <<(union)
    #   `<<` is a COMPUTE operator paired with a union merge: right operand
    #   where present, else carry the left forward
    #   (EDGE paper appendix). Its ASCII name
    #   is `update`, shipped as a default compute op in the UDF registry.
    #   This previously read take_right(union), which returns the RIGHT
    #   operand's empty value where the right is absent and so drops the
    #   carried-forward stack entry. The supports of T and Sprime happen to
    #   be disjoint here so the two coincided for this algorithm, but the
    #   encoding said the wrong thing. See TODO.md Bugs.
    push = Einsum(
        output_tensor="S",
        output_ranks=[i_plus_one(), rv("v")],
        expression=BinaryApp(
            label=1,
            lhs=inp("T", rv("i"), rv("v")),
            rhs=inp("Sprime", rv("i"), rv("v")),
        ),
        specs=[map_spec(1, "update", "union", rank_list=["v"])],
    )

    # (5) UPDATE-VISITED: P_{i+1, v} = P_{i, v} . Sprime_{i, v} :: Map _v OR(union)
    visit = Einsum(
        output_tensor="P",
        output_ranks=[i_plus_one(), rv("v")],
        expression=BinaryApp(
            label=1,
            lhs=inp("P", rv("i"), rv("v")),
            rhs=inp("Sprime", rv("i"), rv("v")),
        ),
        specs=[map_spec(1, "OR", "union", rank_list=["v"])],
    )

    # (6) STOP: <> : ||S_{i+1}|| == 0
    cascade = NestedCascade(
        einsums=[peek, advance, mask, pop, stamp_step, push, visit],
        stopping_conditions=[
            StoppingCondition(
                rank_variable="i",
                predicate=Comparison(
                    lhs=PropertyApp(
                        name="occupancy",
                        pinned_rank_variables=["i"],
                        operands=[proj("S", i_plus_one())],
                    ),
                    op="==",
                    rhs=TensorProjectionValue(proj=proj("Lit_0")),
                ),
            ),
        ],
    )

    return Program(
        declarations=[
            g,
            s,
            p,
            f,
            n,
            u,
            t,
            sprime,
            one,
            stamp,
            lit_0,
            lit_true,
        ],
        initialization=init,
        main_edge=MainEdge(cascade=cascade),
    )


if __name__ == "__main__":
    program = build_dfs()
    sys.stdout.write(program.model_dump_json(indent=2))
