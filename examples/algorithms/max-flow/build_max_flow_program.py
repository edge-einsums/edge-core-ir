"""Construct a Max-flow EDGE program in IR form, dump JSON to stdout.

Source expression: the corrected "Full-Edge" max-flow / push-relabel cascade
in ``Push_Relabel_Max_Flow.ipynb`` (EDGE tutorial zoo). The surface forms live
next to this file in ``einsum.md`` and ``einsum.edge``; coverage metadata is in
``metadata.md``; the diff against the previous version of this artifact is in
``CHANGES.md``.

Einsum IDs E01-E24 match the tutorial and the step-through visualizer, so a
row here, a line in einsum.md and a step in the viz all refer to the same
Einsum.

Two encoding notes:

- ``delta`` is written ``Delta`` here. Tensor names in the IR must match
  ``^[A-Z][A-Za-z0-9_]*$``, so the tutorial's lowercase name cannot be carried
  over verbatim.
- The partial-update operator ``<<`` is emitted as a ``UserDefinedComputeOp``
  named ``update``. Compute builtins are limited to ``+ - * /``, and the
  operator-name pattern rejects ``<<`` outright. See CHANGES.md.

Semantic assumption the cascade relies on: a stored value equal to a tensor's
``empty`` value is treated as ABSENT. Nothing in the IR records this; an
evaluator must be told.
"""

from __future__ import annotations

import sys
from typing import Any

from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, NestedCascade, StoppingCondition
from edge_ir.ir.expr import (
    AnonymousTensor,
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
    CoordinateOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import Comparison, PropertyApp, TensorProjectionValue
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration


def rv(name: str) -> RankVariable:
    return RankVariable(name=name)


def rc(value: int | str) -> RankConstantLiteral:
    return RankConstantLiteral(value=value)


def i_plus_one() -> RankArith:
    return RankArith(op="+", lhs=rv("i"), rhs=rc(1))


def proj(tensor: str, *ranks: Any) -> TensorProjection:
    return TensorProjection(tensor=tensor, ranks=list(ranks))


def inp(tensor: str, *ranks: Any) -> InputTensor:
    return InputTensor(proj=proj(tensor, *ranks))


def unary_not(tensor: str, *ranks: Any) -> UnaryApp:
    return UnaryApp(op=BuiltinUnaryOp(symbol="not"), operand=inp(tensor, *ranks))


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


def tensor_decl(
    name: str,
    ranks: list[tuple[str, int | str | None]],
    dtype: str,
    empty_value: Any,
) -> TensorDeclaration:
    return TensorDeclaration(
        name=name,
        ranks=[
            RankDeclaration(name=rank_name, shape=shape) for rank_name, shape in ranks
        ],
        data_type=BuiltinDataType(name=dtype),  # type: ignore[arg-type]
        empty_value=empty_value,
    )


def scalar_decl(
    name: str, dtype: str, empty_value: Any, value: Any = None
) -> TensorDeclaration:
    """A zero-rank tensor standing in for a literal.

    ``value`` carries the literal itself. Without it a consumer can only see
    the declaration's ``empty_value`` and has no way to learn that ``One`` is
    1 -- which makes the artifact un-executable without out-of-band knowledge.
    See docs/design_decisions.md D21.
    """
    decl = tensor_decl(name, [], dtype, empty_value)
    return decl.model_copy(update={"value": value})


def copy_scalar_to(output: str, output_ranks: list[Any], scalar: str) -> Einsum:
    return Einsum(
        output_tensor=output,
        output_ranks=output_ranks,
        expression=inp(scalar),
        specs=[],
    )


def unary_reduce(
    output: str,
    output_ranks: list[Any],
    tensor: str,
    tensor_ranks: list[Any],
    identity_tensor: str,
    compute_name: str,
    merge_name: str,
    reduce_ranks: list[str],
) -> Einsum:
    """Encode a current-IR-compatible reduction of one tensor.

    The IR currently attaches ReduceSpec to BinaryApp labels only, so
    this helper introduces a scalar identity operand as a transparent
    second input. This is a workaround documented in CHANGES.md.

    The synthetic binary needs a Map spec too, since the evaluator requires
    one on every BinaryApp. ``take_left(take_left)`` makes it transparent. The
    merge keeps a point iff the tensor is present there, whatever the
    identity operand holds, and the compute passes the tensor's value on.
    """

    return Einsum(
        output_tensor=output,
        output_ranks=output_ranks,
        expression=BinaryApp(
            label=1,
            lhs=inp(tensor, *tensor_ranks),
            rhs=inp(identity_tensor),
        ),
        specs=[
            map_spec(1, "take_left", "take_left"),
            reduce_spec(1, compute_name, merge_name, reduce_ranks),
        ],
    )


def build_max_flow() -> Program:
    declarations = [
        tensor_decl("G", [("U", "|V|"), ("V", "|V|")], "int", 0),
        tensor_decl("C", [("U", "|V|"), ("V", "|V|")], "int", 0),
        tensor_decl("F", [("I", None), ("U", "|V|"), ("V", "|V|")], "int", 0),
        tensor_decl("R", [("I", None), ("U", "|V|"), ("V", "|V|")], "int", 0),
        tensor_decl("E", [("I", None), ("U", "|V|")], "int", 0),
        # D's empty value is +inf, NOT 0. Height 0 is a real, common height
        # (every non-source vertex starts there), so it cannot double as the
        # absent marker: the neighbour-height gather in E21 reads D through an
        # intersect, and a height-0 neighbour declared empty would be dropped
        # before reaching the min in E22, relabelling too high.
        tensor_decl("D", [("I", None), ("U", "|V|")], "int", "inf"),
        tensor_decl("Act", [("I", None), ("U", "|V|")], "bool", False),
        tensor_decl("S", [("U", "|V|")], "bool", False),
        tensor_decl("T", [("U", "|V|")], "bool", False),
        # FS: the saturated source edges, before antisymmetrization.
        tensor_decl("FS", [("I", None), ("U", "|V|"), ("V", "|V|")], "int", 0),
        tensor_decl("NST", [("U", "|V|")], "bool", False),
        tensor_decl("ActR", [("I", None), ("U", "|V|"), ("V", "|V|")], "bool", False),
        tensor_decl("Lbl", [("I", None), ("U", "|V|"), ("V", "|V|")], "bool", False),
        tensor_decl("Adm", [("I", None), ("U", "|V|"), ("V", "|V|")], "bool", False),
        # AdmPost: E17's post-push admissibility. Kept apart from Adm so
        # E09's next-round Adm_{i+1} starts empty; see E17.
        tensor_decl(
            "AdmPost", [("I", None), ("U", "|V|"), ("V", "|V|")], "bool", False
        ),
        tensor_decl(
            "PushCand",
            [("I", None), ("U", "|V|"), ("V", "|V|")],
            "bool",
            False,
        ),
        # `delta` in the tutorial; IR tensor names must start uppercase.
        tensor_decl("Delta", [("I", None), ("U", "|V|"), ("V", "|V|")], "int", 0),
        tensor_decl("InPush", [("I", None), ("U", "|V|")], "int", 0),
        tensor_decl("OutPush", [("I", None), ("U", "|V|")], "int", 0),
        tensor_decl("HasAdm", [("I", None), ("U", "|V|")], "bool", False),
        tensor_decl("Rel", [("I", None), ("U", "|V|")], "bool", False),
        # NeiLbl, MinNeiLbl and NewD all hold HEIGHTS, so they inherit D's
        # empty value of +inf for the same reason D does: height 0 is real and
        # common, and declaring it empty would drop every height-0 neighbour
        # before E22's min ever sees it. Declaring these 0 reintroduces
        # exactly the bug that fixing D was meant to remove -- one step later
        # in the cascade.
        tensor_decl("NeiLbl", [("I", None), ("U", "|V|"), ("V", "|V|")], "int", "inf"),
        tensor_decl("MinNeiLbl", [("I", None), ("U", "|V|")], "int", "inf"),
        tensor_decl("NewD", [("I", None), ("U", "|V|")], "int", "inf"),
        # Zero and One are literals in operand position (E06/E19 `> 0`, E04c,
        # the D_1 init, E08/E23 `+ 1`), so they must be PRESENT. A tensor
        # cannot hold its own empty value, so with empty == value the
        # evaluator stores nothing and every intersect against them drops
        # the point. empty_value is None, the D21 rule for literals. Their
        # other use, as the identity operand of unary_reduce, is unaffected
        # because that map's take_left merge ignores the right operand.
        scalar_decl("Zero", "int", None, value=0),
        scalar_decl("One", "int", None, value=1),
        # |V| is graph data, so it has no literal value here; a host supplies
        # it alongside G and C.
        scalar_decl("VertexCount", "int", 0),
        scalar_decl("FalseConst", "bool", False, value=False),
        # Identity for E22's min reduction. +inf, matching D's empty value.
        scalar_decl("MinIdentity", "int", "inf", value="inf"),
        # Lit_0: synthesized zero-rank tensor for the stopping-condition
        # literal `0`. The other side of the comparison is `occupancy(Act)`,
        # a PropertyApp — inherit data_type from the property's RETURN type
        # (int), NOT from Act (which is bool). For empty_value, int is
        # ambiguous (could be 0, +inf, -inf depending on context), so
        # default to None.
        TensorDeclaration(
            name="Lit_0",
            ranks=[],
            data_type=BuiltinDataType(name="int"),
            empty_value=None,
            value=0,
        ),
    ]

    initialization = Initialization(
        einsums=[
            # F_{0,u,v} = 0
            copy_scalar_to("F", [rc(0), rv("u"), rv("v")], "Zero"),
            # E_{0,u} = 0
            copy_scalar_to("E", [rc(0), rv("u")], "Zero"),
            # D_{1,u} = 0 -- generation 1, not 0: the preflow einsums E01-E04
            # write generation 1, and the iterative part reads D_{i} from
            # i = 1 onward.
            copy_scalar_to("D", [rc(1), rv("u")], "Zero"),
            # D_{1,s} = |V|, expressed through the host-provided S mask.
            Einsum(
                output_tensor="D",
                output_ranks=[rc(1), rv("u")],
                expression=BinaryApp(
                    label=1,
                    lhs=inp("S", rv("u")),
                    rhs=inp("VertexCount"),
                ),
                specs=[map_spec(1, "take_right", "intersect")],
            ),
        ]
    )

    einsums = [
        # E01: FS_{1,u,v} = S_u * C_{u,v}
        #      saturate every edge out of the source: f(s,v) = c(s,v)
        Einsum(
            output_tensor="FS",
            output_ranks=[rc(1), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("S", rv("u")),
                rhs=inp("C", rv("u"), rv("v")),
            ),
            specs=[map_spec(1, "*", "intersect")],
        ),
        # E02: F_{1,u,v} = FS_{1,u,v} - FS_{1,v,u}
        #      antisymmetrize: the reverse edge carries the negative flow, so
        #      f(v,s) = -c(s,v). This is what lets E03 be a single reduction.
        Einsum(
            output_tensor="F",
            output_ranks=[rc(1), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("FS", rc(1), rv("u"), rv("v")),
                rhs=inp("FS", rc(1), rv("v"), rv("u")),
            ),
            specs=[map_spec(1, "-", "union")],
        ),
        # E03: E_{1,v} = sum_u F_{1,u,v}
        #      excess is net inflow; outflow is already stored as negative
        #      entries, so no separate In/Out pair is needed.
        #
        #      F is read in its declared (U, V) order and the reduction runs
        #      over the SOURCE end: "sum the flow on every edge whose head is
        #      v". Writing it the other way round -- E_{1,u} = sum_v F_{1,v,u}
        #      -- computes exactly the same numbers, but reads as though F were
        #      transposed, which it is not. The cost is that the vertex is
        #      named v here and u in most other einsums; the variable name is
        #      local to the einsum and E's rank is still U either way.
        unary_reduce(
            "E",
            [rc(1), rv("v")],
            "F",
            [rc(1), rv("u"), rv("v")],
            "Zero",
            "+",
            "union",
            ["u"],
        ),
        # E04 case lowering. The written order is
        #     0        if u = s
        #     C_{v,u}  if v = s and u != s
        #     C_{u,v}  otherwise
        # and later einsums overwrite earlier ones, so the arms are emitted in
        # REVERSE priority: otherwise, then v=s, then u=s. That makes the u=s
        # arm win at the (s,s) cell, matching the written order even when the
        # graph has a source self-loop.
        # E04a: otherwise arm, R_{1,u,v} = C_{u,v}
        Einsum(
            output_tensor="R",
            output_ranks=[rc(1), rv("u"), rv("v")],
            expression=inp("C", rv("u"), rv("v")),
            specs=[],
        ),
        # E04b: source column, R_{1,u,s} = C_{s,u}
        Einsum(
            output_tensor="R",
            output_ranks=[rc(1), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("S", rv("v")),
                rhs=inp("C", rv("v"), rv("u")),
            ),
            specs=[map_spec(1, "take_right", "intersect")],
        ),
        # E04c: source row, R_{1,s,v} = 0
        Einsum(
            output_tensor="R",
            output_ranks=[rc(1), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("S", rv("u")),
                rhs=inp("Zero"),
            ),
            specs=[map_spec(1, "take_right", "intersect")],
        ),
        # E05: NST_u = not S_u AND not T_u
        Einsum(
            output_tensor="NST",
            output_ranks=[rv("u")],
            expression=BinaryApp(
                label=1,
                lhs=unary_not("S", rv("u")),
                rhs=unary_not("T", rv("u")),
            ),
            specs=[map_spec(1, "AND", "intersect")],
        ),
        # E06: Act_{i,u} = NST_u <-(int) (E_{i,u} >(int) 0)_{i,u}
        #      internal vertices with positive excess
        Einsum(
            output_tensor="Act",
            output_ranks=[rv("i"), rv("u")],
            expression=BinaryApp(
                label=1,
                lhs=inp("NST", rv("u")),
                rhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=2,
                        lhs=inp("E", rv("i"), rv("u")),
                        rhs=inp("Zero"),
                    ),
                    ranks=[rv("i"), rv("u")],
                ),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                map_spec(2, "gt", "intersect"),
            ],
        ),
        # E07: ActR_{i,u,v} = Act_{i,u} <-(int) R_{i,u,v}
        #      residual edges leaving an active vertex; a saturated edge has
        #      R = 0 = empty, so the intersect drops it
        Einsum(
            output_tensor="ActR",
            output_ranks=[rv("i"), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("Act", rv("i"), rv("u")),
                rhs=inp("R", rv("i"), rv("u"), rv("v")),
            ),
            specs=[map_spec(1, "take_left", "intersect")],
        ),
        # E08: Lbl_{i,u,v} = D_{i,u} ==(int) (D_{i,v} + 1)_{i,v}
        #      height rule D(u) == D(v)+1. The merge is INTERSECT: with a
        #      union merge an absent height would be compared as though it
        #      were present.
        Einsum(
            output_tensor="Lbl",
            output_ranks=[rv("i"), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("D", rv("i"), rv("u")),
                rhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=2,
                        lhs=inp("D", rv("i"), rv("v")),
                        rhs=inp("One"),
                    ),
                    ranks=[rv("i"), rv("v")],
                ),
            ),
            specs=[
                map_spec(1, "eq", "intersect"),
                map_spec(2, "+", "intersect"),
            ],
        ),
        # E09: Adm_{i,u,v} = ActR_{i,u,v} AND Lbl_{i,u,v}
        Einsum(
            output_tensor="Adm",
            output_ranks=[rv("i"), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("ActR", rv("i"), rv("u"), rv("v")),
                rhs=inp("Lbl", rv("i"), rv("u"), rv("v")),
            ),
            specs=[map_spec(1, "AND", "intersect")],
        ),
        # E10: choose one admissible edge per pushing vertex. This selector is
        #      a resource constraint, not a serialization point: a vertex has
        #      one pool of excess and cannot spend it on several edges at once.
        #      The Map spec makes the synthetic binary transparent (see
        #      unary_reduce), so Adm's True reaches Populate unchanged.
        Einsum(
            output_tensor="PushCand",
            output_ranks=[rv("i"), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("Adm", rv("i"), rv("u"), rv("v")),
                rhs=inp("One"),
            ),
            specs=[
                map_spec(1, "take_left", "take_left"),
                PopulateSpec(
                    label=1,
                    rank_list=["v"],
                    compute_op=compute("pick-admissible-edge"),
                    coord_op=CoordinateOp(name="select-one-admissible-v"),
                ),
            ],
        ),
        # E11: Delta = min(excess, residual) on the selected edges
        Einsum(
            output_tensor="Delta",
            output_ranks=[rv("i"), rv("u"), rv("v")],
            expression=BinaryApp(
                label=2,
                lhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=1,
                        lhs=inp("E", rv("i"), rv("u")),
                        rhs=inp("R", rv("i"), rv("u"), rv("v")),
                    ),
                    ranks=[rv("i"), rv("u"), rv("v")],
                ),
                rhs=inp("PushCand", rv("i"), rv("u"), rv("v")),
            ),
            specs=[
                map_spec(1, "min", "intersect"),
                map_spec(2, "take_left", "intersect"),
            ],
        ),
        # E12: F_{i+1} adds Delta forward and subtracts it on the reverse
        #      edge, preserving antisymmetry
        Einsum(
            output_tensor="F",
            output_ranks=[i_plus_one(), rv("u"), rv("v")],
            expression=BinaryApp(
                label=2,
                lhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=1,
                        lhs=inp("F", rv("i"), rv("u"), rv("v")),
                        rhs=inp("Delta", rv("i"), rv("u"), rv("v")),
                    ),
                    ranks=[rv("i"), rv("u"), rv("v")],
                ),
                rhs=inp("Delta", rv("i"), rv("v"), rv("u")),
            ),
            specs=[
                map_spec(1, "+", "union"),
                map_spec(2, "-", "union"),
            ],
        ),
        # E13: InPush_{i,v} = sum_u Delta_{i,u,v}
        #      Same convention as E03: inflow reduces over the source end, so
        #      Delta is read in its declared (U, V) order. E14 below is outflow
        #      and already reads Delta_{i,u,v}, reducing the head end.
        unary_reduce(
            "InPush",
            [rv("i"), rv("v")],
            "Delta",
            [rv("i"), rv("u"), rv("v")],
            "Zero",
            "+",
            "union",
            ["u"],
        ),
        # E14: OutPush_{i,u} = sum_v Delta_{i,u,v}
        unary_reduce(
            "OutPush",
            [rv("i"), rv("u")],
            "Delta",
            [rv("i"), rv("u"), rv("v")],
            "Zero",
            "+",
            "union",
            ["v"],
        ),
        # E15: E_{i+1} = E_i + InPush - OutPush
        Einsum(
            output_tensor="E",
            output_ranks=[i_plus_one(), rv("u")],
            expression=BinaryApp(
                label=2,
                lhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=1,
                        lhs=inp("E", rv("i"), rv("u")),
                        rhs=inp("InPush", rv("i"), rv("u")),
                    ),
                    ranks=[rv("i"), rv("u")],
                ),
                rhs=inp("OutPush", rv("i"), rv("u")),
            ),
            specs=[
                map_spec(1, "+", "union"),
                map_spec(2, "-", "union"),
            ],
        ),
        # E16: R_{i+1} loses the pushed capacity forward and gains it back
        Einsum(
            output_tensor="R",
            output_ranks=[i_plus_one(), rv("u"), rv("v")],
            expression=BinaryApp(
                label=2,
                lhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=1,
                        lhs=inp("R", rv("i"), rv("u"), rv("v")),
                        rhs=inp("Delta", rv("i"), rv("u"), rv("v")),
                    ),
                    ranks=[rv("i"), rv("u"), rv("v")],
                ),
                rhs=inp("Delta", rv("i"), rv("v"), rv("u")),
            ),
            specs=[
                map_spec(1, "-", "union"),
                map_spec(2, "+", "union"),
            ],
        ),
        # E17: admissibility recomputed from the post-push residual. Lbl is
        #      still generation i because no height has changed yet this round.
        #      einsum.md writes this as Adm_{i+1}. It is written to AdmPost
        #      instead. E09 writes Adm_{i+1} again in the next round, and an
        #      assignment only adds values, so E17's entries would survive
        #      into that round's push. They include inactive vertices, the
        #      source among them, and edges to vertices that have just
        #      relabelled. AdmPost is read only by E18.
        Einsum(
            output_tensor="AdmPost",
            output_ranks=[i_plus_one(), rv("u"), rv("v")],
            expression=BinaryApp(
                label=1,
                lhs=inp("R", i_plus_one(), rv("u"), rv("v")),
                rhs=inp("Lbl", rv("i"), rv("u"), rv("v")),
            ),
            specs=[map_spec(1, "take_right", "intersect")],
        ),
        # E18: HasAdm_{i+1,u} = OR_v AdmPost_{i+1,u,v}
        unary_reduce(
            "HasAdm",
            [i_plus_one(), rv("u")],
            "AdmPost",
            [i_plus_one(), rv("u"), rv("v")],
            "FalseConst",
            "OR",
            "union",
            ["v"],
        ),
        # E19: Act_{i+1}. Recomputed because the push changed E; the anonymous
        #      operand is bound at generation i+1, matching the E it reads.
        Einsum(
            output_tensor="Act",
            output_ranks=[i_plus_one(), rv("u")],
            expression=BinaryApp(
                label=1,
                lhs=inp("NST", rv("u")),
                rhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=2,
                        lhs=inp("E", i_plus_one(), rv("u")),
                        rhs=inp("Zero"),
                    ),
                    ranks=[i_plus_one(), rv("u")],
                ),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                map_spec(2, "gt", "intersect"),
            ],
        ),
        # E20: Rel_{i+1,u} = still active AND nothing admissible left
        Einsum(
            output_tensor="Rel",
            output_ranks=[i_plus_one(), rv("u")],
            expression=BinaryApp(
                label=1,
                lhs=inp("Act", i_plus_one(), rv("u")),
                rhs=unary_not("HasAdm", i_plus_one(), rv("u")),
            ),
            specs=[map_spec(1, "AND", "intersect")],
        ),
        # E21: heights of the residual neighbours of relabelling vertices
        Einsum(
            output_tensor="NeiLbl",
            output_ranks=[rv("i"), rv("u"), rv("v")],
            expression=BinaryApp(
                label=2,
                lhs=AnonymousTensor(
                    expression=BinaryApp(
                        label=1,
                        lhs=inp("R", i_plus_one(), rv("u"), rv("v")),
                        rhs=inp("Rel", i_plus_one(), rv("u")),
                    ),
                    ranks=[rv("i"), rv("u"), rv("v")],
                ),
                rhs=inp("D", rv("i"), rv("v")),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                map_spec(2, "take_right", "intersect"),
            ],
        ),
        # E22: MinNeiLbl_{i,u} = min_v NeiLbl_{i,u,v}
        unary_reduce(
            "MinNeiLbl",
            [rv("i"), rv("u")],
            "NeiLbl",
            [rv("i"), rv("u"), rv("v")],
            "MinIdentity",
            "min",
            "union",
            ["v"],
        ),
        # E23: NewD_{i,u} = (MinNeiLbl_{i,u} + 1)_{i,u}
        Einsum(
            output_tensor="NewD",
            output_ranks=[rv("i"), rv("u")],
            expression=BinaryApp(
                label=1,
                lhs=inp("MinNeiLbl", rv("i"), rv("u")),
                rhs=inp("One"),
            ),
            specs=[map_spec(1, "+", "intersect")],
        ),
        # E24: D_{i+1,u} = D_{i,u} <<(union) NewD_{i,u}
        #      PARTIAL UPDATE. The union merge keeps every vertex present, and
        #      `update` returns the right operand where it exists and the left
        #      one otherwise -- so vertices that did not relabel keep their
        #      height. A take_left/intersect encoding would retain only the
        #      relabelled vertices and silently drop every other height.
        Einsum(
            output_tensor="D",
            output_ranks=[i_plus_one(), rv("u")],
            expression=BinaryApp(
                label=1,
                lhs=inp("D", rv("i"), rv("u")),
                rhs=inp("NewD", rv("i"), rv("u")),
            ),
            specs=[map_spec(1, "update", "union")],
        ),
    ]

    stopping = StoppingCondition(
        rank_variable="i",
        predicate=Comparison(
            lhs=PropertyApp(
                name="occupancy",
                pinned_rank_variables=["i"],
                operands=[proj("Act", i_plus_one())],
            ),
            op="==",
            rhs=TensorProjectionValue(proj=proj("Lit_0")),
        ),
    )

    return Program(
        declarations=declarations,
        initialization=initialization,
        main_edge=MainEdge(
            cascade=NestedCascade(einsums=einsums, stopping_conditions=[stopping])
        ),
    )


if __name__ == "__main__":
    program = build_max_flow()
    sys.stdout.write(program.model_dump_json(indent=2))
