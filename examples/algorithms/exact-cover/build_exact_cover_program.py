"""Construct Knuth's Algorithm X as an EDGE program in IR form; dump JSON.

SOURCE OF TRUTH
---------------
``_algorithms/set_cover.md`` in the edge-algorithms repository -- the
``edge_expression`` block, which is the SEQUENTIAL cascade. That file also
carries a parallel variant with an extra worker rank ``W``; this builder
encodes the sequential one, because that is the cascade the worked examples
trace and the only one an answer can be checked against.

Einsum IDs E01-E19 match the tutorial's cascade line for line. Where one
written line needs more than one IR Einsum, the parts carry letters (E02a,
E02b) exactly as max-flow's E04a/b/c do, so a line in the tutorial, a row
here and a step in the visualiser still refer to the same thing.

WHAT THE ALGORITHM DOES, IN ONE PARAGRAPH
-----------------------------------------
Backtracking search without a mutable matrix. A stack ``S`` holds candidate
rows tagged with DEPTH STAMPS, sigma(d, r) = d*|R| + r, which pack the search
depth and the row into one integer. Each iteration pops the largest stamp,
recovers its depth by integer division, truncates the path to that depth, and
appends the popped row. The active rows and columns are then RECONSTRUCTED
from the path by matrix-vector products against the static matrix ``A`` --
never maintained incrementally, which is what lets backtracking be a
truncation rather than an undo log.

THREE PLACES THE TUTORIAL DOES NOT DETERMINE THE ENCODING
---------------------------------------------------------
These are recorded rather than quietly resolved, because the tutorial is the
authority and each is a question for its author. All three are also listed in
the workbench's ``known_ir_violations.json``.

1. SIX TENSORS ARE USED BUT NOT DECLARED. The ``Tensors`` block declares
   sixteen; the cascade also reads ``F``, ``d*``, ``TPath``,
   ``NewPathEntry``, ``Col`` and ``Row``. Their ranks follow from use, but
   their data types and empty values do not, so the declarations below are
   this builder's inference and are marked as such.

2. E04's ``<`` MUST PRESERVE A VALUE, NOT PRODUCE A BOOLEAN. Written:

       TPath_{i,r} = Path_{i,r} . d*_i :: /\\ <(^)

   The registry's ``lt`` returns a boolean. But E06 feeds TPath into a partial
   update against ``NewPathEntry`` to form ``Path_{i+1}``, whose entries are
   DEPTHS -- and the worked example confirms it: at i = 2, Path = {R3:1, R2:2}
   and d* = 2, and Path_{3} is {R3:1, R0:2}, so the surviving entry kept its
   depth 1 rather than becoming True. So the operator here is a filtered
   take_left, and it is emitted under the name ``take_left_if_less`` rather
   than as ``lt``. A DIFFERENT NAME IS DELIBERATE: the operator-comparison
   gate will report this as a divergence from the surface ``<``, which it is,
   and it should stay visible until the tutorial says which it means.

3. E12's ``||ActiveCol|| == 0`` IS A PROPERTY, AND PROPERTIES LIVE IN
   STOPPING CONDITIONS. ``occupancy`` is available to a StoppingCondition,
   not to an Einsum expression, so the same claim is built in two steps: an
   OR-reduction over the columns, then a negation. Equivalent, and the
   equivalence is the reason it is allowed to differ.

Semantic assumption, as in every cascade here: a stored value equal to a
tensor's ``empty`` value is ABSENT. E07 depends on it directly -- an absent
path entry reads as -1, compares false against 0, and lands on SelRow's empty
value of False, which is what keeps unselected rows out.
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
    RankValue,
    Occupancy,
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
    UserDefinedUnaryOp,
)
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.stopping import Comparison, PropertyApp, TensorProjectionValue
from edge_ir.ir.tensor import BuiltinDataType, RankDeclaration, TensorDeclaration

# E04's filtered take_left. Named apart from both the tutorial's `<` and the
# registry's `lt` so the divergence is reported rather than absorbed; see the
# module docstring.
TAKE_LEFT_IF_LESS = "take_left_if_less"


# --------------------------------------------------------------------------
# the same small helpers max-flow's builder uses
# --------------------------------------------------------------------------
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


def anon(expression: Any, *ranks: Any) -> AnonymousTensor:
    return AnonymousTensor(expression=expression, ranks=list(ranks))


def unary_not(tensor: str, *ranks: Any) -> UnaryApp:
    return UnaryApp(op=BuiltinUnaryOp(symbol="not"), operand=inp(tensor, *ranks))


def compute(name: str) -> BuiltinComputeOp | UserDefinedComputeOp:
    if name in {"+", "-", "*", "/"}:
        return BuiltinComputeOp(symbol=name)  # type: ignore[arg-type]
    return UserDefinedComputeOp(name=name)


def merge(name: str) -> BuiltinMergeOp:
    return BuiltinMergeOp(symbol=name)  # type: ignore[arg-type]


def map_spec(label: int, compute_name: str, merge_name: str) -> MapSpec:
    return MapSpec(
        label=label,
        rank_list=None,
        compute_op=compute(compute_name),
        merge_op=merge(merge_name),
    )


def reduce_spec(
    label: int, compute_name: str, merge_name: str, rank_list: list[str]
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
        ranks=[RankDeclaration(name=n, shape=s) for n, s in ranks],
        data_type=BuiltinDataType(name=dtype),  # type: ignore[arg-type]
        empty_value=empty_value,
    )


def scalar_decl(
    name: str, dtype: str, empty_value: Any, value: Any = None
) -> TensorDeclaration:
    """A zero-rank tensor standing in for a literal (design decision D21)."""
    return tensor_decl(name, [], dtype, empty_value).model_copy(
        update={"value": value}
    )


def build_exact_cover() -> Program:
    declarations = [
        # ---- declared by the tutorial's `Tensors` block -------------------
        tensor_decl("A", [("R", "|R|"), ("C", "|C|")], "bool", False),
        tensor_decl("S", [("I", None), ("R", "|R|")], "int", -1),
        tensor_decl("Path", [("I", None), ("R", "|R|")], "int", -1),
        tensor_decl("SelRow", [("I", None), ("R", "|R|")], "bool", False),
        tensor_decl("CovCol", [("I", None), ("C", "|C|")], "bool", False),
        tensor_decl("ActiveCol", [("I", None), ("C", "|C|")], "bool", False),
        tensor_decl("ConflictRow", [("I", None), ("R", "|R|")], "bool", False),
        tensor_decl("ActiveRow", [("I", None), ("R", "|R|")], "bool", False),
        tensor_decl("Success", [("I", None)], "bool", False),
        tensor_decl("Sol", [("I", None), ("R", "|R|")], "bool", False),
        tensor_decl("RowCounts", [("I", None), ("C", "|C|")], "int", 0),
        tensor_decl("ActiveCounts", [("I", None), ("C", "|C|")], "int", 0),
        tensor_decl("PickedCol", [("I", None), ("C", "|C|")], "bool", False),
        tensor_decl("Candidates", [("I", None), ("R", "|R|")], "bool", False),
        tensor_decl("NewChoices", [("I", None), ("R", "|R|")], "int", -1),
        tensor_decl("T", [("I", None), ("R", "|R|")], "int", -1),
        # ---- USED BY THE CASCADE, NOT DECLARED BY THE TUTORIAL ------------
        # Gap 1 in the module docstring. Ranks follow from use; the data types
        # and empty values below are inferred, not stated.
        #
        # F: the peeked stamp. Inherits S's integer type and -1, because it is
        # one of S's entries surviving the populate.
        tensor_decl("F", [("I", None), ("R", "|R|")], "int", -1),
        # DStar: the popped choice's depth, `d*` in the tutorial. Depth 0 is
        # the empty path and never appears as a pop target (the first pop is
        # at depth 1), so 0 is safe as the empty value here -- unlike heights
        # in max-flow, where 0 was a real value and had to be +inf.
        tensor_decl("DStar", [("I", None)], "int", 0),
        # TPath: the path truncated to depths < d*. Same type as Path, which
        # is what gap 2 in the docstring is about.
        tensor_decl("TPath", [("I", None), ("R", "|R|")], "int", -1),
        tensor_decl("Keep", [("I", None), ("R", "|R|")], "bool", False),
        tensor_decl("NewPathEntry", [("I", None), ("R", "|R|")], "int", -1),
        # Col and Row: the universes. Present at every column / every row, so
        # that E09 and E11 have something to subtract the covered and
        # conflicting sets FROM. A host supplies them; they cannot be derived
        # from A, because a column no row covers still exists.
        tensor_decl("Col", [("C", "|C|")], "bool", False),
        tensor_decl("Row", [("R", "|R|")], "bool", False),
        # ---- introduced by lowering ---------------------------------------
        # ---- the stack seed ------------------------------------------------
        # S_0, which the tutorial states only in prose ("initial candidates at
        # depth 1 (first column branch choices)"). Supplied by the host
        # alongside A, the way max-flow's S and T masks are.
        tensor_decl("InitS", [("R", "|R|")], "int", -1),
        # ---- literals (D21: a literal must be PRESENT, so empty is None) ---
        scalar_decl("Zero", "int", None, value=0),
        scalar_decl("One", "int", None, value=1),
        # |R|, the stamp modulus. Problem data, so no literal value here.
        scalar_decl("RowCount", "int", 0),
        # The stopping condition's literal 0. Its type comes from occupancy's
        # RETURN type (int), not from S.
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
            # S_{0,r} = the host's seed: the rows covering the first branch
            # column, each stamped at depth 1.
            Einsum(
                output_tensor="S",
                output_ranks=[rc(0), rv("r")],
                expression=inp("InitS", rv("r")),
                specs=[],
            ),
            # Path_0 = empty. Nothing is written: an absent tensor IS the
            # empty path, and writing -1 everywhere would say the same thing
            # less directly.
        ]
    )

    einsums = [
        # ---------------------------------------------------------------
        # peek and pop
        # ---------------------------------------------------------------
        # E01: F_{i,r*} = S_{i,r} <<<_{r*} 1(select-max-val)
        #      Pop the deepest choice. The largest stamp is the deepest
        #      because sigma is monotone in depth: d*|R| + r with r < |R|
        #      means depth dominates the row index, so `max` over stamps is
        #      depth-first search and nothing else needs to enforce it.
        Einsum(
            output_tensor="F",
            output_ranks=[rv("i"), rv("r")],
            expression=BinaryApp(label=1, lhs=inp("S", rv("i"), rv("r")),
                                 rhs=inp("One")),
            specs=[
                map_spec(1, "take_left", "take_left"),
                # `1(select-max-val)`: the 1 glyph is the populate COMPUTE op
                # (`identity`), the parenthesised name is the COORDINATE op.
                PopulateSpec(
                    label=1,
                    rank_list=["r"],
                    compute_op=compute("identity"),
                    coord_op=CoordinateOp(name="select-max-val"),
                ),
            ],
        ),
        # E02: d*_i = floor(F_{i,r} / |R|) :: \/ max(u)
        #      One Einsum. The floor is a unary, and EDGE defines a unary as a
        #      degenerate Map against 1 with take-left merge (paper 5.5,
        #      Eq. 30), which is how it applies to the parenthesized quotient:
        #        d*_i = ((F_{i,r} .^1 |R|)_{i,r} .^2 1)
        #               :: /\^1 /(^) /\^2 floor(<-) \/^2_r max(u)
        Einsum(
            output_tensor="DStar",
            output_ranks=[rv("i")],
            expression=BinaryApp(
                label=2,
                lhs=anon(
                    BinaryApp(label=1, lhs=inp("F", rv("i"), rv("r")),
                              rhs=inp("RowCount")),
                    rv("i"),
                    rv("r"),
                ),
                rhs=inp("One"),
            ),
            specs=[
                map_spec(1, "/", "intersect"),
                map_spec(2, "floor", "take_left"),
                reduce_spec(2, "max", "union", ["r"]),
            ],
        ),
        # E03: T_{i,r} = S_{i,r} . not F_{i,r} :: /\ <-(^)
        #      The stack with the popped entry removed. `not F` is a presence
        #      test, so the intersect keeps exactly the entries the populate
        #      did not select.
        Einsum(
            output_tensor="T",
            output_ranks=[rv("i"), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("S", rv("i"), rv("r")),
                rhs=unary_not("F", rv("i"), rv("r")),
            ),
            specs=[map_spec(1, "take_left", "intersect")],
        ),
        # ---------------------------------------------------------------
        # update path
        # ---------------------------------------------------------------
        # E04: TPath_{i,r} = Path_{i,r} . d*_i :: /\ <(^)
        #      BACKTRACKING, and it is one Einsum. Every choice made at depth
        #      >= d* is abandoned by not being copied forward; there is no
        #      undo, because nothing was ever destructively applied.
        #
        #      See gap 2 in the module docstring for why the operator is
        #      `take_left_if_less` and not `lt`.
        Einsum(
            output_tensor="Keep",
            output_ranks=[rv("i"), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("Path", rv("i"), rv("r")),
                rhs=inp("DStar", rv("i")),
            ),
            specs=[map_spec(1, "lt", "intersect")],
        ),
        # E05: TPath_{i,r} = Path_{i,r} . Keep_{i,r} :: /\ <-(^)
        Einsum(
            output_tensor="TPath",
            output_ranks=[rv("i"), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("Path", rv("i"), rv("r")),
                rhs=inp("Keep", rv("i"), rv("r")),
            ),
            specs=[map_spec(1, "take_left", "intersect")],
        ),
        # E05: NewPathEntry_{i,r} = F_{i,r} . d*_i :: /\ ->(^)
        #      The popped row, recorded at the depth it was chosen. take_right
        #      keeps d*'s value at the one r where F is present.
        Einsum(
            output_tensor="NewPathEntry",
            output_ranks=[rv("i"), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("F", rv("i"), rv("r")),
                rhs=inp("DStar", rv("i")),
            ),
            specs=[map_spec(1, "take_right", "intersect")],
        ),
        # E06: Path_{i+1,r} = TPath_{i,r} . NewPathEntry_{i,r} :: /\ <<(u)
        Einsum(
            output_tensor="Path",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("TPath", rv("i"), rv("r")),
                rhs=inp("NewPathEntry", rv("i"), rv("r")),
            ),
            specs=[map_spec(1, "update", "union")],
        ),
        # ---------------------------------------------------------------
        # active subset computation -- the heart of this formulation
        # ---------------------------------------------------------------
        # E07: SelRow_{i+1,r} = Path_{i+1,r} . 0 :: /\ >=(u)
        #      Which rows are on the path at all, discarding their depths.
        #
        #      THE MERGE IS INTERSECT HERE, NOT THE UNION THE TUTORIAL WRITES
        #      -- gap 4 in the module docstring. Under union, `Zero` is
        #      present at every r, so every row enters the map and the absent
        #      Path entries reach `ge` as None, which raises. Under intersect
        #      the support is Path's, and since every path entry holds a depth
        #      of at least 1 the comparison is True wherever it runs: the two
        #      readings agree on the answer, and only one of them executes.
        Einsum(
            output_tensor="SelRow",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("Path", i_plus_one(), rv("r")),
                rhs=inp("Zero"),
            ),
            specs=[map_spec(1, "ge", "union")],   # as the tutorial writes it
        ),
        # E08: CovCol_{i+1,c} = A_{r,c} . SelRow_{i+1,r} :: /\ <-(^) \/ ANY(u)
        #      The columns the path covers. A matrix-vector product against
        #      the STATIC matrix -- this is the reconstruction that replaces
        #      Dancing Links' cover/uncover.
        Einsum(
            output_tensor="CovCol",
            output_ranks=[i_plus_one(), rv("c")],
            expression=BinaryApp(
                label=1,
                lhs=inp("A", rv("r"), rv("c")),
                rhs=inp("SelRow", i_plus_one(), rv("r")),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                reduce_spec(1, "ANY", "union", ["r"]),
            ],
        ),
        # E09: ActiveCol_{i+1,c} = Col_c . not CovCol_{i+1,c} :: /\ <-(^)
        Einsum(
            output_tensor="ActiveCol",
            output_ranks=[i_plus_one(), rv("c")],
            expression=BinaryApp(
                label=1,
                lhs=inp("Col", rv("c")),
                rhs=unary_not("CovCol", i_plus_one(), rv("c")),
            ),
            specs=[map_spec(1, "take_left", "intersect")],
        ),
        # E10: ConflictRow_{i+1,r} = A_{r,c} . CovCol_{i+1,c}
        #                            :: /\ <-(^) \/ ANY(u)
        #      Rows overlapping an already-covered column. In an EXACT cover
        #      every column is covered once, so any such row is dead in this
        #      branch -- which is also why sibling branches are disjoint and
        #      the stack never needs to re-push a shallower alternative.
        Einsum(
            output_tensor="ConflictRow",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("A", rv("r"), rv("c")),
                rhs=inp("CovCol", i_plus_one(), rv("c")),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                reduce_spec(1, "ANY", "union", ["c"]),
            ],
        ),
        # E12: ActiveRow_{i+1,r}
        #        = (Row_r .^1 not ConflictRow_{i+1,r})_{i+1,r} .^2 not SelRow_{i+1,r}
        #          :: /\^1 <-(^) /\^2 <-(^)
        #      The tutorial's three-way product, parenthesized and labelled:
        #      one Einsum, as written, with each binary its own label.
        Einsum(
            output_tensor="ActiveRow",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=2,
                lhs=anon(
                    BinaryApp(
                        label=1,
                        lhs=inp("Row", rv("r")),
                        rhs=unary_not("ConflictRow", i_plus_one(), rv("r")),
                    ),
                    i_plus_one(),
                    rv("r"),
                ),
                rhs=unary_not("SelRow", i_plus_one(), rv("r")),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                map_spec(2, "take_left", "intersect"),
            ],
        ),
        # ---------------------------------------------------------------
        # success evaluation
        # ---------------------------------------------------------------
        # E13: Success_{i+1} = ||ActiveCol_{i+1}|| == 0
        #      One Einsum, as written: the occupancy of the columns left to
        #      cover, compared with 0. `Occupancy` is the expression form of
        #      the ||.|| stopping conditions already use.
        Einsum(
            output_tensor="Success",
            output_ranks=[i_plus_one()],
            expression=BinaryApp(
                label=1,
                lhs=Occupancy(operand=proj("ActiveCol", i_plus_one())),
                rhs=inp("Zero"),
            ),
            specs=[map_spec(1, "eq", "intersect")],
        ),
        # E13: Sol_{i+1,r} = SelRow_{i+1,r} . Success_{i+1} :: /\ <-(^)
        #      A solution is emitted only on the iterations where Success
        #      holds; on every other iteration Sol is absent, which is how one
        #      run reports SEVERAL solutions -- example 1 finds two, example 3
        #      finds two among sixteen iterations.
        Einsum(
            output_tensor="Sol",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("SelRow", i_plus_one(), rv("r")),
                rhs=inp("Success", i_plus_one()),
            ),
            specs=[map_spec(1, "take_left", "intersect")],
        ),
        # ---------------------------------------------------------------
        # branch and push
        # ---------------------------------------------------------------
        # E14: RowCounts_{i+1,c} = A_{r,c} . ActiveRow_{i+1,r}
        #                          :: /\ <-(^) \/ +(u)
        #      How many active rows still cover each column. The reduce is `+`
        #      rather than ANY, so booleans are counted, not merged.
        Einsum(
            output_tensor="RowCounts",
            output_ranks=[i_plus_one(), rv("c")],
            expression=BinaryApp(
                label=1,
                lhs=inp("A", rv("r"), rv("c")),
                rhs=inp("ActiveRow", i_plus_one(), rv("r")),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                reduce_spec(1, "+", "union", ["r"]),
            ],
        ),
        # E15: ActiveCounts_{i+1,c} = RowCounts . ActiveCol :: /\ <-(^)
        #      Restrict the counts to columns still needing cover, so E16's
        #      minimum cannot pick an already-covered column (whose count
        #      would be 0 and would win every time).
        Einsum(
            output_tensor="ActiveCounts",
            output_ranks=[i_plus_one(), rv("c")],
            expression=BinaryApp(
                label=1,
                lhs=inp("RowCounts", i_plus_one(), rv("c")),
                rhs=inp("ActiveCol", i_plus_one(), rv("c")),
            ),
            specs=[map_spec(1, "take_left", "intersect")],
        ),
        # E16: PickedCol_{i+1,c*} = ActiveCounts_{i+1,c}
        #                           <<<_{c*} 1(select-min-val)
        #      Knuth's S heuristic: branch on the most constrained column.
        #      Unlike max-flow's `select-any`, this one RANKS -- the minimum
        #      is part of the specification, not a free choice.
        Einsum(
            output_tensor="PickedCol",
            output_ranks=[i_plus_one(), rv("c")],
            expression=BinaryApp(
                label=1,
                lhs=inp("ActiveCounts", i_plus_one(), rv("c")),
                rhs=inp("One"),
            ),
            specs=[
                map_spec(1, "take_left", "take_left"),
                PopulateSpec(
                    label=1,
                    rank_list=["c"],
                    compute_op=compute("identity"),
                    coord_op=CoordinateOp(name="select-min-val"),
                ),
            ],
        ),
        # E18: Candidates_{i+1,r}
        #        = (A_{r,c} .^1 PickedCol_{i+1,c})_{i+1,r} .^2 ActiveRow_{i+1,r}
        #          :: /\^1 <-(^) \/^1_c ANY(u) /\^2 <-(^)
        #      As written: the inner product and its reduction over c are the
        #      parenthesized part (label 1), then the mask by ActiveRow
        #      (label 2). One Einsum.
        Einsum(
            output_tensor="Candidates",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=2,
                lhs=anon(
                    BinaryApp(
                        label=1,
                        lhs=inp("A", rv("r"), rv("c")),
                        rhs=inp("PickedCol", i_plus_one(), rv("c")),
                    ),
                    i_plus_one(),
                    rv("r"),
                ),
                rhs=inp("ActiveRow", i_plus_one(), rv("r")),
            ),
            specs=[
                map_spec(1, "take_left", "intersect"),
                reduce_spec(1, "ANY", "union", ["c"]),
                map_spec(2, "take_left", "intersect"),
            ],
        ),
        # E19: NewChoices_{i+1,r} = Candidates_{i+1,r} . sigma(d*_i + 1, r)
        #                           :: /\ ->(^)
        #      with sigma(d, r) = d*|R| + r written out in place:
        #        Candidates .^4 (((d*_i .^1 1)_i .^2 |R|)_i .^3 r)_{i,r}
        #        :: /\^1 +(^) /\^2 *(^) /\^3 +(^) /\^4 ->(^)
        #      `r` is the rank variable read as a value (EDGE 5.10).
        Einsum(
            output_tensor="NewChoices",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=4,
                lhs=inp("Candidates", i_plus_one(), rv("r")),
                rhs=anon(
                    BinaryApp(
                        label=3,
                        lhs=anon(
                            BinaryApp(
                                label=2,
                                lhs=anon(
                                    BinaryApp(label=1, lhs=inp("DStar", rv("i")),
                                              rhs=inp("One")),
                                    rv("i"),
                                ),
                                rhs=inp("RowCount"),
                            ),
                            rv("i"),
                        ),
                        rhs=RankValue(expr=rv("r")),
                    ),
                    rv("i"),
                    rv("r"),
                ),
            ),
            specs=[
                map_spec(1, "+", "intersect"),
                map_spec(2, "*", "intersect"),
                map_spec(3, "+", "intersect"),
                map_spec(4, "take_right", "intersect"),
            ],
        ),
        # E19: S_{i+1,r} = T_{i,r} . NewChoices_{i+1,r} :: /\ <<(u)
        #      The popped stack plus the new siblings. The union keeps the
        #      shallower alternatives that E03 left in T, which is what makes
        #      this a stack rather than a frontier.
        Einsum(
            output_tensor="S",
            output_ranks=[i_plus_one(), rv("r")],
            expression=BinaryApp(
                label=1,
                lhs=inp("T", rv("i"), rv("r")),
                rhs=inp("NewChoices", i_plus_one(), rv("r")),
            ),
            specs=[map_spec(1, "update", "union")],
        ),
    ]

    # <>: ||S_{i+1}|| == 0 -- stop when the stack empties, which means the
    # search tree is exhausted. Note this is NOT "stop at the first solution":
    # Success can hold on an iteration and the search continues, which is how
    # every solution is enumerated.
    stopping = StoppingCondition(
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
    )

    return Program(
        declarations=declarations,
        initialization=initialization,
        main_edge=MainEdge(
            cascade=NestedCascade(einsums=einsums, stopping_conditions=[stopping])
        ),
    )


if __name__ == "__main__":
    sys.stdout.write(build_exact_cover().model_dump_json(indent=2))
