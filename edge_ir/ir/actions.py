"""Computation specs for the three EDGE actions: Map, Reduce, Populate.

A computation spec describes one action at one iteration-space point.
Each spec packages the action verb (Map, Reduce, Populate) with the
operators it needs (compute, merge or coordinate) and configuration
(rank list, binary label).

Per the grammar:
    <computation-spec> := <map-reduce> ^label _rank-list <compute-op> "(" <merge-op> ")"
                        | <map-reduce> ^label            <compute-op> "(" <merge-op> ")"
                        | <populate>   ^label _rank-list <compute-op> "(" <coord-op> ")"

The three actions follow the paper Section 6.4 to 6.6 semantics. See
op.py for the operator categories these specs use.

Map and Reduce keep an OPTIONAL rank_list. When omitted, the
mapped/reduced rank set is derived from the iteration spec:
  - Reduce ranks = iteration ranks that do not appear in the output
    (reduce keeps one reduction state per output coordinate, so every
    iteration rank absent from the output is collapsed).
  - Map ranges over the full iteration space; per the formal semantics
    (canonical), the merge operator -- not a rank list -- decides which
    iteration-space points are effectual.
The list is kept rather than always derived because the set is not
always recoverable: an opaque user-defined output RVE (e.g. f(a, w)) can
route iteration points to output coordinates in a way the IR cannot
derive structurally. When a rank_list IS supplied, the Layer 2 validator
cross-checks it against the derived set and hints to the user when it
does not match what the Einsum is doing.

Populate keeps a rank_list with a different operator pairing (compute +
coordinate). That list may be EMPTY: `[]` is the no-`*` default-
assignment case (output coordinate taken from the iteration point, no
mutable rank); a non-empty list names the `*`-marked mutable ranks.

Labels. A `label` links a spec to a `BinaryApp` of the same label in the
expression tree. It is REQUIRED on `MapSpec` and OPTIONAL on `ReduceSpec`
and `PopulateSpec`. The asymmetry is the formal semantics', not a
convenience: Map genuinely combines two operands, so it names the binary
that pairs them. Reduce's two operands are the reduction state and the map
temporary (Reduce semantics, "Inputs and Outputs"), and Populate's
are the reduction temporary and the current output tensor
(Populate semantics, "Inputs and Outputs") -- neither refers to a
binary at all. The published EBNF requires a binary-label on every
computation spec; the formal semantics do not, and where the two disagree
the formal definition wins. `label=None` reads as "this action applies to
the whole right-hand side", which is what a single-operand Einsum such as
`N_{i,d} = SN_{i,s,d} :: reduce +(union)` needs.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from edge_ir.ir.base import IRBase
from edge_ir.ir.op import ComputeOp, CoordinateOp, MergeOp

################################################################################
# Map and Reduce specs
################################################################################


class MapSpec(IRBase):
    """Map action: pointwise application of compute_op, with merge_op
    determining which iteration-space points are effectual.

    Per paper Section 6.4. The label links this spec to a binary
    operation in the expression tree (BinaryApp.label). The rank_list is
    optional: when omitted, map ranges over the full iteration space (per
    the canonical formal semantics, the merge operator selects the
    effectual points). When supplied, the Layer 2 validator checks it
    against the derived set and hints on any mismatch.
    """

    kind: Literal["map"] = "map"
    label: int
    rank_list: list[str] | None = None
    compute_op: ComputeOp
    merge_op: MergeOp


class ReduceSpec(IRBase):
    """Reduce action: collapse the reduced ranks using compute_op, with
    merge_op handling the empty-value cases.

    Per paper Section 6.5. The label optionally links this spec to a binary
    operation; `None` means the reduce applies to the whole right-hand side,
    which is the single-operand case (`N_{i,d} = SN_{i,s,d} :: reduce`).
    Reduce's own operands are the reduction state and the map temporary, so
    it never structurally requires a binary. The rank_list is optional:
    when omitted, the reduced set
    is derived as the iteration ranks that do not appear in the output
    (reduce keeps one reduction state per output coordinate). It is kept
    rather than always derived because an opaque output RVE (e.g.
    f(a, w)) can make the set non-derivable; when supplied, the Layer 2
    validator cross-checks it and hints on mismatch.
    """

    kind: Literal["reduce"] = "reduce"
    label: int | None = None
    rank_list: list[str] | None = None
    compute_op: ComputeOp
    merge_op: MergeOp


################################################################################
# Populate spec
################################################################################


class PopulateSpec(IRBase):
    """Populate action: produce the output coordinate set from a
    computed value, using coordinate_op to determine the new
    coordinates.

    Per paper Section 6.6. The label optionally links this spec to a binary
    operation; `None` means the populate applies to the whole right-hand
    side, which is the single-operand case
    (`F_{i,v*} = S_{i,v} :: populate`). Populate's own operands are the
    reduction temporary and the current output tensor, so it never
    structurally requires a binary. The rank_list names the mutable
    (`*`-marked) ranks being
    populated and may be empty: an empty list `[]` is the default-
    assignment case (no `*`, output coordinate taken from the iteration
    point), while a non-empty list names the populated mutable ranks.
    """

    kind: Literal["populate"] = "populate"
    label: int | None = None
    rank_list: list[str] = Field(min_length=0)
    compute_op: ComputeOp
    coord_op: CoordinateOp


################################################################################
# ComputationSpec union
################################################################################


ComputationSpec = Annotated[
    MapSpec | ReduceSpec | PopulateSpec,
    Field(discriminator="kind"),
]
