"""The Reduce action.

Mirrors the EDGE semantics of Reduce.

Reduce does not write the output tensor -- Populate does. Reduce keeps one
*reduction state* per output tensor coordinate (the "shadow tensor" of
the Reduce semantics) and folds MapTmp into it.

Three details of the EDGE semantics that a casual implementation gets wrong, and
that this module implements literally:

1. **Reduce walks the whole iteration space, not just MapTmp's support.**
   It "iterates over the same iteration space as Map, queries whether Map
   produced a non-empty value at each point". A point where MapTmp is
   *absent* still reaches the merge operator as ``b_m = False``, and with an
   intersect-like merge that absence *invalidates* the accumulated state.
2. **A fresh reduction state is seeded with the compute operator's
   identity**, and the merge's left flag is ``b_s = (dv_s not in E^Z)`` --
   computed from that identity, not from "have we seen anything yet".
3. **When merge says False the state is set to the empty value**, not left
   alone: "in Reduce, the merge operator determines not only whether the
   incoming MapTmp value is used, but also whether the current reduction
   state remains valid".

The identity is the one thing the IR does not carry. Built-in compute
operators get theirs from ``op_properties`` (``+`` is 0, ``*`` is 1);
user-defined ones must declare it on their registry entry. An operator with
no identity and no declared one is an error, not a guess.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from edge_ir.evaluator.actions.map import apply_compute, merge_decides
from edge_ir.evaluator.context import ReduceContext
from edge_ir.evaluator.environment import Environment
from edge_ir.evaluator.errors import UnderSpecifiedProgramError
from edge_ir.evaluator.spaces import Point, is_empty_value
from edge_ir.ir.actions import ReduceSpec
from edge_ir.ir.op import BuiltinComputeOp, ComputeOp, UserDefinedComputeOp
from edge_ir.runtime.op_properties import COMPUTE_OP_PROPERTIES
from edge_ir.udf.categories import UdfCategory
from edge_ir.udf.decl import EMPTY_IDENTITY


@dataclass
class ReductionTemporary:
    """``RedTmp`` -- the final, non-empty reduction states, by output coord."""

    values: dict[Point, Any] = field(default_factory=dict)
    """The surviving ``(cs^Z, dv)`` pairs."""

    witness: dict[Point, Point] = field(default_factory=dict)
    """One iteration-space point that contributed to each output coordinate.

    Populate's operators may consult ``is``; a RedTmp element can be fed by
    many points, so the first contributing point (in the order Reduce ran)
    is kept as the representative. Recorded explicitly because it is a
    choice, not something the EDGE semantics pin down.
    """

    def __len__(self) -> int:
        return len(self.values)


def reduction_identity(
    op: ComputeOp,
    *,
    env: Environment,
    output_empty: Any,
    where: str,
) -> Any:
    """``1_reduce`` -- the identity a fresh reduction state is seeded with."""
    if isinstance(op, BuiltinComputeOp):
        props = COMPUTE_OP_PROPERTIES.get(op.symbol)
        if props is None or props.identity is None:
            raise UnderSpecifiedProgramError(
                f"{where}: the built-in compute operator {op.symbol!r} has no "
                f"two-sided identity, so a reduction using it has nothing to "
                f"seed its state with. Reduce with an operator that has one "
                f"(+ or *), or register a user-defined operator declaring an "
                f"identity."
            )
        return props.identity
    if isinstance(op, UserDefinedComputeOp):
        decl = env.functions.decl(UdfCategory.REDUCE_COMPUTE, op.name)
        if not decl.has_identity:
            raise UnderSpecifiedProgramError(
                f"{where}: the user-defined reduce operator {op.name!r} is "
                f"registered without an identity element, but Reduce seeds a "
                f"fresh reduction state with one. Register it as "
                f"identity=<value>, or identity=EMPTY_IDENTITY if the operator "
                f"simply takes the first contribution."
            )
        if decl.identity is EMPTY_IDENTITY:
            return output_empty
        return decl.identity
    raise UnderSpecifiedProgramError(f"unknown compute operator: {type(op).__name__}")


def reduce_action(
    spec: ReduceSpec | None,
    *,
    points: Sequence[Point],
    axes: tuple[str, ...],
    output_coords: Mapping[Point, Point],
    map_tmp: Mapping[Point, Any],
    output_empty: Any,
    env: Environment,
    where: str,
) -> ReductionTemporary:
    """Fold MapTmp into one reduction state per output coordinate.

    ``points`` are the iteration-space points *for which the Einsum
    projection is defined* -- a point whose output coordinate fell outside
    ``CS^Z`` produces no MapTmp and is not observed by Reduce
    (EDGE semantics, iteration-space constraints).

    ``spec`` may be ``None``: an Einsum with no Reduce spec performs no
    aggregation, so every output coordinate must receive at most one MapTmp
    element. A collision there is an under-specified program, not something
    to resolve by picking a winner.
    """
    red = ReductionTemporary()
    if spec is None:
        return _no_reduce(
            points=points,
            output_coords=output_coords,
            map_tmp=map_tmp,
            red=red,
            where=where,
        )

    identity = reduction_identity(
        spec.compute_op, env=env, output_empty=output_empty, where=where
    )
    state: dict[Point, Any] = {}

    for point in points:
        cs_z = output_coords[point]
        materialized = cs_z in state
        dv_s = state[cs_z] if materialized else identity
        b_s = not is_empty_value(dv_s, output_empty)
        b_m = point in map_tmp
        dv = map_tmp[point] if b_m else output_empty

        guard = merge_decides(spec.merge_op, b_s, b_m, where=where)
        if guard:
            ctx = ReduceContext(
                point=point,
                axes=axes,
                output_coord=cs_z,
                identity=identity,
                state_present=b_s,
                maptmp_present=b_m,
                output_empty=output_empty,
            )
            state[cs_z] = apply_compute(
                spec.compute_op,
                dv_s,
                dv,
                env=env,
                category=UdfCategory.REDUCE_COMPUTE,
                ctx=ctx,
                where=where,
            )
        else:
            state[cs_z] = output_empty
        if b_m and cs_z not in red.witness:
            red.witness[cs_z] = point

    for cs_z, value in state.items():
        if not is_empty_value(value, output_empty):
            red.values[cs_z] = value
            red.witness.setdefault(cs_z, ())
    return red


def _no_reduce(
    *,
    points: Sequence[Point],
    output_coords: Mapping[Point, Point],
    map_tmp: Mapping[Point, Any],
    red: ReductionTemporary,
    where: str,
) -> ReductionTemporary:
    seen: dict[Point, Point] = {}
    for point in points:
        if point not in map_tmp:
            continue
        cs_z = output_coords[point]
        if cs_z in seen:
            raise UnderSpecifiedProgramError(
                f"{where}: two iteration-space points, {seen[cs_z]!r} and "
                f"{point!r}, both produce output coordinate {cs_z!r}, but the "
                f"Einsum declares no Reduce spec, so there is no rule for "
                f"combining them. Add a Reduce spec (a compute operator and a "
                f"merge operator) to say how these contributions aggregate."
            )
        seen[cs_z] = point
        red.values[cs_z] = map_tmp[point]
        red.witness[cs_z] = point
    return red
