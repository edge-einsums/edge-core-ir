"""The Populate action.

Mirrors the EDGE semantics of Populate.

Populate is the only action that touches the output tensor. It processes one
RedTmp element at a time and, for each, does five things:

1. **fiber-key extraction** -- drop the ``*``-marked rank coordinates from
   ``cs^Z`` to get ``fk``;
2. **fiber selection** -- retrieve ``EnumFiber_Z(fk)``, the enumerated
   subtensor, with full coordinate tuples;
3. **coordinate operator** -- a user-defined function maps that fiber plus
   the incoming ``(cs^R, dv_r)`` to a ``write-del`` set of
   ``(coordinate, {Write, Delete, None})`` pairs;
4. **fiber update** -- apply the deletes, then run the compute operator once
   per ``Write`` coordinate to produce the value written there;
5. **splice** -- replace that one fiber in ``Z`` and leave the rest alone.

The semantic constraint of step 3 is enforced here rather than trusted: a
coordinate may be marked ``Write`` only if it holds the empty value in the
*pre-state*, and a coordinate deleted this round may not also be written
this round -- together ``W and (dom(S) or D) = {}``. The EDGE semantics say an
implementation "should raise an error (at compile or runtime) if a violation
is detected", so :class:`PopulateConstraintViolation` is that error.

Assignment (no ``*`` ranks) is the degenerate case: the fiber is the single
coordinate ``cs^Z``, the coordinate operator marks it ``Write``, and the
compute operator passes the RedTmp value through. The EDGE semantics'
default-assignment example walks exactly those steps and performs no emptiness
check, so the pre-state constraint is *not* applied on that path -- an
assignment overwrites. That divergence between the general constraint and
the assignment sidebar is recorded as an open question.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from edge_ir.evaluator.actions.map import apply_compute
from edge_ir.evaluator.actions.reduce import ReductionTemporary
from edge_ir.evaluator.context import PopulateAction, PopulateContext
from edge_ir.evaluator.environment import Environment
from edge_ir.evaluator.errors import (
    PopulateConstraintViolation,
    UnderSpecifiedProgramError,
)
from edge_ir.evaluator.spaces import Point
from edge_ir.evaluator.tensors import Tensor
from edge_ir.ir.actions import PopulateSpec
from edge_ir.udf.categories import UdfCategory


def populate_action(
    spec: PopulateSpec | None,
    *,
    red_tmp: ReductionTemporary,
    order: Sequence[Point],
    axes: tuple[str, ...],
    output: Tensor,
    mutable: tuple[int, ...],
    env: Environment,
    where: str,
) -> None:
    """Update ``output`` in place from the reduction temporary.

    ``order`` is the sequence of output coordinates to process, already put
    in traversal order by the schedule. Populate is a fold -- each
    invocation sees the ``Z`` the previous one produced -- so this order is
    a real semantic degree of freedom whenever two RedTmp elements share a
    fiber.
    """
    if spec is None or not mutable:
        _assign(red_tmp, order, output)
        return

    coord_impl = env.functions.impl(UdfCategory.COORDINATE, spec.coord_op.name)

    for cs_z in order:
        dv_r = red_tmp.values[cs_z]
        witness = red_tmp.witness.get(cs_z, ())
        fiber_key = output.filter_coords(cs_z, mutable)
        fiber = output.enum_fiber(fiber_key, mutable)
        pre_state = {entry.coord for entry in fiber if entry.present}

        ctx = PopulateContext(
            point=witness,
            axes=axes,
            output_coord=cs_z,
            value=dv_r,
            fiber_key=fiber_key,
            fiber=fiber,
            mutable_positions=mutable,
            target_coord=None,
            output_empty=output.empty_value,
        )
        write_del = coord_impl.call(fiber, cs_z, dv_r, ctx=ctx)
        deletes, writes = _partition(write_del, where=where)
        _check_constraints(
            writes=writes, deletes=deletes, pre_state=pre_state, where=where
        )

        for coord in deletes:
            output.delete(coord)
        for coord in writes:
            value = apply_compute(
                spec.compute_op,
                coord,
                dv_r,
                env=env,
                category=UdfCategory.POPULATE_COMPUTE,
                ctx=ctx.with_target(coord),
                where=where,
            )
            output.write(coord, value)


def _assign(
    red_tmp: ReductionTemporary, order: Sequence[Point], output: Tensor
) -> None:
    """Default assignment: copy each RedTmp element to its own coordinate."""
    for cs_z in order:
        output.write(cs_z, red_tmp.values[cs_z])


def _partition(
    write_del: Iterable[Any], *, where: str
) -> tuple[list[Point], list[Point]]:
    deletes: list[Point] = []
    writes: list[Point] = []
    for item in write_del:
        try:
            coord, action = item
        except (TypeError, ValueError) as exc:
            raise UnderSpecifiedProgramError(
                f"{where}: a coordinate operator returned {item!r}; every "
                f"element of the write-del set must be a "
                f"(coordinate, action) pair."
            ) from exc
        parsed = PopulateAction(action)
        if parsed is PopulateAction.DELETE:
            deletes.append(tuple(coord))
        elif parsed is PopulateAction.WRITE:
            writes.append(tuple(coord))
    return deletes, writes


def _check_constraints(
    *,
    writes: Sequence[Point],
    deletes: Sequence[Point],
    pre_state: set[Point],
    where: str,
) -> None:
    occupied = [c for c in writes if c in pre_state]
    if occupied:
        raise PopulateConstraintViolation(
            f"{where}: the coordinate operator marked {occupied!r} Write, but "
            f"those coordinates already hold a value in the pre-state of this "
            f"round. Populate may write only where Z is empty "
            f"(W and dom(S) = {{}})."
        )
        # (A coordinate may be deleted this round and written in a LATER one.)
    both = sorted(set(writes) & set(deletes))
    if both:
        raise PopulateConstraintViolation(
            f"{where}: the coordinate operator marked {both!r} both Write and "
            f"Delete in the same round. A location deleted in this round "
            f"cannot also be written in it (D and W = {{}})."
        )
