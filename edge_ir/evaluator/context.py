"""Action contexts handed to user-defined operators.

The EDGE semantics give each action's compute operator more inputs than
just the two operand values:

- Map: ``compute(dv_A, dv_B, cs^A, cs^B, is)`` -- the operand
  values, their coordinate tuples, and the iteration-space point.
- Reduce: the reduction state, the MapTmp value, the output
  coordinate ``cs^Z``, the iteration-space point, and the reduction
  identity.
- Populate: the coordinate operator sees the enumerated
  fiber, the incoming RedTmp element, and the point; the compute operator
  sees the target coordinate, the RedTmp value, and the point.

Passing all of that positionally to every user function would make the
common two-argument operator unwriteable. Instead the extras are bundled
into a frozen context object and passed as a keyword ``ctx`` only to the
functions that declare it (see ``edge_ir.udf.decl.wants_context``). So
``lambda a, b: a + b`` stays legal, while ``update`` -- the ``<<``
operator, which must know whether its right operand was actually present --
takes ``ctx`` and reads ``ctx.right_present``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from edge_ir.evaluator.spaces import Point
from edge_ir.evaluator.tensors import FiberEntry


@dataclass(frozen=True)
class MapContext:
    """Everything Map's compute operator may consult at one point."""

    point: Point
    """``is`` -- the iteration-space point, aligned to :attr:`axes`."""

    axes: tuple[str, ...]
    """The iteration space's rank variable names, in order."""

    left_coord: Point | None
    right_coord: Point | None
    left_present: bool
    right_present: bool
    left_empty: Any
    right_empty: Any
    output_empty: Any

    @property
    def point_map(self) -> dict[str, Any]:
        """The point as ``{rank variable: coordinate}``."""
        return dict(zip(self.axes, self.point, strict=False))


@dataclass(frozen=True)
class ReduceContext:
    """Everything Reduce's compute operator may consult at one point."""

    point: Point
    axes: tuple[str, ...]
    output_coord: Point
    """``cs^Z`` -- the output coordinate this reduction state belongs to."""

    identity: Any
    """``1_reduce`` -- the identity the state was seeded with."""

    state_present: bool
    """``b_s`` -- the reduction state currently holds a non-empty value."""

    maptmp_present: bool
    """``b_m`` -- MapTmp is defined at this iteration-space point."""

    output_empty: Any

    @property
    def point_map(self) -> dict[str, Any]:
        return dict(zip(self.axes, self.point, strict=False))


class PopulateAction(str, Enum):
    """An action a coordinate operator assigns to one fiber coordinate."""

    WRITE = "Write"
    DELETE = "Delete"
    NONE = "None"


@dataclass(frozen=True)
class PopulateContext:
    """Everything Populate's coordinate and compute operators may consult."""

    point: Point
    axes: tuple[str, ...]
    output_coord: Point
    """``cs^R`` -- the coordinate carried by the incoming RedTmp element."""

    value: Any
    """``dv_r`` -- the value carried by the incoming RedTmp element."""

    fiber_key: Point
    fiber: tuple[FiberEntry, ...]
    """``EnumFiber_Z(fk)`` -- the fiber as it stands before this round."""

    mutable_positions: tuple[int, ...]
    """Rank positions of ``Z`` marked ``*``; the fiber varies over these."""

    target_coord: Point | None
    """The coordinate being written, when the compute operator is running."""

    output_empty: Any

    @property
    def point_map(self) -> dict[str, Any]:
        return dict(zip(self.axes, self.point, strict=False))

    def with_target(self, coord: Point) -> PopulateContext:
        return PopulateContext(
            point=self.point,
            axes=self.axes,
            output_coord=self.output_coord,
            value=self.value,
            fiber_key=self.fiber_key,
            fiber=self.fiber,
            mutable_positions=self.mutable_positions,
            target_coord=coord,
            output_empty=self.output_empty,
        )
