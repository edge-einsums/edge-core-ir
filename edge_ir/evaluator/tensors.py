"""Tensors as partial functions, existence, and fibers.

Mirrors the EDGE semantics sections
"Tensor", "Fibers and Subtensors", and
"Existence Check".

The EDGE semantics are explicit about the shape of a tensor::

    T : CS^T -> DS^T_ne          (partial)
    DS^T_ne = DS^T \\ E^T

so a tensor's codomain is the *non-empty* data space. Two consequences the
rest of the evaluator leans on:

1. **A tensor can never hold its own empty value.** Storing ``e^T`` at a
   coordinate is not a representable state -- it is the same thing as the
   coordinate being absent. :meth:`Tensor.write` therefore deletes rather
   than storing when handed a value in ``E^T``, and construction filters
   the same way.
2. **Existence and non-emptiness are the same predicate.** ``Exists_T(c)``
   is ``c in dom(T)``, and because the codomain excludes ``E^T`` there is
   no third state "present but empty".

A fiber is partial application (currying): fixing some rank coordinates
yields a subtensor over the remaining ranks. Populate needs the
*enumerated* form, which keeps full coordinate tuples -- see
:meth:`Tensor.enum_fiber`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from edge_ir.evaluator.errors import UnderSpecifiedProgramError
from edge_ir.evaluator.spaces import (
    Coordinate,
    CoordinateSpace,
    Point,
    is_empty_value,
    sort_key,
)

################################################################################
# Enumerated fiber entries
################################################################################


@dataclass(frozen=True)
class FiberEntry:
    """One element of an enumerated fiber: full coordinate, value, presence.

    ``present`` records whether the coordinate is in ``dom(Z)``. When it is
    False, ``value`` is the tensor's empty value.

    Why presence is carried explicitly: the formal Populate definition
    writes the enumerated fiber as
    ``{(cs, dv) | ... Z(cs) = dv}``, which reads as "present coordinates
    only", but the worked fibers example enumerates the whole
    rank coordinate set, and the Write constraint ("a coordinate may be
    marked Write only if it maps to the empty value of Z") is unsatisfiable
    unless absent coordinates are visible to the coordinate operator. We
    follow the sidebar and expose presence, so a coordinate operator can
    honour the constraint. This divergence is recorded as an open question.
    """

    coord: Point
    value: Any
    present: bool


################################################################################
# Tensor
################################################################################


class Tensor:
    """A tensor: a partial function from its coordinate space to ``DS_ne``.

    The graph is stored sparsely. Absence *is* emptiness; there is no
    sentinel comparison anywhere in this class.
    """

    __slots__ = ("name", "space", "empty_value", "_graph")

    def __init__(
        self,
        name: str,
        space: CoordinateSpace,
        empty_value: Any,
        graph: Mapping[Point, Any] | None = None,
    ) -> None:
        self.name = name
        self.space = space
        self.empty_value = empty_value
        self._graph: dict[Point, Any] = {}
        if graph:
            for coord, value in graph.items():
                self.write(coord, value)

    # -- construction ------------------------------------------------------

    def copy(self) -> Tensor:
        """A shallow copy sharing the coordinate space."""
        clone = Tensor(self.name, self.space, self.empty_value)
        clone._graph = dict(self._graph)
        return clone

    def renamed(self, name: str) -> Tensor:
        clone = self.copy()
        clone.name = name
        return clone

    # -- the function itself -----------------------------------------------

    def _check_arity(self, coord: Point, verb: str) -> None:
        """A coordinate of the wrong length is a malformed access, not a miss.

        ``Exists_T(c)`` is defined for ``c`` in ``CS^T``, whose elements are
        N-tuples for an N-rank tensor (EDGE semantics, "Tensor Coordinate
        Space Sets"). A 2-tuple is not a point of a 3-rank tensor's coordinate
        space at all -- it is a projection written with the wrong number of
        subscripts. Answering "absent" there would let a whole cascade
        evaluate to empty with no diagnostic, so it raises instead.
        """
        if len(coord) != len(self.space.ranks):
            raise UnderSpecifiedProgramError(
                f"tensor {self.name!r} has {len(self.space.ranks)} rank(s) "
                f"({', '.join(self.space.rank_names) or 'none'}) but was "
                f"{verb} at {coord!r}, which has {len(coord)}. A projection "
                f"must carry one rank expression per declared rank."
            )

    def exists(self, coord: Point) -> bool:
        """``Exists_T(c)`` -- True iff ``T`` is defined at ``coord``.

        False for any coordinate inside ``CS^T`` that carries no value, since
        ``dom(T) subset-of CS^T``. A coordinate of the WRONG ARITY is not a
        miss -- see :meth:`_check_arity`.
        """
        self._check_arity(coord, "queried")
        return coord in self._graph

    def __call__(self, coord: Point) -> Any:
        """``T(c)``. Defined only where :meth:`exists`."""
        self._check_arity(coord, "read")
        try:
            return self._graph[coord]
        except KeyError:
            raise UnderSpecifiedProgramError(
                f"tensor {self.name!r} is not defined at {coord!r}; "
                f"guard the access with exists() or use value_or_empty()."
            ) from None

    def value_or_empty(self, coord: Point) -> Any:
        """``T(c)`` where defined, otherwise the tensor's empty value.

        This is the ``dv_A(is)`` / ``dv_B(is)`` substitution in the Map semantics:
        when a merge operator lets a point through despite a missing
        operand, the missing operand is represented by its empty value.
        """
        self._check_arity(coord, "read")
        return self._graph.get(coord, self.empty_value)

    # -- mutation ----------------------------------------------------------

    def write(self, coord: Point, value: Any) -> None:
        """Place ``value`` at ``coord``.

        Writing a value in ``E^T`` deletes the coordinate instead: the
        codomain is ``DS^T_ne``, so "present with the empty value" is not a
        representable state.

        A coordinate outside ``CS^T`` is refused. ``dom(T) subset-of CS^T``
        (EDGE semantics, Existence Check), so a tensor cannot hold a
        value at a coordinate its own declaration does not admit. Without this
        check, host input data at an out-of-space coordinate is stored and then
        reads back as PRESENT, which quietly rescues exactly the declaration
        error -- a rank declared with the wrong shape -- that rule 6 of the
        Layer 2 validator plan exists to catch.
        """
        self._check_arity(coord, "written")
        if not self.space.contains(coord):
            raise UnderSpecifiedProgramError(
                f"tensor {self.name!r} was written at {coord!r}, which is "
                f"outside its coordinate space "
                f"{self.name}^{{{','.join(self.space.rank_names)}}}. Check the "
                f"declared rank shapes against the data being supplied."
            )
        if is_empty_value(value, self.empty_value):
            self._graph.pop(coord, None)
            return
        self._graph[coord] = value

    def delete(self, coord: Point) -> None:
        self._check_arity(coord, "deleted at")
        self._graph.pop(coord, None)

    def clear_where(self, predicate: Any) -> None:
        """Delete every coordinate for which ``predicate(coord)`` is True."""
        for coord in [c for c in self._graph if predicate(c)]:
            del self._graph[coord]

    # -- observation -------------------------------------------------------

    def occupancy(self) -> int:
        """``||T||`` -- the number of coordinates holding a non-empty value."""
        return len(self._graph)

    def __len__(self) -> int:
        return len(self._graph)

    def __contains__(self, coord: object) -> bool:
        return coord in self._graph

    def domain(self) -> frozenset[Point]:
        """``dom(T)``."""
        return frozenset(self._graph)

    def items(self) -> list[tuple[Point, Any]]:
        """The graph of ``T``, in deterministic coordinate order."""
        return sorted(
            self._graph.items(),
            key=lambda kv: tuple(sort_key(c) for c in kv[0]),
        )

    def __iter__(self) -> Iterator[Point]:
        return iter(sorted(self._graph, key=lambda c: tuple(sort_key(x) for x in c)))

    # -- fibers ------------------------------------------------------------

    def filter_coords(self, coord: Point, mutable: Iterable[int]) -> Point:
        """``filter-coords(cs^Z)`` -- drop the ``*``-marked rank coordinates.

        The result is the *fiber key*: the part of the output coordinate
        that stays fixed while Populate updates the mutable fiber.
        """
        drop = set(mutable)
        return tuple(c for i, c in enumerate(coord) if i not in drop)

    def enum_fiber(
        self, fiber_key: Point, mutable: Iterable[int]
    ) -> tuple[FiberEntry, ...]:
        """``EnumFiber_Z(fk)`` -- the enumerated fiber for one fiber key.

        Full coordinate tuples are retained, as the EDGE semantics require, and
        each entry carries its presence. The mutable ranks are enumerated
        over their whole rank coordinate sets; the non-mutable coordinates
        come from ``fiber_key`` in declaration order.
        """
        mutable_positions = sorted(mutable)
        fixed_positions = [
            i for i in range(len(self.space.ranks)) if i not in set(mutable_positions)
        ]
        if len(fiber_key) != len(fixed_positions):
            raise UnderSpecifiedProgramError(
                f"fiber key {fiber_key!r} has {len(fiber_key)} coordinate(s) "
                f"but tensor {self.name!r} has {len(fixed_positions)} "
                f"non-mutable rank(s)."
            )
        for pos in mutable_positions:
            if not self.space.ranks[pos].bounded:
                raise UnderSpecifiedProgramError(
                    f"tensor {self.name!r} marks rank "
                    f"{self.space.ranks[pos].rank_name!r} mutable, but its "
                    f"coordinate set is unbounded, so the fiber cannot be "
                    f"enumerated."
                )
        fixed = dict(zip(fixed_positions, fiber_key, strict=True))
        axes = [tuple(self.space.ranks[pos]) for pos in mutable_positions]
        entries: list[FiberEntry] = []
        for combo in product(*axes) if axes else [()]:
            varying = dict(zip(mutable_positions, combo, strict=True))
            coord: Point = tuple(
                fixed[i] if i in fixed else varying[i]
                for i in range(len(self.space.ranks))
            )
            present = self.exists(coord)
            entries.append(
                FiberEntry(
                    coord=coord,
                    value=self._graph[coord] if present else self.empty_value,
                    present=present,
                )
            )
        return tuple(entries)

    # -- misc --------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return (
            self.space.rank_names == other.space.rank_names
            and self._graph == other._graph
        )

    def __hash__(self) -> int:  # pragma: no cover - tensors are mutable
        raise TypeError("Tensor is mutable and not hashable")

    def __repr__(self) -> str:
        shown = ", ".join(f"{c}: {v}" for c, v in self.items()[:6])
        more = ", ..." if len(self._graph) > 6 else ""
        ranks = ",".join(self.space.rank_names)
        return f"Tensor {self.name}^{{{ranks}}} {{{shown}{more}}}"


def scalar_tensor(name: str, value: Any, empty_value: Any = None) -> Tensor:
    """A zero-rank tensor holding ``value`` at the singleton point ``()``.

    Scalars are zero-rank tensors everywhere in this IR (design decision
    D21); this is the constructor the evaluator uses for a
    ``TensorDeclaration`` that carries a literal ``value``.
    """
    space = CoordinateSpace(tensor=name, ranks=())
    tensor = Tensor(name, space, empty_value)
    tensor._graph[()] = value
    return tensor


def coordinate_of(point: Point, indices: Iterable[int]) -> tuple[Coordinate, ...]:
    """Project ``point`` onto the given positions, preserving order."""
    return tuple(point[i] for i in indices)
