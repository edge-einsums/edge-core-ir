"""The sets an Extended Einsum operates over.

Mirrors the EDGE semantics of spaces -- rank coordinate
sets, tensor coordinate spaces, data spaces and empty values -- plus the
shape resolution that turns the IR's symbolic shapes into concrete
integers.

The vocabulary here is the EDGE semantics', deliberately:

- **rank coordinate set** (``RCS^T_R``): the set of valid coordinates for
  rank ``R`` of tensor ``T``. May be dense, sparse, or user-defined.
- **rank shape** (``RS``): the cardinality of a rank coordinate set. The
  word is "shape", never "extent".
- **tensor coordinate space** (``CS^T``): the Cartesian product of a
  tensor's rank coordinate sets; its elements are *points*.
- **empty value set** (``E^T``): the singleton subset of the data space
  identifying the tensor's empty value.

Resolution timing follows ``docs/design_decisions.md`` D22. A shape
symbol (``|V|``, ``N``) is bucket B -- symbolic until iteration-space
construction, resolved here once per run from a :class:`ShapeEnv`. A named
coordinate set (``CoordSetName``) is bucket C -- a reference to a runtime
resource, supplied by the host.

A rank declared with neither a shape nor a coordinate set is *unbounded*:
per the EDGE semantics its rank variable set defaults to the whole numbers. That
is the shape an iterative rank carries (``F^{I,S}`` declares ``I`` with no
shape), and the cascade driver pins it to one generation at a time.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from edge_ir.evaluator.errors import UnboundNameError, UnderSpecifiedProgramError
from edge_ir.ir.tensor import (
    CoordSetAlias,
    CoordSetEnum,
    CoordSetInterval,
    CoordSetName,
    RankDeclaration,
    TensorDeclaration,
)

# A rank coordinate. The grammar's <coordinate> production admits letters
# and digits, so a coordinate is an int or a str (paper Example
# uses character coordinates for vertex ids).
Coordinate = int | str

# A point in a tensor coordinate space, or in the iteration space: an
# ordered tuple of coordinates, one per rank (respectively, per axis).
Point = tuple[Coordinate, ...]

# Shape symbols (|V|, N, M) to their bound integer values.
ShapeEnv = Mapping[str, int]

# Host-supplied named coordinate sets (BFS's `id`, num-paths' `root`).
CoordSetEnv = Mapping[str, Sequence[Coordinate]]


def sort_key(coord: Coordinate) -> tuple[int, str | int]:
    """A total order over coordinates, for deterministic enumeration.

    Coordinates may be ints or strs and a rank coordinate set may in
    principle mix them, which plain ``sorted`` refuses. Ints sort before
    strs; within a kind the natural order applies.
    """
    if isinstance(coord, bool):
        # bool is an int subclass; keep it with the ints but ordered by
        # its integer value so True follows False.
        return (0, int(coord))
    if isinstance(coord, int):
        return (0, coord)
    return (1, coord)


################################################################################
# Rank coordinate sets
################################################################################


@dataclass(frozen=True)
class RankCoordinateSet:
    """``RCS^T_R`` -- the coordinate set of one rank of one tensor.

    ``coords`` is ``None`` when the rank is unbounded: the declaration gave
    neither a shape nor an explicit coordinate set, so the rank variable
    set is the default whole numbers (per the EDGE semantics, "by default, its
    elements are whole numbers"). An unbounded set cannot be enumerated;
    :meth:`contains` still answers membership.
    """

    rank_name: str
    coords: tuple[Coordinate, ...] | None

    @property
    def bounded(self) -> bool:
        """True when this coordinate set can be enumerated."""
        return self.coords is not None

    @property
    def shape(self) -> int:
        """The rank shape ``RS`` -- the cardinality of the coordinate set."""
        if self.coords is None:
            raise UnderSpecifiedProgramError(
                f"rank {self.rank_name!r} is unbounded, so it has no rank "
                f"shape. Declare a shape or a coordinate set, or pin the "
                f"rank (an iterative rank is pinned by the cascade driver)."
            )
        return len(self.coords)

    def contains(self, coord: Coordinate) -> bool:
        """Membership in ``RCS^T_R``."""
        if self.coords is None:
            # Default rank variable set: the whole numbers.
            return isinstance(coord, int) and not isinstance(coord, bool) and coord >= 0
        return coord in self._members

    @property
    def _members(self) -> frozenset[Coordinate]:
        assert self.coords is not None
        return frozenset(self.coords)

    def __iter__(self) -> Iterator[Coordinate]:
        if self.coords is None:
            raise UnderSpecifiedProgramError(
                f"cannot enumerate unbounded rank {self.rank_name!r}: its rank "
                f"variable set is the whole numbers. Declare a shape or a "
                f"coordinate set for it."
            )
        return iter(self.coords)


def _resolve_shape_symbol(sym: str, shape_env: ShapeEnv) -> int:
    if sym not in shape_env:
        raise UnboundNameError(
            f"shape symbol {sym!r} is not bound. Supply it in the shape "
            f"environment, e.g. shape_env={{{sym!r}: 8}}."
        )
    return shape_env[sym]


def _resolve_endpoint(value: int | str, shape_env: ShapeEnv) -> int:
    return value if isinstance(value, int) else _resolve_shape_symbol(value, shape_env)


def resolve_coordinate_set(
    cs: Any,
    *,
    declarations: Mapping[str, TensorDeclaration],
    shape_env: ShapeEnv,
    coord_sets: CoordSetEnv,
    _seen: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[Coordinate, ...]:
    """Resolve any ``CoordinateSet`` variant to a concrete tuple of coordinates.

    Shared by rank declarations and by ``SetMembership`` predicates, which
    draw from the same four-variant union.
    """
    if isinstance(cs, CoordSetEnum):
        return tuple(sorted(cs.coords, key=sort_key))
    if isinstance(cs, CoordSetInterval):
        lo = _resolve_endpoint(cs.lo, shape_env)
        hi = _resolve_endpoint(cs.hi, shape_env)
        if hi < lo:
            raise UnderSpecifiedProgramError(
                f"coordinate interval [{lo}, {hi}) has an upper bound below "
                f"its lower bound."
            )
        return tuple(range(lo, hi))
    if isinstance(cs, CoordSetAlias):
        key = (cs.tensor, cs.rank)
        if key in _seen:
            cycle = " -> ".join(f"{t}.{r}" for t, r in [*_seen, key])
            raise UnderSpecifiedProgramError(f"coordinate-set alias cycle: {cycle}")
        target = declarations.get(cs.tensor)
        if target is None:
            raise UnboundNameError(
                f"coordinate-set alias names {cs.tensor}.{cs.rank}, but no "
                f"tensor named {cs.tensor!r} is declared."
            )
        aliased = next((r for r in target.ranks if r.name == cs.rank), None)
        if aliased is None:
            raise UnboundNameError(
                f"coordinate-set alias names {cs.tensor}.{cs.rank}, but tensor "
                f"{cs.tensor!r} has no rank named {cs.rank!r}."
            )
        resolved = resolve_rank_coord_set(
            aliased,
            declarations=declarations,
            shape_env=shape_env,
            coord_sets=coord_sets,
            _seen=_seen | {key},
        )
        if resolved.coords is None:
            raise UnderSpecifiedProgramError(
                f"coordinate-set alias {cs.tensor}.{cs.rank} resolves to an "
                f"unbounded rank, which has no enumerable coordinate set."
            )
        return resolved.coords
    if isinstance(cs, CoordSetName):
        if cs.name not in coord_sets:
            raise UnboundNameError(
                f"named coordinate set {cs.name!r} was not supplied by the "
                f"host. Pass it in coord_sets."
            )
        return tuple(sorted(coord_sets[cs.name], key=sort_key))
    raise UnderSpecifiedProgramError(
        f"unknown CoordinateSet variant: {type(cs).__name__}"
    )


def resolve_rank_coord_set(
    rank: RankDeclaration,
    *,
    declarations: Mapping[str, TensorDeclaration],
    shape_env: ShapeEnv,
    coord_sets: CoordSetEnv,
    _seen: frozenset[tuple[str, str]] = frozenset(),
) -> RankCoordinateSet:
    """Resolve one :class:`RankDeclaration` to a concrete coordinate set.

    Precedence: an explicit ``coord_set`` wins over ``shape`` (the shape is
    a cardinality hint; the coordinate set is the set itself). With
    neither, the rank is unbounded.
    """
    if rank.coord_set is not None:
        coords = resolve_coordinate_set(
            rank.coord_set,
            declarations=declarations,
            shape_env=shape_env,
            coord_sets=coord_sets,
            _seen=_seen,
        )
        return RankCoordinateSet(rank_name=rank.name, coords=coords)

    if rank.shape is not None:
        size = _resolve_endpoint(rank.shape, shape_env)
        # The dense default coordinate set: DenseCS_R = {r | r in [0, RS)}.
        return RankCoordinateSet(rank_name=rank.name, coords=tuple(range(size)))

    return RankCoordinateSet(rank_name=rank.name, coords=None)


################################################################################
# Tensor coordinate spaces
################################################################################


@dataclass(frozen=True)
class CoordinateSpace:
    """``CS^T`` -- the Cartesian product of a tensor's rank coordinate sets.

    A zero-rank tensor (a scalar) has the singleton coordinate space
    ``{()}``, which is why ``ranks`` may be empty.
    """

    tensor: str
    ranks: tuple[RankCoordinateSet, ...]

    @property
    def rank_names(self) -> tuple[str, ...]:
        return tuple(r.rank_name for r in self.ranks)

    @property
    def bounded(self) -> bool:
        return all(r.bounded for r in self.ranks)

    def contains(self, point: Point) -> bool:
        """True iff ``point`` is a valid point of this coordinate space."""
        if len(point) != len(self.ranks):
            return False
        return all(rcs.contains(c) for rcs, c in zip(self.ranks, point, strict=False))

    def points(self) -> Iterator[Point]:
        """Enumerate every point of ``CS^T`` in deterministic order."""
        if not self.ranks:
            yield ()
            return
        for rcs in self.ranks:
            if not rcs.bounded:
                raise UnderSpecifiedProgramError(
                    f"cannot enumerate the coordinate space of tensor "
                    f"{self.tensor!r}: rank {rcs.rank_name!r} is unbounded."
                )
        yield from product(*(tuple(r) for r in self.ranks))


def coordinate_space(
    decl: TensorDeclaration,
    *,
    declarations: Mapping[str, TensorDeclaration],
    shape_env: ShapeEnv,
    coord_sets: CoordSetEnv,
) -> CoordinateSpace:
    """Build ``CS^T`` for one tensor declaration."""
    return CoordinateSpace(
        tensor=decl.name,
        ranks=tuple(
            resolve_rank_coord_set(
                rank,
                declarations=declarations,
                shape_env=shape_env,
                coord_sets=coord_sets,
            )
            for rank in decl.ranks
        ),
    )


################################################################################
# Data space and the empty value set
################################################################################


def is_empty_value(value: Any, empty_value: Any) -> bool:
    """True iff ``value`` is *the* empty value of a tensor (``value in E^T``).

    ``E^T`` is a singleton, so this is equality against the declared empty
    value -- with two care points:

    - ``nan != nan``, so a NaN empty value is matched by identity of
      NaN-ness rather than ``==``.
    - ``True == 1`` in Python but a Boolean tensor's ``empty=False`` must
      not swallow the integer ``0`` of a different tensor, so bool and
      non-bool are never considered equal.
    """
    if value is None or empty_value is None:
        return value is None and empty_value is None
    if isinstance(value, bool) != isinstance(empty_value, bool):
        return False
    if isinstance(empty_value, float) and empty_value != empty_value:
        return isinstance(value, float) and value != value
    try:
        return bool(value == empty_value)
    except Exception:  # noqa: BLE001 - user-defined types may refuse ==
        return False
