"""Iteration-space construction.

Mirrors the EDGE semantics of spaces
(the iteration space and the rank variable set) together with
the semantics of functions (specifying iteration-space
constraints).

The EDGE semantics define ``IS = R_0 x R_1 x ... x R_{D-1}`` where each ``R_d``
defaults to the whole numbers, and immediately adds: *"This does not mean an
implementation must enumerate all of W."* This module is where that licence
is cashed in. It answers two questions:

**Which axes?** The free rank variables of the Einsum. That is exactly what
``edge_ir.analysis.iteration_space`` computes, and this module calls it
rather than re-deriving it -- design decision D14 keeps the analysis layer
shared between the validator and the evaluator so the two cannot disagree.

**Which coordinates per axis?** The EDGE semantics do not say, because with
``R_d = W`` it does not have to. An executable evaluator must bound the set.
The rule here: for every subscript position where the axis appears *alone*
and as an integer-offset shift, invert the shift against that tensor rank's
coordinate set, then take the UNION of those candidate sets across all
occurrences. Union rather than intersection because over-approximating is
the safe direction -- a surplus point either lands outside ``CS^Z`` (so
``EP`` is undefined there and it produces no MapTmp) or is decided by the
merge operator exactly as a point of ``W`` would be. Under-approximating
would silently drop contributions. An axis with no derivable candidate is a
hard error, not a guess.

**Predicates.** ``Einsum.predicates`` restricts the space. The
paper-canonical reading is "materialize the restricted set, then walk it",
and D23 records that an interpreter may implement either that or
walk-and-skip as long as the observable output matches. Both happen here: a
top-level membership predicate on a bare rank variable narrows the axis
before enumeration (so BFS's ``F_{0, s : s in id}`` with ``|V| = 100`` and
``id = {2,5,7}`` walks three points, not a hundred), and every predicate is
also evaluated per point so that compound and comparison forms are honoured.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from edge_ir.analysis.iteration_space import coord_image
from edge_ir.analysis.iteration_space import iteration_space as analysis_iteration_space
from edge_ir.evaluator.errors import UnderSpecifiedProgramError
from edge_ir.evaluator.projection import evaluate_rank_expression
from edge_ir.evaluator.spaces import (
    Coordinate,
    CoordinateSpace,
    CoordSetEnv,
    Point,
    ShapeEnv,
    resolve_coordinate_set,
    sort_key,
)
from edge_ir.ir.einsum import Einsum
from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    Expression,
    InputTensor,
    RankExpression,
    TensorProjection,
    UnaryApp,
)
from edge_ir.ir.predicate import (
    IterationPredicate,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    PredicateComparison,
    SetMembership,
)
from edge_ir.ir.tensor import TensorDeclaration
from edge_ir.udf.registry import FunctionRegistry

################################################################################
# Walking an Einsum's tensor projections
################################################################################


def input_projections(expr: Expression) -> Iterator[TensorProjection]:
    """Every declared-tensor projection reachable in an expression tree."""
    if isinstance(expr, InputTensor):
        yield expr.proj
    elif isinstance(expr, UnaryApp):
        yield expr.operand.proj
    elif isinstance(expr, BinaryApp):
        yield from input_projections(expr.lhs)
        yield from input_projections(expr.rhs)
    elif isinstance(expr, AnonymousTensor):
        # An anonymous tensor is not declared, so its own rank list gives no
        # coordinate set; its operands still do.
        yield from input_projections(expr.expression)


################################################################################
# The concrete iteration space
################################################################################


@dataclass(frozen=True)
class ConcreteIterationSpace:
    """An Einsum's iteration space, materialized as a set of points.

    ``axes`` names the free rank variables in a fixed order and every point
    is a tuple aligned to it. ``pinned`` records rank variables the cascade
    has bound to a single coordinate (the iterative rank), which are
    therefore not axes but are still visible to every rank expression.
    """

    axes: tuple[str, ...]
    axis_sets: tuple[tuple[Coordinate, ...], ...]
    points: tuple[Point, ...]
    pinned: Mapping[str, Coordinate]

    def point_map(self, point: Point) -> dict[str, Coordinate]:
        """A point as ``{rank variable: coordinate}``, pinned ranks included."""
        bound = dict(self.pinned)
        bound.update(zip(self.axes, point, strict=False))
        return bound

    def __len__(self) -> int:
        return len(self.points)


def _axis_candidates(
    einsum: Einsum,
    axis: str,
    spaces: Mapping[str, CoordinateSpace],
) -> list[tuple[Coordinate, ...]]:
    """Candidate coordinate sets for one axis, one per invertible occurrence."""
    candidates: list[tuple[Coordinate, ...]] = []
    occurrences: list[tuple[str, list[RankExpression]]] = [
        (einsum.output_tensor, list(einsum.output_ranks))
    ]
    occurrences.extend(
        (proj.tensor, list(proj.ranks)) for proj in input_projections(einsum.expression)
    )
    for tensor_name, subscripts in occurrences:
        space = spaces.get(tensor_name)
        if space is None:
            continue
        for position, subscript in enumerate(subscripts):
            if position >= len(space.ranks):
                continue
            image = coord_image(subscript)
            if image.axes != frozenset({axis}) or not image.is_shift:
                continue
            rcs = space.ranks[position]
            if not rcs.bounded:
                continue
            assert rcs.coords is not None
            offset = image.offset
            shifted = tuple(
                c - offset
                for c in rcs.coords
                if isinstance(c, int) and not isinstance(c, bool)
            )
            if offset != 0 and len(shifted) != len(rcs.coords):
                # A labelled (non-integer) coordinate set cannot be shifted.
                continue
            candidates.append(shifted if offset != 0 else rcs.coords)
    return candidates


def _narrowing_membership(
    predicate: IterationPredicate | None,
) -> dict[str, SetMembership]:
    """Top-level positive memberships on a bare rank variable, by axis.

    Only these can safely narrow an axis before enumeration: a negated
    membership, a comparison, or anything under a disjunction has to be
    evaluated per point.
    """
    from edge_ir.ir.expr import RankVariable

    found: dict[str, SetMembership] = {}

    def walk(node: IterationPredicate) -> None:
        if isinstance(node, SetMembership):
            if not node.negated and isinstance(node.member, RankVariable):
                found[node.member.name] = node
        elif isinstance(node, LogicalAnd):
            for operand in node.operands:
                walk(operand)

    if predicate is not None:
        walk(predicate)
    return found


def predicate_holds(
    node: IterationPredicate,
    point: Mapping[str, Coordinate],
    *,
    declarations: Mapping[str, TensorDeclaration],
    shape_env: ShapeEnv,
    coord_sets: CoordSetEnv,
    registry: FunctionRegistry,
) -> bool:
    """Evaluate a restricted-iteration predicate at one point."""

    def recurse(child: IterationPredicate) -> bool:
        return predicate_holds(
            child,
            point,
            declarations=declarations,
            shape_env=shape_env,
            coord_sets=coord_sets,
            registry=registry,
        )

    if isinstance(node, SetMembership):
        value = evaluate_rank_expression(
            node.member, point, shape_env=shape_env, registry=registry
        )
        members = resolve_coordinate_set(
            node.coord_set,
            declarations=declarations,
            shape_env=shape_env,
            coord_sets=coord_sets,
        )
        inside = value in members
        return not inside if node.negated else inside
    if isinstance(node, PredicateComparison):
        lhs = evaluate_rank_expression(
            node.lhs, point, shape_env=shape_env, registry=registry
        )
        rhs = evaluate_rank_expression(
            node.rhs, point, shape_env=shape_env, registry=registry
        )
        return _compare(node.op, lhs, rhs)
    if isinstance(node, LogicalAnd):
        return all(recurse(operand) for operand in node.operands)
    if isinstance(node, LogicalOr):
        return any(recurse(operand) for operand in node.operands)
    if isinstance(node, LogicalNot):
        return not recurse(node.operand)
    raise UnderSpecifiedProgramError(
        f"unknown IterationPredicate variant: {type(node).__name__}"
    )


def _compare(op: str, lhs: Coordinate, rhs: Coordinate) -> bool:
    if op == "==":
        return bool(lhs == rhs)
    if op == "!=":
        return bool(lhs != rhs)
    try:
        if op == ">":
            return bool(lhs > rhs)  # type: ignore[operator]
        if op == "<":
            return bool(lhs < rhs)  # type: ignore[operator]
        if op == ">=":
            return bool(lhs >= rhs)  # type: ignore[operator]
        if op == "<=":
            return bool(lhs <= rhs)  # type: ignore[operator]
    except TypeError as exc:
        raise UnderSpecifiedProgramError(
            f"cannot order {lhs!r} {op} {rhs!r}: mixed coordinate kinds."
        ) from exc
    raise UnderSpecifiedProgramError(f"unknown comparison operator {op!r}")


def build_iteration_space(
    einsum: Einsum,
    *,
    declarations: Sequence[TensorDeclaration],
    spaces: Mapping[str, CoordinateSpace],
    shape_env: ShapeEnv,
    coord_sets: CoordSetEnv,
    registry: FunctionRegistry,
    pinned: Mapping[str, Coordinate] | None = None,
) -> ConcreteIterationSpace:
    """Materialize the (possibly predicate-restricted) iteration space."""
    pinned = dict(pinned or {})
    decls = {d.name: d for d in declarations}

    analysis = analysis_iteration_space(einsum, declarations)
    axes = tuple(sorted(a for a in analysis.axes if a not in pinned))

    predicate = einsum.predicates[0] if einsum.predicates else None
    narrowing = _narrowing_membership(predicate)

    axis_sets: list[tuple[Coordinate, ...]] = []
    for axis in axes:
        if axis in narrowing:
            members = resolve_coordinate_set(
                narrowing[axis].coord_set,
                declarations=decls,
                shape_env=shape_env,
                coord_sets=coord_sets,
            )
            axis_sets.append(tuple(sorted(set(members), key=sort_key)))
            continue
        candidates = _axis_candidates(einsum, axis, spaces)
        if not candidates:
            raise UnderSpecifiedProgramError(
                f"cannot bound the rank variable set of axis {axis!r} in the "
                f"Einsum writing {einsum.output_tensor!r}. It never appears as "
                f"a bare (or integer-shifted) subscript of a tensor rank whose "
                f"coordinate set is known, so its rank variable set stays the "
                f"whole numbers and cannot be enumerated. Give the rank a "
                f"shape or an explicit coordinate set, or pin the axis."
            )
        union: set[Coordinate] = set()
        for cand in candidates:
            union.update(cand)
        axis_sets.append(tuple(sorted(union, key=sort_key)))

    points: list[Point] = []
    for combo in product(*axis_sets) if axes else [()]:
        if predicate is not None:
            bound = dict(pinned)
            bound.update(zip(axes, combo, strict=False))
            if not predicate_holds(
                predicate,
                bound,
                declarations=decls,
                shape_env=shape_env,
                coord_sets=coord_sets,
                registry=registry,
            ):
                continue
        points.append(tuple(combo))

    return ConcreteIterationSpace(
        axes=axes,
        axis_sets=tuple(axis_sets),
        points=tuple(points),
        pinned=pinned,
    )
