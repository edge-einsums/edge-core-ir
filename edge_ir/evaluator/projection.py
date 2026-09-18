"""Rank variable expressions and the Einsum projection.

Mirrors the EDGE semantics sections
"Rank Variable Expression", "Einsum Coordinate Set" and the Einsum projection.

A rank variable expression (RVE), written ``Gamma^T_R`` in the EDGE semantics, is a
*function from a point in the iteration space to one rank coordinate of one
tensor*. It is many-to-one. Every tensor rank in an Einsum has its own RVE,
and :func:`project` bundles a tensor's RVEs into ``Gamma^T(is)``, the tuple
of coordinates addressing that tensor at this point.

The Einsum projection ``EP : IS -> ECS`` is *partial*: it is undefined at a
point whose output coordinate falls outside ``CS^Z``. Input coordinates may
fall outside their tensor coordinate spaces without making ``EP``
undefined -- there the existence predicate is simply False and the merge
operator decides what happens. That asymmetry is the whole of
:func:`einsum_projection`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from edge_ir.evaluator.errors import UnboundNameError, UnderSpecifiedProgramError
from edge_ir.evaluator.spaces import Coordinate, Point, ShapeEnv
from edge_ir.ir.expr import (
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankExpression,
    RankFunction,
    RankVariable,
    TensorProjection,
)
from edge_ir.udf.categories import UdfCategory
from edge_ir.udf.registry import FunctionRegistry

# Point map: rank variable name -> the coordinate it takes at this point.
PointMap = Mapping[str, Coordinate]


def evaluate_rank_expression(
    node: RankExpression,
    point: PointMap,
    *,
    shape_env: ShapeEnv,
    registry: FunctionRegistry,
) -> Coordinate:
    """Evaluate one RVE at a point in the iteration space.

    The five variants map onto the four D22 resolution buckets:
    ``RankVariable`` is bucket D (per iteration point), ``RankConstantLiteral``
    is bucket A (pure structure), ``RankConstantShapeSym`` is bucket B
    (resolved once from the shape environment), ``RankArith`` inherits from
    its children, and ``RankFunction`` is bucket C (a reference to a runtime
    resource -- the registry).
    """
    if isinstance(node, RankVariable):
        if node.name not in point:
            raise UnderSpecifiedProgramError(
                f"rank variable {node.name!r} is not bound at this iteration-"
                f"space point (bound: {sorted(point)}). It appears in a "
                f"subscript but is not an axis of the Einsum's iteration space."
            )
        return point[node.name]

    if isinstance(node, RankConstantLiteral):
        return node.value

    if isinstance(node, RankConstantShapeSym):
        if node.name not in shape_env:
            raise UnboundNameError(
                f"shape symbol {node.name!r} appears in a rank expression but "
                f"is not bound. Supply it in the shape environment."
            )
        return shape_env[node.name]

    if isinstance(node, RankArith):
        lhs = evaluate_rank_expression(
            node.lhs, point, shape_env=shape_env, registry=registry
        )
        rhs = evaluate_rank_expression(
            node.rhs, point, shape_env=shape_env, registry=registry
        )
        return _arith(node.op, lhs, rhs)

    if isinstance(node, RankFunction):
        args = [
            evaluate_rank_expression(arg, point, shape_env=shape_env, registry=registry)
            for arg in node.args
        ]
        impl = registry.impl(UdfCategory.RANK_MAPPING, node.func_name)
        result = impl.call(*args)
        if not isinstance(result, (int, str)):
            raise UnderSpecifiedProgramError(
                f"rank-mapping function {node.func_name!r} returned "
                f"{result!r}, which is not a coordinate."
            )
        return result

    raise UnderSpecifiedProgramError(
        f"unknown RankExpression variant: {type(node).__name__}"
    )


def _arith(op: str, lhs: Coordinate, rhs: Coordinate) -> Coordinate:
    if not isinstance(lhs, int) or not isinstance(rhs, int):
        raise UnderSpecifiedProgramError(
            f"rank arithmetic {lhs!r} {op} {rhs!r} needs integer coordinates; "
            f"a labelled (string) coordinate has no arithmetic."
        )
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    if op in ("/", "//"):
        if rhs == 0:
            raise UnderSpecifiedProgramError("division by zero in a rank expression")
        return lhs // rhs
    if op == "%":
        if rhs == 0:
            raise UnderSpecifiedProgramError("modulo by zero in a rank expression")
        return lhs % rhs
    raise UnderSpecifiedProgramError(f"unknown rank-arithmetic operator {op!r}")


def project(
    proj: TensorProjection,
    point: PointMap,
    *,
    shape_env: ShapeEnv,
    registry: FunctionRegistry,
) -> Point:
    """``Gamma^T(is)`` -- the coordinate tuple addressing ``T`` at this point."""
    return tuple(
        evaluate_rank_expression(r, point, shape_env=shape_env, registry=registry)
        for r in proj.ranks
    )


@dataclass(frozen=True)
class EinsumProjection:
    """One point's worth of ``EP(is)``: the output coordinate, or undefined.

    ``defined`` is False exactly when ``Gamma^Z(is)`` fell outside
    ``CS^Z`` -- the point then produces no MapTmp element and is not
    observed by Reduce.
    """

    point: Point
    output_coord: Point
    defined: bool


def einsum_projection(
    output_ranks: list[RankExpression],
    point: PointMap,
    ordered_point: Point,
    *,
    output_space: Any,
    shape_env: ShapeEnv,
    registry: FunctionRegistry,
) -> EinsumProjection:
    """Evaluate the output RVEs and check the result lies in ``CS^Z``."""
    coord = tuple(
        evaluate_rank_expression(r, point, shape_env=shape_env, registry=registry)
        for r in output_ranks
    )
    return EinsumProjection(
        point=ordered_point,
        output_coord=coord,
        defined=output_space.contains(coord),
    )
