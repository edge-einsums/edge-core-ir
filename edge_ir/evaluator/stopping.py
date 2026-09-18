"""Evaluating stopping conditions.

A stopping condition halts a cascade's iterative rank when its predicate
holds. The IR carries the predicate (``edge_ir/ir/stopping.py``); the
exclusion-guard semantics live here, per that module's own note: the diamond
is the paper's exclusion guard ``RV_i = { i | ... and not diamond(i) }``, not
a `break`.

Two readings the evaluator has to pin down, both recorded as open questions
in ``docs/stopping_condition_open_questions.md``:

- **Timing (Q5).** Every worked example writes the predicate over generation
  ``i+1`` (``||F_{i+1}|| == 0``, ``D_{i+1} == D_i``), which does not exist
  until generation ``i`` has run. So the predicate is evaluated with ``i``
  bound to the generation just completed, and a true result stops the
  cascade there.
- **Comparison lifting (Q3).** A comparison between multi-element operands
  holds iff *every* point of the shared space satisfies it. That is the only
  reading the paper's examples support -- each is a convergence test.

``occupancy`` (``||T||``) counts the non-empty coordinates of an operand
with the pinned ranks held fixed. It is the only built-in property, which
matches the paper exactly.
"""

from __future__ import annotations

from typing import Any

from edge_ir.evaluator.environment import Environment
from edge_ir.evaluator.errors import UnderSpecifiedProgramError
from edge_ir.evaluator.projection import evaluate_rank_expression
from edge_ir.evaluator.spaces import Coordinate, Point, sort_key
from edge_ir.ir.expr import TensorProjection
from edge_ir.ir.stopping import (
    Comparison,
    ComparisonOp,
    DiamondBooleanApp,
    Predicate,
    PropertyApp,
    RankExpressionValue,
    StoppingCondition,
    TensorProjectionValue,
    ValueExpression,
)
from edge_ir.udf.categories import UdfCategory

# A value expression evaluates either to a single value, or to a slice: a
# mapping from the coordinates left free by the projection to their values.
Slice = dict[Point, Any]


def _prefix(
    proj: TensorProjection,
    bound: dict[str, Coordinate],
    *,
    env: Environment,
) -> tuple[Point, int]:
    """Evaluate a projection's subscripts; return them and how many there are.

    A stopping-condition projection is often a *partial* application:
    ``F_{i+1}`` names only the first rank of ``F^{I,S}``. The remaining
    ranks stay free, and the result is a fiber.
    """
    coords = tuple(
        evaluate_rank_expression(
            r, bound, shape_env=env.shape_env, registry=env.functions
        )
        for r in proj.ranks
    )
    return coords, len(coords)


def _slice_of(
    proj: TensorProjection, bound: dict[str, Coordinate], *, env: Environment
) -> Slice:
    """The fiber a projection names, as {free coordinates: value}."""
    tensor = env.tensor(proj.tensor)
    prefix, fixed = _prefix(proj, bound, env=env)
    out: Slice = {}
    for coord, value in tensor.items():
        if coord[:fixed] == prefix:
            out[coord[fixed:]] = value
    return out


def _occupancy(
    node: PropertyApp, bound: dict[str, Coordinate], *, env: Environment
) -> int:
    if len(node.operands) != 1:
        raise UnderSpecifiedProgramError(
            f"occupancy was applied to {len(node.operands)} operands. "
            f"||.|| is defined as the non-empty count of ONE tensor; the IR "
            f"carries no operator saying how several operands combine. Name "
            f"the combined expression in its own Einsum and take the "
            f"occupancy of that."
        )
    return len(_slice_of(node.operands[0], bound, env=env))


def _evaluate_value(
    node: ValueExpression, bound: dict[str, Coordinate], *, env: Environment
) -> tuple[Slice | None, Any]:
    """Evaluate a value expression to either a slice or a single value."""
    if isinstance(node, PropertyApp):
        return None, _occupancy(node, bound, env=env)
    if isinstance(node, RankExpressionValue):
        return None, evaluate_rank_expression(
            node.expr, bound, shape_env=env.shape_env, registry=env.functions
        )
    if isinstance(node, TensorProjectionValue):
        tensor = env.tensor(node.proj.tensor)
        prefix, fixed = _prefix(node.proj, bound, env=env)
        if fixed == len(tensor.space.ranks):
            # Fully applied: a single value (a scalar literal, or one cell).
            return None, tensor.value_or_empty(prefix)
        return _slice_of(node.proj, bound, env=env), None
    raise UnderSpecifiedProgramError(
        f"unknown ValueExpression variant: {type(node).__name__}"
    )


def _compare_scalars(op: ComparisonOp, lhs: Any, rhs: Any) -> bool:
    if op == "==":
        return bool(lhs == rhs)
    if op == "!=":
        return bool(lhs != rhs)
    try:
        if op == ">":
            return bool(lhs > rhs)
        if op == "<":
            return bool(lhs < rhs)
        if op == ">=":
            return bool(lhs >= rhs)
        if op == "<=":
            return bool(lhs <= rhs)
    except TypeError as exc:
        raise UnderSpecifiedProgramError(
            f"cannot order {lhs!r} {op} {rhs!r} in a stopping condition."
        ) from exc
    raise UnderSpecifiedProgramError(f"unknown comparison operator {op!r}")


def evaluate_predicate(
    predicate: Predicate,
    bound: dict[str, Coordinate],
    *,
    env: Environment,
    empty_lhs: Any = None,
    empty_rhs: Any = None,
) -> bool:
    """Evaluate a stopping-condition predicate under the given binding."""
    if isinstance(predicate, Comparison):
        lhs_slice, lhs_value = _evaluate_value(predicate.lhs, bound, env=env)
        rhs_slice, rhs_value = _evaluate_value(predicate.rhs, bound, env=env)
        if lhs_slice is None and rhs_slice is None:
            return _compare_scalars(predicate.op, lhs_value, rhs_value)
        # "All points satisfy" lifting. A missing coordinate on either side
        # contributes its tensor's empty value, so a shape difference shows
        # up as a disagreement rather than being silently skipped.
        keys = sorted(
            set(lhs_slice or {}) | set(rhs_slice or {}),
            key=lambda c: tuple(sort_key(x) for x in c),
        )
        for key in keys:
            left = lhs_slice.get(key, empty_lhs) if lhs_slice is not None else lhs_value
            right = (
                rhs_slice.get(key, empty_rhs) if rhs_slice is not None else rhs_value
            )
            if not _compare_scalars(predicate.op, left, right):
                return False
        return True

    if isinstance(predicate, DiamondBooleanApp):
        impl = env.functions.impl(UdfCategory.BOOLEAN, predicate.name)
        operands = [_slice_of(p, bound, env=env) for p in predicate.operands]
        return bool(impl.call(*operands))

    raise UnderSpecifiedProgramError(
        f"unknown stopping Predicate variant: {type(predicate).__name__}"
    )


def should_stop(
    condition: StoppingCondition,
    generation: int,
    *,
    env: Environment,
) -> bool:
    """True when this condition halts the cascade after ``generation``.

    ``condition.rank_variable`` is bound to ``generation`` -- the one just
    completed -- so a predicate written over ``i+1`` reads the generation
    that generation produced.
    """
    bound: dict[str, Coordinate] = {condition.rank_variable: generation}
    empties = _comparison_empties(condition.predicate, env)
    return evaluate_predicate(
        condition.predicate, bound, env=env, empty_lhs=empties[0], empty_rhs=empties[1]
    )


def _comparison_empties(predicate: Predicate, env: Environment) -> tuple[Any, Any]:
    if not isinstance(predicate, Comparison):
        return None, None

    def empty_of(node: ValueExpression) -> Any:
        if isinstance(node, TensorProjectionValue):
            return env.empty_value(node.proj.tensor)
        return None

    return empty_of(predicate.lhs), empty_of(predicate.rhs)
