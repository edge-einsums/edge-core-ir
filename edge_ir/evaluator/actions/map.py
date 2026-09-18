"""The Map action.

Mirrors the EDGE semantics of Map.

Map is a partial function ``IS -> DS^Z_ne``. At one point it:

1. applies the Einsum projection to get the operand coordinate tuples,
2. applies the existence predicate to each operand,
3. hands the two Booleans to the **merge** operator, which decides whether a
   computation happens at all,
4. if merge said yes, applies the **compute** operator to the operand values
   -- substituting a missing operand's empty value, since merge let the point
   through despite it,
5. and drops the result if it lands in ``E^Z``.

Merge decides existence; compute decides value. The two never mix.

The EDGE semantics present Map for a two-input Einsum "for clarity". The IR's
expression tree nests, so this module evaluates the tree bottom-up: every
node yields a :class:`MapResult` (a value plus a presence flag), a leaf
``InputTensor`` gets its presence from the tensor's existence predicate, and
each ``BinaryApp`` applies the merge/compute pair from the ``MapSpec`` its
label names. A unary operator is a degenerate Map, per paper Section 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from edge_ir.evaluator.context import MapContext
from edge_ir.evaluator.environment import Environment
from edge_ir.evaluator.errors import (
    UnderSpecifiedProgramError,
    UnsupportedConstructError,
)
from edge_ir.evaluator.projection import PointMap, project
from edge_ir.evaluator.spaces import Point, is_empty_value
from edge_ir.ir.actions import ComputationSpec, MapSpec
from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    Expression,
    InputTensor,
    RankValue,
    UnaryApp,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    BuiltinUnaryOp,
    ComputeOp,
    MergeOp,
    UnaryOp,
    UserDefinedComputeOp,
    UserDefinedMergeOp,
    UserDefinedUnaryOp,
)
from edge_ir.runtime.op_properties import get_merge_props
from edge_ir.udf.categories import UdfCategory


@dataclass(frozen=True)
class MapResult:
    """One node's contribution at one point: a value and whether it exists.

    ``present=False`` means the (sub)expression produced no value here --
    the partial function is undefined at this point. ``value`` then carries
    the relevant empty value, which is what an enclosing merge will hand to
    an enclosing compute if it lets the point through anyway.
    """

    value: Any
    present: bool
    coord: Point | None = None
    empty: Any = None


def merge_decides(op: MergeOp, left: bool, right: bool, *, where: str) -> bool:
    """Apply a merge operator's truth table to the two presence flags."""
    if isinstance(op, BuiltinMergeOp):
        return get_merge_props(op.symbol).truth_table[(left, right)]
    if isinstance(op, UserDefinedMergeOp):
        raise UnderSpecifiedProgramError(
            f"{where} uses the user-defined merge operator {op.name!r}, but "
            f"merge operators are built-in only: the 16 truth tables in "
            f"edge_ir/runtime/op_properties.py. Use one of those."
        )
    raise UnderSpecifiedProgramError(f"unknown merge operator: {type(op).__name__}")


def apply_compute(
    op: ComputeOp,
    left: Any,
    right: Any,
    *,
    env: Environment,
    category: UdfCategory,
    ctx: Any,
    where: str,
) -> Any:
    """Apply a compute operator to two operand values."""
    if isinstance(op, BuiltinComputeOp):
        return _builtin_compute(op.symbol, left, right, where=where)
    if isinstance(op, UserDefinedComputeOp):
        impl = env.functions.impl(category, op.name)
        return impl.call(left, right, ctx=ctx)
    raise UnderSpecifiedProgramError(f"unknown compute operator: {type(op).__name__}")


def _builtin_compute(symbol: str, left: Any, right: Any, *, where: str) -> Any:
    try:
        if symbol == "+":
            return left + right
        if symbol == "-":
            return left - right
        if symbol == "*":
            return left * right
        if symbol == "/":
            if right == 0:
                raise UnderSpecifiedProgramError(
                    f"{where}: division by zero in the built-in compute operator '/'."
                )
            return left / right
    except TypeError as exc:
        raise UnderSpecifiedProgramError(
            f"{where}: built-in compute operator {symbol!r} cannot combine "
            f"{left!r} and {right!r}."
        ) from exc
    raise UnderSpecifiedProgramError(f"unknown built-in compute operator {symbol!r}")


def apply_unary(op: UnaryOp, value: Any, *, env: Environment, where: str) -> Any:
    if isinstance(op, BuiltinUnaryOp):
        return not value
    if isinstance(op, UserDefinedUnaryOp):
        return env.functions.impl(UdfCategory.UNARY, op.name).call(value)
    raise UnderSpecifiedProgramError(f"unknown unary operator: {type(op).__name__}")


def _reject_nested_reduction(
    expr: AnonymousTensor, *, axes: tuple[str, ...], where: str
) -> None:
    """Refuse an anonymous tensor whose rank list contracts an axis away.

    ``(A_{m,k} . X_k)_m`` names an anonymous tensor over ``m`` alone while its
    inner expression ranges over ``m`` and ``k``, so ``k`` is reduced INSIDE
    the parentheses. The reference evaluator applies its single Reduce spec at
    the Einsum's output coordinate, not at an inner level, so it cannot honour
    that. Evaluating the inner expression pointwise and keeping the rank list
    unread would broadcast instead of contracting -- a wrong number, silently.
    """
    from edge_ir.analysis.iteration_space import coord_image, operand_subscripts

    declared: set[str] = set()
    for rank in expr.ranks:
        declared |= coord_image(rank).axes
    mentioned: set[str] = set()
    for subscript in operand_subscripts(expr.expression):
        mentioned |= coord_image(subscript).axes

    dropped = sorted((mentioned & set(axes)) - declared)
    if dropped:
        raise UnsupportedConstructError(
            f"{where}: the anonymous tensor is written over "
            f"({', '.join(sorted(declared)) or 'no ranks'}) but its inner "
            f"expression ranges over {sorted(mentioned)}, so "
            f"{dropped} would be reduced away inside the parentheses. This "
            f"evaluator applies an Einsum's Reduce spec at the output "
            f"coordinate, not at a nested level, so it cannot evaluate that "
            f"contraction -- and it will not broadcast instead, because that "
            f"silently computes a different quantity. Split the inner "
            f"expression into its own Einsum, which names the intermediate "
            f"and gives the reduction somewhere to attach."
        )


def _derived_unary_merge(op: UnaryOp, *, env: Environment, where: str) -> MergeOp:
    """The merge of the degenerate Map a unary sugars into, when unstated.

    An operator that maps empty to empty visits only the non-empty points
    of its operand (``take_left``); one that maps empty to non-empty must
    visit the points where the operand is empty (``not_left``). The
    built-in ``not`` is the EDGE semantics' own example of the second kind.
    """
    if isinstance(op, BuiltinUnaryOp):
        return BuiltinMergeOp(symbol="not_left")
    if isinstance(op, UserDefinedUnaryOp):
        decl = env.functions.decl(UdfCategory.UNARY, op.name)
        if decl.maps_empty_to_empty:
            return BuiltinMergeOp(symbol="take_left")
        return BuiltinMergeOp(symbol="not_left")
    raise UnderSpecifiedProgramError(
        f"{where}: cannot derive a merge operator for the unary "
        f"{type(op).__name__}; set UnaryApp.merge_op explicitly."
    )


def _spec_index(specs: list[ComputationSpec]) -> dict[int, MapSpec]:
    by_label: dict[int, MapSpec] = {}
    for spec in specs:
        if isinstance(spec, MapSpec):
            if spec.label in by_label:
                raise UnderSpecifiedProgramError(
                    f"two Map specs share the label {spec.label}; a binary's "
                    f"label must name exactly one Map spec."
                )
            by_label[spec.label] = spec
    return by_label


def map_action(
    expression: Expression,
    point: PointMap,
    ordered_point: Point,
    *,
    axes: tuple[str, ...],
    specs: list[ComputationSpec],
    output_empty: Any,
    env: Environment,
    where: str,
) -> MapResult:
    """Evaluate one Einsum's expression tree at one iteration-space point."""
    return _eval(
        expression,
        point,
        ordered_point,
        axes=axes,
        map_specs=_spec_index(specs),
        output_empty=output_empty,
        env=env,
        where=where,
    )


def _eval(
    expr: Expression,
    point: PointMap,
    ordered_point: Point,
    *,
    axes: tuple[str, ...],
    map_specs: dict[int, MapSpec],
    output_empty: Any,
    env: Environment,
    where: str,
) -> MapResult:
    if isinstance(expr, InputTensor):
        name = expr.proj.tensor
        tensor = env.tensor(name)
        coord = project(
            expr.proj,
            point,
            shape_env=env.shape_env,
            registry=env.functions,
        )
        present = tensor.exists(coord)
        return MapResult(
            value=tensor.value_or_empty(coord),
            present=present,
            coord=coord,
            empty=tensor.empty_value,
        )

    if isinstance(expr, UnaryApp):
        # A unary is syntactic sugar for a degenerate Map against the
        # all-ones tensor 1 (EDGE's decomposition of unary functions):
        #     not B_m   ==   (B_m .^2 1)_m :: map not(not_left)
        # The right operand is 1, so it is present at every point; the
        # merge operator therefore decides presence purely from the
        # operand's, and WHICH merge it is depends on how the unary treats
        # the empty value. Getting this wrong inverts every mask in the
        # BFS family, so it is never guessed silently: the IR carries it,
        # and when the IR leaves it None it is derived from the operator.
        operand = _eval(
            expr.operand,
            point,
            ordered_point,
            axes=axes,
            map_specs=map_specs,
            output_empty=output_empty,
            env=env,
            where=where,
        )
        merge = expr.merge_op or _derived_unary_merge(expr.op, env=env, where=where)
        guard = merge_decides(merge, operand.present, True, where=f"{where} (unary)")
        if not guard:
            return MapResult(value=operand.empty, present=False, empty=operand.empty)
        value = apply_unary(expr.op, operand.value, env=env, where=where)
        return MapResult(
            value=value,
            present=True,
            coord=operand.coord,
            empty=operand.empty,
        )

    if isinstance(expr, BinaryApp):
        spec = map_specs.get(expr.label)
        if spec is None:
            raise UnderSpecifiedProgramError(
                f"{where}: the binary labelled {expr.label} has no Map spec. "
                f"A binary combines two operands pointwise, so it needs a "
                f"compute operator and a merge operator; the Einsum's specs "
                f"carry Map labels {sorted(map_specs)}."
            )
        left = _eval(
            expr.lhs,
            point,
            ordered_point,
            axes=axes,
            map_specs=map_specs,
            output_empty=output_empty,
            env=env,
            where=where,
        )
        right = _eval(
            expr.rhs,
            point,
            ordered_point,
            axes=axes,
            map_specs=map_specs,
            output_empty=output_empty,
            env=env,
            where=where,
        )
        guard = merge_decides(
            spec.merge_op,
            left.present,
            right.present,
            where=f"{where} (binary {expr.label})",
        )
        if not guard:
            return MapResult(value=output_empty, present=False, empty=output_empty)
        ctx = MapContext(
            point=ordered_point,
            axes=axes,
            left_coord=left.coord,
            right_coord=right.coord,
            left_present=left.present,
            right_present=right.present,
            left_empty=left.empty,
            right_empty=right.empty,
            output_empty=output_empty,
        )
        value = apply_compute(
            spec.compute_op,
            left.value,
            right.value,
            env=env,
            category=UdfCategory.MAP_COMPUTE,
            ctx=ctx,
            where=f"{where} (binary {expr.label})",
        )
        # "If the computed value lies in the empty space of the output, the
        # MapTmp element is simply omitted at this point."
        return MapResult(
            value=value,
            present=not is_empty_value(value, output_empty),
            empty=output_empty,
        )

    if isinstance(expr, RankValue):
        # A rank expression read as a value ("Rank Variables as Tensors",
        # in EDGE): at each point the coordinate itself is
        # cast into the data space. Every point of the iteration space
        # yields a coordinate, so the operand exists everywhere.
        from edge_ir.evaluator.projection import evaluate_rank_expression

        value = evaluate_rank_expression(
            expr.expr, point, shape_env=env.shape_env, registry=env.functions
        )
        return MapResult(value=value, present=True, empty=None)

    if isinstance(expr, AnonymousTensor):
        # An anonymous tensor gives an inner expression an output shape. Its
        # binaries' specs live on the enclosing Einsum (a flat label
        # namespace), so evaluating it at this point is just evaluating the
        # inner expression -- PROVIDED its rank list is a re-projection of the
        # same point rather than a fresh reduction. When the rank list drops an
        # axis the inner expression mentions, a contraction was written at this
        # level, and this evaluator does not implement nested reduction. It
        # says so rather than broadcasting, which would silently return a
        # number that is not the one the Einsum asks for.
        _reject_nested_reduction(expr, axes=axes, where=where)
        inner = _eval(
            expr.expression,
            point,
            ordered_point,
            axes=axes,
            map_specs=map_specs,
            output_empty=output_empty,
            env=env,
            where=where,
        )
        return inner

    raise UnderSpecifiedProgramError(
        f"unknown Expression variant: {type(expr).__name__}"
    )
