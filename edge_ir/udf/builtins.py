"""The default function registry.

Everything here is a *default*, not a language primitive: each entry is an
ordinary registration that a user could have written, and any of them can
be replaced by passing ``allow_override=True``. The genuine primitives of
EDGE are the four built-in compute symbols (``+ - * /``), the 16 merge
operators and the built-in unary ``not``; those live in the IR and in
``edge_ir/runtime/op_properties.py``, not here.

Two entries deserve their own note.

``update`` is the ``<<`` operator: *the right operand where present, else
carry the left operand forward*. EDGE defines it as a compute
operator, not a merge (EDGE paper appendix),
paired with a union merge. It cannot be spelled ``<<`` in the
IR -- ``REGEX_USER_DEFINED_NAME`` admits no glyphs -- so ``update`` is its
ASCII name. Shipping it is the fix ``TODO.md`` records for the Bellman-Ford
and DFS artifacts, which currently mis-encode it as ``take_right``: those
two differ exactly on the (left present, right absent) case, which is
"distance unchanged this round", i.e. most vertices most rounds.

``ANY`` declares ``identity=EMPTY_IDENTITY``. It selects one of the
contributions reaching an output coordinate, so it has no identity element
of its own; seeding the reduction state with ``e^Z`` makes the merge
operator's left presence flag ``b_s`` False on first touch, which is what
the Reduce semantics describe for a freshly created state.
"""

from __future__ import annotations

import math
from typing import Any

from edge_ir.evaluator.context import PopulateAction, PopulateContext
from edge_ir.evaluator.spaces import is_empty_value
from edge_ir.udf.categories import UdfCategory
from edge_ir.udf.decl import EMPTY_IDENTITY
from edge_ir.udf.registry import FunctionRegistry

################################################################################
# Compute operators
################################################################################


def _empties(ctx: Any) -> tuple[Any, Any]:
    """The empty values of this operator's two operands.

    Under Map the operands are two different tensors, so each has its own
    empty value. Under Reduce both the reduction state and the MapTmp value
    live in the output tensor's data space, so both empties are ``e^Z``.
    """
    left = getattr(ctx, "left_empty", None)
    if hasattr(ctx, "left_empty"):
        return left, ctx.right_empty
    return ctx.output_empty, ctx.output_empty


def take_left(left: Any, right: Any) -> Any:
    """``<-`` -- the left operand's value."""
    return left


def take_right(left: Any, right: Any) -> Any:
    """``->`` -- the right operand's value."""
    return right


def update(left: Any, right: Any, *, ctx: Any) -> Any:
    """``<<`` -- the right operand where present, else carry the left forward."""
    return right if ctx.right_present else left


def logical_or(left: Any, right: Any, *, ctx: Any) -> bool:
    """``OR``, handling the empty value a union merge feeds it.

    A merge operator that admits a point with one operand missing hands the
    compute that operand's EMPTY VALUE (Map and Reduce semantics). A naive
    ``bool(left) or bool(right)`` then reads a missing operand as
    ``bool(e)``, which is wrong whenever the empty value is truthy -- BFS
    declares ``F`` with ``empty = inf``, and ``bool(inf)`` is True, so a
    naive OR would report every coordinate reached. An absent operand
    contributes OR's identity, False.
    """
    left_empty, right_empty = _empties(ctx)
    lhs = False if is_empty_value(left, left_empty) else bool(left)
    rhs = False if is_empty_value(right, right_empty) else bool(right)
    return lhs or rhs


def logical_and(left: Any, right: Any, *, ctx: Any) -> bool:
    """``AND``; an absent operand contributes AND's identity, True."""
    left_empty, right_empty = _empties(ctx)
    lhs = True if is_empty_value(left, left_empty) else bool(left)
    rhs = True if is_empty_value(right, right_empty) else bool(right)
    return lhs and rhs


def any_of(left: Any, right: Any, *, ctx: Any) -> Any:
    """``ANY`` -- select one of the contributions.

    "One of" is only meaningful among the NON-empty contributions: under a
    union merge Reduce invokes compute even at points where MapTmp is
    absent, handing it ``e^Z``, and returning that would discard everything
    accumulated so far. So an empty operand is skipped. When both are
    non-empty either is a legal answer and the incumbent is kept, which
    makes the choice deterministic; a program whose contributions to one
    output coordinate DIFFER is genuinely order-dependent there, and the
    randomized iteration-order mode is what surfaces that.
    """
    left_empty, right_empty = _empties(ctx)
    if is_empty_value(right, right_empty):
        return left
    if is_empty_value(left, left_empty):
        return right
    return left


def _extreme(pick: Any) -> Any:
    """Build an empty-skipping ``min`` / ``max``."""

    def op(left: Any, right: Any, *, ctx: Any) -> Any:
        left_empty, right_empty = _empties(ctx)
        if is_empty_value(right, right_empty):
            return left
        if is_empty_value(left, left_empty):
            return right
        return pick(left, right)

    return op


def _cmp(name: str) -> Any:
    ops = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
    }
    return ops[name]


################################################################################
# Coordinate operators
################################################################################


def assign(fiber: Any, red_coord: Any, red_value: Any, *, ctx: Any) -> Any:
    """Default assignment: Write at the RedTmp's own coordinate.

    This is the no-``*`` case of Populate (Populate semantics,
    default assignment): the fiber holds exactly the one coordinate, and
    the coordinate operator marks it Write.
    """
    del fiber, red_value, ctx
    return [(red_coord, PopulateAction.WRITE)]


def _select(kind: str, extreme: str) -> Any:
    """Build a ``select-{min,max}-{val,coord}`` coordinate operator.

    The fiber's incumbent entries and the incoming RedTmp element compete;
    the winner is kept and every other present coordinate is deleted. The
    ``-val`` forms rank by data value, the ``-coord`` forms by the mutable
    coordinate itself.
    """

    def op(fiber: Any, red_coord: Any, red_value: Any, *, ctx: PopulateContext) -> Any:
        candidates: list[tuple[Any, Any]] = [
            (entry.value, entry.coord) for entry in fiber if entry.present
        ]
        candidates.append((red_value, red_coord))
        if kind == "val":
            key = lambda cv: (cv[0], cv[1])  # noqa: E731
        else:
            key = lambda cv: (cv[1], cv[0])  # noqa: E731
        winner = (max if extreme == "max" else min)(candidates, key=key)
        winning_coord = winner[1]
        actions: list[tuple[Any, PopulateAction]] = []
        for entry in fiber:
            if entry.coord == winning_coord:
                # Already holding the winning value: leave it alone.
                actions.append(
                    (
                        entry.coord,
                        PopulateAction.NONE if entry.present else PopulateAction.WRITE,
                    )
                )
            elif entry.present:
                actions.append((entry.coord, PopulateAction.DELETE))
            else:
                actions.append((entry.coord, PopulateAction.NONE))
        del ctx
        return actions

    return op


################################################################################
# Populate compute operators
################################################################################


def populate_pass_through(target_coord: Any, red_value: Any, *, ctx: Any) -> Any:
    """The assignment compute operator: return the RedTmp's value unchanged."""
    del target_coord, ctx
    return red_value


def populate_coordinate_value(
    target_coord: Any, red_value: Any, *, ctx: PopulateContext
) -> Any:
    """Write the winning mutable COORDINATE as the value.

    This is the value-as-coordinate pattern: a Populate whose result is
    *where* something was found rather than *what* was found. The mutable
    coordinates are read off the target coordinate at the positions the
    Populate spec marked with ``*``; a single mutable rank yields a bare
    coordinate rather than a one-tuple.
    """
    del red_value
    picked = tuple(target_coord[i] for i in ctx.mutable_positions)
    return picked[0] if len(picked) == 1 else picked


################################################################################
# The default registry
################################################################################


def default_registry() -> FunctionRegistry:
    """A registry seeded with the operators the paper's examples use."""
    reg = FunctionRegistry()

    # -- map compute -------------------------------------------------------
    for cat in (UdfCategory.MAP_COMPUTE, UdfCategory.REDUCE_COMPUTE):
        reg.register_python(
            "take_left", cat, take_left, arity=2, doc="left operand's value"
        )
        reg.register_python(
            "take_right", cat, take_right, arity=2, doc="right operand's value"
        )
        reg.register_python(
            "update",
            cat,
            update,
            arity=2,
            wants_context=True,
            identity=EMPTY_IDENTITY,
            doc="<< : right where present, else carry the left forward",
        )
        reg.register_python(
            "OR",
            cat,
            logical_or,
            arity=2,
            identity=False,
            associative=True,
            commutative=True,
        )
        reg.register_python(
            "AND",
            cat,
            logical_and,
            arity=2,
            identity=True,
            associative=True,
            commutative=True,
        )
        reg.register_python(
            "ANY",
            cat,
            any_of,
            arity=2,
            identity=EMPTY_IDENTITY,
            associative=True,
            commutative=True,
            doc="select one of the contributions",
        )
        reg.register_python(
            "min",
            cat,
            _extreme(min),
            arity=2,
            wants_context=True,
            identity=math.inf,
            associative=True,
            commutative=True,
        )
        reg.register_python(
            "max",
            cat,
            _extreme(max),
            arity=2,
            wants_context=True,
            identity=-math.inf,
            associative=True,
            commutative=True,
        )
        # Relational compute operators. The paper writes these as glyphs
        # (`<`, `>`, `==`/`≡`) which the IR cannot spell -- neither
        # BuiltinComputeOp.symbol (which is only + - * /) nor
        # REGEX_USER_DEFINED_NAME admits them -- so each ships under a short
        # name AND the spelled-out name the existing artifacts already use.
        # Note `≡` is ALSO the paper's glyph for the xnor MERGE operator;
        # these are the COMPUTE flavour, and the two must never be confused.
        for cmp_name, alias in (
            ("eq", "equals"),
            ("ne", "not_equals"),
            ("lt", "less_than"),
            ("le", "less_equal"),
            ("gt", "greater_than"),
            ("ge", "greater_equal"),
        ):
            reg.register_python(cmp_name, cat, _cmp(cmp_name), arity=2)
            reg.register_python(alias, cat, _cmp(cmp_name), arity=2)

    # -- populate compute --------------------------------------------------
    reg.register_python(
        "pass_through",
        UdfCategory.POPULATE_COMPUTE,
        populate_pass_through,
        arity=2,
        wants_context=True,
        doc="assignment: the RedTmp value, unchanged",
    )
    reg.register_python(
        "identity",
        UdfCategory.POPULATE_COMPUTE,
        populate_pass_through,
        arity=2,
        wants_context=True,
        doc="alias of pass_through; the ASCII name for the paper's 1 glyph",
    )
    reg.register_python(
        "coordinate_value",
        UdfCategory.POPULATE_COMPUTE,
        populate_coordinate_value,
        arity=2,
        wants_context=True,
        doc="value-as-coordinate: write the winning mutable coordinate",
    )

    # -- coordinate --------------------------------------------------------
    reg.register_python(
        "assign", UdfCategory.COORDINATE, assign, arity=3, wants_context=True
    )
    for extreme in ("min", "max"):
        for kind in ("val", "coord"):
            op = _select(kind, extreme)
            for name in (
                f"select-{extreme}-{kind}",
                f"select_{extreme}_{kind}",
                f"{extreme}-{kind}-1",
            ):
                reg.register_python(
                    name,
                    UdfCategory.COORDINATE,
                    op,
                    arity=3,
                    wants_context=True,
                    doc=f"keep the {extreme}imum {kind}; delete the rest",
                )

    # -- rank mapping ------------------------------------------------------
    reg.register_python("min", UdfCategory.RANK_MAPPING, min, arity=2, is_affine=False)
    reg.register_python("max", UdfCategory.RANK_MAPPING, max, arity=2, is_affine=False)

    return reg
