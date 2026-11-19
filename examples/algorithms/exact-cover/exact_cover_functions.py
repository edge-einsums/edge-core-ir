"""The user-defined operators Algorithm X names, registered for the evaluator.

The IR carries operator names only; a FunctionRegistry gives them meaning.
The default registry already defines everything this cascade uses -- including
``select-max-val`` and ``select-min-val`` for the two populates, ``floor`` for
the depth extraction, and ``ANY`` and ``update`` -- except one.

``take_left_if_less``: E04's operator
-------------------------------------
The tutorial writes the path truncation as::

    TPath_{i,r} = Path_{i,r} . d*_i :: /\\ <(^)

and the bare ``<`` is ambiguous in a way that matters. The registry's ``lt``
returns a BOOLEAN, which would make TPath a mask. But TPath is fed straight
into E06's partial update to form ``Path_{i+1}``, whose entries are DEPTHS,
and the tutorial's own worked example settles it: at i = 2 the path is
{R3:1, R2:2} with d* = 2, and Path_3 is {R3:1, R0:2}. The surviving entry kept
its depth of 1. It did not become True.

So the operator is a FILTERED take_left: the left value where it is less than
the right, absent otherwise.

WHY IT IS NOT REGISTERED AS AN ALIAS OF `lt`, AND NOT NAMED `lt`
----------------------------------------------------------------
Because it is not ``lt``, and calling it ``lt`` would make the artifact claim
the tutorial says something it does not. The workbench's operator-comparison
gate reads the surface ``einsum.edge`` and the IR side by side and reports any
operator that differs -- that gate exists because a silently substituted merge
operator once passed every other check. Emitting a distinct name here means
the gate reports this divergence, and it should keep reporting it until the
tutorial's author says which reading is intended.

A filtered take_left may well deserve to be a builtin; three of the
comparisons would have the same shape. That is a question for the registry,
not something to settle inside one algorithm's function file.

Import note: ``edge_ir.evaluator`` must be imported before ``edge_ir.udf``.
The reverse order hits a circular import.
"""

from __future__ import annotations

import edge_ir.evaluator  # noqa: F401  (must precede edge_ir.udf)
from edge_ir.udf import FunctionRegistry, UdfCategory, default_registry

TAKE_LEFT_IF_LESS = "take_left_if_less"

def take_left_if_less(left, right, *, ctx):
    """The left operand's value where it is strictly less than the right.

    HOW A COMPUTE OPERATOR SAYS "NOTHING HERE"
    ------------------------------------------
    By returning the OUTPUT TENSOR'S EMPTY VALUE, which `ctx.output_empty`
    supplies. A compute operator's return is stored as the value at that
    point; there is no separate absence channel, and the empty-is-absent rule
    is what turns a value into no-entry.

    Returning None instead does not mean absent -- it stores None, which then
    reaches the next operator as an operand and raises there rather than here.
    That is how this was first written, and E07's `ge` is where it surfaced,
    two Einsums downstream of the mistake.

    Reading the empty value from the context rather than writing -1 is what
    keeps this operator usable on a tensor whose empty value is something
    else.
    """
    if left is None or right is None:
        return ctx.output_empty
    return left if left < right else ctx.output_empty


def floor_left(left, right):
    """`floor` as the compute operator of a degenerate Map (paper 5.5):
    the floor of the left operand; the right operand is the `1` that the
    desugaring of a unary supplies, and is ignored."""
    import math

    return None if left is None else math.floor(left)


def exact_cover_registry() -> FunctionRegistry:
    """The default registry plus E04's filtered take_left, and E02's floor as
    the compute operator of a degenerate Map."""
    registry = default_registry()
    registry.register_python(
        "floor",
        UdfCategory.MAP_COMPUTE,
        floor_left,
        arity=2,
        doc="E02: the unary floor, desugared as a Map against 1 (paper 5.5).",
    )
    for category in (UdfCategory.MAP_COMPUTE, UdfCategory.REDUCE_COMPUTE):
        registry.register_python(
            TAKE_LEFT_IF_LESS,
            category,
            take_left_if_less,
            arity=2,
            wants_context=True,
            doc=(
                "E04 path truncation: the left value where left < right, "
                "absent otherwise. The tutorial writes this as `<`; see this "
                "module's docstring for why it is not the boolean `lt`."
            ),
        )
    return registry
