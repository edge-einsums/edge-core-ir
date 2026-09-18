"""Affine analysis on rank expressions.

A rank expression is *affine* when it can be expressed as an integer
linear combination of rank variables plus a constant. Affine-ness
matters to downstream consumers that prefer (or require) affine maps,
e.g. an MLIR ``affine`` dialect backend.

This pass is a side analysis: results live in a dict keyed by
``id(node)``, NOT on the IR node itself. The core IR (``edge_ir.ir``)
stays frozen and pure-syntax; this module reads it and returns plain
Python values.

Rules (moved here from the old ``RankArith.is_affine`` docstring):

- ``RankVariable`` -> ``AFFINE``
- ``RankConstantLiteral`` -> ``AFFINE``
- ``RankConstantShapeSym`` -> ``AFFINE``
- ``RankArith`` with ``+`` or ``-`` -> ``AFFINE`` iff both children are
  ``AFFINE``
- ``RankArith`` with ``*`` -> ``AFFINE`` iff one child is a
  ``RankConstantLiteral`` or ``RankConstantShapeSym`` and the other
  child is ``AFFINE``
- ``RankArith`` with ``/``, ``//``, or ``%`` -> ``NON_AFFINE``
- ``RankFunction`` -> ``AFFINE`` if its user-declared ``is_affine`` is
  ``True``, ``NON_AFFINE`` if ``False``, otherwise ``UNKNOWN``

A node whose children are a mix of ``AFFINE`` and ``UNKNOWN`` (with no
``NON_AFFINE`` evidence) is ``UNKNOWN``: we cannot prove affine without
knowing the opaque function's behavior, but we also cannot disprove it.
"""

from __future__ import annotations

from enum import Enum

from edge_ir.ir.expr import (
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankExpression,
    RankFunction,
    RankVariable,
)


class Affinity(str, Enum):
    """Result of the affine-analysis pass for one rank-expression node."""

    AFFINE = "affine"
    NON_AFFINE = "non_affine"
    UNKNOWN = "unknown"


def analyze_affinity(node: RankExpression) -> Affinity:
    """Return the :class:`Affinity` of a single rank-expression tree.

    See module docstring for the full rule set.
    """
    if isinstance(node, RankVariable):
        return Affinity.AFFINE
    if isinstance(node, RankConstantLiteral | RankConstantShapeSym):
        return Affinity.AFFINE
    if isinstance(node, RankFunction):
        if node.is_affine is True:
            return Affinity.AFFINE
        if node.is_affine is False:
            return Affinity.NON_AFFINE
        return Affinity.UNKNOWN
    if isinstance(node, RankArith):
        if node.op in ("/", "//", "%"):
            return Affinity.NON_AFFINE
        lhs = analyze_affinity(node.lhs)
        rhs = analyze_affinity(node.rhs)
        if node.op in ("+", "-"):
            return _combine_add_sub(lhs, rhs)
        return _combine_mul(node.lhs, lhs, node.rhs, rhs)
    raise TypeError(f"unknown RankExpression variant: {type(node).__name__}")


def affinity_table(root: RankExpression) -> dict[int, Affinity]:
    """Walk one rank-expression tree, returning ``{id(node): Affinity}``.

    Keys are Python ``id()`` values, so the table is only meaningful
    while every node in ``root`` is still alive in memory. Do NOT
    serialize this table; recompute it after deserialization. If a
    persisted form is ever required, key by structural path (a tuple of
    field-name/index steps from the root) instead of ``id()``.
    """
    table: dict[int, Affinity] = {}
    _walk(root, table)
    return table


def _walk(node: RankExpression, table: dict[int, Affinity]) -> None:
    table[id(node)] = analyze_affinity(node)
    if isinstance(node, RankArith):
        _walk(node.lhs, table)
        _walk(node.rhs, table)
    elif isinstance(node, RankFunction):
        for arg in node.args:
            _walk(arg, table)


def _combine_add_sub(lhs: Affinity, rhs: Affinity) -> Affinity:
    if lhs is Affinity.NON_AFFINE or rhs is Affinity.NON_AFFINE:
        return Affinity.NON_AFFINE
    if lhs is Affinity.AFFINE and rhs is Affinity.AFFINE:
        return Affinity.AFFINE
    return Affinity.UNKNOWN


def _combine_mul(
    lhs_node: RankExpression,
    lhs: Affinity,
    rhs_node: RankExpression,
    rhs: Affinity,
) -> Affinity:
    if lhs is Affinity.NON_AFFINE or rhs is Affinity.NON_AFFINE:
        return Affinity.NON_AFFINE
    lhs_is_const = isinstance(lhs_node, RankConstantLiteral | RankConstantShapeSym)
    rhs_is_const = isinstance(rhs_node, RankConstantLiteral | RankConstantShapeSym)
    if lhs_is_const and rhs is Affinity.AFFINE:
        return Affinity.AFFINE
    if rhs_is_const and lhs is Affinity.AFFINE:
        return Affinity.AFFINE
    if lhs_is_const and rhs is Affinity.UNKNOWN:
        return Affinity.UNKNOWN
    if rhs_is_const and lhs is Affinity.UNKNOWN:
        return Affinity.UNKNOWN
    return Affinity.NON_AFFINE
