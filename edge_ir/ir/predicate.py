"""Restricted-iteration predicates.

A predicate restricts which coordinates an einsum walks over. The predicate
sits on Einsum.predicates and is evaluated as part of the iteration-space
construction — see docs/parsing.md and docs/usecase.md for the surface
syntax and worked examples, docs/design_decisions.md D22 for why this is
a separate union from stopping.Predicate.

There are five node types:
- SetMembership: a coordinate expression is in (or not in) a coordinate set.
- PredicateComparison: a comparison between two coordinate-valued expressions.
- LogicalAnd / LogicalOr / LogicalNot: Boolean composition.

The union IterationPredicate discriminates by the "kind" field.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from edge_ir.ir.base import IRBase
from edge_ir.ir.expr import RankExpression
from edge_ir.ir.stopping import ComparisonOp
from edge_ir.ir.tensor import CoordinateSet


class SetMembership(IRBase):
    """A coordinate expression is in (or not in) a coordinate set.

    `member` is the coordinate-valued thing being tested (typically a
    RankVariable like "s", but admits any RankExpression — e.g. RankArith
    "s+1" or RankFunction "f(s)" — so the predicate can carry shifted
    or function-mapped membership tests).

    `coord_set` is any CoordinateSet variant: an explicit enum, an
    interval, an alias to another tensor's rank, or a named runtime set
    (CoordSetName) the host supplies at execution time.

    `negated=False` means ∈; `negated=True` means ∉.
    """

    kind: Literal["set_membership"] = "set_membership"
    member: RankExpression
    coord_set: CoordinateSet
    negated: bool = False


class PredicateComparison(IRBase):
    """A comparison between two coordinate-valued expressions.

    Both lhs and rhs are RankExpression (the existing union: RankVariable
    | RankConstantLiteral | RankConstantShapeSym | RankArith | RankFunction).
    Tensor-projection LHS is structurally rejected — predicates restrict
    iteration based on coordinate values, not tensor values.

    The op alphabet is the same six as Comparison.op in stopping.py
    (==, !=, >, <, >=, <=); ComparisonOp is reused.
    """

    kind: Literal["comparison"] = "comparison"
    lhs: RankExpression
    op: ComparisonOp
    rhs: RankExpression


class LogicalAnd(IRBase):
    """All operand predicates hold."""

    kind: Literal["and"] = "and"
    operands: list[IterationPredicate] = Field(min_length=2)


class LogicalOr(IRBase):
    """At least one operand predicate holds."""

    kind: Literal["or"] = "or"
    operands: list[IterationPredicate] = Field(min_length=2)


class LogicalNot(IRBase):
    """The operand predicate does NOT hold."""

    kind: Literal["not"] = "not"
    operand: IterationPredicate


IterationPredicate = Annotated[
    SetMembership | PredicateComparison | LogicalAnd | LogicalOr | LogicalNot,
    Field(discriminator="kind"),
]
