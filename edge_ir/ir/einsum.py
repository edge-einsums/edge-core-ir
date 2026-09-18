"""Einsums and nested cascades.

Per the grammar:
    <einsum> := <iteration-spec> | <iteration-spec> "::" <computation-spec-list>
    <iteration-spec> := <output-tensor> "=" <expression>

    <einsum-cascade> := <einsum-cascade> <einsum> | <einsum>

    <nested-cascade> := <einsum-cascade>
                      | <nested-cascade> <stopping-condition>

The <stopping-condition> production is structured (a predicate tree,
not just a boolean-function name) and is defined in
edge_ir/ir/stopping.py; it is re-exported from this module for
backwards compatibility of the import path.

A named Einsum has a tensor projection as output and shares structure
with AnonymousTensor (defined in expr.py): both have an expression and
an output rank-expression list. The named Einsum additionally has the
output tensor name and the computation-spec list; AnonymousTensor does
not carry specs (all specs for an Einsum live at the Einsum level, in
a single flat label namespace covering every binary in the expression
tree, including those nested inside AnonymousTensors).

The NestedCascade is the top-level structure of the main Einsum
section. It's a sequence of einsums optionally followed by stopping
conditions for iteration.

Per paper Section 7.3.1, when a stopping condition is present, default
generational-rank-variable conventions apply: the variable is `i`,
its initial value is 0, and its rank-variable set is implicit. The IR
does not encode these defaults; consumers apply them when needed.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from edge_ir.ir import patterns
from edge_ir.ir.actions import ComputationSpec
from edge_ir.ir.base import IRBase
from edge_ir.ir.expr import Expression, RankExpression
from edge_ir.ir.predicate import IterationPredicate
from edge_ir.ir.stopping import StoppingCondition
from edge_ir.ir.tensor import CoordinateSet

__all__ = [
    "Einsum",
    "IterativeRankSpec",
    "NestedCascade",
    "StoppingCondition",
]

################################################################################
# Einsum
################################################################################


class Einsum(IRBase):
    """A single Einsum: output = expression, with computation specs.

    Per the grammar:
        <einsum> := <iteration-spec>
                  | <iteration-spec> "::" <computation-spec-list>
        <iteration-spec> := <output-tensor> "=" <expression>

    The output is a tensor projection (a tensor name plus its rank
    expressions). The expression is the right-hand side. The specs are
    the list of computation specs (Map, Reduce, Populate) that apply
    to the binary operations within the expression. The specs list may
    be empty (when the expression is just an InputTensor or a UnaryApp,
    no binary operations exist).

    Specs live here, at the Einsum level, even for binaries that appear
    inside a nested AnonymousTensor in the expression: binary labels
    form a single flat namespace within an Einsum, and each BinaryApp's
    label links to the matching ComputationSpec in this list regardless
    of nesting depth.

    This shares structure with AnonymousTensor (in expr.py): both have
    an expression and an output rank-expression list. They differ in
    that Einsum has an output_tensor name (mandatory) and the specs
    list, while AnonymousTensor has neither (its output is the
    anonymous tensor itself, used as an operand by the surrounding
    expression, and its binaries' specs live on the enclosing Einsum).
    """

    output_tensor: str = Field(pattern=patterns.REGEX_TENSOR_NAME)
    output_ranks: list[RankExpression]
    expression: Expression
    specs: list[ComputationSpec]
    predicates: list[IterationPredicate] = []

    @model_validator(mode="after")
    def _predicates_list_zero_or_one(self) -> Einsum:
        """An Einsum can hold at most one predicate in its `predicates` list.

        Why the limit: we want exactly one canonical way to write "this is my
        iteration restriction." If we let the list hold any number of items
        and said "they're all ANDed together," then a predicate with two
        clauses could be written two ways — as two list items, or as one
        LogicalAnd of two children — and the IR would have two shapes for
        the same meaning. That's bad for tools that compare or rewrite IRs.

        To restrict iteration with multiple conditions, build one combined
        predicate using LogicalAnd, LogicalOr, or LogicalNot, and put that
        single combined predicate in the list. The list is either empty (no
        restriction) or holds exactly one predicate (possibly a composition).
        """
        if len(self.predicates) > 1:
            raise ValueError(
                f"Einsum.predicates has {len(self.predicates)} items, "
                f"but it can hold at most 1. To combine multiple conditions, "
                f"wrap them in a LogicalAnd, LogicalOr, or LogicalNot so "
                f"the whole thing fits in one slot."
            )
        return self


################################################################################
# Iterative rank
################################################################################


class IterativeRankSpec(IRBase):
    """The three things an iterative cascade specifies about its rank.

    The EDGE syntax: "If a cascade is
    iterative, we additionally specify (a) a generational rank variable,
    (b) its initial value, and (c) its rank variable set. If these are
    omitted, the default generational rank variable is `i`, the default
    initial value is `0`, and the default rank variable set is
    `RV_i = { i | i in [0, I) and not (diamond)(i) }`."

    All three defaults are real defaults, so `NestedCascade.iterative_rank`
    may be omitted entirely and consumers apply them. It is present when a
    cascade departs from them -- max-flow's preflow, for instance, indexes
    its first computed generation `1`, not `0`.

    `coord_set` is the rank variable set. When `None` the set is the default
    above: the whole numbers, less the generations the diamond excludes.
    """

    rank_variable: str = Field(pattern=patterns.REGEX_RANK_VARIABLE)
    initial_value: int = 0
    coord_set: CoordinateSet | None = None


################################################################################
# NestedCascade
################################################################################


class NestedCascade(IRBase):
    """A nested cascade: a sequence of Einsums with optional stopping
    conditions for iteration.

    Per the grammar:
        <nested-cascade> := <einsum-cascade>
                          | <nested-cascade> <stopping-condition>
        <einsum-cascade> := <einsum-cascade> <einsum> | <einsum>

    Conceptually, a NestedCascade is a list of Einsums (the cascade)
    followed by zero or more stopping conditions that apply to the
    iteration of the cascade.

    The IR represents a NestedCascade flatly: a list of einsums and a
    list of stopping conditions. The grammar's recursive shape
    (NestedCascade -> NestedCascade StoppingCondition) is normalized to
    this flat form by the lowering pass.

    For nested cascades within nested cascades (the "nested" in the
    name): the einsums list can contain Einsums whose expressions
    contain AnonymousTensors. There is no separate "sub-cascade" type
    at this level; nesting happens through AnonymousTensor in the
    expression tree. The specs for binaries inside a nested
    AnonymousTensor live on the enclosing Einsum's specs list (a flat
    label namespace) — they were never part of this cascade's
    structure either way; the cascade only holds the list of Einsums
    and their stopping conditions.
    """

    einsums: list[Einsum] = Field(min_length=1)
    stopping_conditions: list[StoppingCondition] = []
    iterative_rank: IterativeRankSpec | None = None
    """The cascade's iterative rank, when it departs from the defaults.

    `None` means the paper's defaults apply: rank variable `i`, initial
    value `0`, rank variable set the whole numbers less the generations the
    diamond excludes. See :class:`IterativeRankSpec`.
    """
