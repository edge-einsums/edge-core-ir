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

from typing import Annotated, Literal

from pydantic import Field, model_validator

from edge_ir.ir import patterns
from edge_ir.ir.actions import ComputationSpec
from edge_ir.ir.base import IRBase
from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    Expression,
    InputTensor,
    Occupancy,
    RankExpression,
    RankValue,
    UnaryApp,
)
from edge_ir.ir.predicate import IterationPredicate
from edge_ir.ir.stopping import Predicate, StoppingCondition
from edge_ir.ir.tensor import CoordinateSet

__all__ = [
    "CaseArm",
    "Cases",
    "ConditionalArm",
    "ConditionalCascade",
    "Einsum",
    "IterativeRankSpec",
    "NestedCascade",
    "StoppingCondition",
    "flatten",
]

################################################################################
# Case statements (LOCAL PATCH, edge-workbench issue 8)
################################################################################


class CaseArm(IRBase):
    """One branch of a case statement.

    EDGE's selective computations: "Each branch of a case statement is
    itself a valid EDGE expression, with its own modifier if needed. At each
    point in the iteration space, [the statement] selects the appropriate
    sub-Einsum expression."

    So an arm carries its own expression AND its own Map/Reduce specs (its
    binary labels are a namespace of its own), plus the condition saying
    which region of the iteration space it covers:

    - ``where``: a predicate over rank variables (``n == j+1``). The region
      is known before any tensor is read -- docs/cases.md, Pattern 1.
    - ``when``: a predicate over tensor values (``TopDown_i == True``),
      read at each point -- docs/cases.md, Pattern 2. "EDGE allows any
      Boolean predicate as conditions for each case."
    - both: their conjunction (Pattern 3). Neither: ``otherwise``.

    ``expression=None`` is the arm written ``empty``: the region is named,
    and nothing is written there.
    """

    expression: Expression | None = None
    specs: list[ComputationSpec] = []
    where: IterationPredicate | None = None
    when: Predicate | None = None


class Cases(IRBase):
    """A case statement: different expressions for different regions of one
    output tensor's iteration space, in ONE Einsum.

    Per the paper a case statement is syntactic sugar: each arm is an Einsum
    into a temporary restricted to its region, and the temporaries are
    combined with the update operator. Kept as a node (the "middle path" of
    docs/cases.md) so the Einsum the author wrote stays one Einsum writing
    its output once, which is what single assignment asks for.

    Order convention (docs/cases.md asks for one): arms are tested in the
    order written and a point belongs to the FIRST arm whose condition holds,
    so ``otherwise`` -- an arm with no condition -- is whatever no earlier arm
    claimed.

    Only valid as the whole right-hand side of an Einsum
    (``Einsum.expression``); it is deliberately not a member of the
    ``Expression`` union, so it cannot be an operand.
    """

    kind: Literal["cases"] = "cases"
    arms: list[CaseArm] = Field(min_length=1)


EinsumExpression = Annotated[
    InputTensor
    | UnaryApp
    | BinaryApp
    | AnonymousTensor
    | RankValue
    | Occupancy
    | Cases,
    Field(discriminator="kind"),
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
    expression: EinsumExpression
    """The right-hand side. A ``Cases`` here is a case statement: each arm
    carries its own Map/Reduce specs, and ``specs`` below then holds at most
    the Populate spec, which applies to the Einsum as a whole."""
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

    einsums: list[Einsum | ConditionalCascade | NestedCascade] = Field(min_length=1)
    """The cascade's body, in order. An entry is an Einsum, a conditional
    cascade (one of several Einsum sequences, chosen per generation), or an
    inner cascade with its own iterative rank and stopping condition (LOCAL
    PATCH, edge-workbench issues 9 and 10). :func:`flatten` gives the
    Einsums alone, in written order."""
    stopping_conditions: list[StoppingCondition] = []
    iterative_rank: IterativeRankSpec | None = None
    """The cascade's iterative rank, when it departs from the defaults.

    `None` means the paper's defaults apply: rank variable `i`, initial
    value `0`, rank variable set the whole numbers less the generations the
    diamond excludes. See :class:`IterativeRankSpec`.
    """


################################################################################
# Conditional cascades (LOCAL PATCH, edge-workbench issue 9)
################################################################################


class ConditionalArm(IRBase):
    """One alternative of a conditional cascade: a condition and the Einsums
    that run when it holds.

    ``when`` is a predicate over tensor values, evaluated once, with the
    enclosing cascades' iterative ranks bound to the current generation,
    BEFORE any Einsum of the arm runs. ``None`` is ``otherwise``.

    ``macro`` records which macro the arm's Einsums were expanded from
    (``edge_ir.ir.macro``), for tools that show the program; it has no
    effect on evaluation, because a macro is a lexical substitution.
    """

    when: Predicate | None = None
    einsums: list[Einsum | ConditionalCascade | NestedCascade] = Field(min_length=1)
    macro: str | None = None


class ConditionalCascade(IRBase):
    """A case statement over CASCADES: per generation, exactly one arm's
    Einsums run (none, if no condition holds).

    This is what a multi-output case statement whose arms are cascade-form
    macro invocations means once the macros are expanded:

        (Tree_{i+1,s,d}, F_{i+1,d}, TopDown_{i+1}) =
            { TOP-DOWN(...)   if TopDown_i == True
            { BOTTOM-UP(...)  if TopDown_i == False

    Arms are tested in the order written; the first whose condition holds
    runs. The arms usually write the same tensors, which is one write per
    generation, not two: only one arm runs.
    """

    kind: Literal["conditional"] = "conditional"
    arms: list[ConditionalArm] = Field(min_length=1)


def flatten(
    entries: list[Einsum | ConditionalCascade | NestedCascade] | NestedCascade,
) -> list[Einsum]:
    """Every Einsum of a cascade body, depth first, in written order.

    Conditional arms and inner cascades are descended into, so index ``k`` of
    the result names the same Einsum for every consumer (validators, the
    evaluator's trace, tools that label Einsums).
    """
    if isinstance(entries, NestedCascade):
        entries = entries.einsums
    out: list[Einsum] = []
    for entry in entries:
        if isinstance(entry, Einsum):
            out.append(entry)
        elif isinstance(entry, ConditionalCascade):
            for arm in entry.arms:
                out.extend(flatten(arm.einsums))
        else:
            out.extend(flatten(entry.einsums))
    return out


NestedCascade.model_rebuild()
ConditionalArm.model_rebuild()
ConditionalCascade.model_rebuild()
