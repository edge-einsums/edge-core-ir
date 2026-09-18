"""Structured stopping conditions.

A stopping condition halts a nested cascade when a Boolean predicate
holds. The predicate is either a comparison between two value
expressions or a direct call to a user-defined function that returns
Boolean.

Per the paper, the diamond notation `<>_<rank> : <predicate>` names
the generational rank whose iteration is halted. Plain `<>` in input
is shorthand for "the cascade's only generational rank"; the lowering
pass resolves this to an explicit rank name before constructing the
IR. The IR always carries an explicit `rank_variable`.

The two predicate variants are:

  Comparison:
    <value-expression> <comparison-op> <value-expression>

    Comparison operators are lifted to "all points satisfy": for
    multi-element operands, the comparison holds iff every point in
    the shared iteration space satisfies the elementwise predicate.

  DiamondBooleanApp:
    <name>(<operand-tensor>, ...)

    A user-defined function called at the predicate level. It must
    return Boolean. The IR does not enforce the return type; the
    Layer 2 validator does.

A ValueExpression is one of:
  - TensorProjectionValue: a tensor slice (wraps TensorProjection).
  - RankExpressionValue: a rank-variable expression (wraps RankExpression).
  - PropertyApp: a property extractor applied to one or more tensor
    projections. Currently the only built-in property is occupancy.

Scalars in comparisons are represented as TensorProjectionValue
referencing a synthesized zero-rank TensorDeclaration whose value field
carries the literal. See docs/parsing.md for the lowering rule.

PropertyApp and DiamondBooleanApp both carry a `pinned_rank_variables`
list. The list always includes the enclosing StoppingCondition's
rank_variable, plus any additional ranks the user fixed. Lowering is
responsible for prepending the enclosing rank; the IR validates the
shape but not the prepending contract.

This module imports from expr.py (for TensorProjection and
RankExpression) but is not imported back by expr.py, so there is no
circular-import risk.

Notes from the EDGE-semantics review (for the future validator and
interpreter):

  - The "all points satisfy" lifting for Comparison is the
    convergence-test reading the paper's examples use: `D_{i+1} ≡ D_i`
    means ∀d. D[i+1, d] = D[i, d].

  - Comparing a multi-element projection against a scalar (e.g.
    `X == 0` over all elements) is not the same as
    `occupancy(X) == 0`: they differ whenever X's empty value isn't 0
    (as in BFS, where F's empty value is ∞). The IR permits both; they
    are distinct computations.

  - The diamond is the paper's exclusion guard
    (`RV_i = { i | … ∧ ¬◇(i) }`, paper §6 syntax), not a `break`. The
    IR just carries the predicate; the exclude-this-generation
    semantics live in the interpreter.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from edge_ir.ir import patterns
from edge_ir.ir.base import IRBase
from edge_ir.ir.expr import RankExpression, TensorProjection

################################################################################
# Value-expression wrappers
################################################################################


class TensorProjectionValue(IRBase):
    """A tensor projection used as a value expression.

    Wraps a TensorProjection to give it a `kind` discriminator for
    membership in the ValueExpression union. Mirrors the InputTensor
    pattern used in expr.py for the Expression union.
    """

    kind: Literal["tensor_value"] = "tensor_value"
    proj: TensorProjection


class RankExpressionValue(IRBase):
    """A rank-variable expression used as a value expression.

    Wraps a RankExpression (itself a discriminated union of
    RankVariable, RankConstantLiteral, RankConstantShapeSym, RankArith,
    RankFunction) to give it a `kind` discriminator for the outer
    ValueExpression union.
    """

    kind: Literal["rank_value"] = "rank_value"
    expr: RankExpression


################################################################################
# PropertyApp: built-in tensor-property extractors
################################################################################


class PropertyApp(IRBase):
    """A property extractor applied to one or more tensor projections.

    The name field is restricted to the closed set of built-in
    properties. Currently the only built-in is `occupancy`, which
    returns the number of non-empty coordinates in the operand's
    iteration space (with the pinned ranks held fixed).

    Per the lowering contract, `pinned_rank_variables` includes the
    enclosing StoppingCondition.rank_variable as its first element.
    Any additional ranks listed are held fixed when evaluating the
    property.

    To add a new built-in property, extend the Literal type for `name`.
    The interpreter and validator look up properties by this name.
    """

    kind: Literal["property_app"] = "property_app"
    # function name
    name: Literal["occupancy"]

    # list of pinned variables
    pinned_rank_variables: list[str] = Field(min_length=1)
    operands: list[TensorProjection] = Field(min_length=1)


################################################################################
# ValueExpression union
################################################################################


ValueExpression = Annotated[
    TensorProjectionValue | RankExpressionValue | PropertyApp,
    Field(discriminator="kind"),
]


################################################################################
# Predicate variants
################################################################################


ComparisonOp = Literal["==", "!=", ">", "<", ">=", "<="]


class Comparison(IRBase):
    """A comparison between two value expressions.

    The comparison operator is from the closed set
    {==, !=, >, <, >=, <=}. The paper uses Unicode glyphs
    (the identity, not-equal, and inequality signs); the IR uses
    ASCII. The lowering pass rewrites Unicode forms to ASCII.

    Semantics: when both sides are multi-element, the comparison holds
    iff every point in the shared iteration space satisfies the
    elementwise predicate.
    """

    kind: Literal["comparison"] = "comparison"
    lhs: ValueExpression
    op: ComparisonOp
    rhs: ValueExpression


class DiamondBooleanApp(IRBase):
    """A user-defined Boolean-returning function at the predicate level.

    The whole stopping condition is `name(operand, operand, ...)`,
    where `name` is a user-defined function that must return Boolean.
    The IR does not enforce the return type; the Layer 2 validator
    does.

    Per the lowering contract, `pinned_rank_variables` includes the
    enclosing StoppingCondition.rank_variable as its first element.

    There are no built-in DiamondBooleanApp names. All names resolve
    through the user-defined function registry.
    """

    kind: Literal["diamond_boolean_app"] = "diamond_boolean_app"
    # This is the name of the user defined function
    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)
    # These are inputs to the function above (in "name")
    pinned_rank_variables: list[str] = Field(min_length=1)

    # input tensors
    operands: list[TensorProjection] = Field(min_length=1)


################################################################################
# Predicate union
################################################################################


Predicate = Annotated[
    Comparison | DiamondBooleanApp,
    Field(discriminator="kind"),
]


################################################################################
# StoppingCondition
################################################################################


class StoppingCondition(IRBase):
    """A stopping condition for a nested cascade.

    Halts iteration over the named generational rank when the
    predicate holds. The rank_variable is always explicit (the
    lowering pass resolves any shorthand from input).

    Per paper Section 7.3.1, when a stopping condition is present, the
    default generational-rank-variable is "i", with initial value 0.
    """

    rank_variable: str = Field(pattern=patterns.REGEX_RANK_VARIABLE)
    predicate: Predicate
