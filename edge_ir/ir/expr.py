"""Rank expressions and tensor projections.

Rank expressions are the things that appear in a tensor's subscripts
in an Einsum expression. They map iteration-space points to coordinate
space.

This module covers the right-hand-side machinery for tensor accesses:
the five kinds of rank expressions and the TensorProjection that wraps
them with a tensor name, plus the recursive Expression tree (InputTensor,
UnaryApp, BinaryApp, AnonymousTensor, RankValue).

The five rank-expression variants are:
- RankVariable: a bare iteration variable, e.g., m or sm
- RankConstantLiteral: an integer literal or a label from a declared
  char/string-valued coord set, e.g., 0, 1, "a"
- RankConstantShapeSym: a reference to a declared shape parameter
  (|V|, N, M, ...); the shape resolver substitutes the bound integer
  at iteration-space construction
- RankArith: structured binary arithmetic, e.g., m + 1, k * |V|, m % N
- RankFunction: a user-defined function applied to rank variables,
  e.g., min(a, w) for non-affine mappings

Per the grammar:
    <rank-expression> := <rank-variable>
                       | <rank-mapping-function> "(" <rank-variable-list> ")"

The grammar treats all non-bare RVEs uniformly under
<rank-mapping-function>. The IR splits them: structured arithmetic
expressions (RankArith) preserve their tree shape so that downstream
consumers (egglog rewriter, MLIR backend, static analysis) can reason
about them. Opaque user-defined functions (RankFunction) remain a
fallback for non-affine or otherwise unstructured mappings like
min(a, w) from connected components.

Per paper Section 7.3.5 for non-affine RVEs and Section 6 for the
formal semantics of how RVEs map points to coordinates.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from edge_ir.ir import patterns
from edge_ir.ir.base import IRBase
from edge_ir.ir.op import MergeOp, UnaryOp

################################################################################
# RankVariable
################################################################################


class RankVariable(IRBase):
    """A bare rank variable in an iteration spec or tensor projection.

    Per the grammar production:
        <rank-variable> := /[a-z]+/ /[A-Za-z0-9]*/

    Lowercase start, then any alphanumerics. So "m", "k", "sm", "dm" are
    all valid. Distinct from rank names (uppercase, used in tensor
    declarations). Even when the same letter is used (e.g., a tensor
    rank named M with a rank variable also written m), the IR keeps
    them as separate concepts.
    """

    kind: Literal["var"] = "var"
    name: str = Field(pattern=patterns.REGEX_RANK_VARIABLE)


################################################################################
# RankConstant variants
################################################################################


class RankConstantLiteral(IRBase):
    """A literal coordinate value in a rank arithmetic expression.

    The value is either:
    - An integer literal (e.g., 0, 1, 2, -3) -- e.g., the `1` in `m + 1`
    - A label from a declared char/string-valued coord set (e.g., "a"
      when a rank's CoordSetEnum lists ["a","b","c"]). Fully static; a
      value in the rank's coord space.

    Distinct from RankConstantShapeSym, which references a declared
    shape parameter that the shape resolver substitutes at
    iteration-space construction.
    """

    kind: Literal["literal"] = "literal"
    value: int | str
    # Resolved (was a TODO asking for "any data type that matches the
    # coordinate set of the rank name"): `int | str` already IS the full
    # coordinate space. A coordinate is whatever a rank coordinate set can
    # hold, and CoordSetEnum.coords is typed `list[str | int]` -- the
    # grammar's <coordinate> production is /[A-Za-z0-9]+/, so labels are
    # strings and dense coordinates are ints. Widening further would admit
    # values no coordinate set can contain. Whether a particular literal is
    # type-compatible with the rank it indexes is a Layer 2 check (it needs
    # the declaration, which this node does not carry).


class RankConstantShapeSym(IRBase):
    """A reference to a declared shape parameter (|V|, N, M, ...).

    The shape resolver substitutes the bound integer at
    iteration-space construction. Used in expressions like
    ``k * |V|`` (where ``|V|`` is a RankConstantShapeSym).
    """

    kind: Literal["shape_sym"] = "shape_sym"
    name: str


# Internal isinstance convenience: a "rank constant" in the loose sense
# is either flavor. Python 3.10+ supports ``isinstance(x, A | B)``
# directly via this alias.
RankConstant = RankConstantLiteral | RankConstantShapeSym


################################################################################
# RankArith (structured arithmetic)
################################################################################


class RankArith(IRBase):
    """Binary arithmetic on rank expressions.

    Supports +, -, *, /, //, % as standard arithmetic operators on
    integer-valued rank expressions. Recursive: lhs and rhs can be any
    RankExpression, allowing arbitrary nesting like (m + 1) * 2 or
    (k * |V|) % N.

    Whether a particular RankArith tree is affine is a *derived* fact,
    not stored on the node. See ``edge_ir.analysis.affine`` for the
    analysis pass that computes it into a side table.
    """

    kind: Literal["arith"] = "arith"
    op: Literal["+", "-", "*", "/", "//", "%"]
    lhs: RankExpression
    rhs: RankExpression


################################################################################
# RankFunction (opaque user-defined mapping)
################################################################################


class RankFunction(IRBase):
    """A user-defined function applied to one or more rank expressions.

    Per the grammar:
        <rank-expression> := <rank-mapping-function> "(" <rank-variable-list> ")"
        <rank-mapping-function> := <user-defined-function>

    The grammar production lists arguments as a <rank-variable-list>, but
    the IR intentionally accepts full rank-variable-expressions (RVEs) as
    arguments: any RankExpression, i.e. variables, constants, structured
    arithmetic, or nested functions. This is a deliberate extension over
    the published grammar, consistent with how the IR already accepts
    structured RankArith where the grammar folds everything under
    <rank-mapping-function> (see the module docstring above). So
    min(a, 0), f(m, |V|), f(m + 1), and f(g(x)) are all valid.

    Used when the mapping from iteration space to coordinate space is
    not expressible as structured arithmetic. The canonical example is
    min(a, w) from connected components (paper Section 7.3.5), which
    is non-affine and best left opaque.

    The is_affine field is populated by the Layer 2 validator. For
    opaque functions, the validator cannot derive affine-ness from
    structure; it relies on user annotation if any. None means "not
    analyzed yet"; the validator may set it to False as a conservative
    default for unannotated user functions.
    """

    kind: Literal["func"] = "func"
    func_name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)
    args: list[RankExpression] = Field(min_length=1)
    # this is a black-box so the user has to give us this info
    is_affine: bool | None = None


# A rank expression is one of the five above. Discriminated by "kind".
# Structured arithmetic (RankArith) preserves tree shape so downstream
# consumers can reason about it; opaque functions (RankFunction) are
# the fallback for non-affine or unstructured mappings.
RankExpression = Annotated[
    RankVariable
    | RankConstantLiteral
    | RankConstantShapeSym
    | RankArith
    | RankFunction,
    Field(discriminator="kind"),
]


################################################################################
# TensorProjection
################################################################################


class TensorProjection(IRBase):
    """A tensor name plus a list of rank expressions.

    Per the grammar:
        <tensor-projection> := <tensor-name>_<rank-expression-list>

    In math notation: A[m, k], B[k, n], Z[m, n]. The tensor name
    identifies which tensor; the rank expressions tell you how
    iteration variables map to coordinates of that tensor.

    For scalars (0-rank tensors), the ranks list is empty. The
    grammar permits this; the iteration spec for a scalar
    projection has no iteration variables.
    """

    tensor: str = Field(pattern=patterns.REGEX_TENSOR_NAME)
    ranks: list[RankExpression]


################################################################################
# Expression tree (recursive)
################################################################################


class InputTensor(IRBase):
    """A tensor projection used as an operand in an expression.

    Wraps a TensorProjection. This is the leaf case of an Expression
    tree: a reference to a declared input tensor at a specific
    iteration-space mapping.
    """

    kind: Literal["input"] = "input"
    proj: TensorProjection


class UnaryApp(IRBase):
    """A unary operator applied to an input tensor.

    Per the grammar:
        <expression> := <unary-operator> <input-tensor>

    The unary operator is either the built-in negation or a user-defined
    function. UnaryOp is defined in op.py.

    Note: the grammar restricts the unary's operand to <input-tensor>,
    not arbitrary <expression>. This is a deliberate language constraint
    (per paper Section 5.5: unary functions are syntactic sugar for a
    degenerate Map), and this field is typed to match it.

    An anonymous tensor is NOT an escape hatch here -- `AnonymousTensor` is
    not an `InputTensor`, so it is rejected in this field too. (An earlier
    version of this docstring said otherwise; it was wrong, and the door it
    named does not exist.) To apply a unary to a sub-expression, name the
    sub-expression with its own Einsum and apply the unary to that tensor:

        Diff_{v,xy} = L_{v,xy} . Lg_{xy} :: map -(intersect)
        AD_{v,xy}   = abs(Diff_{v,xy})

    which is the paper-legal rewrite. Whether the language SHOULD admit
    unary-on-expression is an open question; the IR and the EBNF currently
    agree that it does not.
    """

    kind: Literal["unary"] = "unary"
    op: UnaryOp
    operand: InputTensor
    merge_op: MergeOp | None = None
    """The merge operator of the degenerate Map this unary sugars into.

    EDGE's decomposition of unary functions is
    explicit that this is NOT optional information:

        "every unary function on some tensor A decomposes into an Einsum
        with input tensors A and B, where B = 1. The compute operator is
        the unary operator itself; THE MERGE OPERATOR DEPENDS ON HOW THAT
        OPERATOR TREATS THE EMPTY VALUE. For unary operators that map empty
        to empty, such as negation or squaring, the map action uses the
        take-left merge operator to visit only the non-empty points of A.
        For unary operators that map empty to non-empty, such as logical
        complement, the map action must instead visit the points where A is
        empty, using the not-take-left merge operator. The merge operator
        therefore matters even for unary operators."

    and its worked desugaring is
    `Z_m = A_m .^1 (B_m .^2 1)_m :: AND^1  not^2(not_left)`.

    The distinction is load-bearing, not decorative. `not P` under
    `take_left` exists where P exists; under `not_left` it exists where P
    does NOT. Every masking step in the BFS family
    (`F_{i+1,d} = T_{i,d} . not P_{i,d} :: map take_left(intersect)`) wants
    the second reading -- with the first it computes `T and P` instead of
    `T minus P`, the exact complement of the intended mask.

    `None` means "derive it from the operator": the built-in `not` maps
    empty to non-empty, so it derives `not_left`; a user-defined unary
    derives from the `maps_empty_to_empty` flag on its registry
    declaration, which defaults to True (hence `take_left`). Set the field
    explicitly to override.
    """


class BinaryApp(IRBase):
    """A binary operation between two expressions, tagged with a label.

    Per the grammar:
        <expression> := <expression> "." <binary-label> <expression>

    The label is an integer that links this binary to its corresponding
    computation spec(s) in the surrounding Einsum's spec list. The IR
    does not store the actual compute/merge operators here; those live
    on the matching ComputationSpec.

    Recursive: lhs and rhs can be any Expression variant.
    """

    kind: Literal["binary"] = "binary"
    label: int
    lhs: Expression
    rhs: Expression


class AnonymousTensor(IRBase):
    """A parenthesized sub-expression used as an operand.

    Per the grammar:
        <anonymous-tensor> := "(" <expression> ")" <rank-expression-list>

    An anonymous tensor wraps an expression and gives it an output
    shape (the rank-expression list — the subscript on the closing
    paren, e.g. (...)_{m,n}). The enclosing expression consumes it as
    an operand, in the same syntactic position an InputTensor would
    occupy; but it is not a declared tensor — the value at each point
    comes from evaluating the inner expression under the relevant
    computation specs from the outer Einsum.

    Computation specs do NOT live on AnonymousTensor. All specs for an
    Einsum live at the outer Einsum level. Labels are flat within an
    Einsum: every binary in the entire expression tree (including
    binaries inside nested AnonymousTensors) has a unique label in a
    single global namespace, and the label on each BinaryApp links to
    the corresponding ComputationSpec in the enclosing Einsum's spec
    list regardless of how deeply nested the BinaryApp is.

    Validation that every binary has a matching spec, that every spec
    has a matching binary, and that labels are globally unique within
    an Einsum, is a Layer 2 concern; the IR does not enforce label
    coverage or uniqueness structurally.
    """

    kind: Literal["anonymous"] = "anonymous"
    expression: Expression
    ranks: list[RankExpression]


class RankValue(IRBase):
    """A rank expression read as a VALUE, not as a subscript.

    Per EDGE's "Rank Variables as Tensors":
    EDGE allows a rank variable to appear as a
    scalar operand inside an expression -- at each point of the iteration
    space the coordinate itself is cast into the data space.

    This is the leaf the rest of the Expression union has no room for.
    ``InputTensor`` reads a declared tensor; a zero-rank ``TensorDeclaration``
    carrying a literal ``value`` (design decision D21) is a CONSTANT and
    cannot vary with the iteration point. A rank value varies with the point
    by construction, which is exactly what the DFS order stamp needs::

        sigma(i, v) = i * |V| + v      ->  RankValue(expr=RankArith(...))
        Post_{i+1,v} = Post_{i,v} . (Fin_{i,v} . i)

    and what max-flow's ``D_{0, u : u = s} = |V|`` needs (a shape symbol in
    value position).

    A rank value is always present: every point of the iteration space
    yields a coordinate, so the operand's existence predicate is True
    everywhere. Its "empty value" is therefore never consulted.

    Relationship to ``stopping.RankExpressionValue``: same idea, different
    union. That one wraps a ``RankExpression`` for the stopping-condition
    ``ValueExpression`` union; this one wraps it for the expression tree.
    They are kept separate for the same reason D22 keeps the two predicate
    unions separate -- the surrounding operand vocabularies differ.
    """

    kind: Literal["rank_value"] = "rank_value"
    expr: RankExpression


# An Expression is one of the five above. Discriminated by "kind".
# This is recursive: BinaryApp.lhs and .rhs are Expressions; AnonymousTensor
# also contains an Expression. Pydantic handles the recursion via the
# `from __future__ import annotations` lazy evaluation at the top of the
# file.
Expression = Annotated[
    InputTensor | UnaryApp | BinaryApp | AnonymousTensor | RankValue,
    Field(discriminator="kind"),
]
