"""Iteration space of an Einsum as coordinate images over rank variables.

Step 0 of the fusion-analysis pass. This is the VAL-6 prereq.

The semantics:

- **IS axes = free rank variables.** The iteration space of an Einsum is
  the set of free ``RankVariable`` names reachable in its ``output_ranks``
  *and* in every ``InputTensor.proj.ranks`` across the expression tree
  (recursing ``BinaryApp`` / ``UnaryApp`` / ``AnonymousTensor``). We
  descend INTO ``RankArith`` / ``RankFunction`` to collect the variables
  they mention, but the *result coordinate* of an RVE does not itself
  become a new axis. So the BFS filter written in bare-reduce EDGE

      F_{i+1,d} = T_{i,d} . not P_{i,d}         (reduces nothing here)

  has axes ``{i, d}`` -- NOT ``{i, i+1, d}``: the ``i+1`` in the output
  contributes the variable ``i`` (already an axis), not a fresh axis.
  A rank absent from the output but present in an input (a reduced rank)
  IS an axis: ``X_m = Y_{m,n}`` (reduce n) has axes ``{m, n}``.

- **Matching is on coordinate IMAGES with offsets.** For each tensor-rank
  subscript we record a :class:`CoordImage` = (axis variables, affine
  offset). ``m`` -> ``({m}, 0)``; ``m + 1`` -> ``({m}, +1)``. Per N1
  (the paper is silent; the default chosen here) we keep the original variables and
  record the offset rather than coining a fresh axis: ``p + s`` ->
  ``({p, s}, 0)``, not a new axis ``c = p + s``.

- **Opaque non-invertible RVE output.** An opaque ``RankFunction`` in an
  output rank (e.g. ``min(a, w)`` from connected components) leaves its
  argument variables in the iteration space as reduced ranks, and marks
  ``reduce_nonderivable`` -- "reduce required, rank-set non-derivable".
  We do NOT compute the reduced set by subtraction (VAL-3 ruling in
  ``docs/core_ir_decisions_pass.md``).

This is a pure side analysis in the shape of ``analysis/affine.py``: it
reads ``edge_ir.ir`` and returns plain dataclasses; it mutates nothing on
the frozen IR.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from edge_ir.ir.einsum import Einsum
from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    Expression,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankExpression,
    RankFunction,
    RankValue,
    RankVariable,
    UnaryApp,
)
from edge_ir.ir.tensor import TensorDeclaration

################################################################################
# CoordImage
################################################################################


@dataclass(frozen=True)
class CoordImage:
    """The image of one tensor-rank subscript.

    - ``axes``: the free rank variables the subscript mentions.
    - ``offset``: the additive integer constant. Only meaningful (a true
      "generational" shift) when :attr:`is_shift` holds.
    - ``is_shift``: the subscript is an integer-offset shift of its axes,
      i.e. it has the form ``(sum of distinct axes, each coefficient +1)
      + integer offset``. False for scaling / division / modulo arithmetic
      and for opaque ``RankFunction`` s -- there ``offset`` is 0 and
      carries no meaning.

    Examples: ``m`` -> ``({m}, 0, is_shift=True)``; ``m + 1`` ->
    ``({m}, +1, is_shift=True)``; ``m - k`` -> ``({m, k}, 0,
    is_shift=False)`` (``k`` has coefficient -1); ``k * |V|`` ->
    ``({k}, 0, is_shift=False)``; ``min(a, w)`` -> ``({a, w}, 0,
    is_shift=False)``.
    """

    axes: frozenset[str]
    offset: int
    is_shift: bool


def coord_image(node: RankExpression) -> CoordImage:
    """Compute the :class:`CoordImage` of a single rank-expression tree."""
    if isinstance(node, RankVariable):
        return CoordImage(axes=frozenset({node.name}), offset=0, is_shift=True)
    if isinstance(node, RankConstantLiteral):
        offset = node.value if isinstance(node.value, int) else 0
        return CoordImage(axes=frozenset(), offset=offset, is_shift=True)
    if isinstance(node, RankConstantShapeSym):
        # A symbolic constant (|V|, N). We do not resolve it to an integer
        # here, so its contribution to the offset is unknown: not a shift.
        return CoordImage(axes=frozenset(), offset=0, is_shift=False)
    if isinstance(node, RankArith):
        lhs = coord_image(node.lhs)
        rhs = coord_image(node.rhs)
        axes = lhs.axes | rhs.axes
        if node.op == "+":
            return CoordImage(
                axes=axes,
                offset=lhs.offset + rhs.offset,
                is_shift=lhs.is_shift and rhs.is_shift,
            )
        if node.op == "-":
            # Subtracting an axis flips its coefficient to -1, so the
            # result is a shift only when the right side has no axes.
            return CoordImage(
                axes=axes,
                offset=lhs.offset - rhs.offset,
                is_shift=lhs.is_shift and rhs.is_shift and not rhs.axes,
            )
        # *, /, //, % : not an integer-offset shift of its axes.
        return CoordImage(axes=axes, offset=0, is_shift=False)
    if isinstance(node, RankFunction):
        arg_axes: set[str] = set()
        for arg in node.args:
            arg_axes |= coord_image(arg).axes
        return CoordImage(axes=frozenset(arg_axes), offset=0, is_shift=False)
    raise TypeError(f"unknown RankExpression variant: {type(node).__name__}")


################################################################################
# IterationSpace
################################################################################


@dataclass(frozen=True)
class IterationSpace:
    """The iteration space of one Einsum.

    - ``axes``: the free rank variables the Einsum ranges over (reduced
      ranks included). This is what boundary classification compares.
    - ``output_images``: the coordinate images of ``output_ranks``, in
      order. Used to align an intermediate tensor across a boundary and to
      compare generational images (producer write vs consumer read).
    - ``reduce_nonderivable``: an opaque ``RankFunction`` appears in an
      output rank, so the reduced rank-set cannot be derived by
      subtraction (VAL-3). The argument variables stay in ``axes``.
    """

    axes: frozenset[str]
    output_images: tuple[CoordImage, ...]
    reduce_nonderivable: bool


def iteration_space(
    einsum: Einsum,
    declarations: Sequence[TensorDeclaration],
) -> IterationSpace:
    """Build the :class:`IterationSpace` of ``einsum``.

    ``declarations`` is the settled VAL-6 signature: it is reserved for
    resolving rank variables to concrete coordinate sets. Increment 1
    matches on variable-name images with offsets (the resolved consult),
    so coordinate-set resolution is deferred and ``declarations`` is not
    consulted yet.
    """
    del declarations  # reserved for coordinate-set resolution (deferred).

    output_images = tuple(coord_image(r) for r in einsum.output_ranks)

    axes: set[str] = set()
    for img in output_images:
        axes |= img.axes
    for subscript in _input_subscripts(einsum.expression):
        axes |= coord_image(subscript).axes

    reduce_nonderivable = any(_contains_rank_function(r) for r in einsum.output_ranks)

    return IterationSpace(
        axes=frozenset(axes),
        output_images=output_images,
        reduce_nonderivable=reduce_nonderivable,
    )


################################################################################
# Expression-tree walkers
################################################################################


def operand_subscripts(expr: Expression) -> Iterator[RankExpression]:
    """Every rank expression reachable in operand position in ``expr``.

    Public entry point to the same walk :func:`iteration_space` uses, so a
    consumer that needs "which rank variables does this sub-expression
    mention" cannot drift from the axis rule.
    """
    yield from _input_subscripts(expr)


def _input_subscripts(expr: Expression) -> Iterator[RankExpression]:
    """Yield every rank expression reachable in operand position in ``expr``.

    That is every subscript of every ``InputTensor``, plus the wrapped
    expression of every ``RankValue``. A ``RankValue`` reads a rank
    expression as a value rather than as a subscript, but the rank variables
    it mentions are iterated over just the same, so they are axes: in
    ``Post_{i+1,v} = Post_{i,v} . (Fin_{i,v} . i)`` the value-position ``i``
    is the same axis the subscripts use.
    """
    if isinstance(expr, InputTensor):
        yield from expr.proj.ranks
    elif isinstance(expr, UnaryApp):
        yield from expr.operand.proj.ranks
    elif isinstance(expr, BinaryApp):
        yield from _input_subscripts(expr.lhs)
        yield from _input_subscripts(expr.rhs)
    elif isinstance(expr, AnonymousTensor):
        yield from _input_subscripts(expr.expression)
    elif isinstance(expr, RankValue):
        yield expr.expr


def _contains_rank_function(node: RankExpression) -> bool:
    """True if ``node`` contains an opaque ``RankFunction`` anywhere."""
    if isinstance(node, RankFunction):
        return True
    if isinstance(node, RankArith):
        return _contains_rank_function(node.lhs) or _contains_rank_function(node.rhs)
    return False
