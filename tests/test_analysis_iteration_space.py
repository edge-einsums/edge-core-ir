"""Tests for ``edge_ir.analysis.iteration_space`` (Step 0 of the fusion pass).

These pin the coordinate-image semantics:

- IS axes = the free rank VARIABLES reachable in the output and every input
  subscript (reduced ranks included), not a fresh axis per rank-expression.
- Matching is on coordinate IMAGES with integer offsets: ``m + 1`` is axis
  ``m`` at offset +1, not a new axis and not a bare copy.
- An opaque ``RankFunction`` in an output rank trips ``reduce_nonderivable``;
  a structured ``RankArith`` does NOT.
"""

from __future__ import annotations

import pytest

from edge_ir.analysis.iteration_space import (
    CoordImage,
    coord_image,
    iteration_space,
)
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
    RankVariable,
    TensorProjection,
    UnaryApp,
)
from edge_ir.ir.op import UserDefinedUnaryOp

################################################################################
# Builders
################################################################################


def _var(name: str) -> RankVariable:
    return RankVariable(name=name)


def _inp(tensor: str, *ranks: RankExpression) -> InputTensor:
    return InputTensor(proj=TensorProjection(tensor=tensor, ranks=list(ranks)))


def _einsum(
    output_tensor: str,
    output_ranks: list[RankExpression],
    expression: Expression,
) -> Einsum:
    return Einsum(
        output_tensor=output_tensor,
        output_ranks=output_ranks,
        expression=expression,
        specs=[],
    )


################################################################################
# coord_image
################################################################################


@pytest.mark.parametrize(
    ("node", "axes", "offset", "is_shift"),
    [
        # A bare rank variable is its own axis at offset 0.
        (_var("m"), frozenset({"m"}), 0, True),
        # m + 1 / m - 1: single-axis integer shift, offset +/- 1.
        (
            RankArith(op="+", lhs=_var("m"), rhs=RankConstantLiteral(value=1)),
            frozenset({"m"}),
            1,
            True,
        ),
        (
            RankArith(op="-", lhs=_var("m"), rhs=RankConstantLiteral(value=1)),
            frozenset({"m"}),
            -1,
            True,
        ),
        # m - k: subtracting an axis flips its coefficient to -1, so this is
        # NOT an integer-offset shift; both variables stay in the axis set.
        (
            RankArith(op="-", lhs=_var("m"), rhs=_var("k")),
            frozenset({"m", "k"}),
            0,
            False,
        ),
        # k * |V|: scaling by a shape symbol is not a shift; offset collapses
        # to 0 and only the variable k survives as an axis.
        (
            RankArith(op="*", lhs=_var("k"), rhs=RankConstantShapeSym(name="|V|")),
            frozenset({"k"}),
            0,
            False,
        ),
        # k % N and m // 2: modulo / floor-division are not shifts.
        (
            RankArith(op="%", lhs=_var("k"), rhs=RankConstantShapeSym(name="N")),
            frozenset({"k"}),
            0,
            False,
        ),
        (
            RankArith(op="//", lhs=_var("m"), rhs=RankConstantLiteral(value=2)),
            frozenset({"m"}),
            0,
            False,
        ),
        # min(a, w): an opaque RankFunction leaves its argument variables in
        # the axis set but is never a shift.
        (
            RankFunction(func_name="min", args=[_var("a"), _var("w")]),
            frozenset({"a", "w"}),
            0,
            False,
        ),
        # A bare shape symbol contributes no axis and an unknown offset, so it
        # is not a shift.
        (RankConstantShapeSym(name="|V|"), frozenset(), 0, False),
    ],
)
def test_coord_image_maps_rank_expression_to_axes_offset_and_shift(
    node: RankExpression,
    axes: frozenset[str],
    offset: int,
    is_shift: bool,
) -> None:
    image = coord_image(node)
    assert image == CoordImage(axes=axes, offset=offset, is_shift=is_shift)


def test_coord_image_p_plus_s_is_multi_axis_shift_of_both_variables() -> None:
    """p + s carries BOTH variables as axes at offset 0.

    Per the ``is_shift`` contract ("sum of distinct axes, each coefficient
    +1, plus integer offset") a ``+`` of two variables is still a shift; the
    load-bearing facts for boundary alignment are the two-element axis set and
    the zero offset, distinguishing it from the single-axis ``m + 1``.
    """
    image = coord_image(RankArith(op="+", lhs=_var("p"), rhs=_var("s")))
    assert image.axes == frozenset({"p", "s"})
    assert image.offset == 0


################################################################################
# iteration_space
################################################################################


def test_iteration_space_includes_reduced_rank_from_input_only() -> None:
    """X_m = Y_{m,n} (reduce n): axes are {m, n}, not just the output {m}."""
    einsum = _einsum("X", [_var("m")], _inp("Y", _var("m"), _var("n")))
    space = iteration_space(einsum, [])
    assert space.axes == frozenset({"m", "n"})
    assert space.reduce_nonderivable is False


def test_iteration_space_recurses_through_unary_and_anonymous_tensors() -> None:
    """Axes are gathered from input subscripts nested inside UnaryApp and
    AnonymousTensor operands, not just the top-level inputs."""
    unary = UnaryApp(
        op=UserDefinedUnaryOp(name="exp"),
        operand=_inp("A", _var("m"), _var("n")),
    )
    anon = AnonymousTensor(
        expression=_inp("B", _var("p"), _var("q")),
        ranks=[_var("m")],
    )
    einsum = _einsum(
        "T",
        [_var("m")],
        BinaryApp(label=1, lhs=unary, rhs=anon),
    )
    space = iteration_space(einsum, [])
    assert space.axes == frozenset({"m", "n", "p", "q"})


def test_iteration_space_records_output_images_with_generational_offset() -> None:
    """A generational output F_{i+1,d} records images (i offset +1, d offset 0)
    while contributing only the variable i as an axis."""
    einsum = _einsum(
        "F",
        [RankArith(op="+", lhs=_var("i"), rhs=RankConstantLiteral(value=1)), _var("d")],
        _inp("T", _var("i"), _var("d")),
    )
    space = iteration_space(einsum, [])
    assert space.output_images == (
        CoordImage(axes=frozenset({"i"}), offset=1, is_shift=True),
        CoordImage(axes=frozenset({"d"}), offset=0, is_shift=True),
    )
    assert space.axes == frozenset({"i", "d"})


def test_iteration_space_opaque_output_rank_function_trips_reduce_nonderivable() -> (
    None
):
    """An opaque min(a, w) in an OUTPUT rank marks reduce_nonderivable and
    keeps its argument variables as axes (the reduced set is not derived by
    subtraction, per VAL-3)."""
    einsum = _einsum(
        "C",
        [RankFunction(func_name="min", args=[_var("a"), _var("w")])],
        _inp("D", _var("a"), _var("w")),
    )
    space = iteration_space(einsum, [])
    assert space.reduce_nonderivable is True
    assert {"a", "w"} <= space.axes


def test_iteration_space_structured_arith_output_rank_is_derivable() -> None:
    """A structured RankArith (m + 1) in an output rank must NOT trip
    reduce_nonderivable -- only opaque RankFunctions do."""
    einsum = _einsum(
        "F",
        [RankArith(op="+", lhs=_var("m"), rhs=RankConstantLiteral(value=1))],
        _inp("T", _var("m")),
    )
    space = iteration_space(einsum, [])
    assert space.reduce_nonderivable is False
