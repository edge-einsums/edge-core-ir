"""Tests for the affine-analysis side pass.

The affine-analysis pass lives in ``edge_ir.analysis.affine`` and
returns either a single :class:`Affinity` for a node or a side table
(``{id(node): Affinity}``) for a whole rank-expression tree. These
tests pin the *rules*, not the API shape — the IR datamodel tests
already cover construction.

Rules under test (per the module docstring):

- ``RankVariable`` -> AFFINE
- ``RankConstantLiteral`` -> AFFINE
- ``RankConstantShapeSym`` -> AFFINE
- ``RankArith`` ``+`` / ``-`` -> AFFINE iff both children are AFFINE
- ``RankArith`` ``*`` -> AFFINE iff one child is a constant
  (``RankConstantLiteral`` or ``RankConstantShapeSym``) and the other
  child is AFFINE
- ``RankArith`` ``/`` / ``//`` / ``%`` -> NON_AFFINE
- ``RankFunction`` -> AFFINE / NON_AFFINE / UNKNOWN driven by the
  user-declared ``is_affine`` (True / False / None)
"""

from __future__ import annotations

import pytest

from edge_ir.analysis.affine import Affinity, affinity_table, analyze_affinity
from edge_ir.ir.expr import (
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankFunction,
    RankVariable,
)

################################################################################
# Leaves
################################################################################


def test_rank_variable_is_affine() -> None:
    assert analyze_affinity(RankVariable(name="m")) is Affinity.AFFINE


def test_rank_constant_int_is_affine() -> None:
    assert analyze_affinity(RankConstantLiteral(value=1)) is Affinity.AFFINE


def test_rank_constant_shape_dsl_is_affine() -> None:
    """Shape-DSL placeholders (e.g. ``|V|``) are still constants."""
    assert analyze_affinity(RankConstantShapeSym(name="|V|")) is Affinity.AFFINE


################################################################################
# RankArith: +, -
################################################################################


def test_var_plus_const_is_affine() -> None:
    """m + 1: classic affine shift."""
    expr = RankArith(
        op="+",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=1),
    )
    assert analyze_affinity(expr) is Affinity.AFFINE


def test_var_minus_var_is_affine() -> None:
    """m - n: still an integer linear combination."""
    expr = RankArith(
        op="-",
        lhs=RankVariable(name="m"),
        rhs=RankVariable(name="n"),
    )
    assert analyze_affinity(expr) is Affinity.AFFINE


################################################################################
# RankArith: *
################################################################################


def test_var_times_const_is_affine() -> None:
    """m * 2: constant scale -> affine."""
    expr = RankArith(
        op="*",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=2),
    )
    assert analyze_affinity(expr) is Affinity.AFFINE


def test_const_times_var_is_affine() -> None:
    """2 * m: order of const/var must not matter."""
    expr = RankArith(
        op="*",
        lhs=RankConstantLiteral(value=2),
        rhs=RankVariable(name="m"),
    )
    assert analyze_affinity(expr) is Affinity.AFFINE


def test_var_times_shape_sym_is_affine() -> None:
    """m * |V|: a shape symbol counts as a constant in the affine rule, so
    multiplying it by an affine sub-expression is still affine."""
    expr = RankArith(
        op="*",
        lhs=RankVariable(name="m"),
        rhs=RankConstantShapeSym(name="|V|"),
    )
    assert analyze_affinity(expr) is Affinity.AFFINE


def test_shape_sym_times_var_is_affine() -> None:
    """|V| * m: order does not matter when the constant is a shape symbol."""
    expr = RankArith(
        op="*",
        lhs=RankConstantShapeSym(name="|V|"),
        rhs=RankVariable(name="m"),
    )
    assert analyze_affinity(expr) is Affinity.AFFINE


def test_var_times_var_is_non_affine() -> None:
    """m * n: a quadratic combination is NOT affine."""
    expr = RankArith(
        op="*",
        lhs=RankVariable(name="m"),
        rhs=RankVariable(name="n"),
    )
    assert analyze_affinity(expr) is Affinity.NON_AFFINE


################################################################################
# RankArith: /, //, %
################################################################################


@pytest.mark.parametrize("op", ["/", "//", "%"])
def test_division_and_modulo_are_non_affine(op: str) -> None:
    """m / 2, m // 2, m % 2: none of these are affine, even with a
    constant divisor / modulus."""
    expr = RankArith(
        op=op,  # type: ignore[arg-type]
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=2),
    )
    assert analyze_affinity(expr) is Affinity.NON_AFFINE


################################################################################
# Nesting
################################################################################


def test_nested_affine_is_affine() -> None:
    """(m + 1) * 2: an affine sub-tree scaled by a constant is affine."""
    inner = RankArith(
        op="+",
        lhs=RankVariable(name="m"),
        rhs=RankConstantLiteral(value=1),
    )
    outer = RankArith(
        op="*",
        lhs=inner,
        rhs=RankConstantLiteral(value=2),
    )
    assert analyze_affinity(outer) is Affinity.AFFINE


def test_nested_non_affine_taints_parent() -> None:
    """(m * n) + 1: a NON_AFFINE child makes the sum NON_AFFINE."""
    inner = RankArith(
        op="*",
        lhs=RankVariable(name="m"),
        rhs=RankVariable(name="n"),
    )
    outer = RankArith(
        op="+",
        lhs=inner,
        rhs=RankConstantLiteral(value=1),
    )
    assert analyze_affinity(outer) is Affinity.NON_AFFINE


################################################################################
# RankFunction (user-declared)
################################################################################


def test_rank_function_declared_affine() -> None:
    f = RankFunction(
        func_name="shift",
        args=[RankVariable(name="m")],
        is_affine=True,
    )
    assert analyze_affinity(f) is Affinity.AFFINE


def test_rank_function_declared_non_affine() -> None:
    f = RankFunction(
        func_name="min",
        args=[RankVariable(name="a"), RankVariable(name="w")],
        is_affine=False,
    )
    assert analyze_affinity(f) is Affinity.NON_AFFINE


def test_rank_function_undeclared_is_unknown() -> None:
    """No user annotation -> we genuinely don't know. The analyzer must
    NOT conservatively assume NON_AFFINE here; that's a separate
    policy choice for downstream consumers."""
    f = RankFunction(
        func_name="f",
        args=[RankVariable(name="m")],
    )
    assert f.is_affine is None  # precondition
    assert analyze_affinity(f) is Affinity.UNKNOWN


################################################################################
# affinity_table: walks the whole tree
################################################################################


def test_affinity_table_keys_every_subnode() -> None:
    """For ``(m + 1) * 2`` the table should hold entries for:
    the outer ``*``, the inner ``+``, ``m``, ``1`` (inside the ``+``),
    and ``2`` (outside the ``+``) — five distinct nodes, each with the
    correct Affinity."""
    m = RankVariable(name="m")
    one = RankConstantLiteral(value=1)
    two = RankConstantLiteral(value=2)
    inner = RankArith(op="+", lhs=m, rhs=one)
    outer = RankArith(op="*", lhs=inner, rhs=two)

    table = affinity_table(outer)

    assert table[id(outer)] is Affinity.AFFINE
    assert table[id(inner)] is Affinity.AFFINE
    assert table[id(m)] is Affinity.AFFINE
    assert table[id(one)] is Affinity.AFFINE
    assert table[id(two)] is Affinity.AFFINE
    # No extra spurious entries.
    assert len(table) == 5


def test_affinity_table_walks_both_constant_variants() -> None:
    """Both RankConstantLiteral and RankConstantShapeSym appear in the
    side table as AFFINE leaves; the walk must not skip either variant."""
    m = RankVariable(name="m")
    literal_one = RankConstantLiteral(value=1)
    shape_sym = RankConstantShapeSym(name="|V|")
    inner = RankArith(op="+", lhs=m, rhs=literal_one)
    outer = RankArith(op="*", lhs=inner, rhs=shape_sym)

    table = affinity_table(outer)

    assert table[id(outer)] is Affinity.AFFINE
    assert table[id(inner)] is Affinity.AFFINE
    assert table[id(m)] is Affinity.AFFINE
    assert table[id(literal_one)] is Affinity.AFFINE
    assert table[id(shape_sym)] is Affinity.AFFINE
    # outer, inner, m, literal_one, shape_sym = 5 entries.
    assert len(table) == 5


def test_affinity_table_records_non_affine_subtree() -> None:
    """For ``(m * n) + 1`` the inner ``*`` should be NON_AFFINE and the
    outer ``+`` should inherit that, while the leaves stay AFFINE."""
    m = RankVariable(name="m")
    n = RankVariable(name="n")
    one = RankConstantLiteral(value=1)
    inner = RankArith(op="*", lhs=m, rhs=n)
    outer = RankArith(op="+", lhs=inner, rhs=one)

    table = affinity_table(outer)

    assert table[id(outer)] is Affinity.NON_AFFINE
    assert table[id(inner)] is Affinity.NON_AFFINE
    assert table[id(m)] is Affinity.AFFINE
    assert table[id(n)] is Affinity.AFFINE
    assert table[id(one)] is Affinity.AFFINE


def test_affinity_table_walks_into_rank_function_args() -> None:
    """A RankFunction's args are themselves rank expressions and must
    appear in the table with their own Affinity, independently of the
    function's user-declared verdict."""
    arg = RankVariable(name="m")
    f = RankFunction(func_name="shift", args=[arg], is_affine=True)

    table = affinity_table(f)

    assert table[id(f)] is Affinity.AFFINE
    assert table[id(arg)] is Affinity.AFFINE
