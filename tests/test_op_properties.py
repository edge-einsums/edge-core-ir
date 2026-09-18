"""Tests for edge_ir.runtime.op_properties.

Locks down the operator-properties registry: sizes, contents, truth tables
for the 16 merge operators, lookup helpers, frozen-dataclass behavior, and
cross-registry consistency between the property tables and the Unicode
alias dicts.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from edge_ir.runtime.op_properties import (
    COMPUTE_OP_PROPERTIES,
    COMPUTE_OP_UNICODE,
    MERGE_OP_PROPERTIES,
    MERGE_OP_UNICODE,
    UNARY_OP_UNICODE,
    ComputeOpProperties,
    MergeOpProperties,
    get_compute_props,
    get_merge_props,
    render_compute_unicode,
    render_merge_unicode,
)

# The 16 merge ASCII names (Appendix A).
_MERGE_NAMES: list[str] = [
    "pass_through",
    "no_pass",
    "intersect",
    "take_left_only",
    "take_left",
    "take_right_only",
    "take_right",
    "xor",
    "union",
    "nor",
    "xnor",
    "not_right",
    "not_left",
    "implies_left",
    "implies_right",
    "nand",
]

# The 4 compute ASCII names.
_COMPUTE_NAMES: set[str] = {"+", "-", "*", "/"}


def _tt(nn: bool, ny: bool, yn: bool, yy: bool) -> dict[tuple[bool, bool], bool]:
    """Build a truth-table dict from four bits in NN, NY, YN, YY order."""
    return {
        (False, False): nn,
        (False, True): ny,
        (True, False): yn,
        (True, True): yy,
    }


# Expected truth-table / commutative / includes_neither for each merge op.
# Order: NN, NY, YN, YY (where N=False, Y=True).
_MERGE_EXPECTATIONS: list[tuple[str, dict[tuple[bool, bool], bool], bool, bool]] = [
    ("pass_through", _tt(True, True, True, True), True, True),
    ("no_pass", _tt(False, False, False, False), True, False),
    ("intersect", _tt(False, False, False, True), True, False),
    ("take_left_only", _tt(False, False, True, False), False, False),
    ("take_left", _tt(False, False, True, True), False, False),
    ("take_right_only", _tt(False, True, False, False), False, False),
    ("take_right", _tt(False, True, False, True), False, False),
    ("xor", _tt(False, True, True, False), True, False),
    ("union", _tt(False, True, True, True), True, False),
    ("nor", _tt(True, False, False, False), True, True),
    ("xnor", _tt(True, False, False, True), True, True),
    ("not_right", _tt(True, False, True, False), False, True),
    ("not_left", _tt(True, True, False, False), False, True),
    ("implies_left", _tt(True, False, True, True), False, True),
    ("implies_right", _tt(True, True, False, True), False, True),
    ("nand", _tt(True, True, True, False), True, True),
]


################################################################################
# Registry sizes
################################################################################


def test_compute_op_properties_has_four_entries() -> None:
    assert len(COMPUTE_OP_PROPERTIES) == 4


def test_merge_op_properties_has_sixteen_entries() -> None:
    assert len(MERGE_OP_PROPERTIES) == 16


def test_compute_op_unicode_has_four_entries() -> None:
    assert len(COMPUTE_OP_UNICODE) == 4


def test_merge_op_unicode_has_sixteen_entries() -> None:
    assert len(MERGE_OP_UNICODE) == 16


def test_unary_op_unicode_has_one_entry() -> None:
    assert len(UNARY_OP_UNICODE) == 1


################################################################################
# Compute-operator property contents
################################################################################


@pytest.mark.parametrize(
    "symbol, expected_commutative, expected_associative, expected_identity",
    [
        ("+", True, True, 0),
        ("-", False, False, None),
        ("*", True, True, 1),
        ("/", False, False, None),
    ],
)
def test_compute_op_properties_match_spec(
    symbol: str,
    expected_commutative: bool,
    expected_associative: bool,
    expected_identity: int | float | None,
) -> None:
    """Each compute operator must report the algebraic properties given in
    the EDGE spec: commutativity, associativity, and two-sided identity."""
    props = COMPUTE_OP_PROPERTIES[symbol]
    assert props.symbol == symbol
    assert props.commutative is expected_commutative
    assert props.associative is expected_associative
    assert props.identity == expected_identity


def test_multiplication_has_zero_as_left_and_right_absorbing() -> None:
    """`*` is the only compute op with absorbing elements; both sides are 0."""
    props = COMPUTE_OP_PROPERTIES["*"]
    assert props.left_absorbing == 0
    assert props.right_absorbing == 0


@pytest.mark.parametrize("symbol", ["+", "-", "/"])
def test_non_multiplicative_compute_ops_have_no_absorbing_elements(
    symbol: str,
) -> None:
    """Only `*` declares absorbing elements; the other three must be None on
    both sides — guards against accidental cross-contamination if someone
    edits the registry."""
    props = COMPUTE_OP_PROPERTIES[symbol]
    assert props.left_absorbing is None
    assert props.right_absorbing is None


################################################################################
# Merge-operator truth-table walk
################################################################################


@pytest.mark.parametrize(
    "name, expected_truth_table, expected_commutative, expected_includes_neither",
    [(n, tt, c, inc) for (n, tt, c, inc) in _MERGE_EXPECTATIONS],
    ids=[name for (name, _, _, _) in _MERGE_EXPECTATIONS],
)
def test_merge_op_truth_table_and_flags_match_spec(
    name: str,
    expected_truth_table: dict[tuple[bool, bool], bool],
    expected_commutative: bool,
    expected_includes_neither: bool,
) -> None:
    """Every merge operator's full 4-entry truth table, commutativity flag,
    and includes_neither flag must exactly match the Appendix A spec.

    The truth table is the operator's *defining* semantics, so this is the
    most load-bearing test in the file: a flipped bit here changes which
    coordinates appear in the output fibertree.
    """
    props = MERGE_OP_PROPERTIES[name]
    assert props.symbol == name
    assert props.truth_table == expected_truth_table
    assert props.commutative is expected_commutative
    assert props.includes_neither is expected_includes_neither


################################################################################
# Spot checks on includes_neither
################################################################################


def test_pass_through_includes_neither_is_true() -> None:
    """pass_through emits a coordinate even when both inputs are empty —
    this is the marker that the operator is empty-stable."""
    assert MERGE_OP_PROPERTIES["pass_through"].includes_neither is True


def test_intersect_includes_neither_is_false() -> None:
    """intersect drops the (empty, empty) case — its output is the
    intersection of the two presence sets."""
    assert MERGE_OP_PROPERTIES["intersect"].includes_neither is False


def test_nor_includes_neither_is_true() -> None:
    """nor is one of the upper-half operators that admits (empty, empty)."""
    assert MERGE_OP_PROPERTIES["nor"].includes_neither is True


################################################################################
# Lookup helpers
################################################################################


def test_get_compute_props_returns_registry_entry_identity() -> None:
    """The helper must return the *same object* held by the registry; no
    copying, so callers can rely on identity comparisons."""
    assert get_compute_props("+") is COMPUTE_OP_PROPERTIES["+"]


def test_get_compute_props_raises_key_error_on_unknown_symbol() -> None:
    with pytest.raises(KeyError):
        get_compute_props("nope")


def test_get_merge_props_intersect_is_commutative() -> None:
    assert get_merge_props("intersect").commutative is True


def test_get_merge_props_raises_key_error_on_unknown_symbol() -> None:
    with pytest.raises(KeyError):
        get_merge_props("not-a-real-op")


def test_render_compute_unicode_multiplication_is_times_sign() -> None:
    """`*` renders as the Unicode multiplication sign U+00D7, not 'x' or '*'."""
    assert render_compute_unicode("*") == "×"


def test_render_merge_unicode_intersect_is_intersection_sign() -> None:
    """intersect renders as the Unicode intersection sign U+2229."""
    assert render_merge_unicode("intersect") == "∩"


################################################################################
# Cross-registry consistency
################################################################################


def test_compute_property_keys_match_unicode_keys() -> None:
    """Each operator known to the property registry must have a Unicode
    rendering, and vice versa — guards against drift if one side is updated
    but the other is forgotten."""
    assert set(COMPUTE_OP_PROPERTIES.keys()) == set(COMPUTE_OP_UNICODE.keys())


def test_merge_property_keys_match_unicode_keys() -> None:
    assert set(MERGE_OP_PROPERTIES.keys()) == set(MERGE_OP_UNICODE.keys())


def test_compute_keys_are_exactly_the_four_arithmetic_symbols() -> None:
    """Pin the canonical compute-symbol set: `+`, `-`, `*`, `/`."""
    assert set(COMPUTE_OP_PROPERTIES.keys()) == _COMPUTE_NAMES


def test_merge_keys_are_exactly_the_sixteen_appendix_a_names() -> None:
    """Pin the canonical merge-name set from Appendix A. Adding or removing
    an operator must be a deliberate spec change, caught here."""
    assert set(MERGE_OP_PROPERTIES.keys()) == set(_MERGE_NAMES)


################################################################################
# Frozen-dataclass behavior
################################################################################


def test_compute_op_properties_is_frozen() -> None:
    """ComputeOpProperties is declared `frozen=True`; mutation must raise
    FrozenInstanceError (or AttributeError on some Python versions)."""
    props = COMPUTE_OP_PROPERTIES["+"]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        props.commutative = False  # type: ignore[misc]


def test_merge_op_properties_is_frozen() -> None:
    """MergeOpProperties is declared `frozen=True`; mutation must raise."""
    props = MERGE_OP_PROPERTIES["intersect"]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        props.commutative = False  # type: ignore[misc]


################################################################################
# Module-level coupling smoke test
################################################################################


def test_op_properties_module_does_not_depend_on_ir_module() -> None:
    """Importing op_properties from a fresh interpreter state must not
    pull in any edge_ir.ir.* module — the registry is supposed to be a
    leaf module that the IR-side code consults, not the other way around.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import edge_ir.runtime.op_properties  # noqa: F401\n"
        "leaked = [m for m in sys.modules if m.startswith('edge_ir.ir')]\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_compute_op_properties_constructor_field_order() -> None:
    """ComputeOpProperties must accept (symbol, commutative, associative)
    with the optional identity/absorbing fields defaulting to None."""
    p = ComputeOpProperties(symbol="x", commutative=True, associative=False)
    assert p.symbol == "x"
    assert p.commutative is True
    assert p.associative is False
    assert p.identity is None
    assert p.left_absorbing is None
    assert p.right_absorbing is None


def test_merge_op_properties_constructor_positional_order() -> None:
    """MergeOpProperties is constructed positionally throughout the registry
    as (symbol, truth_table, commutative, includes_neither). Lock that
    ordering so a future field reorder doesn't silently shift values."""
    tt = _tt(False, True, True, False)
    p = MergeOpProperties("x", tt, True, False)
    assert p.symbol == "x"
    assert p.truth_table == tt
    assert p.commutative is True
    assert p.includes_neither is False
