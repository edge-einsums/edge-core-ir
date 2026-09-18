"""Operator properties registry.

Maps built-in operators to their algebraic properties: commutativity,
associativity, identity element, and empty-value handling. Also records
the Unicode-to-ASCII name mapping for operators that the paper writes
in mathematical notation but the IR represents in ASCII. Consumers
(evaluator, egglog rewriter, future static analyses, paper-style
pretty-printers) consult this registry; the IR itself does not.

The IR is the discriminator for what operator is being used (the
ASCII symbol on a BuiltinComputeOp, etc.). The registry is the lookup
table for what that symbol means semantically and how to render it in
mathematical notation if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

################################################################################
# Unicode aliases (for consumers that render in paper notation)
################################################################################


# Compute operators. ASCII name (used in IR) -> Unicode symbol (paper).
COMPUTE_OP_UNICODE: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": "×",  # multiplication sign
    "/": "÷",  # division sign
}


# Merge operators. ASCII name (used in IR) -> Unicode symbol (paper).
# Symbols come from Appendix A of the EDGE paper.
MERGE_OP_UNICODE: dict[str, str] = {
    "pass_through": "T",
    "no_pass": "F",
    "intersect": "∩",
    "take_left_only": "⊖l",
    "take_left": "←",
    "take_right_only": "⊖r",
    "take_right": "→",
    "xor": "⊕",
    "union": "∪",
    "nor": "∪̄",
    "xnor": "≡",
    "not_right": "↛",
    "not_left": "↚",
    "implies_left": "⇐",
    "implies_right": "⇒",
    "nand": "∩̄",
}


# Unary operators.
UNARY_OP_UNICODE: dict[str, str] = {
    "not": "¬",
}


################################################################################
# Compute operator properties
################################################################################


@dataclass(frozen=True)
class ComputeOpProperties:
    """Properties of a compute operator.

    - commutative: f(a, b) == f(b, a)
    - associative: f(f(a, b), c) == f(a, f(b, c))
    - identity: a value e such that f(a, e) == f(e, a) == a; None if
      no two-sided identity exists
    - left_absorbing: a value z such that f(z, a) == z for all a; None
      if no left absorbing element
    - right_absorbing: similarly for right
    """

    symbol: str
    commutative: bool
    associative: bool
    identity: int | float | None = None
    left_absorbing: int | float | None = None
    right_absorbing: int | float | None = None


COMPUTE_OP_PROPERTIES: dict[str, ComputeOpProperties] = {
    "+": ComputeOpProperties(
        symbol="+",
        commutative=True,
        associative=True,
        identity=0,
    ),
    "-": ComputeOpProperties(
        symbol="-",
        commutative=False,
        associative=False,
    ),
    "*": ComputeOpProperties(
        symbol="*",
        commutative=True,
        associative=True,
        identity=1,
        left_absorbing=0,
        right_absorbing=0,
    ),
    "/": ComputeOpProperties(
        symbol="/",
        commutative=False,
        associative=False,
    ),
}


################################################################################
# Merge operator properties
################################################################################


@dataclass(frozen=True)
class MergeOpProperties:
    """Properties of a merge operator.

    Merge operators are 2-input boolean functions (left presence,
    right presence) producing output presence. They have a truth table
    that determines the output for each input combination.

    - truth_table: maps (left_present, right_present) to output_present
    - commutative: swapping inputs gives the same result
    - includes_neither: True if (False, False) is in the output set
      (i.e., the operator admits points where both operands are empty;
      affects whether compute is invoked with empty values per
      paper Section 6.4)
    """

    symbol: str
    truth_table: dict[tuple[bool, bool], bool]
    commutative: bool
    includes_neither: bool


def _tt(nn: bool, ny: bool, yn: bool, yy: bool) -> dict[tuple[bool, bool], bool]:
    """Helper: truth table from four bits (no/no, no/yes, yes/no, yes/yes)."""
    return {
        (False, False): nn,
        (False, True): ny,
        (True, False): yn,
        (True, True): yy,
    }


MERGE_OP_PROPERTIES: dict[str, MergeOpProperties] = {
    "pass_through": MergeOpProperties(
        "pass_through", _tt(True, True, True, True), True, True
    ),
    "no_pass": MergeOpProperties(
        "no_pass", _tt(False, False, False, False), True, False
    ),
    "intersect": MergeOpProperties(
        "intersect", _tt(False, False, False, True), True, False
    ),
    "take_left_only": MergeOpProperties(
        "take_left_only", _tt(False, False, True, False), False, False
    ),
    "take_left": MergeOpProperties(
        "take_left", _tt(False, False, True, True), False, False
    ),
    "take_right_only": MergeOpProperties(
        "take_right_only", _tt(False, True, False, False), False, False
    ),
    "take_right": MergeOpProperties(
        "take_right", _tt(False, True, False, True), False, False
    ),
    "xor": MergeOpProperties("xor", _tt(False, True, True, False), True, False),
    "union": MergeOpProperties("union", _tt(False, True, True, True), True, False),
    "nor": MergeOpProperties("nor", _tt(True, False, False, False), True, True),
    "xnor": MergeOpProperties("xnor", _tt(True, False, False, True), True, True),
    "not_right": MergeOpProperties(
        "not_right", _tt(True, False, True, False), False, True
    ),
    "not_left": MergeOpProperties(
        "not_left", _tt(True, True, False, False), False, True
    ),
    "implies_left": MergeOpProperties(
        "implies_left", _tt(True, False, True, True), False, True
    ),
    "implies_right": MergeOpProperties(
        "implies_right", _tt(True, True, False, True), False, True
    ),
    "nand": MergeOpProperties("nand", _tt(True, True, True, False), True, True),
}


################################################################################
# Lookup helpers
################################################################################


def get_compute_props(symbol: str) -> ComputeOpProperties:
    """Look up properties for a built-in compute operator.

    Raises KeyError if symbol is not a built-in compute operator.
    """
    return COMPUTE_OP_PROPERTIES[symbol]


def get_merge_props(symbol: str) -> MergeOpProperties:
    """Look up properties for a built-in merge operator.

    Raises KeyError if symbol is not a built-in merge operator.
    """
    return MERGE_OP_PROPERTIES[symbol]


def render_compute_unicode(symbol: str) -> str:
    """Render a built-in compute operator in the paper's Unicode notation."""
    return COMPUTE_OP_UNICODE[symbol]


def render_merge_unicode(symbol: str) -> str:
    """Render a built-in merge operator in the paper's Unicode notation."""
    return MERGE_OP_UNICODE[symbol]
