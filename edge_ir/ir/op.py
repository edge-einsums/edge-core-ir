"""Operators in EDGE expressions.

Four categories per paper Section 5.4 and the grammar:
- Compute operators: combine two values to produce one (+, -, *, /, or UDF)
- Merge operators: combine presence flags from two operands; 16 builtins
  per Appendix A, plus user-defined
- Coordinate operators: produce a coordinate set from a value and the
  current output fiber; user-defined only
- Unary operators: transform a single tensor element (negation builtin,
  plus user-defined)

All operator names are ASCII identifiers in the IR. The Unicode
symbols used in the paper (e.g., the math symbols for merge
operators) are translated to ASCII names by the future parser. The
IR-side ASCII names are listed below; the registry in
op_properties.py records the Unicode-to-ASCII mapping for any
consumer that needs to render in paper notation.

Each category has its own discriminated union of builtin (with a fixed
symbol) and user-defined (with a name). Operator properties
(commutativity, associativity, identity, etc.) are NOT stored on the
operator nodes themselves; they live in the registry at
edge_ir/runtime/op_properties.py and are consulted by the evaluator and
future egglog rewriter.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from edge_ir.ir import patterns
from edge_ir.ir.base import IRBase

################################################################################
# Compute operators
################################################################################


class BuiltinComputeOp(IRBase):
    """Built-in compute operator.

    Per the grammar:
        <compute-operator> := "+" | "-" | "*" | "/" | <user-defined-function>

    The Unicode symbols in the paper (multiplication, division) are
    normalized to ASCII (*, /) by the parser before reaching the IR.
    """

    kind: Literal["builtin"] = "builtin"
    symbol: Literal["+", "-", "*", "/"]


class UserDefinedComputeOp(IRBase):
    """User-defined compute operator, identified by name."""

    kind: Literal["user_defined"] = "user_defined"
    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)


ComputeOp = Annotated[
    BuiltinComputeOp | UserDefinedComputeOp,
    Field(discriminator="kind"),
]


################################################################################
# Merge operators (16 builtins per Appendix A)
################################################################################


class BuiltinMergeOp(IRBase):
    """Built-in merge operator. There are 16 of these, one for each
    truth table on two Boolean inputs (presence flags for left and
    right operands).

    The 16 ASCII names map to the paper's Unicode symbols as recorded
    in op_properties.py. The truth table for each is in Appendix A
    of the paper.

    Note on naming: the paper's `take_left` symbol can appear as either a
    merge operator (one of the 16 built-ins listed here as
    "take_left") or as a user-defined compute operator (a function
    that returns its left argument value). These are different
    operator categories despite sharing the paper symbol; the IR
    distinguishes them by which field they appear in (a MapSpec's
    merge_op vs its compute_op). Same for "take_right".

    Names (in paper appendix order):
    - "pass_through": output present for all inputs
    - "no_pass": output never present
    - "intersect": output present iff both inputs present
    - "take_left_only": present iff left only
    - "take_left": present iff left present
    - "take_right_only": present iff right only
    - "take_right": present iff right present
    - "xor": present iff exactly one input present
    - "union": present iff at least one input present
    - "nor": present iff neither input present
    - "xnor": present iff both inputs same presence
    - "not_right": present iff right not present
    - "not_left": present iff left not present
    - "implies_left": present iff right implies left
    - "implies_right": present iff left implies right
    - "nand": present iff not both inputs present
    """

    kind: Literal["builtin"] = "builtin"
    symbol: Literal[
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


class UserDefinedMergeOp(IRBase):
    """User-defined merge operator, identified by name."""

    kind: Literal["user_defined"] = "user_defined"
    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)


MergeOp = Annotated[
    BuiltinMergeOp | UserDefinedMergeOp,
    Field(discriminator="kind"),
]


################################################################################
# Coordinate operators
################################################################################


class CoordinateOp(IRBase):
    """Coordinate operator for the Populate action.

    Per the grammar:
        <coordinate-operator> := <user-defined-function>

    All coordinate operators are user-defined (no built-in). Used only
    in the Populate action to produce a coordinate set from a computed
    value and the current output fiber state.
    """

    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)


################################################################################
# Unary operators
################################################################################


class BuiltinUnaryOp(IRBase):
    """Built-in unary operator: negation.

    Per the grammar:
        <unary-operator> := "not" | <user-defined-function>

    The paper uses the Unicode negation symbol; the IR uses the ASCII
    name "not".
    """

    kind: Literal["builtin"] = "builtin"
    symbol: Literal["not"]


class UserDefinedUnaryOp(IRBase):
    """User-defined unary operator (e.g., AND-with-something, sqrt, etc.)."""

    kind: Literal["user_defined"] = "user_defined"
    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)


UnaryOp = Annotated[
    BuiltinUnaryOp | UserDefinedUnaryOp,
    Field(discriminator="kind"),
]
