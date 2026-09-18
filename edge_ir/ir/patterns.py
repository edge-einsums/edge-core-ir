"""
Regex patterns for names in the IR.
Keep this file in sync with the grammar. When the grammar changes,
update here first, then propagate to IR nodes.
Patterns are from my EDGE ebnf file.
I probably shoudl have used lark for this?
"""

from __future__ import annotations

# <rank-name> := /[A-Z]+/
# Uppercase letters only. Used for rank names in tensor declarations
# and rank variables in expressions.
REGEX_RANK_NAME = r"^[A-Z]+$"

# <tensor-name> := /[A-Z]+/ /[A-Za-z0-9_]*/
# Starts with an uppercase letter, followed by any alphanumeric or
# underscore. Underscore is permitted so parser-synthesized literal
# names like `Lit_0`, `Lit_c` can live alongside user-declared tensor
# names without collision. Used for tensor names in declarations and
# tensor projections.
REGEX_TENSOR_NAME = r"^[A-Z][A-Za-z0-9_]*$"

# <coordinate> := /[A-Za-z0-9]+/
# Used for explicit coordinate values in coordinate-set enumerations
# (e.g., {a, b, c, d, e} for vertex labels in a graph).
REGEX_COORDINATE = r"^[A-Za-z0-9]+$"

# <user-defined-function> := /[A-Za-z][A-Za-z0-9_-]*/
# Identifier-like syntax extended to allow hyphens, matching the
# paper's own UDF examples (e.g. "select-any-s", "select-min-s",
# "user-rand-func"). Used for compute, merge, coordinate, and rank-
# mapping function names, plus user-defined data type names.
# IR-side first-char allows a leading underscore as well; EBNF is
# stricter (letter only). See debug.md Finding 4.
REGEX_USER_DEFINED_NAME = r"^[A-Za-z_][A-Za-z0-9_-]*$"

# <binary-label> := /[0-9]+/
# Numeric label on binary operators in expressions. Used for matching
# binary operations to their computation specs.
REGEX_BINARY_LABEL = r"^[0-9]+$"

# <rank-variable> := /[a-z]+/ /[A-Za-z0-9]*/
# Lowercase start, then any alphanumerics. Used for rank variables in
# expressions and iteration specs.
REGEX_RANK_VARIABLE = r"^[a-z]+[A-Za-z0-9]*$"
