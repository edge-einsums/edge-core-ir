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
#
# This is the DECLARED form: a rank a user wrote down. Ranks produced by
# a transform use REGEX_DERIVED_RANK_NAME below; the two are deliberately
# kept apart so a name tells you which it is. Do not widen this pattern to
# admit digits -- that would make `C1` ambiguous between a declared rank
# literally named "C1" and the outer tile of `C`.
REGEX_RANK_NAME = r"^[A-Z]+$"

# <derived-rank-name> := <rank-name> ("." /[0-9]+/)+
# A rank PRODUCED by a transform rather than declared by a user.
# Partitioning `C` yields `C.1` (outer tile) and `C.0` (inner tile); the
# dot marks the name as generated.
#
# The form nests, so the partition tree is recoverable from the name:
# partitioning `C.1` again gives `C.1.0`, the inner tile of the outer
# tile. A convention that instead appends bare digits (`C1`, `C0`) loses
# this -- `C10` cannot be read back unambiguously.
#
# Consumers that need the provenance as data rather than as a string
# should carry it in fields alongside the name; this pattern only governs
# what a derived name may look like.
REGEX_DERIVED_RANK_NAME = r"^[A-Z]+(\.[0-9]+)+$"

# Either form, for fields that accept a rank name from any source.
REGEX_ANY_RANK_NAME = r"^[A-Z]+(\.[0-9]+)*$"

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
