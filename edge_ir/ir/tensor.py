"""
This class represents Tensors in Einsums.
It is generated from the "declaration" section of EDGE.

- DataType: The type of data stored in the tensor (e.g., int, float, bool).
- RankDeclaration: one rank with a name and (optional) shape.
  Note that it can be unboudned.
- TensorDeclaration: The tensor declaration
    G^{S≡|V|, D≡|V|} → integer, empty=0
    F^{S≡|V|} → float, empty=∞
    P^{D≡|V|} → Boolean, empty=False
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator

from edge_ir.ir import patterns
from edge_ir.ir.base import IRBase

# JSON has no Infinity/NaN literals, so non-finite floats can't survive
# strict-JSON serialization. We encode them as reserved string sentinels
# on empty_value (the only Any-typed field where they're expected).
# Reserved sentinels:  "inf", "-inf", "nan".
_FLOAT_SENTINELS: dict[str, float] = {
    "inf": math.inf,
    "-inf": -math.inf,
    "nan": math.nan,
    "infinity": math.inf,
    "-infinity": -math.inf,
}


def _decode_float_sentinel(v: Any) -> Any:
    """Decode a reserved float-sentinel string to its float value.

    See ``docs/keywords.md`` (§Float JSON sentinels) for the reserved
    set. Non-string inputs and strings outside the set pass through
    unchanged.
    """
    if isinstance(v, str) and v in _FLOAT_SENTINELS:
        return _FLOAT_SENTINELS[v]
    return v


def _encode_float_sentinel(v: Any) -> Any:
    """Encode a non-finite float as its reserved sentinel string.

    Inverse of ``_decode_float_sentinel``. Finite floats and non-float
    inputs pass through unchanged.
    """
    if isinstance(v, float) and not math.isfinite(v):
        if math.isnan(v):
            return "nan"
        return "inf" if v > 0 else "-inf"
    return v


class BuiltinDataType(IRBase):
    """A built-in data type: int, float, or bool."""

    kind: Literal["builtin"] = "builtin"
    name: Literal["int", "float", "bool"]


class UserDefinedDataType(IRBase):
    """A user-defined data type, identified by name.

    The IR carries only the name. The actual implementation is provided
    by the user at runtime via a type registry. The host language is
    not constrained: the implementation may be in Python, C, Rust, or
    any other language the user's binding mechanism supports. The IR
    is opaque to where the type's behavior lives.

    The empty_value field on TensorDeclaration is opaque for user-defined
    types: it can be any JSON-representable value, and the registered
    implementation is responsible for interpreting it.

    Answering the note that used to sit here ("we have to validate this in
    the interpreter -- or do we need a second validator upstream, because I
    don't want a compiler to have to deal with validation?"): the upstream
    instinct was right, and that second validator is the Layer 2 validator.
    Layer 1 (this Pydantic IR) checks structure only; Layer 2 runs after it
    and before ANY backend -- evaluator or compiler -- and does the
    cross-reference and type checks, including that an `empty_value` is a
    valid instance of its declared `data_type`. See D14 (the derivations are
    shared so two backends cannot disagree) and `edge_ir/validator/README.md`.
    So a compiler does not deal with validation; it consumes an
    already-validated Program. The reference evaluator still fails loudly on
    anything Layer 2 has not yet been taught to catch, rather than guessing.
    """

    kind: Literal["user_defined"] = "user_defined"
    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)


# DataType is a union: builtin or user-defined. The "kind"
# field tells Pydantic (and any future consumer like egglog or a backend
# compiler) which variant to construct from JSON.
# The Annotated alias does two things: it constrains what is admissible as a
# "data type", and it lets any class write `data_type: DataType` in its
# annotations instead of repeating the union and its discriminator.
DataType = Annotated[
    BuiltinDataType | UserDefinedDataType,  # union
    Field(discriminator="kind"),
]


################################################################################
# Rank Coordinate sets
################################################################################


# Instead of just a shape, we can also have an explicit coordinate set,
# which is either an enumeration, an interval,
# an alias to another tensor's coordinate set, or a named runtime set
# (CoordSetName) whose contents the host supplies at execution time.
# This is used for sparse and generational ranks.
class CoordSetEnum(IRBase):
    """Explicit enumeration of coordinates.

    Used when the rank coordinate set is explicit, e.g.,
    {0, 1, 5, 7} for a sparse coordinate set or
    {a, b, c, d, e} for vertex labels.

    Coordinates can be strings or integers. The grammar production
    <coordinate> := /[A-Za-z0-9]+/ permits both forms; in JSON we
    accept either.
    """

    kind: Literal["enum"] = "enum"
    coords: list[str | int]


class CoordSetInterval(IRBase):
    """Interval [lo, hi).

    Both endpoints are part of the same shape mini-DSL as
    RankDeclaration.shape: integers (e.g., 0, 5) or strings
    (e.g., "|V|", "N"). The evaluator resolves strings to
    concrete integers when constructing iteration spaces.

    Used for explicit dense ranges and for generative ranks
    where we know the start (e.g., i = 0) but not the end.
    """

    kind: Literal["interval"] = "interval"
    lo: int | str
    hi: int | str


class CoordSetAlias(IRBase):
    """Alias another tensor's rank coordinate set.

    Used for declarations like D^G = S^G, which say "rank D of the
    tensor being declared has the same coordinate set as rank S of
    tensor G."
    """

    kind: Literal["alias"] = "alias"
    tensor: str = Field(pattern=patterns.REGEX_TENSOR_NAME)
    rank: str = Field(pattern=patterns.REGEX_RANK_NAME)


class CoordSetName(IRBase):
    """A coordinate set referenced by name; contents bound at runtime by the host.

    Same convention as tensor names: the IR carries the string, the host
    runtime resolves it. No "runtime" tag on the IR. Admissible in both
    RankDeclaration.coord_set (a tensor declaring "my rank's coord set is
    whatever this name resolves to") and SetMembership.coord_set (a
    predicate testing membership in a host-supplied set, e.g. BFS's `id`).
    """

    kind: Literal["named"] = "named"
    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)


# A coordinate set is one of the four above. Discriminated by "kind".
CoordinateSet = Annotated[
    CoordSetEnum | CoordSetInterval | CoordSetAlias | CoordSetName,
    Field(discriminator="kind"),
]


# We define a rank as a rank name with a shape. Works for dense.
# If shape is None, it is either unbounded or a CSOverride (see below)
class RankDeclaration(IRBase):
    """One rank in a tensor declaration.

    A rank's coordinate space is described by the combination of
    `shape` and `coord_set`. Both are independently optional;
    together they specify what the iteration space for this rank
    looks like.

    `shape` gives a cardinality hint when one is known up front:
    an integer literal (e.g., 5), or a string from the shape
    mini-DSL (e.g., "|V|"). May be None for unbounded or
    generational ranks where the cardinality emerges from
    execution rather than being declared.

    `coord_set` gives an explicit coordinate set when needed:
    an enumeration (sparse subset, or non-integer labels), an
    interval (explicit dense range), or an alias (defer to
    another rank's coord set). When None, the rank uses the
    default dense integer coords [0, shape) implied by shape.
    """

    name: str = Field(pattern=patterns.REGEX_RANK_NAME)
    shape: int | str | None = None
    coord_set: CoordinateSet | None = None
    iterative: bool = False
    """This rank is the tensor's ITERATIVE rank (the paper's `I` in `F^{I,S}`).

    An iterative rank is the one a cascade advances a generation at a time:
    the cascade pins it to a single coordinate per iteration, so its
    coordinate set emerges from execution rather than being declared. That
    is why iterative ranks are the ranks declared with neither a shape nor a
    coordinate set -- but "no shape" alone is ambiguous (an unbounded rank
    is also shapeless), so the flag says which one it is rather than making
    consumers guess. The surface syntax marks it: widest-path's
    `tensor F[gen I, R=N]` writes it `gen`.

    Terminology: EDGE texts use both "generational
    rank" and "iterative rank". The EDGE syntax
    says "if a cascade is ITERATIVE" and then "a GENERATIONAL rank
    variable" in the same sentence; later EDGE text says
    "iterative rank". This IR uses "iterative"; older docstrings in this
    package still say "generational" and mean the same thing.
    """


################################################################################
# Coordinate sets
################################################################################


class TensorDeclaration(IRBase):
    """A complete tensor declaration.

    Maps to the grammar production:
        <tensor-name>^<shape-definition> -> <data-type>, empty=<empty-value>

    Scalars (0-rank tensors) are valid per paper Section 6: a scalar is a
    function from the singleton coordinate space {()} to the non-empty
    data space. This is why ranks has no min_length constraint.

    The empty_value is opaque from the IR's perspective. For builtin
    types, the evaluator will check that empty_value is a Python int,
    float, or bool. For user-defined types, the registered
    implementation is responsible for validating empty_value. This
    deferral keeps the IR pure data and lets the type system stay
    extensible.

    JSON encoding of non-finite floats: standard JSON has no Infinity
    or NaN literal, so float('inf'), float('-inf'), and float('nan')
    are serialized as the reserved strings "inf", "-inf", "nan" and
    decoded back to the corresponding float on load. User-defined
    types must not use those exact strings as legitimate empty values.
    """

    name: str = Field(pattern=patterns.REGEX_TENSOR_NAME)
    ranks: list[RankDeclaration]
    data_type: DataType
    # Required: every tensor must declare an empty value. It MAY be
    # null/None (an explicit "empty is null"), but it cannot be omitted.
    # To instead allow omission -- a missing empty_value defaulting to
    # null -- switch to the commented line below. Tradeoff: that conflates
    # "didn't specify" with "deliberately null"; the required form forces
    # an explicit choice and matches the paper's always-declared `empty=`.
    #     empty_value: Any = None
    empty_value: Any
    value: Any | None = None

    @field_validator("empty_value", mode="before")
    @classmethod
    def _decode_empty_value(cls, v: Any) -> Any:
        return _decode_float_sentinel(v)

    @field_serializer("empty_value")
    def _encode_empty_value(self, v: Any) -> Any:
        return _encode_float_sentinel(v)

    @field_validator("value", mode="before")
    @classmethod
    def _decode_value(cls, v: Any) -> Any:
        return _decode_float_sentinel(v)

    @field_serializer("value")
    def _encode_value(self, v: Any) -> Any:
        return _encode_float_sentinel(v)

    @model_validator(mode="after")
    def _value_only_for_zero_rank(self) -> TensorDeclaration:
        """The `value` field can only be set on a zero-rank tensor.

        Why: a zero-rank tensor has exactly one coordinate point (the empty
        tuple), so one `value` is a complete description of its contents. A
        tensor with ranks has many coordinate points; one `value` field
        can't describe a value at every point — that's what tensor data is
        for, supplied at runtime, not at declaration time.
        """
        if self.value is not None and self.ranks:
            raise ValueError(
                f"TensorDeclaration {self.name!r} has {len(self.ranks)} ranks "
                f"and also a `value` set. The `value` field is only allowed "
                f"on zero-rank tensors (tensors with no ranks). For tensors "
                f"with ranks, the actual data is supplied at runtime, not on "
                f"the declaration."
            )
        return self
