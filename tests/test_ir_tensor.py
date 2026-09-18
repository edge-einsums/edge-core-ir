"""Tests for tensor declaration IR nodes."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import TypeAdapter, ValidationError

from edge_ir.ir.tensor import (
    # DataType for Data Space
    BuiltinDataType,
    CoordinateSet,
    # Rank Coordinate Set
    CoordSetAlias,
    CoordSetEnum,
    CoordSetInterval,
    CoordSetName,
    RankDeclaration,
    #  Tensor Declaration
    TensorDeclaration,
    UserDefinedDataType,
)

################################################################################
# BuiltinDataType
################################################################################


def test_builtin_int() -> None:
    dt = BuiltinDataType(name="int")
    assert dt.kind == "builtin"
    assert dt.name == "int"


def test_builtin_float() -> None:
    dt = BuiltinDataType(name="float")
    assert dt.name == "float"


def test_builtin_bool() -> None:
    dt = BuiltinDataType(name="bool")
    assert dt.name == "bool"


def test_builtin_rejects_unknown_name() -> None:
    with pytest.raises(ValidationError):
        # should raise an error b/c this is not in my list of
        # built-in datatypes.
        BuiltinDataType(name="complex")  # type: ignore[arg-type]


def test_builtin_round_trips_through_json() -> None:
    dt = BuiltinDataType(name="int")
    serialized = dt.model_dump_json()
    deserialized = BuiltinDataType.model_validate_json(serialized)
    assert dt == deserialized


################################################################################
# UserDefinedDataType
################################################################################


def test_user_defined_type() -> None:
    dt = UserDefinedDataType(name="VertexLabel")
    assert dt.kind == "user_defined"
    assert dt.name == "VertexLabel"


def test_user_defined_accepts_underscore_start() -> None:
    dt = UserDefinedDataType(name="_private_type")
    assert dt.name == "_private_type"


def test_user_defined_rejects_digit_start() -> None:
    with pytest.raises(ValidationError):
        UserDefinedDataType(name="1Type")


def test_user_defined_data_type_accepts_hyphen() -> None:
    """Per Finding 4, REGEX_USER_DEFINED_NAME was relaxed to permit
    hyphens in the body so paper examples like 'select-any-s' are
    legal. A hyphen in the body of a user-defined type name must be
    accepted (leading hyphen is still rejected)."""
    dt = UserDefinedDataType(name="My-Type")
    assert dt.name == "My-Type"


def test_user_defined_data_type_rejects_space() -> None:
    """Spaces are not in REGEX_USER_DEFINED_NAME's character class,
    so they must still be rejected even after the hyphen relaxation."""
    with pytest.raises(ValidationError):
        UserDefinedDataType(name="My Type")


def test_user_defined_round_trips_through_json() -> None:
    dt = UserDefinedDataType(name="VertexLabel")
    serialized = dt.model_dump_json()
    deserialized = UserDefinedDataType.model_validate_json(serialized)
    assert dt == deserialized


################################################################################
# RankDeclaration: shape only
################################################################################


def test_rank_declaration_with_string_shape() -> None:
    """Shape is a string in the mini-DSL (e.g., '|V|')."""
    r = RankDeclaration(name="S", shape="|V|")
    assert r.name == "S"
    assert r.shape == "|V|"
    assert r.coord_set is None


def test_rank_declaration_with_int_shape() -> None:
    r = RankDeclaration(name="K", shape=3)
    assert r.shape == 3


def test_rank_declaration_unbounded() -> None:
    """Unbounded ranks have no shape (e.g., generational ranks)."""
    r = RankDeclaration(name="I", shape=None)
    assert r.shape is None


def test_rank_declaration_default_shape_is_none() -> None:
    r = RankDeclaration(name="M")
    assert r.shape is None
    assert r.coord_set is None


def test_rank_declaration_rejects_lowercase_name() -> None:
    with pytest.raises(ValidationError):
        RankDeclaration(name="lowercase")


def test_rank_declaration_rejects_mixed_case_name() -> None:
    with pytest.raises(ValidationError):
        RankDeclaration(name="Sx")


# I may want to change this later. For example: m.0 or m0
def test_rank_declaration_rejects_digits_in_name() -> None:
    with pytest.raises(ValidationError):
        RankDeclaration(name="S1")


def test_rank_declaration_round_trips_through_json() -> None:
    r = RankDeclaration(name="S", shape="|V|")
    serialized = r.model_dump_json()
    deserialized = RankDeclaration.model_validate_json(serialized)
    assert r == deserialized


################################################################################
# CoordinateSet variants
################################################################################


def test_coord_set_enum_strings() -> None:
    cs = CoordSetEnum(coords=["a", "b", "c", "d", "e"])
    assert cs.kind == "enum"
    assert cs.coords == ["a", "b", "c", "d", "e"]


def test_coord_set_enum_ints() -> None:
    cs = CoordSetEnum(coords=[0, 1, 5, 7])
    assert cs.coords == [0, 1, 5, 7]


def test_coord_set_enum_round_trips() -> None:
    cs = CoordSetEnum(coords=["a", "b", "c"])
    serialized = cs.model_dump_json()
    deserialized = CoordSetEnum.model_validate_json(serialized)
    assert cs == deserialized


def test_coord_set_interval_int_bounds() -> None:
    cs = CoordSetInterval(lo=0, hi=5)
    assert cs.lo == 0
    assert cs.hi == 5


def test_coord_set_interval_string_bounds() -> None:
    """Bounds can be expressions in the shape mini-DSL."""
    cs = CoordSetInterval(lo=0, hi="|V|")
    assert cs.hi == "|V|"


def test_coord_set_interval_round_trips() -> None:
    cs = CoordSetInterval(lo="N", hi="M")
    serialized = cs.model_dump_json()
    deserialized = CoordSetInterval.model_validate_json(serialized)
    assert cs == deserialized


def test_coord_set_alias() -> None:
    """D^G = S^G: rank D aliases rank S of tensor G."""
    cs = CoordSetAlias(tensor="G", rank="S")
    assert cs.tensor == "G"
    assert cs.rank == "S"


def test_coord_set_alias_round_trips() -> None:
    cs = CoordSetAlias(tensor="G", rank="S")
    serialized = cs.model_dump_json()
    deserialized = CoordSetAlias.model_validate_json(serialized)
    assert cs == deserialized


def test_coord_set_alias_rejects_lowercase_tensor() -> None:
    with pytest.raises(ValidationError):
        CoordSetAlias(tensor="g", rank="S")


def test_coord_set_alias_rejects_lowercase_rank() -> None:
    with pytest.raises(ValidationError):
        CoordSetAlias(tensor="G", rank="s")


################################################################################
# CoordSetName: host-supplied named coordinate set
################################################################################


def test_coord_set_name_basic_construction() -> None:
    """A CoordSetName carries an identifier name and a 'named' kind tag."""
    cs = CoordSetName(name="id")
    assert cs.kind == "named"
    assert cs.name == "id"


def test_coord_set_name_round_trips_through_json() -> None:
    cs = CoordSetName(name="id")
    serialized = cs.model_dump_json()
    deserialized = CoordSetName.model_validate_json(serialized)
    assert cs == deserialized


def test_coord_set_name_rejects_special_chars() -> None:
    """A name containing characters outside REGEX_USER_DEFINED_NAME's
    character class must be rejected."""
    with pytest.raises(
        ValidationError,
        match=r"String should match pattern '\^\[A-Za-z_\]\[A-Za-z0-9_-\]\*\$'",
    ):
        CoordSetName(name="<<")


def test_coord_set_name_rejects_empty_string() -> None:
    """An empty string violates REGEX_USER_DEFINED_NAME (requires >=1
    char)."""
    with pytest.raises(
        ValidationError,
        match=r"String should match pattern '\^\[A-Za-z_\]\[A-Za-z0-9_-\]\*\$'",
    ):
        CoordSetName(name="")


def test_coord_set_name_admissible_in_rank_declaration_coord_set() -> None:
    """A RankDeclaration's coord_set field accepts a CoordSetName (the
    union grew to four variants); round-trips preserve it."""
    r = RankDeclaration(
        name="S",
        shape="|V|",
        coord_set=CoordSetName(name="some_set"),
    )
    assert isinstance(r.coord_set, CoordSetName)
    assert r.coord_set.name == "some_set"
    deserialized = RankDeclaration.model_validate_json(r.model_dump_json())
    assert deserialized == r
    assert isinstance(deserialized.coord_set, CoordSetName)


def test_coord_set_union_round_trips_all_four_variants() -> None:
    """A list mixing all four CoordinateSet variants survives a round
    trip through TypeAdapter[list[CoordinateSet]]: each one dispatches
    back to the same concrete type."""
    adapter: TypeAdapter[list[CoordinateSet]] = TypeAdapter(list[CoordinateSet])
    variants: list[CoordinateSet] = [
        CoordSetEnum(coords=["a", "b"]),
        CoordSetInterval(lo=0, hi="|V|"),
        CoordSetAlias(tensor="G", rank="S"),
        CoordSetName(name="id"),
    ]
    dumped = adapter.dump_json(variants)
    parsed = adapter.validate_json(dumped)
    assert len(parsed) == 4
    assert isinstance(parsed[0], CoordSetEnum)
    assert isinstance(parsed[1], CoordSetInterval)
    assert isinstance(parsed[2], CoordSetAlias)
    assert isinstance(parsed[3], CoordSetName)
    assert parsed == variants


def test_coord_set_dispatches_to_named_in_rank_declaration() -> None:
    """A JSON-serialized RankDeclaration with coord_set kind='named'
    must dispatch back to CoordSetName."""
    raw = json.dumps(
        {
            "name": "S",
            "shape": "|V|",
            "coord_set": {"kind": "named", "name": "id"},
        }
    )
    rank = RankDeclaration.model_validate_json(raw)
    assert isinstance(rank.coord_set, CoordSetName)
    assert rank.coord_set.name == "id"


################################################################################
# RankDeclaration with explicit coord_set
################################################################################


def test_rank_declaration_with_enum_coord_set() -> None:
    """A dense rank with explicit non-integer labels (e.g., vertex names)."""
    r = RankDeclaration(
        name="S",
        shape="|V|",
        coord_set=CoordSetEnum(coords=["a", "b", "c", "d", "e"]),
    )
    assert r.coord_set is not None
    assert r.coord_set.kind == "enum"


def test_rank_declaration_with_interval_coord_set() -> None:
    """An explicit interval, e.g., for generational ranks."""
    r = RankDeclaration(
        name="I",
        shape=None,
        coord_set=CoordSetInterval(lo=0, hi="|V|"),
    )
    assert r.coord_set is not None
    assert r.coord_set.kind == "interval"


def test_rank_declaration_with_alias_coord_set() -> None:
    """D^G = S^G: rank D of G aliases rank S of G."""
    r = RankDeclaration(
        name="D",
        shape="|V|",
        coord_set=CoordSetAlias(tensor="G", rank="S"),
    )
    assert r.coord_set is not None
    assert r.coord_set.kind == "alias"


def test_rank_declaration_with_coord_set_round_trips() -> None:
    r = RankDeclaration(
        name="S",
        shape="|V|",
        coord_set=CoordSetEnum(coords=["a", "b", "c"]),
    )
    serialized = r.model_dump_json()
    deserialized = RankDeclaration.model_validate_json(serialized)
    assert r == deserialized


################################################################################
# CoordinateSet discriminated dispatch from JSON
################################################################################


def test_coord_set_dispatches_to_enum() -> None:
    raw_json = json.dumps(
        {
            "name": "S",
            "shape": "|V|",
            "coord_set": {"kind": "enum", "coords": ["a", "b", "c"]},
        }
    )
    rank = RankDeclaration.model_validate_json(raw_json)
    assert isinstance(rank.coord_set, CoordSetEnum)
    assert rank.coord_set.coords == ["a", "b", "c"]


def test_coord_set_dispatches_to_interval() -> None:
    raw_json = json.dumps(
        {
            "name": "K",
            "shape": None,
            "coord_set": {"kind": "interval", "lo": 0, "hi": 10},
        }
    )
    rank = RankDeclaration.model_validate_json(raw_json)
    assert isinstance(rank.coord_set, CoordSetInterval)
    assert rank.coord_set.hi == 10


def test_coord_set_dispatches_to_alias() -> None:
    raw_json = json.dumps(
        {
            "name": "D",
            "shape": "|V|",
            "coord_set": {"kind": "alias", "tensor": "G", "rank": "S"},
        }
    )
    rank = RankDeclaration.model_validate_json(raw_json)
    assert isinstance(rank.coord_set, CoordSetAlias)
    assert rank.coord_set.rank == "S"


################################################################################
# TensorDeclaration: builtin types
################################################################################


def test_tensor_declaration_g_from_bfs() -> None:
    """The G tensor from BFS Example 2: graph as a 2D integer tensor."""
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    assert g.name == "G"
    assert len(g.ranks) == 2
    assert g.empty_value == 0


def test_tensor_declaration_p_from_bfs() -> None:
    """The P tensor from BFS Example 2: visited set as a Boolean tensor."""
    p = TensorDeclaration(
        name="P",
        ranks=[RankDeclaration(name="D", shape="|V|")],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    assert p.empty_value is False


def test_tensor_declaration_scalar() -> None:
    """Scalars (0-rank tensors) are valid per paper Section 6."""
    s = TensorDeclaration(
        name="X",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    assert s.ranks == []


def test_tensor_declaration_rejects_lowercase_name() -> None:
    with pytest.raises(ValidationError):
        TensorDeclaration(
            name="lowercase",
            ranks=[RankDeclaration(name="M")],
            data_type=BuiltinDataType(name="int"),
            empty_value=0,
        )


def test_tensor_declaration_round_trips_through_json() -> None:
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    serialized = g.model_dump_json()
    deserialized = TensorDeclaration.model_validate_json(serialized)
    assert g == deserialized


@pytest.mark.skip(reason="Layer 2 validator not implemented yet")
def test_layer2_rejects_string_empty_for_int_tensor() -> None:
    """Layer 2 validator should reject string empty_value for int tensor.

    The IR layer is intentionally opaque to empty_value semantics for
    user-defined types; type-consistency is checked upstream of consumers.
    When Layer 2 lands, this test becomes its specification.
    """
    pass


################################################################################
# TensorDeclaration with explicit coord sets on ranks
################################################################################


def test_tensor_declaration_with_explicit_coord_sets() -> None:
    """G with vertex labels and aliased D rank.

    From paper Example 23: rank S has explicit string coordinates,
    rank D aliases rank S. Each rank carries its own coord_set;
    no separate override list.
    """
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(
                name="S",
                shape="|V|",
                coord_set=CoordSetEnum(coords=["a", "b", "c", "d", "e"]),
            ),
            RankDeclaration(
                name="D",
                shape="|V|",
                coord_set=CoordSetAlias(tensor="G", rank="S"),
            ),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    assert g.ranks[0].coord_set is not None
    assert g.ranks[0].coord_set.kind == "enum"
    assert g.ranks[1].coord_set is not None
    assert g.ranks[1].coord_set.kind == "alias"


def test_tensor_declaration_with_explicit_coord_sets_round_trips() -> None:
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(
                name="D",
                shape="|V|",
                coord_set=CoordSetAlias(tensor="G", rank="S"),
            ),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    serialized = g.model_dump_json()
    deserialized = TensorDeclaration.model_validate_json(serialized)
    assert g == deserialized


################################################################################
# TensorDeclaration: user-defined types
################################################################################


def test_tensor_declaration_with_user_defined_type() -> None:
    """User-defined types are referenced by name; the IR is opaque to them."""
    g = TensorDeclaration(
        name="L",
        ranks=[RankDeclaration(name="V")],
        data_type=UserDefinedDataType(name="VertexLabel"),
        empty_value="",
    )
    assert g.data_type.name == "VertexLabel"
    assert g.empty_value == ""


def test_tensor_declaration_with_complex_empty_value() -> None:
    """Empty values can be arbitrary JSON for user-defined types."""
    s = TensorDeclaration(
        name="State",
        ranks=[RankDeclaration(name="N")],
        data_type=UserDefinedDataType(name="BFSEntry"),
        empty_value={"depth": -1, "parent": -1},
    )
    assert s.empty_value == {"depth": -1, "parent": -1}


def test_tensor_declaration_user_defined_round_trips_through_json() -> None:
    s = TensorDeclaration(
        name="State",
        ranks=[RankDeclaration(name="N")],
        data_type=UserDefinedDataType(name="BFSEntry"),
        empty_value={"depth": -1, "parent": -1},
    )
    serialized = s.model_dump_json()
    deserialized = TensorDeclaration.model_validate_json(serialized)
    assert s == deserialized


################################################################################
# TensorDeclaration: discriminated union dispatch
################################################################################


def test_tensor_declaration_dispatches_to_builtin() -> None:
    """The 'kind' field on JSON tells Pydantic which DataType variant to construct."""
    raw_json = json.dumps(
        {
            "name": "G",
            "ranks": [{"name": "S"}],
            "data_type": {"kind": "builtin", "name": "int"},
            "empty_value": 0,
        }
    )
    decl = TensorDeclaration.model_validate_json(raw_json)
    assert isinstance(decl.data_type, BuiltinDataType)
    assert decl.data_type.name == "int"


def test_tensor_declaration_dispatches_to_user_defined() -> None:
    raw_json = json.dumps(
        {
            "name": "L",
            "ranks": [{"name": "V"}],
            "data_type": {"kind": "user_defined", "name": "VertexLabel"},
            "empty_value": "",
        }
    )
    decl = TensorDeclaration.model_validate_json(raw_json)
    assert isinstance(decl.data_type, UserDefinedDataType)
    assert decl.data_type.name == "VertexLabel"


################################################################################
# TensorDeclaration.value: the new scalar-as-zero-rank-tensor field
################################################################################


def test_tensor_declaration_value_defaults_to_none() -> None:
    """Backward-compatibility: existing non-scalar decls don't supply
    ``value``. The field must default to ``None`` and survive a JSON
    round-trip as ``None``."""
    decl = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    assert decl.value is None
    again = TensorDeclaration.model_validate_json(decl.model_dump_json())
    assert again.value is None


def test_tensor_declaration_zero_rank_with_builtin_value_round_trips() -> None:
    """A zero-rank tensor with a builtin-typed ``value`` survives a JSON
    round-trip with equal model and preserved value semantics."""
    decl = TensorDeclaration(
        name="X",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
        value=5,
    )
    assert decl.value == 5
    assert type(decl.value) is int
    again = TensorDeclaration.model_validate_json(decl.model_dump_json())
    assert again == decl
    assert again.value == 5
    assert type(again.value) is int


def test_tensor_declaration_zero_rank_with_user_defined_value_round_trips() -> None:
    """``value`` is opaque for user-defined types, mirroring the
    ``empty_value`` contract: any JSON-representable Python value goes
    through unchanged."""
    decl = TensorDeclaration(
        name="X",
        ranks=[],
        data_type=UserDefinedDataType(name="char"),
        empty_value="\0",
        value="c",
    )
    assert decl.value == "c"
    again = TensorDeclaration.model_validate_json(decl.model_dump_json())
    assert again == decl
    assert again.value == "c"


def test_tensor_declaration_non_zero_rank_with_value_is_rejected() -> None:
    """``value`` is only valid for zero-rank tensors (ranks == []).
    Supplying a value alongside any rank declaration must raise a
    ValidationError whose message names both ``value`` and the
    zero-rank constraint."""
    with pytest.raises(ValidationError, match=r"value.*zero-rank"):
        TensorDeclaration(
            name="X",
            ranks=[RankDeclaration(name="I")],
            data_type=BuiltinDataType(name="int"),
            empty_value=0,
            value=5,
        )


def test_tensor_declaration_value_inf_serializes_as_sentinel() -> None:
    """``float('inf')`` is encoded as the string sentinel ``"inf"`` in
    JSON (no JSON Infinity literal exists) and decodes back to
    ``math.inf``."""
    decl = TensorDeclaration(
        name="X",
        ranks=[],
        data_type=BuiltinDataType(name="float"),
        empty_value=float("inf"),
        value=float("inf"),
    )
    dumped = decl.model_dump_json()
    raw = json.loads(dumped)
    assert raw["value"] == "inf"
    again = TensorDeclaration.model_validate_json(dumped)
    assert isinstance(again.value, float)
    assert math.isinf(again.value)
    assert again.value > 0


def test_tensor_declaration_value_negative_inf_serializes_as_sentinel() -> None:
    decl = TensorDeclaration(
        name="X",
        ranks=[],
        data_type=BuiltinDataType(name="float"),
        empty_value=float("-inf"),
        value=float("-inf"),
    )
    dumped = decl.model_dump_json()
    raw = json.loads(dumped)
    assert raw["value"] == "-inf"
    again = TensorDeclaration.model_validate_json(dumped)
    assert isinstance(again.value, float)
    assert math.isinf(again.value)
    assert again.value < 0


def test_tensor_declaration_value_nan_serializes_as_sentinel() -> None:
    decl = TensorDeclaration(
        name="X",
        ranks=[],
        data_type=BuiltinDataType(name="float"),
        empty_value=float("nan"),
        value=float("nan"),
    )
    dumped = decl.model_dump_json()
    raw = json.loads(dumped)
    assert raw["value"] == "nan"
    again = TensorDeclaration.model_validate_json(dumped)
    assert isinstance(again.value, float)
    assert math.isnan(again.value)


def test_tensor_declaration_value_discriminator_dispatch_in_list() -> None:
    """A list mixing a zero-rank decl that carries ``value`` with a
    multi-rank decl whose ``value`` is None must round-trip through a
    ``TypeAdapter`` such that each variant deserializes back to a
    TensorDeclaration with the right ranks / value pairing."""
    scalar = TensorDeclaration(
        name="Lit_0",
        ranks=[],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
        value=5,
    )
    tensor = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )

    adapter = TypeAdapter(list[TensorDeclaration])
    dumped = adapter.dump_json([scalar, tensor])
    again = adapter.validate_json(dumped)
    assert len(again) == 2
    assert again[0].name == "Lit_0"
    assert again[0].ranks == []
    assert again[0].value == 5
    assert type(again[0].value) is int
    assert again[1].name == "G"
    assert len(again[1].ranks) == 2
    assert again[1].value is None
