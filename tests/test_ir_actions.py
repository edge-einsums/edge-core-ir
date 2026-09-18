"""Tests for ComputationSpec nodes (Map, Reduce, Populate) in
edge_ir.ir.actions.

These tests pin down the contracts described in the actions docstring
and reasoning notes:

- Map and Reduce share the (compute_op, merge_op) pairing; Populate
  uses (compute_op, coord_op).
- Map's rank_list is optional (None allowed, no min length).
- Reduce's rank_list is optional (None allowed, no min length); the
  bare `\\bigvee` case from paper Section 6.211, 6.234.
- Populate's rank_list is required but may be empty (min_length=0);
  an empty list `[]` is the no-`*` default-assignment case.
- Each spec is a frozen, strict, extra=forbid IRBase model.
- ComputationSpec is a discriminated union over the literal "kind"
  field, so JSON round-trips dispatch to the right concrete class.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from edge_ir.ir.actions import (
    ComputationSpec,
    MapSpec,
    PopulateSpec,
    ReduceSpec,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    CoordinateOp,
    UserDefinedMergeOp,
)

################################################################################
# Shared operator fixtures
################################################################################


@pytest.fixture
def compute_op() -> BuiltinComputeOp:
    """A simple builtin compute op used as a placeholder across specs."""
    return BuiltinComputeOp(symbol="+")


@pytest.fixture
def merge_op() -> BuiltinMergeOp:
    """A simple builtin merge op used as a placeholder across specs."""
    return BuiltinMergeOp(symbol="intersect")


@pytest.fixture
def coord_op() -> CoordinateOp:
    """A simple coordinate op used as a placeholder for Populate specs."""
    return CoordinateOp(name="label_select")


################################################################################
# MapSpec
################################################################################


def test_map_spec_constructs_with_rank_list(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """MapSpec accepts a non-empty rank_list and stores all fields verbatim."""
    spec = MapSpec(
        label=0,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    assert spec.kind == "map"
    assert spec.label == 0
    assert spec.rank_list == ["s"]
    assert spec.compute_op == compute_op
    assert spec.merge_op == merge_op


def test_map_spec_constructs_without_rank_list(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """MapSpec's rank_list is optional; omitting it leaves it as None,
    signalling the default 'parallel over all ranks' behaviour."""
    spec = MapSpec(label=1, compute_op=compute_op, merge_op=merge_op)
    assert spec.rank_list is None


def test_map_spec_accepts_empty_rank_list(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """MapSpec.rank_list has no min_length constraint, so an empty list
    is accepted. If this regresses, MapSpec was given an over-strict
    constraint not specified in the design notes."""
    spec = MapSpec(
        label=0,
        rank_list=[],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    assert spec.rank_list == []


def test_map_spec_accepts_negative_label(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """The label is a plain int with no sign constraint; negative values
    must be accepted."""
    spec = MapSpec(
        label=-3,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    assert spec.label == -3


def test_map_spec_accepts_user_defined_merge_op(
    compute_op: BuiltinComputeOp,
) -> None:
    """MapSpec.merge_op is the MergeOp union, so a UserDefinedMergeOp
    must be accepted alongside a BuiltinMergeOp."""
    udf_merge = UserDefinedMergeOp(name="my_merge")
    spec = MapSpec(
        label=0,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=udf_merge,
    )
    assert spec.merge_op == udf_merge


def test_map_spec_kind_defaults_to_map(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """The Literal['map'] field has a default value, so omitting `kind`
    from the constructor still produces kind=='map'."""
    spec = MapSpec(label=0, compute_op=compute_op, merge_op=merge_op)
    assert spec.kind == "map"


def test_map_spec_is_frozen(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """IRBase is frozen, so MapSpec instances must reject attribute
    assignment after construction."""
    spec = MapSpec(label=0, compute_op=compute_op, merge_op=merge_op)
    with pytest.raises(ValidationError):
        spec.label = 7  # type: ignore[misc]


def test_map_spec_round_trips_through_json(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """A MapSpec dumped to JSON and reloaded must compare equal to the
    original (no field loss, no silent coercion)."""
    spec = MapSpec(
        label=2,
        rank_list=["s", "d"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    serialized = spec.model_dump_json()
    deserialized = MapSpec.model_validate_json(serialized)
    assert spec == deserialized


################################################################################
# ReduceSpec
################################################################################


def test_reduce_spec_constructs_with_rank_list(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """ReduceSpec accepts a single-rank rank_list and stores fields verbatim."""
    spec = ReduceSpec(
        label=0,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    assert spec.kind == "reduce"
    assert spec.rank_list == ["s"]


def test_reduce_spec_constructs_without_rank_list(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """ReduceSpec.rank_list is optional; omitting it leaves it as None,
    which encodes the bare `\\bigvee` case from paper Section 6.211 and
    6.234 (reduce over all reducible ranks at this site, resolved
    downstream)."""
    spec = ReduceSpec(label=1, compute_op=compute_op, merge_op=merge_op)
    assert spec.kind == "reduce"
    assert spec.rank_list is None


def test_reduce_spec_accepts_empty_rank_list(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """ReduceSpec.rank_list has no min_length constraint, so an empty
    list is accepted (mirrors MapSpec exactly). If this regresses,
    ReduceSpec was given an over-strict constraint not specified in the
    paper."""
    spec = ReduceSpec(
        label=0,
        rank_list=[],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    assert spec.rank_list == []


def test_reduce_spec_accepts_multiple_ranks(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """A reduce can collapse over multiple ranks at once; the rank_list
    must accept lists of length >1."""
    spec = ReduceSpec(
        label=0,
        rank_list=["s", "t"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    assert spec.rank_list == ["s", "t"]


def test_reduce_spec_round_trips_through_json(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """A ReduceSpec dumped to JSON and reloaded must compare equal."""
    spec = ReduceSpec(
        label=4,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    serialized = spec.model_dump_json()
    deserialized = ReduceSpec.model_validate_json(serialized)
    assert spec == deserialized


def test_reduce_spec_round_trips_with_none_rank_list(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """A ReduceSpec with no rank_list must survive a JSON round-trip
    cleanly: the omitted (None) rank_list must come back as None, not
    as `[]` or a missing attribute. This locks down the bare-`\\bigvee`
    case at the serialization layer."""
    spec = ReduceSpec(label=5, compute_op=compute_op, merge_op=merge_op)
    assert spec.rank_list is None
    serialized = spec.model_dump_json()
    deserialized = ReduceSpec.model_validate_json(serialized)
    assert spec == deserialized
    assert deserialized.rank_list is None


def test_reduce_spec_rejects_wrong_kind(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """The kind field is Literal['reduce']; passing 'map' must raise so
    that a hand-crafted misclassified payload doesn't silently parse as
    the wrong action."""
    with pytest.raises(ValidationError):
        ReduceSpec(
            kind="map",  # type: ignore[arg-type]
            label=0,
            rank_list=["s"],
            compute_op=compute_op,
            merge_op=merge_op,
        )


################################################################################
# PopulateSpec
################################################################################


def test_populate_spec_constructs_with_rank_list(
    compute_op: BuiltinComputeOp, coord_op: CoordinateOp
) -> None:
    """PopulateSpec uses (compute_op, coord_op) — not merge_op — and
    requires a rank_list specifying the mutable rank."""
    spec = PopulateSpec(
        label=0,
        rank_list=["d"],
        compute_op=compute_op,
        coord_op=coord_op,
    )
    assert spec.kind == "populate"
    assert spec.rank_list == ["d"]
    assert spec.coord_op == coord_op
    assert spec.compute_op == compute_op


def test_populate_spec_accepts_empty_rank_list(
    compute_op: BuiltinComputeOp, coord_op: CoordinateOp
) -> None:
    """PopulateSpec.rank_list has min_length=0; an empty list is the
    no-`*` default-assignment case and must be accepted, preserving []."""
    spec = PopulateSpec(
        label=0,
        rank_list=[],
        compute_op=compute_op,
        coord_op=coord_op,
    )
    assert spec.rank_list == []


def test_populate_spec_empty_rank_list_round_trips_through_json(
    compute_op: BuiltinComputeOp, coord_op: CoordinateOp
) -> None:
    """An empty (default-assignment) populate rank_list survives a JSON
    round-trip as [], not dropped or coerced to None."""
    spec = PopulateSpec(
        label=0,
        rank_list=[],
        compute_op=compute_op,
        coord_op=coord_op,
    )
    back = PopulateSpec.model_validate_json(spec.model_dump_json())
    assert back.rank_list == []


def test_populate_spec_requires_rank_list(
    compute_op: BuiltinComputeOp, coord_op: CoordinateOp
) -> None:
    """PopulateSpec.rank_list has no default, so omitting it must raise."""
    with pytest.raises(ValidationError):
        PopulateSpec(  # type: ignore[call-arg]
            label=0,
            compute_op=compute_op,
            coord_op=coord_op,
        )


def test_populate_spec_rejects_merge_op_in_coord_op_slot(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """The coord_op field is a CoordinateOp (no `kind` discriminator,
    just a name). A BuiltinMergeOp carries a `kind` and a `symbol` —
    with extra=forbid on CoordinateOp the structural mismatch must
    raise. This guards against accidentally pairing Populate with a
    merge operator."""
    with pytest.raises(ValidationError):
        PopulateSpec(
            label=0,
            rank_list=["d"],
            compute_op=compute_op,
            coord_op=merge_op,  # type: ignore[arg-type]
        )


def test_populate_spec_rejects_merge_op_kwarg(
    compute_op: BuiltinComputeOp, merge_op: BuiltinMergeOp
) -> None:
    """PopulateSpec has no `merge_op` field. With extra=forbid, passing
    a `merge_op=` kwarg must raise — confirming the action distinction
    at the field-name level."""
    with pytest.raises(ValidationError):
        PopulateSpec(  # type: ignore[call-arg]
            label=0,
            rank_list=["d"],
            compute_op=compute_op,
            merge_op=merge_op,
        )


def test_populate_spec_round_trips_through_json(
    compute_op: BuiltinComputeOp, coord_op: CoordinateOp
) -> None:
    """A PopulateSpec dumped to JSON and reloaded must compare equal."""
    spec = PopulateSpec(
        label=7,
        rank_list=["d"],
        compute_op=compute_op,
        coord_op=coord_op,
    )
    serialized = spec.model_dump_json()
    deserialized = PopulateSpec.model_validate_json(serialized)
    assert spec == deserialized


################################################################################
# ComputationSpec discriminated union
################################################################################


def test_computation_spec_dispatches_map() -> None:
    """A `kind=map` payload routed through the union must deserialize
    as a MapSpec."""
    adapter: TypeAdapter[ComputationSpec] = TypeAdapter(ComputationSpec)
    parsed = adapter.validate_python(
        {
            "kind": "map",
            "label": 0,
            "rank_list": ["s"],
            "compute_op": {"kind": "builtin", "symbol": "+"},
            "merge_op": {"kind": "builtin", "symbol": "intersect"},
        }
    )
    assert isinstance(parsed, MapSpec)
    assert parsed.label == 0
    assert parsed.rank_list == ["s"]


def test_computation_spec_dispatches_reduce() -> None:
    """A `kind=reduce` payload must deserialize as a ReduceSpec."""
    adapter: TypeAdapter[ComputationSpec] = TypeAdapter(ComputationSpec)
    parsed = adapter.validate_python(
        {
            "kind": "reduce",
            "label": 1,
            "rank_list": ["s"],
            "compute_op": {"kind": "builtin", "symbol": "+"},
            "merge_op": {"kind": "builtin", "symbol": "intersect"},
        }
    )
    assert isinstance(parsed, ReduceSpec)
    assert parsed.label == 1


def test_computation_spec_dispatches_reduce_without_rank_list() -> None:
    """A `kind=reduce` payload with no `rank_list` key must still
    dispatch through the discriminated union to a ReduceSpec, with
    `rank_list` defaulting to None. This is the bare-`\\bigvee` case
    coming in over-the-wire (e.g. from a hand-written or paper-faithful
    JSON IR)."""
    adapter: TypeAdapter[ComputationSpec] = TypeAdapter(ComputationSpec)
    parsed = adapter.validate_python(
        {
            "kind": "reduce",
            "label": 1,
            "compute_op": {"kind": "builtin", "symbol": "+"},
            "merge_op": {"kind": "builtin", "symbol": "intersect"},
        }
    )
    assert isinstance(parsed, ReduceSpec)
    assert parsed.label == 1
    assert parsed.rank_list is None


def test_computation_spec_dispatches_populate() -> None:
    """A `kind=populate` payload must deserialize as a PopulateSpec
    (which carries coord_op, not merge_op)."""
    adapter: TypeAdapter[ComputationSpec] = TypeAdapter(ComputationSpec)
    parsed = adapter.validate_python(
        {
            "kind": "populate",
            "label": 2,
            "rank_list": ["d"],
            "compute_op": {"kind": "builtin", "symbol": "+"},
            "coord_op": {"name": "label_select"},
        }
    )
    assert isinstance(parsed, PopulateSpec)
    assert parsed.coord_op.name == "label_select"


def test_computation_spec_rejects_missing_kind() -> None:
    """Without a `kind` field the discriminated union has no way to
    pick a concrete class; this must raise."""
    adapter: TypeAdapter[ComputationSpec] = TypeAdapter(ComputationSpec)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "label": 0,
                "rank_list": ["s"],
                "compute_op": {"kind": "builtin", "symbol": "+"},
                "merge_op": {"kind": "builtin", "symbol": "intersect"},
            }
        )


def test_computation_spec_rejects_unknown_kind() -> None:
    """Only 'map', 'reduce', 'populate' are valid discriminator values;
    anything else must raise."""
    adapter: TypeAdapter[ComputationSpec] = TypeAdapter(ComputationSpec)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "weird",
                "label": 0,
                "rank_list": ["s"],
                "compute_op": {"kind": "builtin", "symbol": "+"},
                "merge_op": {"kind": "builtin", "symbol": "intersect"},
            }
        )


################################################################################
# Mixed-list round-trip
################################################################################


def test_computation_spec_mixed_list_round_trips_per_spec(
    compute_op: BuiltinComputeOp,
    merge_op: BuiltinMergeOp,
    coord_op: CoordinateOp,
) -> None:
    """Round-trip a list of heterogeneous specs one at a time through
    the discriminated union — the `kind` discriminator must route each
    payload back to its original concrete class with full equality."""
    map_spec = MapSpec(
        label=0,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    reduce_spec = ReduceSpec(
        label=1,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    populate_spec = PopulateSpec(
        label=2,
        rank_list=["d"],
        compute_op=compute_op,
        coord_op=coord_op,
    )
    originals: list[ComputationSpec] = [map_spec, reduce_spec, populate_spec]
    adapter: TypeAdapter[ComputationSpec] = TypeAdapter(ComputationSpec)
    reloaded = [adapter.validate_json(spec.model_dump_json()) for spec in originals]
    assert reloaded == originals


def test_computation_spec_list_adapter_round_trips(
    compute_op: BuiltinComputeOp,
    merge_op: BuiltinMergeOp,
    coord_op: CoordinateOp,
) -> None:
    """A `TypeAdapter[list[ComputationSpec]]` must round-trip a list of
    heterogeneous specs in one shot. The list intentionally includes a
    bare-`\\bigvee` ReduceSpec (no rank_list) so the union-level adapter
    is exercised against the optional-rank_list case as well."""
    map_spec = MapSpec(
        label=0,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    reduce_spec = ReduceSpec(
        label=1,
        rank_list=["s"],
        compute_op=compute_op,
        merge_op=merge_op,
    )
    bare_reduce_spec = ReduceSpec(
        label=3,
        compute_op=compute_op,
        merge_op=merge_op,
    )
    populate_spec = PopulateSpec(
        label=2,
        rank_list=["d"],
        compute_op=compute_op,
        coord_op=coord_op,
    )
    originals: list[ComputationSpec] = [
        map_spec,
        reduce_spec,
        bare_reduce_spec,
        populate_spec,
    ]
    list_adapter: TypeAdapter[list[ComputationSpec]] = TypeAdapter(
        list[ComputationSpec]
    )
    serialized = list_adapter.dump_json(originals)
    reloaded = list_adapter.validate_json(serialized)
    assert reloaded == originals
