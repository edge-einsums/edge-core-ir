"""Tests for the top-level Program IR node and its sub-sections.

Covers:
- `Initialization`: optional list of einsums (default empty).
- `MainEdge`: a wrapper around a mandatory `NestedCascade`.
- `Program`: the full <declarations> <initializations> <main-edge>
  triple, with mandatory non-empty declarations.
"""

from __future__ import annotations

import pytest
from _stopping_helpers import bfs_stopping_condition
from pydantic import ValidationError

from edge_ir.ir.actions import MapSpec
from edge_ir.ir.einsum import Einsum, NestedCascade, StoppingCondition
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
)
from edge_ir.ir.program import Initialization, MainEdge, Program
from edge_ir.ir.tensor import (
    BuiltinDataType,
    RankDeclaration,
    TensorDeclaration,
)

################################################################################
# Helpers
################################################################################


def _simple_tensor_decl(name: str = "A") -> TensorDeclaration:
    return TensorDeclaration(
        name=name,
        ranks=[RankDeclaration(name="M", shape=10)],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )


def _simple_einsum() -> Einsum:
    return Einsum(
        output_tensor="C",
        output_ranks=[RankVariable(name="m")],
        expression=BinaryApp(
            label=1,
            lhs=InputTensor(
                proj=TensorProjection(tensor="A", ranks=[RankVariable(name="m")])
            ),
            rhs=InputTensor(
                proj=TensorProjection(tensor="B", ranks=[RankVariable(name="m")])
            ),
        ),
        specs=[
            MapSpec(
                label=1,
                rank_list=["m"],
                compute_op=BuiltinComputeOp(symbol="+"),
                merge_op=BuiltinMergeOp(symbol="intersect"),
            )
        ],
    )


def _simple_main_edge() -> MainEdge:
    return MainEdge(cascade=NestedCascade(einsums=[_simple_einsum()]))


def _bfs_einsum(output_tensor: str, input_tensor: str, output_var: str = "m") -> Einsum:
    """A second simple einsum shape used to flesh out the BFS-shaped
    program (no semantic claim, just a structurally valid Einsum)."""
    return Einsum(
        output_tensor=output_tensor,
        output_ranks=[RankVariable(name=output_var)],
        expression=InputTensor(
            proj=TensorProjection(
                tensor=input_tensor, ranks=[RankVariable(name=output_var)]
            )
        ),
        specs=[],
    )


def _bfs_program() -> Program:
    """Construct a BFS-shaped Program: 4 declarations, no
    initialization einsums, a 3-einsum cascade with one stopping
    condition. Used by both the construct-succeeds test and the
    round-trip test."""
    g = TensorDeclaration(
        name="G",
        ranks=[
            RankDeclaration(name="S", shape="|V|"),
            RankDeclaration(name="D", shape="|V|"),
        ],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    f = TensorDeclaration(
        name="F",
        ranks=[RankDeclaration(name="S", shape="|V|")],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    p = TensorDeclaration(
        name="P",
        ranks=[RankDeclaration(name="D", shape="|V|")],
        data_type=BuiltinDataType(name="bool"),
        empty_value=False,
    )
    t = TensorDeclaration(
        name="T",
        ranks=[RankDeclaration(name="D", shape="|V|")],
        data_type=BuiltinDataType(name="int"),
        empty_value=0,
    )
    e1 = _simple_einsum()
    e2 = _bfs_einsum("D", "C")
    e3 = _bfs_einsum("E", "D")
    cascade = NestedCascade(
        einsums=[e1, e2, e3],
        stopping_conditions=[bfs_stopping_condition()],
    )
    return Program(
        declarations=[g, f, p, t],
        initialization=Initialization(),
        main_edge=MainEdge(cascade=cascade),
    )


################################################################################
# Initialization
################################################################################


def test_initialization_default_einsums_is_empty_list() -> None:
    """Initialization may be constructed with no arguments; einsums
    defaults to an empty list."""
    init = Initialization()
    assert init.einsums == []


def test_initialization_with_one_einsum() -> None:
    """Initialization carries the einsums list intact."""
    e = _simple_einsum()
    init = Initialization(einsums=[e])
    assert len(init.einsums) == 1
    assert init.einsums[0] == e


def test_initialization_round_trips_via_json() -> None:
    """JSON round-trip preserves the einsums list (including the
    nested expression and specs structures)."""
    init = Initialization(einsums=[_simple_einsum()])
    serialized = init.model_dump_json()
    deserialized = Initialization.model_validate_json(serialized)
    assert init == deserialized
    assert len(deserialized.einsums) == 1
    assert deserialized.einsums[0].output_tensor == "C"


################################################################################
# MainEdge
################################################################################


def test_main_edge_with_single_einsum_cascade() -> None:
    """MainEdge wraps a NestedCascade; the cascade's contents are
    accessible through the wrapper."""
    me = _simple_main_edge()
    assert isinstance(me.cascade, NestedCascade)
    assert len(me.cascade.einsums) == 1
    assert me.cascade.einsums[0].output_tensor == "C"


def test_main_edge_rejects_missing_cascade_kwarg() -> None:
    """The cascade field is mandatory."""
    with pytest.raises(ValidationError):
        MainEdge()  # type: ignore[call-arg]


def test_main_edge_round_trips_via_json() -> None:
    """JSON round-trip preserves the cascade and its discriminated
    sub-structures."""
    me = MainEdge(
        cascade=NestedCascade(
            einsums=[_simple_einsum()],
            stopping_conditions=[bfs_stopping_condition()],
        )
    )
    serialized = me.model_dump_json()
    deserialized = MainEdge.model_validate_json(serialized)
    assert me == deserialized
    assert len(deserialized.cascade.einsums) == 1
    assert len(deserialized.cascade.stopping_conditions) == 1


################################################################################
# Program
################################################################################


def test_program_minimal_construction_succeeds() -> None:
    """The smallest valid Program: one declaration, empty
    initialization, and a single-einsum main edge."""
    prog = Program(
        declarations=[_simple_tensor_decl("A")],
        initialization=Initialization(),
        main_edge=_simple_main_edge(),
    )
    assert len(prog.declarations) == 1
    assert prog.declarations[0].name == "A"
    assert prog.initialization.einsums == []
    assert isinstance(prog.main_edge, MainEdge)


def test_program_rejects_empty_declarations_list() -> None:
    """The declarations list has min_length=1; an empty list is invalid
    (a Program with no declared tensors has nothing to compute over)."""
    with pytest.raises(ValidationError):
        Program(
            declarations=[],
            initialization=Initialization(),
            main_edge=_simple_main_edge(),
        )


def test_program_rejects_missing_declarations_kwarg() -> None:
    """The declarations field is mandatory; omitting it must raise."""
    with pytest.raises(ValidationError):
        Program(  # type: ignore[call-arg]
            initialization=Initialization(),
            main_edge=_simple_main_edge(),
        )


def test_program_rejects_missing_initialization_kwarg() -> None:
    """The initialization field is mandatory (even though
    Initialization itself defaults to empty einsums, the wrapper must
    still be supplied)."""
    with pytest.raises(ValidationError):
        Program(  # type: ignore[call-arg]
            declarations=[_simple_tensor_decl("A")],
            main_edge=_simple_main_edge(),
        )


def test_program_rejects_missing_main_edge_kwarg() -> None:
    """The main_edge field is mandatory."""
    with pytest.raises(ValidationError):
        Program(  # type: ignore[call-arg]
            declarations=[_simple_tensor_decl("A")],
            initialization=Initialization(),
        )


def test_program_bfs_shape_constructs() -> None:
    """A BFS-shaped Program — 4 tensor declarations spanning multiple
    builtin data types and empty-values, an empty Initialization, and
    a 3-einsum cascade with one stopping condition — is structurally
    valid. This is a smoke test for the full integration shape; it
    does not claim to model BFS semantics."""
    prog = _bfs_program()
    assert len(prog.declarations) == 4
    decl_names = [d.name for d in prog.declarations]
    assert decl_names == ["G", "F", "P", "T"]
    assert prog.initialization.einsums == []
    assert len(prog.main_edge.cascade.einsums) == 3
    assert len(prog.main_edge.cascade.stopping_conditions) == 1
    assert prog.main_edge.cascade.stopping_conditions[0].rank_variable == "i"


def test_program_bfs_shape_round_trips_via_json() -> None:
    """The BFS-shaped Program round-trips losslessly through JSON: all
    four declarations, the empty initialization, and the cascade with
    its stopping condition are preserved exactly."""
    prog = _bfs_program()
    serialized = prog.model_dump_json()
    deserialized = Program.model_validate_json(serialized)
    assert prog == deserialized
    assert len(deserialized.declarations) == 4
    assert [d.name for d in deserialized.declarations] == ["G", "F", "P", "T"]
    assert deserialized.initialization.einsums == []
    assert len(deserialized.main_edge.cascade.einsums) == 3
    assert len(deserialized.main_edge.cascade.stopping_conditions) == 1


def test_program_round_trip_preserves_discriminated_union_variants() -> None:
    """After JSON round-trip, every Pydantic discriminated-union member
    in the cascade resolves back to its concrete subclass. This guards
    against the failure mode where a missing/incorrect discriminator
    causes Pydantic to fall back to the union's first variant."""
    prog = _bfs_program()
    serialized = prog.model_dump_json()
    deserialized = Program.model_validate_json(serialized)

    # Declarations carry a DataType discriminated by "kind".
    for decl in deserialized.declarations:
        assert isinstance(decl, TensorDeclaration)
        assert isinstance(decl.data_type, BuiltinDataType)

    # The first cascade einsum's expression is a BinaryApp whose
    # operands are InputTensors, and whose specs entry is a MapSpec.
    first_einsum = deserialized.main_edge.cascade.einsums[0]
    assert isinstance(first_einsum.expression, BinaryApp)
    assert isinstance(first_einsum.expression.lhs, InputTensor)
    assert isinstance(first_einsum.expression.rhs, InputTensor)
    assert isinstance(first_einsum.specs[0], MapSpec)
    # The MapSpec's compute and merge ops are builtins.
    assert isinstance(first_einsum.specs[0].compute_op, BuiltinComputeOp)
    assert isinstance(first_einsum.specs[0].merge_op, BuiltinMergeOp)

    # Stopping condition class is preserved.
    assert isinstance(
        deserialized.main_edge.cascade.stopping_conditions[0],
        StoppingCondition,
    )


def test_program_model_json_schema_is_non_empty_dict() -> None:
    """`Program.model_json_schema()` is the schema-generation gate the
    project plan relies on. It must succeed and return a non-empty
    dict so external consumers (egglog, MLIR backend, JSON validators)
    can introspect the IR."""
    schema = Program.model_json_schema()
    assert isinstance(schema, dict)
    assert len(schema) > 0
