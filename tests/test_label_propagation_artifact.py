"""End-to-end test: label-propagation program JSON round-trips through the IR.

The on-disk artifact at
``examples/algorithms/label-propagation/label_propagation_program.json``
encodes max-label propagation:

    L_{i+1, d} = G_{s, d} . L_{i, s} :: AND_s *(intersect) OR_s max(union)
    <> : L_{i+1} == L_i

Distinguishing features from the other graph artifacts:

- The init block is EMPTY: both G and the generation-0 labels L_0 are
  host-provided data, so there is no init einsum and no synthesized literal
  tensor (no ``Lit_0`` / ``Stamp`` / ``Lit_True``).
- The reduce compute op is the user-defined ``max`` (the dual of
  Bellman-Ford's ``min``); ``max`` is not a builtin compute op.
- The stop is VALUE convergence ``L_{i+1} == L_i`` (two TensorProjectionValue
  of L), NOT the occupancy form the BFS/DFS artifacts use.
"""

from __future__ import annotations

import json
from pathlib import Path

from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankVariable,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.program import Program
from edge_ir.ir.stopping import Comparison, PropertyApp, TensorProjectionValue
from edge_ir.ir.tensor import BuiltinDataType

LP_PROGRAM_PATH = Path(
    "examples/algorithms/label-propagation/label_propagation_program.json"
)


def _load_program() -> Program:
    raw = LP_PROGRAM_PATH.read_text()
    return Program.model_validate_json(raw)


def _i_plus_1() -> RankArith:
    return RankArith(
        op="+",
        lhs=RankVariable(name="i"),
        rhs=RankConstantLiteral(value=1),
    )


def test_cc_artifact_round_trips() -> None:
    raw = LP_PROGRAM_PATH.read_text()
    program = Program.model_validate_json(raw)
    re_serialized = program.model_dump_json(indent=2)
    assert json.loads(raw) == json.loads(re_serialized)


def test_cc_artifact_declarations_are_exactly_G_and_L() -> None:
    """Only the two user tensors are declared. Unlike BFS/DFS/BF, there are
    NO synthesized literal tensors (no Lit_0/Stamp): the stop is value-based,
    not occupancy-based, and the init block is empty, so nothing needs a
    zero-rank literal."""
    program = _load_program()
    names = {decl.name for decl in program.declarations}
    assert names == {"G", "L"}


def test_cc_artifact_tensor_types_are_float_empty_zero() -> None:
    """G and L are both float with empty_value 0.0 (faithful to the source's
    `float, empty = 0.0`)."""
    program = _load_program()
    decls_by_name = {d.name: d for d in program.declarations}
    for name in ("G", "L"):
        decl = decls_by_name[name]
        assert decl.data_type == BuiltinDataType(name="float")
        assert decl.empty_value == 0.0
        assert isinstance(decl.empty_value, float)
    # G has two N-shaped data ranks; L has a generational rank then one N-shaped.
    g = decls_by_name["G"]
    assert [r.shape for r in g.ranks] == ["N", "N"]
    el = decls_by_name["L"]
    assert el.ranks[0].shape is None  # generational rank I, unbounded
    assert el.ranks[1].shape == "N"


def test_cc_artifact_initialization_is_empty() -> None:
    """The init block holds NO einsums. Both G and L_0 are host-provided data
    (L_0 = [1, 2, ...] is fully host-specified with no source restriction), so
    there is nothing to compute at init time. This is the first artifact with
    an empty initialization block."""
    program = _load_program()
    assert program.initialization.einsums == []


def test_cc_artifact_has_one_einsum() -> None:
    program = _load_program()
    assert len(program.main_edge.cascade.einsums) == 1


def test_cc_artifact_einsum_expression_is_G_times_L() -> None:
    """The single einsum writes L at the NEXT generation i+1, d, and its RHS
    expression is the binary G_{s,d} . L_{i,s}: LHS is G projected at (s, d),
    RHS is L projected at the CURRENT generation (i, s). `s` is the source/
    contraction rank shared by both inputs; `d` is preserved to the output."""
    program = _load_program()
    einsum = program.main_edge.cascade.einsums[0]
    assert einsum.output_tensor == "L"
    # Output writes L at the next generation: i+1, d.
    assert einsum.output_ranks == [_i_plus_1(), RankVariable(name="d")]

    expr = einsum.expression
    assert isinstance(expr, BinaryApp)

    # LHS: G_{s, d} -- adjacency, source row s, dest col d.
    assert isinstance(expr.lhs, InputTensor)
    assert expr.lhs.proj.tensor == "G"
    assert expr.lhs.proj.ranks == [
        RankVariable(name="s"),
        RankVariable(name="d"),
    ]

    # RHS: L_{i, s} -- the CURRENT-generation label at the source vertex s.
    assert isinstance(expr.rhs, InputTensor)
    assert expr.rhs.proj.tensor == "L"
    assert expr.rhs.proj.ranks == [
        RankVariable(name="i"),
        RankVariable(name="s"),
    ]


def test_cc_artifact_map_is_mul_intersect_over_s() -> None:
    """The map carries the BUILTIN compute op `*` (multiply the source label by
    the edge weight) and the builtin merge `intersect` (touch a point only
    where both G and L are present -- empty is not zero). It maps over the
    contraction rank `s`."""
    program = _load_program()
    einsum = program.main_edge.cascade.einsums[0]
    map_specs = [s for s in einsum.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    map_spec = map_specs[0]
    assert map_spec.compute_op == BuiltinComputeOp(symbol="*")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="intersect")
    assert map_spec.rank_list == ["s"]


def test_cc_artifact_reduce_is_max_union_over_s() -> None:
    """The reduce carries the USER-DEFINED compute op `max` (keep the largest
    incoming label) and the builtin merge `union`, over the contraction rank
    `s`. `max` is NOT a builtin compute op -- the builtin set is only
    {+, -, *, /} -- so it must be a UserDefinedComputeOp, the dual of
    Bellman-Ford's `min`."""
    program = _load_program()
    einsum = program.main_edge.cascade.einsums[0]
    reduce_specs = [s for s in einsum.specs if isinstance(s, ReduceSpec)]
    assert len(reduce_specs) == 1
    reduce_spec = reduce_specs[0]
    assert reduce_spec.compute_op == UserDefinedComputeOp(name="max")
    assert reduce_spec.merge_op == BuiltinMergeOp(symbol="union")
    assert reduce_spec.rank_list == ["s"]


def test_cc_artifact_contraction_rank_s_not_in_output() -> None:
    """`s` is the contraction rank: it is mapped AND reduced, and must NOT
    survive into the output ranks (otherwise it would not have been reduced).
    The surviving ranks are the generational i (via i+1) and the preserved
    d."""
    program = _load_program()
    einsum = program.main_edge.cascade.einsums[0]
    output_var_names = {
        r.name for r in einsum.output_ranks if isinstance(r, RankVariable)
    }
    assert "s" not in output_var_names
    assert "d" in output_var_names


def test_cc_artifact_stop_is_value_convergence_not_occupancy() -> None:
    """The stop is ``<>_i : L_{i+1} == L_i`` -- value convergence. Both sides
    are TensorProjectionValue of L (the whole tensor at the i+1 and i
    generations). It is NOT an occupancy comparison: there is no PropertyApp
    anywhere, because L converges in value, it never empties."""
    program = _load_program()
    conds = program.main_edge.cascade.stopping_conditions
    assert len(conds) == 1
    sc = conds[0]
    assert sc.rank_variable == "i"
    assert isinstance(sc.predicate, Comparison)
    assert sc.predicate.op == "=="

    # Both sides are plain tensor-value projections of L, not occupancy.
    assert isinstance(sc.predicate.lhs, TensorProjectionValue)
    assert isinstance(sc.predicate.rhs, TensorProjectionValue)
    assert not isinstance(sc.predicate.lhs, PropertyApp)
    assert not isinstance(sc.predicate.rhs, PropertyApp)

    # LHS: L at the next generation i+1; RHS: L at the current generation i.
    assert sc.predicate.lhs.proj.tensor == "L"
    assert sc.predicate.lhs.proj.ranks == [_i_plus_1()]
    assert sc.predicate.rhs.proj.tensor == "L"
    assert sc.predicate.rhs.proj.ranks == [RankVariable(name="i")]
