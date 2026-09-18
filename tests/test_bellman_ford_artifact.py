"""Einsum-derived tests for the Bellman-Ford SSSP artifact.

Every assertion in this file is derived from the EDGE einsum semantics in
``examples/algorithms/bellman-ford/einsum.md`` -- NOT from whatever happens to
sit in ``bellman_ford_program.json``. The question each test answers is
"does the artifact encode what the einsum *means*?" Where the artifact is
known to diverge from the einsum (the ``<<`` UPDATE bug, see ``TODO.md``
Bugs section), the test encodes the EINSUM-CORRECT expectation and is
marked ``xfail`` so it flips to passing once the artifact is fixed.

The four einsums (paper ``cascade:bf``) are:

  (1) RELAX:   N_{i,d} = G_{s,d} . D_{i,s} :: /\\_s +(intersect) \\/_s min(union)
  (2) IMPROVE: C_{i,d} = N_{i,d} . D_{i,d} :: /\\_d <(union)
  (3) RECORD:  NewlyRelaxed_{i,d} = C_{i,d} . N_{i,d} :: /\\_d ->(intersect)
  (4) UPDATE:  D_{i+1,d} = D_{i,d} . NewlyRelaxed_{i,d} :: /\\_d <<(union)

  <> : D_{i+1} == D_i

There is no evaluator yet, so these tests pin the IR *structure* that
encodes the ODE semantics rather than executing the program.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum
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
from edge_ir.ir.stopping import Comparison, TensorProjectionValue
from edge_ir.ir.tensor import BuiltinDataType
from edge_ir.runtime.op_properties import get_merge_props

BF_PROGRAM_PATH = Path("examples/algorithms/bellman-ford/bellman_ford_program.json")


def _load_program() -> Program:
    return Program.model_validate_json(BF_PROGRAM_PATH.read_text())


def _decls_by_name(program: Program) -> dict[str, object]:
    return {d.name: d for d in program.declarations}


def _einsum_for(program: Program, output_tensor: str) -> Einsum:
    """The cascade Einsum whose output is ``output_tensor``.

    Each of N, C, NewlyRelaxed is the output of exactly one cascade einsum.
    D is the output of UPDATE in the cascade (D_0 init lives in the
    initialization block, not the cascade), so this resolves UPDATE for D.
    """
    return next(
        e for e in program.main_edge.cascade.einsums if e.output_tensor == output_tensor
    )


################################################################################
# Round-trip (mirrors test_bfs_artifact.py)
################################################################################


def test_bf_artifact_round_trips() -> None:
    raw = BF_PROGRAM_PATH.read_text()
    program = Program.model_validate_json(raw)
    re_serialized = program.model_dump_json(indent=2)
    assert json.loads(raw) == json.loads(re_serialized)


################################################################################
# Tensor declarations: empty values matter to the EDGE semantics.
# Empty (+inf = unreached) is NOT zero; Boolean empty is False; graph empty
# is 0. einsum.md "Tensors" block.
################################################################################


def test_bf_artifact_has_expected_tensors() -> None:
    """G, C, N, NewlyRelaxed, D must all be declared. Synthesized literals
    (Lit_0) may appear alongside; the test is robust to extra literals."""
    program = _load_program()
    names = {d.name for d in program.declarations}
    assert {"G", "C", "N", "NewlyRelaxed", "D"} <= names


def test_bf_artifact_distance_tensors_empty_is_positive_inf() -> None:
    """D, N, NewlyRelaxed are integer SSSP distances whose empty value is
    +inf (the "unreached / no path yet" marker). +inf, not 0: an empty
    distance is the absence of any path, distinct from a zero-length path.
    einsum.md declares all three with ``empty = inf``."""
    program = _load_program()
    decls = _decls_by_name(program)
    for tensor_name in ("D", "N", "NewlyRelaxed"):
        empty = decls[tensor_name].empty_value  # type: ignore[attr-defined]
        assert isinstance(empty, float), tensor_name
        assert math.isinf(empty), tensor_name
        assert empty > 0, tensor_name


def test_bf_artifact_distance_tensors_are_int() -> None:
    """D, N, NewlyRelaxed are integer-valued. einsum.md: ``-> integer``."""
    program = _load_program()
    decls = _decls_by_name(program)
    for tensor_name in ("D", "N", "NewlyRelaxed"):
        dt = decls[tensor_name].data_type  # type: ignore[attr-defined]
        assert dt == BuiltinDataType(name="int"), tensor_name


def test_bf_artifact_C_is_boolean_with_empty_false() -> None:
    """C is the improvement mask (N < D). It is Boolean, and its empty
    value is False, NOT empty/unset: "no improvement here" is a concrete
    False, the absence of a True. einsum.md: ``C -> Boolean, empty = False``.

    This distinguishes empty-from-False is the wrong framing for C: the
    paper *picks* False as C's empty, so an absent improvement reads as a
    definite "not improved" rather than an unknown."""
    program = _load_program()
    decls = _decls_by_name(program)
    c = decls["C"]
    assert c.data_type == BuiltinDataType(name="bool")  # type: ignore[attr-defined]
    assert c.empty_value is False  # type: ignore[attr-defined]


def test_bf_artifact_G_is_int_with_empty_zero() -> None:
    """G is the weighted adjacency matrix: integer, empty = 0. einsum.md:
    ``G -> integer, empty = 0``. (G's empty is 0, unlike the distance
    tensors' +inf -- a deliberate contrast the artifact must preserve.)"""
    program = _load_program()
    decls = _decls_by_name(program)
    g = decls["G"]
    assert g.data_type == BuiltinDataType(name="int")  # type: ignore[attr-defined]
    assert g.empty_value == 0  # type: ignore[attr-defined]
    assert type(g.empty_value) is int  # type: ignore[attr-defined]


################################################################################
# (1) RELAX: N_{i,d} = G_{s,d} . D_{i,s} :: /\_s +(intersect) \/_s min(union)
#
# The min reduce over s is what makes this Bellman-Ford rather than BFS
# (whose advance reduces with ANY/union -- presence only). BF needs the
# numeric minimum over incoming sources.
################################################################################


def test_bf_artifact_relax_map_is_plus_over_intersect() -> None:
    """RELAX's map adds the edge weight to the source distance, and is
    effectual only where BOTH the edge and a finite source distance exist
    -- hence intersect merge. einsum.md: ``+(intersect)``."""
    program = _load_program()
    relax = _einsum_for(program, "N")
    map_specs = [s for s in relax.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    ms = map_specs[0]
    assert ms.compute_op == BuiltinComputeOp(symbol="+")
    assert ms.merge_op == BuiltinMergeOp(symbol="intersect")


def test_bf_artifact_relax_reduce_is_min_over_union() -> None:
    """RELAX's reduce keeps the MINIMUM candidate over all sources reaching
    d. The compute op is ``min`` (the numeric minimum, NOT BFS's ANY) paired
    with union. einsum.md: ``\\/_s min(union)``. The min vs ANY distinction
    is the whole difference between BF and weighted BFS."""
    program = _load_program()
    relax = _einsum_for(program, "N")
    reduce_specs = [s for s in relax.specs if isinstance(s, ReduceSpec)]
    assert len(reduce_specs) == 1
    rs = reduce_specs[0]
    assert rs.compute_op == UserDefinedComputeOp(name="min")
    assert rs.merge_op == BuiltinMergeOp(symbol="union")


def test_bf_artifact_relax_reduces_over_s() -> None:
    """RELAX reduces over the source rank ``s``: the reduce's rank_list is
    [s]. einsum.md: ``\\/_s``."""
    program = _load_program()
    relax = _einsum_for(program, "N")
    reduce_specs = [s for s in relax.specs if isinstance(s, ReduceSpec)]
    assert len(reduce_specs) == 1
    assert reduce_specs[0].rank_list == ["s"]


def test_bf_artifact_relax_output_does_not_carry_reduced_rank_s() -> None:
    """N_{i,d}: the reduced rank ``s`` is collapsed and must NOT appear in
    N's output ranks. A reduce that left ``s`` in the output would not be a
    reduce at all. einsum.md: ``N_{i,d}`` (output ranks are i, d only)."""
    program = _load_program()
    relax = _einsum_for(program, "N")
    output_var_names = [
        r.name for r in relax.output_ranks if isinstance(r, RankVariable)
    ]
    assert output_var_names == ["i", "d"]
    assert "s" not in output_var_names


def test_bf_artifact_relax_expression_is_G_times_D() -> None:
    """RELAX combines G_{s,d} (lhs) with D_{i,s} (rhs). The source rank s of
    D is contracted against G's source rank. einsum.md:
    ``N_{i,d} = G_{s,d} . D_{i,s}``."""
    program = _load_program()
    relax = _einsum_for(program, "N")
    assert isinstance(relax.expression, BinaryApp)
    lhs = relax.expression.lhs
    rhs = relax.expression.rhs
    assert isinstance(lhs, InputTensor)
    assert isinstance(rhs, InputTensor)
    assert lhs.proj.tensor == "G"
    assert [r.name for r in lhs.proj.ranks if isinstance(r, RankVariable)] == ["s", "d"]
    assert rhs.proj.tensor == "D"
    assert [r.name for r in rhs.proj.ranks if isinstance(r, RankVariable)] == ["i", "s"]


################################################################################
# (2) IMPROVE: C_{i,d} = N_{i,d} . D_{i,d} :: /\_d <(union)
#
# Strict-less-than comparison producing a Boolean mask. < is not a builtin
# compute op in this IR (only + - * /), so it is encoded as the UDF compute
# op ``less_than`` (metadata gap G2). The einsum MEANING is "strict <", so
# we assert the encoding faithfully denotes that, and that the merge is union.
################################################################################


def test_bf_artifact_improve_compute_is_strict_less_than() -> None:
    """IMPROVE's compute is the strict-less-than comparison N < D. Since <
    is not a builtin compute op, the einsum's ``<`` is denoted by the UDF
    compute op named ``less_than`` (the comparison-producing function).
    einsum.md: ``<(union)``; metadata G2."""
    program = _load_program()
    improve = _einsum_for(program, "C")
    map_specs = [s for s in improve.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    assert map_specs[0].compute_op == UserDefinedComputeOp(name="less_than")


def test_bf_artifact_improve_merge_is_union() -> None:
    """IMPROVE merges with union. einsum.md: ``<(union)``."""
    program = _load_program()
    improve = _einsum_for(program, "C")
    map_specs = [s for s in improve.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    assert map_specs[0].merge_op == BuiltinMergeOp(symbol="union")


def test_bf_artifact_improve_expression_is_N_compared_to_D() -> None:
    """IMPROVE compares N_{i,d} (lhs, the candidate) against D_{i,d} (rhs,
    the incumbent). Order matters for a strict ``<``: it must be N < D, not
    D < N. einsum.md: ``C_{i,d} = N_{i,d} . D_{i,d}``."""
    program = _load_program()
    improve = _einsum_for(program, "C")
    assert isinstance(improve.expression, BinaryApp)
    lhs = improve.expression.lhs
    rhs = improve.expression.rhs
    assert isinstance(lhs, InputTensor)
    assert isinstance(rhs, InputTensor)
    assert lhs.proj.tensor == "N"
    assert rhs.proj.tensor == "D"


################################################################################
# (3) RECORD: NewlyRelaxed_{i,d} = C_{i,d} . N_{i,d} :: /\_d ->(intersect)
#
# Take-right: keep N's value where it is present. Intersect: only where BOTH
# the improvement mask C and the candidate N are present. The result is the
# improved distances only.
################################################################################


def test_bf_artifact_record_compute_is_take_right() -> None:
    """RECORD takes the right operand's value (N). einsum.md: ``->`` =
    take-right compute."""
    program = _load_program()
    record = _einsum_for(program, "NewlyRelaxed")
    map_specs = [s for s in record.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    assert map_specs[0].compute_op == UserDefinedComputeOp(name="take_right")


def test_bf_artifact_record_merge_is_intersect() -> None:
    """RECORD is effectual only where C AND N are both present -- intersect.
    einsum.md: ``->(intersect)``. (Contrast UPDATE, which uses union.)"""
    program = _load_program()
    record = _einsum_for(program, "NewlyRelaxed")
    map_specs = [s for s in record.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    assert map_specs[0].merge_op == BuiltinMergeOp(symbol="intersect")


def test_bf_artifact_record_expression_is_C_then_N() -> None:
    """RECORD's lhs is the mask C, rhs is the candidate N; take-right then
    keeps N. einsum.md: ``NewlyRelaxed_{i,d} = C_{i,d} . N_{i,d}``."""
    program = _load_program()
    record = _einsum_for(program, "NewlyRelaxed")
    assert isinstance(record.expression, BinaryApp)
    lhs = record.expression.lhs
    rhs = record.expression.rhs
    assert isinstance(lhs, InputTensor)
    assert isinstance(rhs, InputTensor)
    assert lhs.proj.tensor == "C"
    assert rhs.proj.tensor == "N"


################################################################################
# (4) UPDATE: D_{i+1,d} = D_{i,d} . NewlyRelaxed_{i,d} :: /\_d <<(union)
#
# KNOWN BUG (TODO.md Bugs section): the einsum operator is `<<` ("take the
# right operand's value where right is present, ELSE carry the left operand
# forward"). The artifact builds it as compute take_right + merge union,
# which is NOT `<<`: on the (left present, right absent) case --
# "distance unchanged this round" -- take_right(union) returns the right's
# EMPTY value (+inf) and drops the carried-forward left distance, whereas
# `<<` keeps the left distance. The tests below pin the FAITHFUL `<<`
# semantics and are xfail until the artifact encodes `<<` correctly.
################################################################################


def test_bf_artifact_update_output_is_next_generation_of_D() -> None:
    """UPDATE writes the NEXT generation of D: D_{i+1,d}. The first output
    rank is the arithmetic i+1, the second is d. This advances the single
    distance tensor across iterations. einsum.md: ``D_{i+1,d} = ...``.

    (This part of UPDATE is correct in the artifact -- only the `<<` op
    pairing is buggy, pinned separately below.)"""
    program = _load_program()
    update = _einsum_for(program, "D")
    assert len(update.output_ranks) == 2
    first, second = update.output_ranks
    assert isinstance(first, RankArith)
    assert first.op == "+"
    assert first.lhs == RankVariable(name="i")
    assert first.rhs == RankConstantLiteral(value=1)
    assert second == RankVariable(name="d")


def test_bf_artifact_update_expression_is_D_then_NewlyRelaxed() -> None:
    """UPDATE's lhs is the carried-forward distance D_{i,d}, rhs is the
    NewlyRelaxed candidate. The `<<` operator prefers the rhs where present
    and keeps the lhs otherwise, so this operand order matters.
    einsum.md: ``D_{i+1,d} = D_{i,d} . NewlyRelaxed_{i,d}``."""
    program = _load_program()
    update = _einsum_for(program, "D")
    assert isinstance(update.expression, BinaryApp)
    lhs = update.expression.lhs
    rhs = update.expression.rhs
    assert isinstance(lhs, InputTensor)
    assert isinstance(rhs, InputTensor)
    assert lhs.proj.tensor == "D"
    assert rhs.proj.tensor == "NewlyRelaxed"


def test_bf_artifact_update_implements_shift_left_truth_table() -> None:
    """UPDATE must faithfully implement the `<<` merge over the 4 presence
    cases (einsum.md step 4, ``<<(union)``; def in
    the EDGE paper appendix):

        both empty   -> empty   (no distance either way)
        left only    -> left    (distance unchanged this round; CARRY FORWARD)
        right only   -> right    (newly reached vertex)
        both present -> right    (relaxed to a shorter distance)

    We drive the value-selection that the artifact's *actual* op pairing
    (compute + merge, read off the JSON) would produce, and compare it to
    the `<<` value-selection above. The presence set comes from the merge's
    truth table in op_properties.py; the value comes from the compute op.

    For `take_right(union)`, the (left, right-absent) case is present (union)
    but the value is the right operand -- which is empty -- so the carried
    distance is dropped. That is the divergence this test catches.
    """
    program = _load_program()
    update = _einsum_for(program, "D")
    map_specs = [s for s in update.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    spec = map_specs[0]

    # Sentinel value carried by an EMPTY cell (D's empty = +inf).
    EMPTY = math.inf
    LEFT_VAL = 7  # arbitrary distinct finite distances
    RIGHT_VAL = 3

    def artifact_update(left_present: bool, right_present: bool) -> float:
        """Value UPDATE produces under the artifact's actual op pairing.

        Presence of the output cell is the merge's truth table. When the
        cell is present, its value is the compute op applied to the two
        operands, where an absent operand contributes the EMPTY sentinel.
        """
        merge_sym = spec.merge_op.symbol  # type: ignore[union-attr]
        present = get_merge_props(merge_sym).truth_table[(left_present, right_present)]
        if not present:
            return EMPTY
        left = LEFT_VAL if left_present else EMPTY
        right = RIGHT_VAL if right_present else EMPTY
        compute_name = getattr(spec.compute_op, "name", None) or getattr(
            spec.compute_op, "symbol", None
        )
        if compute_name == "update":
            # `<<`: the right operand where present, else carry the left.
            return right if right_present else left
        if compute_name == "take_right":
            return right
        if compute_name == "take_left":
            return left
        raise AssertionError(f"unmodeled UPDATE compute op: {compute_name!r}")

    def shift_left_expected(left_present: bool, right_present: bool) -> float:
        """`<<` value-selection: right where right present, else left."""
        if not left_present and not right_present:
            return EMPTY
        if right_present:
            return RIGHT_VAL
        return LEFT_VAL  # left present, right absent -> CARRY THE LEFT FORWARD

    for left_present in (False, True):
        for right_present in (False, True):
            assert artifact_update(left_present, right_present) == shift_left_expected(
                left_present, right_present
            ), (left_present, right_present)


################################################################################
# Stopping condition: <> : D_{i+1} == D_i  (convergence)
################################################################################


def test_bf_artifact_stops_when_D_is_unchanged() -> None:
    """The cascade halts when the distance tensor stops changing:
    D_{i+1} == D_i. The IR encodes this as a single Comparison(==) over two
    TensorProjectionValues of D -- lhs projected at i+1, rhs at i. The
    "all points satisfy" lifting of == is exactly the paper's ``D_{i+1} ==
    D_i``. einsum.md: ``<> : D_{i+1} == D_i``."""
    program = _load_program()
    conds = program.main_edge.cascade.stopping_conditions
    assert len(conds) == 1
    sc = conds[0]
    assert sc.rank_variable == "i"
    pred = sc.predicate
    assert isinstance(pred, Comparison)
    assert pred.op == "=="

    # Both sides project the SAME distance tensor D.
    assert isinstance(pred.lhs, TensorProjectionValue)
    assert isinstance(pred.rhs, TensorProjectionValue)
    assert pred.lhs.proj.tensor == "D"
    assert pred.rhs.proj.tensor == "D"

    # lhs is projected at i+1, rhs at i.
    lhs_rank = pred.lhs.proj.ranks[0]
    assert isinstance(lhs_rank, RankArith)
    assert lhs_rank.op == "+"
    assert lhs_rank.lhs == RankVariable(name="i")
    assert lhs_rank.rhs == RankConstantLiteral(value=1)

    rhs_rank = pred.rhs.proj.ranks[0]
    assert rhs_rank == RankVariable(name="i")
