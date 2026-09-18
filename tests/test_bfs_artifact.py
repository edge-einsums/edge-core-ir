"""End-to-end test: BFS program JSON round-trips through the IR.

The on-disk artifact at ``examples/algorithms/bfs/bfs_program.json`` reflects
the reconciliation with ``examples/algorithms/bfs/einsum.md``:

- F's data_type is ``int`` (not ``float``).
- T's empty_value is ``float('inf')`` (not 0).
- The reduce action in Einsum (a) uses
  ``UserDefinedComputeOp(name="ANY")`` (not ``"min"``).
- The stopping-condition rhs is a ``TensorProjectionValue`` pointing to
  a synthesized zero-rank ``TensorDeclaration`` named ``Lit_0`` that
  carries the integer literal ``0`` on its ``value`` field.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from edge_ir.ir.actions import MapSpec, ReduceSpec
from edge_ir.ir.expr import (
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankVariable,
    UnaryApp,
)
from edge_ir.ir.op import (
    BuiltinMergeOp,
    BuiltinUnaryOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.predicate import SetMembership
from edge_ir.ir.program import Program
from edge_ir.ir.stopping import Comparison, PropertyApp, TensorProjectionValue
from edge_ir.ir.tensor import BuiltinDataType, CoordSetName

BFS_PROGRAM_PATH = Path("examples/algorithms/bfs/bfs_program.json")


def _load_program() -> Program:
    raw = BFS_PROGRAM_PATH.read_text()
    return Program.model_validate_json(raw)


def test_bfs_artifact_round_trips() -> None:
    raw = BFS_PROGRAM_PATH.read_text()
    program = Program.model_validate_json(raw)
    re_serialized = program.model_dump_json(indent=2)
    assert json.loads(raw) == json.loads(re_serialized)


def test_bfs_artifact_has_expected_tensors() -> None:
    """All four user tensors must be among the declarations. A
    synthesized literal (e.g. ``Lit_0``) is allowed to appear alongside
    them; the test is robust against any future synthesized literals."""
    program = _load_program()
    names = {decl.name for decl in program.declarations}
    assert {"G", "F", "P", "T"} <= names


def test_bfs_artifact_has_three_einsums() -> None:
    program = _load_program()
    assert len(program.main_edge.cascade.einsums) == 3


def test_bfs_artifact_has_stopping_condition() -> None:
    """The BFS stopping condition is ``<>_i : occupancy(F_{i+1}) == Lit_0``,
    where ``Lit_0`` is the synthesized zero-rank tensor literal whose
    ``value`` carries the integer 0. Its data_type is contextually
    inherited from the ``occupancy`` PropertyApp's RETURN type (int).
    Its empty_value is ``None`` — int's empty is ambiguous (could be 0,
    +inf, -inf depending on algorithm context), so the synthesized
    literal defaults to the None sentinel rather than picking one."""
    program = _load_program()
    conds = program.main_edge.cascade.stopping_conditions
    assert len(conds) == 1
    sc = conds[0]
    assert sc.rank_variable == "i"
    assert isinstance(sc.predicate, Comparison)
    assert sc.predicate.op == "=="
    assert isinstance(sc.predicate.lhs, PropertyApp)
    assert sc.predicate.lhs.name == "occupancy"
    assert sc.predicate.lhs.pinned_rank_variables == ["i"]
    assert [op.tensor for op in sc.predicate.lhs.operands] == ["F"]

    # rhs is a TensorProjectionValue pointing to Lit_0.
    assert isinstance(sc.predicate.rhs, TensorProjectionValue)
    assert sc.predicate.rhs.proj.tensor == "Lit_0"
    assert sc.predicate.rhs.proj.ranks == []

    # Lit_0 must be declared with the expected fields.
    decls_by_name = {d.name: d for d in program.declarations}
    assert "Lit_0" in decls_by_name
    lit_0 = decls_by_name["Lit_0"]
    assert lit_0.ranks == []
    assert lit_0.data_type == BuiltinDataType(name="int")
    assert lit_0.empty_value is None
    assert lit_0.value == 0
    assert type(lit_0.value) is int


################################################################################
# Reconciliation with einsum.md
################################################################################


def test_bfs_artifact_F_data_type_is_int() -> None:
    """F is declared as ``int`` (the einsum.md tex has it as int, not
    float). The artifact must match."""
    program = _load_program()
    decls_by_name = {d.name: d for d in program.declarations}
    f = decls_by_name["F"]
    assert f.data_type == BuiltinDataType(name="int")


def test_bfs_artifact_T_empty_is_inf() -> None:
    """T's empty_value is ``float('inf')`` after reconciliation
    (was 0 before)."""
    program = _load_program()
    decls_by_name = {d.name: d for d in program.declarations}
    t = decls_by_name["T"]
    assert isinstance(t.empty_value, float)
    assert math.isinf(t.empty_value)
    assert t.empty_value > 0


def test_bfs_artifact_einsum_a_reduce_op_is_ANY() -> None:
    """Einsum (a) is the one whose output is T. Its single ReduceSpec
    must carry ``UserDefinedComputeOp(name="ANY")`` after the
    reconciliation (was ``"min"`` before). Einsum (a) is known to have
    exactly one ReduceSpec; the other specs are MapSpecs."""
    program = _load_program()
    einsum_a = next(
        e for e in program.main_edge.cascade.einsums if e.output_tensor == "T"
    )
    reduce_specs = [s for s in einsum_a.specs if isinstance(s, ReduceSpec)]
    assert len(reduce_specs) == 1
    assert reduce_specs[0].compute_op == UserDefinedComputeOp(name="ANY")


################################################################################
# BFS init block: F_0 and P_0 init einsums (restricted-iteration predicates)
################################################################################


def test_bfs_artifact_has_lit_true_declaration() -> None:
    """The BFS artifact synthesizes a zero-rank ``Lit_True`` tensor for
    the P_0 init einsum's RHS (the constant value `True` lifted to a
    scalar tensor). Its data_type is ``bool``; ``empty_value`` is None
    (no canonical empty for a bool literal acting as a scalar source);
    its ``value`` is the Python ``bool`` ``True`` (must be ``bool``, not
    ``int``, since Python's ``True == 1``)."""
    program = _load_program()
    decls_by_name = {d.name: d for d in program.declarations}
    assert "Lit_True" in decls_by_name
    lit_true = decls_by_name["Lit_True"]
    assert lit_true.ranks == []
    assert lit_true.data_type == BuiltinDataType(name="bool")
    assert lit_true.empty_value is None
    assert lit_true.value is True
    assert type(lit_true.value) is bool


def test_bfs_artifact_initialization_has_two_einsums() -> None:
    """The BFS init block now contains two einsums (F_0 and P_0).
    Previously it was empty; this test pins the new length."""
    program = _load_program()
    assert len(program.initialization.einsums) == 2


def test_bfs_artifact_init_einsum_F_shape() -> None:
    """The first init einsum populates F at i=0 over s ∈ id.

    output_tensor='F', output_ranks=[RankConstantLiteral(0), RankVariable('s')],
    expression is an InputTensor projecting Lit_0 (scalar), specs is empty,
    and the predicates list contains exactly one SetMembership saying
    ``s ∈ id`` (negated=False, coord_set=CoordSetName('id'))."""
    program = _load_program()
    init_einsums = program.initialization.einsums
    f_init = next(e for e in init_einsums if e.output_tensor == "F")
    assert f_init.output_ranks == [
        RankConstantLiteral(value=0),
        RankVariable(name="s"),
    ]
    assert isinstance(f_init.expression, InputTensor)
    assert f_init.expression.proj.tensor == "Lit_0"
    assert f_init.specs == []
    assert len(f_init.predicates) == 1
    pred = f_init.predicates[0]
    assert isinstance(pred, SetMembership)
    assert pred.member == RankVariable(name="s")
    assert isinstance(pred.coord_set, CoordSetName)
    assert pred.coord_set.name == "id"
    assert pred.negated is False


def test_bfs_artifact_init_einsum_P_shape() -> None:
    """The second init einsum populates P at i=0 over d ∈ id.

    output_tensor='P', output_ranks=[RankConstantLiteral(0), RankVariable('d')],
    expression is an InputTensor projecting Lit_True (scalar), specs is empty,
    and the predicates list contains exactly one SetMembership saying
    ``d ∈ id`` (negated=False, coord_set=CoordSetName('id'))."""
    program = _load_program()
    init_einsums = program.initialization.einsums
    p_init = next(e for e in init_einsums if e.output_tensor == "P")
    assert p_init.output_ranks == [
        RankConstantLiteral(value=0),
        RankVariable(name="d"),
    ]
    assert isinstance(p_init.expression, InputTensor)
    assert p_init.expression.proj.tensor == "Lit_True"
    assert p_init.specs == []
    assert len(p_init.predicates) == 1
    pred = p_init.predicates[0]
    assert isinstance(pred, SetMembership)
    assert pred.member == RankVariable(name="d")
    assert isinstance(pred.coord_set, CoordSetName)
    assert pred.coord_set.name == "id"
    assert pred.negated is False


################################################################################
# Einsum (a): T_{i,d} = G_{s,d} . F_{i,s} :: AND_s +(intersect) OR_s ANY(union)
#
# The reduce rank `s` is contracted away; it must NOT survive into T's
# output ranks (otherwise it would not have been reduced).
################################################################################


def test_bfs_artifact_einsum_a_reduced_rank_not_in_output() -> None:
    """Einsum (a) reduces over ``s`` (``OR_s``). Per the einsum, T's
    output ranks are ``i, d`` only — ``s`` is the contraction rank and is
    gone from the output. The reduce spec's rank_list must name ``s``,
    and ``s`` must not appear among the output ranks."""
    program = _load_program()
    einsum_a = next(
        e for e in program.main_edge.cascade.einsums if e.output_tensor == "T"
    )
    # Output ranks are exactly i, d (the surviving free ranks).
    assert einsum_a.output_ranks == [
        RankVariable(name="i"),
        RankVariable(name="d"),
    ]
    # The reduced rank is s.
    reduce_specs = [s for s in einsum_a.specs if isinstance(s, ReduceSpec)]
    assert len(reduce_specs) == 1
    assert reduce_specs[0].rank_list == ["s"]
    # s is contracted away: it appears in no output rank.
    output_var_names = {
        r.name for r in einsum_a.output_ranks if isinstance(r, RankVariable)
    }
    assert "s" not in output_var_names


################################################################################
# Einsum (b): F_{i+1,d} = T_{i,d} . NOT P_{i,d} :: AND_d <-(intersect)
#
# The visited-filter, the heart of BFS-not-revisiting. The map's compute
# op is take-left (the ``<-`` arrow: emit the left operand T's value),
# gated by the merge. The merge is intersect: F at i+1 is present only
# where BOTH T (a freshly reached distance) AND (NOT P) (not yet visited)
# are present. The RHS operand is the complement of P: a UnaryApp with a
# ``not`` op over the P projection.
################################################################################


def test_bfs_artifact_einsum_b_map_is_take_left_intersect() -> None:
    """Einsum (b) (output F) carries a single MapSpec whose compute op is
    the user-defined ``take_left`` (the ``<-`` arrow: emit the left
    operand T's value) and whose merge op is the builtin ``intersect``
    (the ``(intersect)``: present only where both operands present). This
    is what makes F_{i+1} carry the distance from T only at not-yet-
    visited destinations."""
    program = _load_program()
    einsum_b = next(
        e for e in program.main_edge.cascade.einsums if e.output_tensor == "F"
    )
    map_specs = [s for s in einsum_b.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    assert [s for s in einsum_b.specs if isinstance(s, ReduceSpec)] == []
    map_spec = map_specs[0]
    assert map_spec.compute_op == UserDefinedComputeOp(name="take_left")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="intersect")


def test_bfs_artifact_einsum_b_rhs_is_complement_of_P() -> None:
    """Einsum (b)'s RHS is ``NOT P_{i,d}``: a UnaryApp whose op is the
    builtin negation (``not``) applied to P projected at ``i, d``. The
    complement is what filters out already-visited vertices; an
    un-negated P (or P at the wrong iteration) would break BFS. The LHS
    is T projected at ``i, d``, and the output writes F at the next
    frontier ``i+1, d``."""
    program = _load_program()
    einsum_b = next(
        e for e in program.main_edge.cascade.einsums if e.output_tensor == "F"
    )
    # Output writes F at the NEXT frontier: i+1, d.
    assert einsum_b.output_ranks == [
        RankArith(
            op="+",
            lhs=RankVariable(name="i"),
            rhs=RankConstantLiteral(value=1),
        ),
        RankVariable(name="d"),
    ]
    expr = einsum_b.expression
    assert isinstance(expr, BinaryApp)

    # LHS: T_{i,d} (the freshly computed distances).
    assert isinstance(expr.lhs, InputTensor)
    assert expr.lhs.proj.tensor == "T"
    assert expr.lhs.proj.ranks == [
        RankVariable(name="i"),
        RankVariable(name="d"),
    ]

    # RHS: NOT P_{i,d} -- the complement of the visited mask.
    assert isinstance(expr.rhs, UnaryApp)
    assert expr.rhs.op == BuiltinUnaryOp(symbol="not")
    assert isinstance(expr.rhs.operand, InputTensor)
    assert expr.rhs.operand.proj.tensor == "P"
    # P is read at the CURRENT iteration i (not i+1): we filter against
    # what was already visited before this step.
    assert expr.rhs.operand.proj.ranks == [
        RankVariable(name="i"),
        RankVariable(name="d"),
    ]


################################################################################
# Einsum (c): P_{i+1,d} = P_{i,d} . F_{i+1,d} :: AND_d OR(union)
#
# The visited-set accumulator. P_{i+1} marks a vertex visited if it was
# already visited (P_i) OR it was reached on this step (F_{i+1}). OR
# compute over a union merge: present at a destination if either operand
# is present, value combined by OR.
################################################################################


def test_bfs_artifact_einsum_c_map_is_OR_union() -> None:
    """Einsum (c) (output P) carries a single MapSpec whose compute op is
    the user-defined ``OR`` and whose merge op is the builtin ``union``
    (present where either operand is present). This is the visited-set
    accumulation: once visited, stay visited."""
    program = _load_program()
    einsum_c = next(
        e for e in program.main_edge.cascade.einsums if e.output_tensor == "P"
    )
    map_specs = [s for s in einsum_c.specs if isinstance(s, MapSpec)]
    assert len(map_specs) == 1
    assert [s for s in einsum_c.specs if isinstance(s, ReduceSpec)] == []
    map_spec = map_specs[0]
    assert map_spec.compute_op == UserDefinedComputeOp(name="OR")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="union")


def test_bfs_artifact_einsum_c_reads_P_at_i_and_F_at_i_plus_1() -> None:
    """Einsum (c) writes P at ``i+1, d`` and reads its two operands at
    distinct iterations: the LHS is P at the CURRENT ``i`` (already
    visited), the RHS is F at the NEXT ``i+1`` (reached this step). Mixing
    these iterations up would either lose accumulated visits or read a
    frontier that does not exist yet."""
    program = _load_program()
    einsum_c = next(
        e for e in program.main_edge.cascade.einsums if e.output_tensor == "P"
    )
    i_plus_1 = RankArith(
        op="+",
        lhs=RankVariable(name="i"),
        rhs=RankConstantLiteral(value=1),
    )
    # Output writes P at i+1, d.
    assert einsum_c.output_ranks == [i_plus_1, RankVariable(name="d")]

    expr = einsum_c.expression
    assert isinstance(expr, BinaryApp)

    # LHS: P_{i,d} -- the prior visited set (current iteration i).
    assert isinstance(expr.lhs, InputTensor)
    assert expr.lhs.proj.tensor == "P"
    assert expr.lhs.proj.ranks == [
        RankVariable(name="i"),
        RankVariable(name="d"),
    ]

    # RHS: F_{i+1,d} -- the newly reached frontier (note: i+1, matching
    # the output's iteration, NOT i).
    assert isinstance(expr.rhs, InputTensor)
    assert expr.rhs.proj.tensor == "F"
    assert expr.rhs.proj.ranks == [i_plus_1, RankVariable(name="d")]
