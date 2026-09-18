"""Einsum-derived tests for the DFS artifact.

Every assertion in this file is justified by the *math* in
``examples/algorithms/dfs/einsum.md`` (the order-stamped-value DFS), NOT by
whatever happens to sit in ``dfs_program.json``. Each test answers the
question "does the artifact encode what the einsum *means*?"

The governing spec (one ``$$ ... $$`` block) is reproduced here so the
assertions can be traced line-by-line:

    Tensors
      G^{S≡|V|, D≡|V|} -> Boolean, empty = False        (static graph)
      S^{I, V≡|V|}     -> integer, empty = -1           (stack; payload = stamp)
      P^{I, V≡|V|}     -> Boolean, empty = False         (visited)
      F^{I, V≡|V|}     -> integer, empty = -1            (one-hot top)
      N^{I, D≡|V|}     -> Boolean, empty = False          (neighbors)
      U^{I, D≡|V|}     -> Boolean, empty = False          (undiscovered nbrs)
      T^{I, V≡|V|}     -> integer, empty = -1             (stack after pop)
      S'^{I, V≡|V|}    -> integer, empty = -1             (newly stamped)

    Stamp:  sigma(i, v) = i * |V| + v       (per-iteration-point value)

    Init (root = root_id):
      S_{0, v : v in root_id} = sigma(0, v)
      P_{0, v : v in root_id} = True

    Cascade (one pop+expand per iteration i):
      (1) PEEK:  F_{i, v*} = S_{i, v}        :: populate_{v*} 1(select-max-val)
      (2) ADV:   N_{i, d}  = G_{s, d} . F_{i, s}
                                             :: Map_s take_left(intersect)
                                                Reduce_s ANY(union)
      (3) MASK:  U_{i, d}  = N_{i, d} . not P_{i, d}
                                             :: Map_d take_left(intersect)
      (4a) POP:  T_{i, v}  = S_{i, v} . not F_{i, v}
                                             :: Map_v take_left(intersect)
      (4b) STAMP: S'_{i, v} = U_{i, v} . sigma(i+1, v)
                                             :: Map_v take_right(intersect)
      (4c) PUSH: S_{i+1, v} = T_{i, v} . S'_{i, v}
                                             :: Map_v <<(union)
      (5) VISIT: P_{i+1, v} = P_{i, v} . S'_{i, v}
                                             :: Map_v OR(union)
      Stop: <> : ||S_{i+1}|| == 0

Three known gaps make some math-correct assertions fail against the
current artifact; those tests are marked ``xfail`` so they flip to
passing once the encoding is fixed:

  G1 -- the rank-as-value stamp ``sigma`` has no IR leaf, so STAMP's
        RHS and the S_0 init RHS are the placeholder tensor ``Stamp``
        rather than the per-point value.
  G3 -- the update merge ``<<`` (PUSH) is encoded as ``take_right(union)``,
        which is NOT ``<<``: ``<<`` carries the LEFT operand forward where
        the right is empty, ``take_right`` drops it. See TODO.md "Bugs".

(G2 -- ``select-max-val`` being a not-yet-registered UDF coordinate op --
 is a runtime-registry gap, not an *encoding* gap: the math says the
 PEEK coordinate op is exactly this user-defined op, so the artifact's
 encoding is faithful and the corresponding test PASSES.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum
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
from edge_ir.ir.stopping import (
    Comparison,
    PropertyApp,
    TensorProjectionValue,
)
from edge_ir.ir.tensor import BuiltinDataType, CoordSetName
from edge_ir.runtime.op_properties import MERGE_OP_PROPERTIES

DFS_PROGRAM_PATH = Path("examples/algorithms/dfs/dfs_program.json")


def _load_program() -> Program:
    return Program.model_validate_json(DFS_PROGRAM_PATH.read_text())


def _decls_by_name(program: Program) -> dict[str, object]:
    return {d.name: d for d in program.declarations}


def _einsum(program: Program, output_tensor: str) -> Einsum:
    """The cascade einsum whose output is ``output_tensor``.

    PUSH (S) and VISIT (P) are distinguished from any other einsum with
    the same output name by uniqueness within the cascade: each math
    step writes a distinct output tensor, so the output name is a
    unique key over the seven cascade einsums.
    """
    matches = [
        e for e in program.main_edge.cascade.einsums if e.output_tensor == output_tensor
    ]
    assert len(matches) == 1, (
        f"expected exactly one cascade einsum writing {output_tensor!r}, "
        f"found {len(matches)}"
    )
    return matches[0]


def _only_map(einsum: Einsum) -> MapSpec:
    maps = [s for s in einsum.specs if isinstance(s, MapSpec)]
    assert len(maps) == 1, f"expected one MapSpec, found {len(maps)}"
    return maps[0]


################################################################################
# Round trip
################################################################################


def test_dfs_artifact_round_trips() -> None:
    """Load -> dump -> load is the identity on the JSON. Guards against
    any lossy field on the IR datamodel for this artifact."""
    raw = DFS_PROGRAM_PATH.read_text()
    program = Program.model_validate_json(raw)
    re_serialized = program.model_dump_json(indent=2)
    assert json.loads(raw) == json.loads(re_serialized)


################################################################################
# Tensor declarations (lines 2-10 of einsum.md)
################################################################################


@pytest.mark.parametrize(
    ("name", "dtype", "empty"),
    [
        # G is the static graph: Boolean, empty False.
        ("G", "bool", False),
        # S is the stack; payload is an integer stamp; empty = -1, chosen
        # OUTSIDE the stamp range (stamps start at sigma(0,0) = 0) so that
        # "empty stack slot" is distinguishable from "stamp 0". This is the
        # EDGE empty-vs-zero distinction: -1 is the empty sentinel, 0 is a
        # real stamp value.
        ("S", "int", -1),
        ("P", "bool", False),
        ("F", "int", -1),
        ("N", "bool", False),
        ("U", "bool", False),
        ("T", "int", -1),
        ("Sprime", "int", -1),
    ],
)
def test_dfs_artifact_tensor_dtype_and_empty(
    name: str, dtype: str, empty: object
) -> None:
    """Each math tensor's declared data type and empty value match the
    declarations block. The integer stack tensors (S, F, T, S') use -1 as
    their empty value precisely so empty != 0 (== stamp sigma(0,0))."""
    decls = _decls_by_name(_load_program())
    assert name in decls, f"{name} must be declared"
    decl = decls[name]
    assert decl.data_type == BuiltinDataType(name=dtype)  # type: ignore[attr-defined]
    assert decl.empty_value == empty  # type: ignore[attr-defined]
    # Guard the empty-vs-zero distinction for the integer stack tensors.
    if dtype == "int":
        assert decl.empty_value != 0  # type: ignore[attr-defined]


def test_dfs_artifact_G_is_static_no_iteration_rank() -> None:
    """G is the static adjacency matrix G^{S, D}: it carries NO
    per-iteration rank I. Only the per-iteration tensors (S, P, F, N, U,
    T, S') carry I. A spurious I rank on the static graph would be the
    seqn:Gf bug recorded in TODO.md, so this pins the correction."""
    g = _decls_by_name(_load_program())["G"]
    rank_names = [r.name for r in g.ranks]  # type: ignore[attr-defined]
    assert rank_names == ["S", "D"]
    assert "I" not in rank_names


@pytest.mark.parametrize(
    ("name", "expected_ranks"),
    [
        ("S", ["I", "V"]),
        ("P", ["I", "V"]),
        ("F", ["I", "V"]),
        ("N", ["I", "D"]),
        ("U", ["I", "D"]),
        ("T", ["I", "V"]),
        ("Sprime", ["I", "V"]),
    ],
)
def test_dfs_artifact_per_iteration_tensors_carry_I(
    name: str, expected_ranks: list[str]
) -> None:
    """Every per-iteration tensor carries the generational rank I first,
    then its value rank (V for stack/visited tensors, D for the
    neighbor-frontier tensors). The math superscripts are exactly these."""
    decl = _decls_by_name(_load_program())[name]
    assert [r.name for r in decl.ranks] == expected_ranks  # type: ignore[attr-defined]


################################################################################
# Initialization: S_{0,v in root_id} = sigma(0,v) ; P_{0,v in root_id} = True
################################################################################


def test_dfs_artifact_init_has_two_einsums() -> None:
    """The init block sets S_0 and P_0: exactly two einsums."""
    program = _load_program()
    outputs = {e.output_tensor for e in program.initialization.einsums}
    assert outputs == {"S", "P"}
    assert len(program.initialization.einsums) == 2


def _init_einsum(program: Program, output_tensor: str) -> Einsum:
    matches = [
        e for e in program.initialization.einsums if e.output_tensor == output_tensor
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("output_tensor", ["S", "P"])
def test_dfs_artifact_init_restricts_to_root_id(output_tensor: str) -> None:
    """Both init einsums are RESTRICTED to v in root_id by a single
    SetMembership predicate (negated=False). The math writes the
    subscript ``v : v in root_id``: only the root coordinate(s) get a
    value at i=0, not the whole V rank. The output rank list is the
    pinned literal 0 (the generational rank) and the free var v."""
    program = _load_program()
    e = _init_einsum(program, output_tensor)
    assert e.output_ranks == [RankConstantLiteral(value=0), RankVariable(name="v")]
    assert len(e.predicates) == 1
    pred = e.predicates[0]
    assert isinstance(pred, SetMembership)
    assert pred.member == RankVariable(name="v")
    assert isinstance(pred.coord_set, CoordSetName)
    assert pred.coord_set.name == "root_id"
    assert pred.negated is False
    # Init is a pure assignment under a predicate -- no map/reduce/populate.
    assert e.specs == []


def test_dfs_artifact_init_P0_is_literal_true() -> None:
    """P_{0,v} = True: the RHS is the boolean literal True lifted to a
    zero-rank tensor. The literal must be a real Python ``bool`` (not the
    int 1) and its declared type bool, since P is a Boolean tensor."""
    program = _load_program()
    p0 = _init_einsum(program, "P")
    assert isinstance(p0.expression, InputTensor)
    lit_name = p0.expression.proj.tensor
    lit = _decls_by_name(program)[lit_name]
    assert lit.data_type == BuiltinDataType(name="bool")  # type: ignore[attr-defined]
    assert lit.value is True  # type: ignore[attr-defined]
    assert type(lit.value) is bool  # type: ignore[attr-defined]


def test_dfs_artifact_init_S0_rhs_is_per_point_stamp() -> None:
    """S_{0,v} = sigma(0,v) = v. The init RHS must be the per-iteration-
    point rank-as-value ``v`` (for i=0, sigma(0,v) collapses to v itself),
    NOT a constant scalar that cannot depend on v. A zero-rank `Stamp`
    placeholder is a CONSTANT (D21: scalars are zero-rank broadcast
    tensors) and cannot encode the per-point value, so this is the G1
    encoding gap. The faithful encoding's RHS depends on the rank variable
    v -- i.e. it is NOT a bare projection of a zero-rank constant tensor."""
    program = _load_program()
    s0 = _init_einsum(program, "S")
    # The math operand sigma(0,v) depends on v. The placeholder `Stamp` is a
    # zero-rank constant (empty ranks, a single broadcast value), so detecting
    # "RHS does not vary with v" catches the gap.
    if isinstance(s0.expression, InputTensor):
        stamp_decl = _decls_by_name(program).get(s0.expression.proj.tensor)
        is_zero_rank_constant = (
            stamp_decl is not None
            and stamp_decl.ranks == []  # type: ignore[attr-defined]
            and s0.expression.proj.ranks == []
        )
        assert not is_zero_rank_constant, (
            "S_0 init RHS is a zero-rank constant tensor; it cannot encode the "
            "per-point stamp sigma(0,v)=v"
        )


################################################################################
# (1) PEEK:  F_{i, v*} = S_{i, v} :: populate_{v*} 1(select-max-val)
################################################################################


def test_dfs_artifact_peek_is_populate_over_v_with_select_max_val() -> None:
    """PEEK is a POPULATE (not a map/reduce): it produces the output
    coordinate set (the one-hot top-of-stack) from a computed value via a
    coordinate op. The mutable rank list is ["v"] (v* in the math); the
    coordinate op is the user-defined ``select-max-val`` argmax-over-stamp.
    The populate carries the stamp value through, so the compute op is the
    value-passing ``take_left`` (the math ``1`` selects, take_left passes
    S's stamp). The presence of a PopulateSpec (and absence of Map/Reduce)
    is what distinguishes PEEK from the surrounding map einsums."""
    program = _load_program()
    peek = _einsum(program, "F")
    assert peek.output_ranks == [RankVariable(name="i"), RankVariable(name="v")]
    pops = [s for s in peek.specs if isinstance(s, PopulateSpec)]
    assert len(pops) == 1, "PEEK must be a populate"
    assert not any(isinstance(s, (MapSpec, ReduceSpec)) for s in peek.specs)
    pop = pops[0]
    assert pop.rank_list == ["v"]
    assert pop.coord_op.name == "select-max-val"
    assert pop.compute_op == UserDefinedComputeOp(name="identity")


################################################################################
# (2) ADVANCE: N_{i,d} = G_{s,d} . F_{i,s} :: Map_s take_left(intersect)
#                                              Reduce_s ANY(union)
################################################################################


def test_dfs_artifact_advance_map_is_take_left_intersect() -> None:
    """ADVANCE's map is the contraction body G . F: take_left(intersect).
    intersect (the ∩ merge) means a point is effectual only where BOTH G
    and F are present -- i.e. only edges out of the peeked top vertex."""
    program = _load_program()
    adv = _einsum(program, "N")
    map_spec = _only_map(adv)
    assert map_spec.compute_op == UserDefinedComputeOp(name="take_left")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="intersect")


def test_dfs_artifact_advance_reduce_is_ANY_union_over_s() -> None:
    """ADVANCE reduces the source rank ``s`` with ANY(union). ANY is the
    boolean OR-reduction (a neighbor d is in N iff ANY source s reaches
    it); union is the merge so empty reduction operands don't block. The
    reduced rank is exactly ``s``."""
    program = _load_program()
    adv = _einsum(program, "N")
    reduces = [s for s in adv.specs if isinstance(s, ReduceSpec)]
    assert len(reduces) == 1
    red = reduces[0]
    assert red.rank_list == ["s"]
    assert red.compute_op == UserDefinedComputeOp(name="ANY")
    assert red.merge_op == BuiltinMergeOp(symbol="union")


def test_dfs_artifact_advance_reduces_s_so_it_vanishes_from_output() -> None:
    """The whole point of the reduce over s: the source rank ``s`` is
    contracted away. It appears in the inputs (G_{s,d}, F_{i,s}) but must
    NOT appear in the output N_{i,d}. The output coordinate set is over
    {i, d} only; s is gone. (populate vs reduce distinction: a populate
    would have produced an output coordinate FROM s; the reduce collapses
    it.)"""
    program = _load_program()
    adv = _einsum(program, "N")
    output_var_names = {r.name for r in adv.output_ranks if isinstance(r, RankVariable)}
    assert output_var_names == {"i", "d"}
    assert "s" not in output_var_names
    # s is genuinely an input rank that got reduced.
    assert isinstance(adv.expression, BinaryApp)
    lhs = adv.expression.lhs
    rhs = adv.expression.rhs
    assert isinstance(lhs, InputTensor) and isinstance(rhs, InputTensor)
    input_rank_names = {
        rv.name
        for proj in (lhs.proj, rhs.proj)
        for rv in proj.ranks
        if isinstance(rv, RankVariable)
    }
    assert "s" in input_rank_names


################################################################################
# (3) MASK: U_{i,d} = N_{i,d} . not P_{i,d} :: Map_d take_left(intersect)
################################################################################


def test_dfs_artifact_mask_is_take_left_intersect_with_negated_P() -> None:
    """MASK keeps neighbors that are NOT yet visited: U = N . (not P),
    Map_d take_left(intersect). The RHS is a UNARY ``not`` over P (the
    discovered set); intersect keeps the point only where both N is
    present and (not P) is present -- i.e. undiscovered neighbors. The
    reduce-rank-free map ranges over d only."""
    program = _load_program()
    mask = _einsum(program, "U")
    map_spec = _only_map(mask)
    assert map_spec.compute_op == UserDefinedComputeOp(name="take_left")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="intersect")
    assert isinstance(mask.expression, BinaryApp)
    rhs = mask.expression.rhs
    assert isinstance(rhs, UnaryApp)
    assert rhs.op == BuiltinUnaryOp(symbol="not")
    assert isinstance(rhs.operand, InputTensor)
    assert rhs.operand.proj.tensor == "P"


################################################################################
# (4a) POP: T_{i,v} = S_{i,v} . not F_{i,v} :: Map_v take_left(intersect)
################################################################################


def test_dfs_artifact_pop_is_masked_removal_not_subtraction() -> None:
    """POP removes the top via MASKING, not via a subtract/remove merge:
    T = S . (not F), Map_v take_left(intersect). intersect keeps a stack
    slot only where S is present AND (not F) is present -- i.e. every slot
    except the popped top. take_left carries S's stamp through. The math
    is explicit that POP uses no ``-`` / ``>>`` removal merge."""
    program = _load_program()
    pop = _einsum(program, "T")
    map_spec = _only_map(pop)
    assert map_spec.compute_op == UserDefinedComputeOp(name="take_left")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="intersect")
    assert isinstance(pop.expression, BinaryApp)
    lhs = pop.expression.lhs
    rhs = pop.expression.rhs
    assert isinstance(lhs, InputTensor) and lhs.proj.tensor == "S"
    assert isinstance(rhs, UnaryApp)
    assert rhs.op == BuiltinUnaryOp(symbol="not")
    assert rhs.operand.proj.tensor == "F"


################################################################################
# (4b) STAMP: S'_{i,v} = U_{i,v} . sigma(i+1,v) :: Map_v take_right(intersect)
################################################################################


def test_dfs_artifact_stamp_map_op_is_take_right_intersect() -> None:
    """STAMP's map OP shape is faithful: take_right(intersect). intersect
    keeps only undiscovered-neighbor coords (where both U and the stamp
    are present), and take_right selects the fresh-stamp value. This part
    of step (4b) is exact; only the RHS operand (the stamp itself) is the
    G1 gap, tested separately."""
    program = _load_program()
    stamp = _einsum(program, "Sprime")
    map_spec = _only_map(stamp)
    assert map_spec.compute_op == UserDefinedComputeOp(name="take_right")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="intersect")


def test_dfs_artifact_stamp_rhs_is_per_point_stamp() -> None:
    """The RHS of STAMP is sigma(i+1, v) = (i+1)*|V| + v, a per-iteration-
    point rank-as-value: it MUST vary with both i and v. The placeholder
    `Stamp` is a zero-rank constant broadcast tensor (D21) and structurally
    cannot encode a value that depends on the iteration point. The faithful
    encoding's RHS is therefore NOT a bare projection of a zero-rank
    constant tensor."""
    program = _load_program()
    stamp = _einsum(program, "Sprime")
    assert isinstance(stamp.expression, BinaryApp)
    rhs = stamp.expression.rhs
    if isinstance(rhs, InputTensor):
        rhs_decl = _decls_by_name(program).get(rhs.proj.tensor)
        is_zero_rank_constant = (
            rhs_decl is not None
            and rhs_decl.ranks == []  # type: ignore[attr-defined]
            and rhs.proj.ranks == []
        )
        assert not is_zero_rank_constant, (
            "STAMP RHS is a zero-rank constant tensor; it cannot encode the "
            "per-point stamp sigma(i+1,v)=(i+1)*|V|+v"
        )


################################################################################
# (4c) PUSH: S_{i+1,v} = T_{i,v} . S'_{i,v} :: Map_v <<(union)
################################################################################


def test_dfs_artifact_push_writes_next_generation_with_union_merge() -> None:
    """PUSH writes the NEXT generation of the stack: output rank is i+1
    (RankArith ``i + 1``), value rank v. The MERGE is union (presence:
    present where either the kept stack T or the freshly stamped S' is
    present). The merge half is faithful; the COMPUTE half (`<<` vs
    take_right) is the G3 bug, tested separately below."""
    program = _load_program()
    push = _einsum(program, "S")
    assert push.output_ranks[0] == RankArith(
        op="+", lhs=RankVariable(name="i"), rhs=RankConstantLiteral(value=1)
    )
    assert push.output_ranks[1] == RankVariable(name="v")
    map_spec = _only_map(push)
    assert map_spec.merge_op == BuiltinMergeOp(symbol="union")


def test_dfs_artifact_push_lhs_is_T_rhs_is_Sprime() -> None:
    """PUSH combines the popped stack T (left) with the freshly stamped
    discoveries S' (right). Operand ORDER matters for ``<<``
    semantics: ``<<`` carries the LEFT forward where the right is empty,
    so swapping operands would change the meaning."""
    program = _load_program()
    push = _einsum(program, "S")
    assert isinstance(push.expression, BinaryApp)
    lhs, rhs = push.expression.lhs, push.expression.rhs
    assert isinstance(lhs, InputTensor) and lhs.proj.tensor == "T"
    assert isinstance(rhs, InputTensor) and rhs.proj.tensor == "Sprime"


def test_dfs_artifact_push_compute_op_encodes_left_shift_not_take_right() -> None:
    """PUSH's compute op must encode ``<<`` (carry-left-where-right-empty),
    NOT ``take_right``.

    The 4-case VALUE truth table for ``<<`` (which operand's value the
    result carries, by (left_present, right_present)):

        (absent,  absent ) -> empty       (no value either side)
        (absent,  present) -> RIGHT       (take the new stamp)
        (present, absent ) -> LEFT        (carry the kept stack forward)  <-- key
        (present, present) -> RIGHT       (the new stamp wins)

    ``take_right`` gives RIGHT (i.e. right's empty value) in the
    (present, absent) row -- it DROPS the left operand there. That single
    row is the bug: under faithful ``<<`` the kept-but-not-rediscovered
    stack slots survive; under take_right they vanish. So the compute op
    in the IR must NOT be ``take_right``; it must be a dedicated ``<<``
    compute op (per TODO.md the fix is a named compute op paired with the
    union merge)."""
    program = _load_program()
    push = _einsum(program, "S")
    map_spec = _only_map(push)
    # The faithful encoding must not be take_right -- that is the bug.
    assert map_spec.compute_op != UserDefinedComputeOp(name="take_right"), (
        "PUSH compute op is take_right, which is not `<<`; it drops the kept "
        "stack on the (left present, right absent) case"
    )
    # And whatever op it IS, the union merge presence pattern must match the
    # `<<` presence pattern (present iff either operand present), confirming
    # `<<` is a COMPUTE op paired with the union MERGE, not a 17th merge symbol.
    union_tt = MERGE_OP_PROPERTIES["union"].truth_table
    assert union_tt == {
        (False, False): False,
        (False, True): True,
        (True, False): True,
        (True, True): True,
    }


################################################################################
# (5) VISIT: P_{i+1,v} = P_{i,v} . S'_{i,v} :: Map_v OR(union)
################################################################################


def test_dfs_artifact_visit_accumulates_visited_with_OR_union() -> None:
    """VISIT accumulates the discovered set monotonically: P_{i+1} =
    P_i OR S'. Map_v OR(union). union admits a point where EITHER the old
    visited set or the new stamp is present; OR computes the boolean
    accumulation. Output is the next generation i+1 over v."""
    program = _load_program()
    visit = _einsum(program, "P")
    assert visit.output_ranks[0] == RankArith(
        op="+", lhs=RankVariable(name="i"), rhs=RankConstantLiteral(value=1)
    )
    assert visit.output_ranks[1] == RankVariable(name="v")
    map_spec = _only_map(visit)
    assert map_spec.compute_op == UserDefinedComputeOp(name="OR")
    assert map_spec.merge_op == BuiltinMergeOp(symbol="union")
    assert isinstance(visit.expression, BinaryApp)
    lhs, rhs = visit.expression.lhs, visit.expression.rhs
    assert isinstance(lhs, InputTensor) and lhs.proj.tensor == "P"
    assert isinstance(rhs, InputTensor) and rhs.proj.tensor == "Sprime"


################################################################################
# Cascade shape: exactly the seven math steps
################################################################################


def test_dfs_artifact_cascade_has_seven_einsums() -> None:
    """The cascade is exactly the seven steps PEEK, ADV, MASK, POP, STAMP,
    PUSH, VISIT -- one pop+expand per iteration."""
    program = _load_program()
    assert len(program.main_edge.cascade.einsums) == 7


################################################################################
# Stopping condition:  <> : ||S_{i+1}|| == 0
################################################################################


def test_dfs_artifact_stops_when_stack_occupancy_hits_zero() -> None:
    """The diamond stops the generational rank i when the NEXT stack
    S_{i+1} is empty: occupancy(S_{i+1}) == 0. occupancy counts non-empty
    coordinates (the empty-vs-zero distinction matters here: a slot whose
    stamp is the literal 0 is OCCUPIED, because S's empty value is -1, not
    0 -- so occupancy counts it). The comparison is against the integer
    literal 0, and the operand projects S at the next generation i+1."""
    program = _load_program()
    conds = program.main_edge.cascade.stopping_conditions
    assert len(conds) == 1
    sc = conds[0]
    assert sc.rank_variable == "i"
    assert isinstance(sc.predicate, Comparison)
    assert sc.predicate.op == "=="

    lhs = sc.predicate.lhs
    assert isinstance(lhs, PropertyApp)
    assert lhs.name == "occupancy"
    assert lhs.pinned_rank_variables == ["i"]
    assert [op.tensor for op in lhs.operands] == ["S"]
    # The occupancy operand is S at the NEXT generation i+1, matching ||S_{i+1}||.
    s_operand = lhs.operands[0]
    assert s_operand.ranks[0] == RankArith(
        op="+", lhs=RankVariable(name="i"), rhs=RankConstantLiteral(value=1)
    )

    rhs = sc.predicate.rhs
    assert isinstance(rhs, TensorProjectionValue)
    lit = _decls_by_name(program)[rhs.proj.tensor]
    assert lit.data_type == BuiltinDataType(name="int")  # type: ignore[attr-defined]
    assert lit.value == 0  # type: ignore[attr-defined]
    assert type(lit.value) is int  # type: ignore[attr-defined]


def test_dfs_artifact_stop_compares_occupancy_not_elementwise_value() -> None:
    """The stop test is ||S_{i+1}|| == 0 (occupancy), which is NOT the same
    as the elementwise S_{i+1} == 0. Because S's empty value is -1, the
    elementwise reading would be ``every present slot holds stamp 0``,
    which is a different (and wrong) condition. Pin that the LHS is an
    occupancy PropertyApp and not a bare tensor-value comparison."""
    program = _load_program()
    sc = program.main_edge.cascade.stopping_conditions[0]
    assert isinstance(sc.predicate, Comparison)
    assert isinstance(sc.predicate.lhs, PropertyApp)
    assert not isinstance(sc.predicate.lhs, TensorProjectionValue)
