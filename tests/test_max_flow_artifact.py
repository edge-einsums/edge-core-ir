"""Einsum-derived artifact tests for the max-flow push-relabel program.

The governing principle: every assertion below is derived from the EINSUM
SEMANTICS in ``examples/algorithms/max-flow/einsum.md`` -- the spec -- NOT from
whatever the builder happens to emit. Each test answers "does the artifact
encode what the einsum *means*?"

The spec was rewritten to match the corrected cascade in the EDGE tutorial
zoo (``Push_Relabel_Max_Flow.ipynb``); ``examples/algorithms/max-flow/CHANGES.md``
records what moved and why. Einsum IDs E01-E24 in the docstrings below match
the tutorial and the step-through visualizer.

Cascade layout: 26 IR einsums for 24 spec einsums, because E04's three-arm
``cases`` has no IR node and is lowered to three sequential writes to R.

Notes on operators that appear in the spec:

  - ``<-`` is the take_left merge (paper glyph) and ``->`` the take_right
    merge. Both appear as *merge* operators; ``take_left`` / ``take_right``
    also appear as user-defined *compute* operators, which is a different
    thing sharing a name. The IR distinguishes them by field.

  - ``<<`` (partial update, E24) is emitted as a user-defined compute op
    named ``update``: compute builtins are limited to ``+ - * /`` and the
    operator-name pattern rejects ``<<`` outright.

  - ``|Act_{i+1}| == 0`` in the stopping condition is cardinality
    (occupancy), the count of non-empty coordinates -- distinct from
    "every value equals 0".

  - A stored value equal to a tensor's declared ``empty`` is treated as
    ABSENT. Several intersect merges below rely on this (a saturated edge
    has R = 0 = empty and drops out). Nothing in the IR records the
    convention; it is an evaluator contract.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum
from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    InputTensor,
    RankArith,
    RankConstantLiteral,
    RankVariable,
    UnaryApp,
)
from edge_ir.ir.op import (
    BuiltinComputeOp,
    BuiltinMergeOp,
    BuiltinUnaryOp,
    CoordinateOp,
    UserDefinedComputeOp,
)
from edge_ir.ir.program import Program
from edge_ir.ir.stopping import Comparison, PropertyApp, TensorProjectionValue
from edge_ir.ir.tensor import BuiltinDataType, TensorDeclaration

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "algorithms" / "max-flow"
)
BUILDER_PATH = ARTIFACT_DIR / "build_max_flow_program.py"
JSON_PATH = ARTIFACT_DIR / "max_flow_program.json"


# Spec einsum ID -> position in the IR cascade. E04's cases lowering occupies
# three slots, which is why the two numberings diverge after it.
EINSUM_INDEX = {
    "E01": 0,  # FS   preflow out of the source
    "E02": 1,  # F    antisymmetrize
    "E03": 2,  # E    excess as net inflow
    "E04a": 3,  # R   otherwise arm
    "E04b": 4,  # R   source column
    "E04c": 5,  # R   source row (highest priority, emitted last)
    "E05": 6,  # NST
    "E06": 7,  # Act_i
    "E07": 8,  # ActR
    "E08": 9,  # Lbl
    "E09": 10,  # Adm_i
    "E10": 11,  # PushCand
    "E11": 12,  # Delta
    "E12": 13,  # F_{i+1}
    "E13": 14,  # InPush
    "E14": 15,  # OutPush
    "E15": 16,  # E_{i+1}
    "E16": 17,  # R_{i+1}
    "E17": 18,  # AdmPost_{i+1} (einsum.md: Adm_{i+1})
    "E18": 19,  # HasAdm
    "E19": 20,  # Act_{i+1}
    "E20": 21,  # Rel
    "E21": 22,  # NeiLbl
    "E22": 23,  # MinNeiLbl
    "E23": 24,  # NewD
    "E24": 25,  # D_{i+1}
}


def _load_program() -> Program:
    """Build the max-flow Program by executing the builder module.

    The builder lives in a non-package directory, so it is loaded via an
    explicit file-based import spec rather than a normal ``import``.
    """
    spec = importlib.util.spec_from_file_location(
        "build_max_flow_program", BUILDER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program: Program = module.build_max_flow()
    return program


def _decls(program: Program) -> dict[str, TensorDeclaration]:
    return {d.name: d for d in program.declarations}


def _cascade_einsums(program: Program) -> list[Einsum]:
    return program.main_edge.cascade.einsums


def _einsum_by_output(einsums: list[Einsum], output: str) -> list[Einsum]:
    return [e for e in einsums if e.output_tensor == output]


def _e(program: Program, spec_id: str) -> Einsum:
    """The cascade einsum implementing a given spec ID (E01 .. E24)."""
    return _cascade_einsums(program)[EINSUM_INDEX[spec_id]]


def _map_specs(einsum: Einsum) -> dict[int, MapSpec]:
    return {s.label: s for s in einsum.specs if isinstance(s, MapSpec)}


def _reduce_specs(einsum: Einsum) -> list[ReduceSpec]:
    return [s for s in einsum.specs if isinstance(s, ReduceSpec)]


def _populate_specs(einsum: Einsum) -> list[PopulateSpec]:
    return [s for s in einsum.specs if isinstance(s, PopulateSpec)]


def _i_plus_one() -> RankArith:
    return RankArith(
        op="+", lhs=RankVariable(name="i"), rhs=RankConstantLiteral(value=1)
    )


def _tensor_of(node: object) -> str:
    assert isinstance(node, InputTensor)
    return node.proj.tensor


def _ranks_of(node: object) -> list[object]:
    assert isinstance(node, InputTensor)
    return list(node.proj.ranks)


################################################################################
# Round-trip and on-disk artifact
################################################################################


def test_max_flow_artifact_round_trips() -> None:
    """The IR survives serialize -> deserialize -> serialize unchanged."""
    program = _load_program()
    first = program.model_dump_json(indent=2)
    reparsed = Program.model_validate_json(first)
    second = reparsed.model_dump_json(indent=2)
    assert json.loads(first) == json.loads(second)


def test_max_flow_json_artifact_matches_builder() -> None:
    """``max_flow_program.json`` on disk is what the builder produces.

    bfs, dfs and bellman-ford all commit a program JSON alongside the
    builder; max-flow now does too, and the two must not drift.
    """
    assert JSON_PATH.exists(), "max_flow_program.json is missing"
    on_disk = json.loads(JSON_PATH.read_text())
    from_builder = json.loads(_load_program().model_dump_json(indent=2))
    assert on_disk == from_builder


def test_max_flow_cascade_has_expected_shape() -> None:
    """24 spec einsums, 26 IR einsums (E04's three case arms), one stop."""
    program = _load_program()
    cascade = program.main_edge.cascade
    assert len(cascade.einsums) == 26
    assert len(cascade.stopping_conditions) == 1


################################################################################
# Declarations: tensors, data types, empty values (empty vs zero)
################################################################################


def test_max_flow_declares_all_spec_tensors() -> None:
    """The spec's Tensors block: G, C, F, R, E, D, Act, S, T.

    Synthesized literals / host scalars and cascade intermediates may appear
    alongside them; the test only requires the superset.
    """
    names = set(_decls(_load_program()))
    assert {"G", "C", "F", "R", "E", "D", "Act", "S", "T"} <= names


def test_max_flow_declares_the_preflow_staging_tensor() -> None:
    """E01 writes FS, the saturated source edges before antisymmetrization.

    FS is what makes E02 expressible: you cannot subtract a tensor's own
    transpose from itself in one einsum, so the saturation is staged.
    """
    decls = _decls(_load_program())
    assert "FS" in decls
    assert [r.name for r in decls["FS"].ranks] == ["I", "U", "V"]


@pytest.mark.parametrize("tensor", ["G", "C", "F", "R", "E", "D"])
def test_max_flow_integer_tensors_have_int_dtype(tensor: str) -> None:
    """Spec declares G, C, F, R, E, D as Integer."""
    decl = _decls(_load_program())[tensor]
    assert decl.data_type == BuiltinDataType(name="int")


@pytest.mark.parametrize("tensor", ["Act", "S", "T"])
def test_max_flow_boolean_tensors_have_bool_dtype(tensor: str) -> None:
    """Spec declares Act, S, T as Boolean."""
    decl = _decls(_load_program())[tensor]
    assert decl.data_type == BuiltinDataType(name="bool")


@pytest.mark.parametrize("tensor", ["Act", "S", "T"])
def test_max_flow_boolean_tensors_empty_is_false(tensor: str) -> None:
    """Spec: the Boolean tensors have empty value ``false``."""
    decl = _decls(_load_program())[tensor]
    assert decl.empty_value is False


@pytest.mark.parametrize("tensor", ["G", "C", "F", "R", "E"])
def test_max_flow_flow_valued_tensors_have_zero_empty(tensor: str) -> None:
    """Spec: ``empty = 0`` for capacity, flow, residual and excess.

    For these, 0 and absent genuinely mean the same thing -- a missing
    capacity, a missing flow, a saturated residual and a vertex with no
    excess all behave identically wherever they are read. That equivalence
    is load-bearing: it is what lets E07's intersect skip saturated edges
    without an explicit positivity guard on R.
    """
    decl = _decls(_load_program())[tensor]
    assert decl.empty_value == 0


def test_max_flow_height_empty_is_infinity_not_zero() -> None:
    """Spec: ``D -> Integer, empty = +inf``. NOT 0.

    Height 0 is a real and common height -- every non-source vertex starts
    there -- so it cannot double as the absent marker. E08's height rule and
    E21's neighbour-height gather both read D through an INTERSECT; with
    ``empty = 0`` a height-0 neighbour would be treated as absent and
    dropped before reaching E22's min, so the vertex would relabel to a
    height above its true lowest neighbour. ``+inf`` is also the right
    identity for that min.
    """
    decl = _decls(_load_program())["D"]
    # The IR coerces the "inf" shorthand to a float; it serializes back to
    # the string "inf", the same convention Bellman-Ford uses for its
    # distance tensors.
    assert decl.empty_value == float("inf")


def test_max_flow_literal_scalars_carry_their_value() -> None:
    """The zero-rank stand-ins for literals must record what they *are*.

    A consumer reading only ``empty_value`` has no way to learn that ``One``
    is 1, which makes the artifact un-executable without out-of-band
    knowledge. ``VertexCount`` is the exception: |V| is graph data supplied
    by the host alongside G and C, so it has no literal value here.
    """
    decls = _decls(_load_program())
    assert decls["Zero"].value == 0
    assert decls["One"].value == 1
    assert decls["FalseConst"].value is False
    # `value` is typed Any and gets no coercion, unlike `empty_value` -- so
    # infinity stays the string "inf". That is also what `empty_value`
    # serializes back to, so the two agree on the wire; a consumer parses
    # "inf" the same way in both fields.
    assert decls["MinIdentity"].value == "inf"
    assert decls["VertexCount"].value is None


@pytest.mark.parametrize("tensor", ["NeiLbl", "MinNeiLbl", "NewD"])
def test_max_flow_height_intermediates_inherit_infinite_empty(tensor: str) -> None:
    """The relabel intermediates hold HEIGHTS, so they inherit D's empty value.

    Same argument as D itself, one step later in the cascade: height 0 is
    real and common. With ``empty = 0``, a height-0 neighbour gathered into
    ``NeiLbl`` is treated as absent and never reaches E22's ``min``; a
    ``MinNeiLbl`` of 0 is likewise dropped, so E23 produces no ``NewD`` and
    the vertex does not relabel at all.

    Fixing D but leaving these at 0 reintroduces exactly the bug that fixing
    D was meant to remove. Caught by replaying the interpreter against the
    reference trace.
    """
    decl = _decls(_load_program())[tensor]
    assert decl.empty_value == float("inf")


def test_max_flow_min_reduction_identity_is_infinity() -> None:
    """E22 reduces with ``min``, so its identity operand must be +inf.

    A 0 identity would silently floor every relabel at 0 -- the same class
    of mistake as declaring D's empty value 0.
    """
    decl = _decls(_load_program())["MinIdentity"]
    assert decl.empty_value == float("inf")


@pytest.mark.parametrize("literal", ["Zero", "One"])
def test_max_flow_operand_literals_have_no_empty_value(literal: str) -> None:
    """``Zero`` and ``One`` are operands of intersect merges, so they must exist.

    E06/E19 test ``E > 0`` and E08/E23 add ``+ 1`` through intersects, and
    the D_1 init copies ``0``. A tensor cannot hold its own empty value, so
    a literal declared with ``empty == value`` is absent and every one of
    those intersects visits nothing. On the reference evaluator that left
    ``Act_1`` empty and the cascade stopped after the preflow. The literals
    take ``empty_value = None``, the D21 rule for synthesized literals.
    """
    decl = _decls(_load_program())[literal]
    assert decl.ranks == []
    assert decl.empty_value is None
    assert decl.value is not None


################################################################################
# Initialization block
################################################################################


def test_max_flow_initializes_flow_and_excess_at_generation_zero() -> None:
    """Init ``F_{0,u,v} = 0`` and ``E_{0,u} = 0``."""
    program = _load_program()
    inits = program.initialization.einsums
    f_init = _einsum_by_output(inits, "F")[0]
    e_init = _einsum_by_output(inits, "E")[0]
    assert f_init.output_ranks == [
        RankConstantLiteral(value=0),
        RankVariable(name="u"),
        RankVariable(name="v"),
    ]
    assert e_init.output_ranks == [
        RankConstantLiteral(value=0),
        RankVariable(name="u"),
    ]


def test_max_flow_initializes_height_at_generation_one() -> None:
    """Init ``D_{1,u} = 0``, at generation 1 rather than 0.

    E01-E04 write generation 1 (the preflow round), and the iterative body
    reads ``D_{i}`` from i = 1 onward. Initializing D at generation 0 would
    leave generation 1 unwritten for every non-source vertex.
    """
    program = _load_program()
    d_inits = _einsum_by_output(program.initialization.einsums, "D")
    assert d_inits, "no D initialization"
    for einsum in d_inits:
        assert einsum.output_ranks[0] == RankConstantLiteral(value=1)


def test_max_flow_initialization_sets_source_height_to_vertex_count() -> None:
    """Init ``D_{1,u:u=s} = |V|``: the source's height starts at |V|.

    Lowered through the host S mask as ``S_u take_right VertexCount`` --
    where S is present (the source) the right operand is taken, and the
    intersect keeps it only there.
    """
    program = _load_program()
    d_inits = _einsum_by_output(program.initialization.einsums, "D")
    source_height = next(e for e in d_inits if isinstance(e.expression, BinaryApp))
    assert source_height.output_ranks == [
        RankConstantLiteral(value=1),
        RankVariable(name="u"),
    ]
    expr = source_height.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "S"
    assert isinstance(expr.rhs, InputTensor)
    specs = _map_specs(source_height)
    assert set(specs) == {1}
    assert specs[1].compute_op == UserDefinedComputeOp(name="take_right")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


################################################################################
# E01-E04: the preflow round
################################################################################


def test_max_flow_E01_preflow_saturates_source_edges_into_FS() -> None:
    """E01: ``FS_{1,u,v} = S_u * C_{u,v} :: AND *(intersect)``.

    Preflow saturates every source edge. The merge is intersect: output
    present only where BOTH S_u and C_{u,v} are present, i.e. only on edges
    leaving the source. Union here would emit capacities on every edge.

    The output is FS, not F: F must be antisymmetric, and E02 builds it.
    """
    einsum = _e(_load_program(), "E01")
    assert einsum.output_tensor == "FS"
    assert einsum.output_ranks == [
        RankConstantLiteral(value=1),
        RankVariable(name="u"),
        RankVariable(name="v"),
    ]
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "S"
    assert _tensor_of(expr.rhs) == "C"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == BuiltinComputeOp(symbol="*")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E02_flow_is_antisymmetric() -> None:
    """E02: ``F_{1,u,v} = FS_{1,u,v} - FS_{1,v,u} :: AND -(union)``.

    Both operands are FS, one transposed. At a saturated source edge (s,v)
    this stores +c(s,v); at the reverse cell (v,s) it stores -c(s,v). That
    antisymmetry is what makes E03 a single reduction instead of an
    in-minus-out pair, and it is preserved by E12.

    The merge is UNION: the reverse cell has no forward entry, so an
    intersect would drop exactly the negative half being constructed.
    """
    einsum = _e(_load_program(), "E02")
    assert einsum.output_tensor == "F"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "FS"
    assert _tensor_of(expr.rhs) == "FS"
    # The right operand is the transpose: ranks u,v on the left, v,u on right.
    u, v = RankVariable(name="u"), RankVariable(name="v")
    assert _ranks_of(expr.lhs)[1:] == [u, v]
    assert _ranks_of(expr.rhs)[1:] == [v, u]
    specs = _map_specs(einsum)
    assert specs[1].compute_op == BuiltinComputeOp(symbol="-")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="union")


def test_max_flow_E03_excess_is_net_inflow_in_one_reduction() -> None:
    """E03: ``E_{1,u} = F_{1,v,u} :: OR +(union)``.

    Because F is antisymmetric, summing the column (v,u) already nets the
    outflow off -- outgoing flow is stored as negative entries on the
    reverse cells. So excess needs one reduction, not an In/Out pair
    followed by a subtraction.

    The reduced rank is v; u survives into the output.
    """
    einsum = _e(_load_program(), "E03")
    assert einsum.output_tensor == "E"
    assert einsum.output_ranks == [
        RankConstantLiteral(value=1),
        RankVariable(name="u"),
    ]
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "F"
    # Column read: F_{1,v,u}, not F_{1,u,v}.
    assert _ranks_of(expr.lhs)[1:] == [RankVariable(name="v"), RankVariable(name="u")]
    reduces = _reduce_specs(einsum)
    assert len(reduces) == 1
    assert reduces[0].rank_list == ["v"]
    assert reduces[0].compute_op == BuiltinComputeOp(symbol="+")
    assert reduces[0].merge_op == BuiltinMergeOp(symbol="union")


def test_max_flow_E03_does_not_use_a_separate_in_out_pair() -> None:
    """E03 supersedes the old In/Out formulation, which must be gone.

    In and Out only existed to compute in-minus-out on a non-antisymmetric
    F. With E02 in place they are dead, and leaving them declared would
    invite an evaluator to keep computing them.
    """
    decls = _decls(_load_program())
    assert "In" not in decls
    assert "Out" not in decls


def test_max_flow_E04_lowers_cases_to_three_writes_to_R() -> None:
    """E04 is a three-arm ``cases``; the IR has no cases node.

    It is lowered to three sequential einsums writing R at generation 1,
    relying on later-overwrites-earlier.
    """
    program = _load_program()
    for spec_id in ("E04a", "E04b", "E04c"):
        einsum = _e(program, spec_id)
        assert einsum.output_tensor == "R"
        assert einsum.output_ranks[0] == RankConstantLiteral(value=1)


def test_max_flow_E04_source_row_arm_is_emitted_last() -> None:
    """E04 arm ORDER, and the reason it matters.

    The spec's written order is::

        0        if u = s
        C_{v,u}  if v = s and u != s
        C_{u,v}  otherwise

    with the u=s arm highest priority. Sequential lowering means the LAST
    write wins, so the arms must be emitted in reverse priority: otherwise,
    then the source column, then the source row.

    This is only observable at the (s,s) cell -- a source self-loop -- but
    getting it backwards silently contradicts the written spec there.
    """
    program = _load_program()
    source_row = _e(program, "E04c")
    expr = source_row.expression
    assert isinstance(expr, BinaryApp)
    # The source-row arm is the one masked on S_u (not S_v) producing 0.
    assert _tensor_of(expr.lhs) == "S"
    assert _ranks_of(expr.lhs) == [RankVariable(name="u")]
    assert _tensor_of(expr.rhs) == "Zero"
    # ... and it comes after the source-column arm in cascade order.
    assert EINSUM_INDEX["E04c"] > EINSUM_INDEX["E04b"] > EINSUM_INDEX["E04a"]


def test_max_flow_E04_default_arm_is_capacity() -> None:
    """E04 otherwise arm: ``R_{1,u,v} = C_{u,v}``, an unmasked copy."""
    einsum = _e(_load_program(), "E04a")
    assert _tensor_of(einsum.expression) == "C"
    assert einsum.specs == []


def test_max_flow_E04_source_column_arm_gets_reverse_capacity() -> None:
    """E04 source column: ``R_{1,u,s} = C_{s,u}``.

    Pushing s->u consumes forward residual and creates residual back to the
    source, so the reverse cell starts at the forward capacity.
    """
    einsum = _e(_load_program(), "E04b")
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "S"
    assert _ranks_of(expr.lhs) == [RankVariable(name="v")]
    assert _tensor_of(expr.rhs) == "C"
    assert _ranks_of(expr.rhs) == [RankVariable(name="v"), RankVariable(name="u")]
    specs = _map_specs(einsum)
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


################################################################################
# E05-E09: selecting admissible edges
################################################################################


def test_max_flow_E05_nst_is_neither_source_nor_sink() -> None:
    """E05: ``NST_u = not S_u AND not T_u :: AND AND(intersect)``.

    Source and sink are never active: the source starts with no excess to
    spend and the sink is where flow is meant to accumulate.
    """
    einsum = _e(_load_program(), "E05")
    assert einsum.output_tensor == "NST"
    assert einsum.output_ranks == [RankVariable(name="u")]
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert isinstance(expr.lhs, UnaryApp)
    assert expr.lhs.op == BuiltinUnaryOp(symbol="not")
    assert isinstance(expr.rhs, UnaryApp)
    assert expr.rhs.op == BuiltinUnaryOp(symbol="not")
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="AND")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


@pytest.mark.parametrize("spec_id", ["E06", "E19"])
def test_max_flow_active_is_internal_and_excess_positive(spec_id: str) -> None:
    """E06 / E19: ``Act = NST_u <-(int) (E > 0)``.

    Two labels: the inner one tests excess against zero, the outer one
    intersects with the internal-vertex mask and takes the LEFT value --
    Act is a Boolean, so it carries NST's truth, not the excess.
    """
    einsum = _e(_load_program(), spec_id)
    assert einsum.output_tensor == "Act"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "NST"
    assert isinstance(expr.rhs, AnonymousTensor)
    inner = expr.rhs.expression
    assert isinstance(inner, BinaryApp)
    assert _tensor_of(inner.lhs) == "E"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="take_left")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")
    assert specs[2].compute_op == UserDefinedComputeOp(name="gt")
    assert specs[2].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E19_binds_the_anonymous_operand_at_the_next_generation() -> None:
    """E19's anonymous operand must be bound at i+1, matching the E it reads.

    E19 recomputes Act from ``E_{i+1}`` -- the excess the push just
    produced. Binding the parenthesised subexpression at generation i while
    its operand is at i+1 mismatches the rank it is annotated with.
    """
    einsum = _e(_load_program(), "E19")
    assert einsum.output_ranks[0] == _i_plus_one()
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    anon = expr.rhs
    assert isinstance(anon, AnonymousTensor)
    inner = anon.expression
    assert isinstance(inner, BinaryApp)
    assert _ranks_of(inner.lhs)[0] == _i_plus_one()
    assert anon.ranks[0] == _i_plus_one()


def test_max_flow_E06_reads_the_current_generation() -> None:
    """E06 is the pre-push Act, reading ``E_{i}`` and writing ``Act_{i}``."""
    einsum = _e(_load_program(), "E06")
    assert einsum.output_ranks[0] == RankVariable(name="i")
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    anon = expr.rhs
    assert isinstance(anon, AnonymousTensor)
    inner = anon.expression
    assert isinstance(inner, BinaryApp)
    assert _ranks_of(inner.lhs)[0] == RankVariable(name="i")


def test_max_flow_E07_active_residual_takes_active_over_intersect() -> None:
    """E07: ``ActR_{i,u,v} = Act_{i,u} <-(intersect) R_{i,u,v}``.

    Residual edges leaving an active vertex. The intersect is doing real
    work in both directions: it drops rows for inactive vertices, and --
    because R = 0 equals R's empty value -- it also drops saturated edges,
    which is why no positivity guard on R is needed.
    """
    einsum = _e(_load_program(), "E07")
    assert einsum.output_tensor == "ActR"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "Act"
    assert _tensor_of(expr.rhs) == "R"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="take_left")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E08_height_rule_uses_intersect_not_union() -> None:
    """E08: ``Lbl_{i,u,v} = D_{i,u} ==(intersect) (D_{i,v} + 1)_{i,v}``.

    The height rule D(u) == D(v)+1. Both merges are INTERSECT, and that is
    the point: under a union merge a vertex with no height would be
    compared as though it had one, and could satisfy the rule spuriously.
    A height only participates when it exists.
    """
    einsum = _e(_load_program(), "E08")
    assert einsum.output_tensor == "Lbl"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "D"
    assert isinstance(expr.rhs, AnonymousTensor)
    inner = expr.rhs.expression
    assert isinstance(inner, BinaryApp)
    assert _tensor_of(inner.lhs) == "D"
    assert _tensor_of(inner.rhs) == "One"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="eq")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")
    assert specs[2].compute_op == BuiltinComputeOp(symbol="+")
    assert specs[2].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E09_admissible_is_active_residual_and_height_rule() -> None:
    """E09: ``Adm_{i,u,v} = ActR AND Lbl :: AND AND(intersect)``."""
    einsum = _e(_load_program(), "E09")
    assert einsum.output_tensor == "Adm"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "ActR"
    assert _tensor_of(expr.rhs) == "Lbl"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="AND")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


################################################################################
# E10-E16: the push step
################################################################################


def test_max_flow_E10_pushcand_is_a_populate_selecting_one_v_per_u() -> None:
    """E10: ``PushCand_{i,u,v*} = Adm <<<_{v*} 1(pick-admissible-edge)``.

    A populate, reducing the admissible set to one edge per pushing vertex.
    This selector is a RESOURCE CONSTRAINT, not a serialization point: a
    vertex has a single pool of excess and cannot spend it on several edges
    at once. It is therefore the one selector that cannot be removed to
    gain parallelism.

    The selector is free to pick any admissible edge; correctness does not
    depend on which.
    """
    einsum = _e(_load_program(), "E10")
    assert einsum.output_tensor == "PushCand"
    populates = _populate_specs(einsum)
    assert len(populates) == 1
    assert populates[0].rank_list == ["v"]
    assert populates[0].compute_op == UserDefinedComputeOp(name="pick-admissible-edge")
    assert populates[0].coord_op == CoordinateOp(name="select-one-admissible-v")


@pytest.mark.parametrize("spec_id", ["E03", "E10", "E13", "E14", "E18", "E22"])
def test_max_flow_synthetic_binaries_carry_a_transparent_map(spec_id: str) -> None:
    """The synthetic ``X . identity`` binaries need a Map spec of their own.

    E03/E13/E14/E18/E22 (the reductions) and E10 (the populate) pair their
    one real operand with a scalar so the spec has a binary label to name.
    The evaluator requires a Map spec on every binary. ``take_left`` as the
    merge keeps a point iff the real operand is present, and ``take_left``
    as the compute passes its value on, so the binary adds nothing.
    """
    einsum = _e(_load_program(), spec_id)
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    specs = _map_specs(einsum)
    assert set(specs) == {expr.label}
    assert specs[expr.label].compute_op == UserDefinedComputeOp(name="take_left")
    assert specs[expr.label].merge_op == BuiltinMergeOp(symbol="take_left")


def test_max_flow_E11_delta_is_min_excess_residual_on_selected_edges() -> None:
    """E11: ``Delta = (E min(int) R)_{i,u,v} <-(int) PushCand``.

    You can push no more than the vertex has and no more than the edge
    admits, hence min. The outer take_left keeps the computed amount while
    the intersect with PushCand restricts it to the one chosen edge.
    """
    einsum = _e(_load_program(), "E11")
    assert einsum.output_tensor == "Delta"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert isinstance(expr.lhs, AnonymousTensor)
    inner = expr.lhs.expression
    assert isinstance(inner, BinaryApp)
    assert {_tensor_of(inner.lhs), _tensor_of(inner.rhs)} == {"E", "R"}
    assert _tensor_of(expr.rhs) == "PushCand"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="min")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")
    assert specs[2].compute_op == UserDefinedComputeOp(name="take_left")
    assert specs[2].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E12_flow_update_preserves_antisymmetry() -> None:
    """E12: ``F_{i+1} = (F_i + Delta)_{i,u,v} - Delta_{i,v,u}``.

    Delta is added on the forward cell and subtracted on the reverse one,
    so the antisymmetry E02 established survives every push. Both merges
    are union because a cell may be touched by only one of the three terms.
    """
    einsum = _e(_load_program(), "E12")
    assert einsum.output_tensor == "F"
    assert einsum.output_ranks[0] == _i_plus_one()
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert isinstance(expr.lhs, AnonymousTensor)
    inner = expr.lhs.expression
    assert isinstance(inner, BinaryApp)
    assert _tensor_of(inner.lhs) == "F"
    assert _tensor_of(inner.rhs) == "Delta"
    assert _tensor_of(expr.rhs) == "Delta"
    u, v = RankVariable(name="u"), RankVariable(name="v")
    assert _ranks_of(inner.rhs)[1:] == [u, v]
    assert _ranks_of(expr.rhs)[1:] == [v, u]
    specs = _map_specs(einsum)
    assert specs[1].compute_op == BuiltinComputeOp(symbol="+")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="union")
    assert specs[2].compute_op == BuiltinComputeOp(symbol="-")
    assert specs[2].merge_op == BuiltinMergeOp(symbol="union")


def test_max_flow_E13_inpush_reduces_over_senders() -> None:
    """E13: ``InPush_{i,u} = sum_v Delta_{i,v,u}`` -- flow arriving at u."""
    einsum = _e(_load_program(), "E13")
    assert einsum.output_tensor == "InPush"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "Delta"
    assert _ranks_of(expr.lhs)[1:] == [RankVariable(name="v"), RankVariable(name="u")]
    reduces = _reduce_specs(einsum)
    assert reduces[0].rank_list == ["v"]
    assert reduces[0].compute_op == BuiltinComputeOp(symbol="+")


def test_max_flow_E14_outpush_reduces_over_receivers() -> None:
    """E14: ``OutPush_{i,u} = sum_v Delta_{i,u,v}`` -- flow leaving u.

    Same reduction as E13 with the operand transposed; that transposition
    is the only difference between the two.
    """
    einsum = _e(_load_program(), "E14")
    assert einsum.output_tensor == "OutPush"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "Delta"
    assert _ranks_of(expr.lhs)[1:] == [RankVariable(name="u"), RankVariable(name="v")]
    reduces = _reduce_specs(einsum)
    assert reduces[0].rank_list == ["v"]


def test_max_flow_E15_excess_update_adds_received_subtracts_sent() -> None:
    """E15: ``E_{i+1} = (E_i + InPush)_{i,u} - OutPush``."""
    einsum = _e(_load_program(), "E15")
    assert einsum.output_tensor == "E"
    assert einsum.output_ranks[0] == _i_plus_one()
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert isinstance(expr.lhs, AnonymousTensor)
    inner = expr.lhs.expression
    assert isinstance(inner, BinaryApp)
    assert _tensor_of(inner.lhs) == "E"
    assert _tensor_of(inner.rhs) == "InPush"
    assert _tensor_of(expr.rhs) == "OutPush"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == BuiltinComputeOp(symbol="+")
    assert specs[2].compute_op == BuiltinComputeOp(symbol="-")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="union")
    assert specs[2].merge_op == BuiltinMergeOp(symbol="union")


def test_max_flow_E16_residual_update_mirrors_the_flow_update() -> None:
    """E16: ``R_{i+1} = (R_i - Delta)_{i,u,v} + Delta_{i,v,u}``.

    The signs are the mirror of E12: capacity spent forward is capacity
    gained backward.
    """
    einsum = _e(_load_program(), "E16")
    assert einsum.output_tensor == "R"
    assert einsum.output_ranks[0] == _i_plus_one()
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert isinstance(expr.lhs, AnonymousTensor)
    inner = expr.lhs.expression
    assert isinstance(inner, BinaryApp)
    assert _tensor_of(inner.lhs) == "R"
    u, v = RankVariable(name="u"), RankVariable(name="v")
    assert _ranks_of(inner.rhs)[1:] == [u, v]
    assert _ranks_of(expr.rhs)[1:] == [v, u]
    specs = _map_specs(einsum)
    assert specs[1].compute_op == BuiltinComputeOp(symbol="-")
    assert specs[2].compute_op == BuiltinComputeOp(symbol="+")


################################################################################
# E17-E24: the relabel step
################################################################################


def test_max_flow_E17_admissible_after_push_uses_the_new_residual() -> None:
    """E17: ``Adm_{i+1} = R_{i+1} ->(intersect) Lbl_{i}``.

    Recomputed because the push changed R -- this is the bulk-synchronous
    stand-in for a sequential read-after-write. Lbl is still generation i:
    no height has changed yet this round. take_right carries Lbl's Boolean.

    The IR writes it to ``AdmPost``, not ``Adm``. This departs from
    einsum.md. E09 writes ``Adm_{i+1}`` again in the next round, and an
    Einsum's assignment only adds values, so a shared slice carries E17's
    entries into that round's push. On the reference evaluator that let
    the source push a negative amount before the split.
    """
    einsum = _e(_load_program(), "E17")
    assert einsum.output_tensor == "AdmPost"
    assert einsum.output_ranks[0] == _i_plus_one()
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "R"
    assert _ranks_of(expr.lhs)[0] == _i_plus_one()
    assert _tensor_of(expr.rhs) == "Lbl"
    assert _ranks_of(expr.rhs)[0] == RankVariable(name="i")
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="take_right")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E18_hasadm_reduces_v_with_or() -> None:
    """E18: ``HasAdm_{i+1,u} = OR_v Adm_{i+1,u,v}``."""
    einsum = _e(_load_program(), "E18")
    assert einsum.output_tensor == "HasAdm"
    assert einsum.output_ranks[0] == _i_plus_one()
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "AdmPost"
    assert _ranks_of(expr.lhs)[0] == _i_plus_one()
    reduces = _reduce_specs(einsum)
    assert reduces[0].rank_list == ["v"]
    assert reduces[0].compute_op == UserDefinedComputeOp(name="OR")
    assert reduces[0].merge_op == BuiltinMergeOp(symbol="union")


def test_max_flow_E20_relabel_is_active_with_no_admissible_edge() -> None:
    """E20: ``Rel_{i+1,u} = Act_{i+1,u} AND not HasAdm_{i+1,u}``.

    Written at generation i+1, and reading generation i+1 operands: this is
    the post-push state, which is the only state in which "still active but
    stuck" is a meaningful question.
    """
    einsum = _e(_load_program(), "E20")
    assert einsum.output_tensor == "Rel"
    assert einsum.output_ranks[0] == _i_plus_one()
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "Act"
    assert _ranks_of(expr.lhs)[0] == _i_plus_one()
    assert isinstance(expr.rhs, UnaryApp)
    assert expr.rhs.op == BuiltinUnaryOp(symbol="not")
    assert _tensor_of(expr.rhs.operand) == "HasAdm"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="AND")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E21_neighbor_labels_for_relabel_vertices() -> None:
    """E21: ``NeiLbl = (R_{i+1} <-(int) Rel_{i+1})_{i,u,v} ->(int) D_{i,v}``.

    Gathers the heights of residual neighbours, restricted to the vertices
    that are about to relabel. Rel is read at i+1, matching where E20 wrote
    it. The final take_right carries the neighbour's height as the value.
    """
    einsum = _e(_load_program(), "E21")
    assert einsum.output_tensor == "NeiLbl"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert isinstance(expr.lhs, AnonymousTensor)
    inner = expr.lhs.expression
    assert isinstance(inner, BinaryApp)
    assert _tensor_of(inner.lhs) == "R"
    assert _ranks_of(inner.lhs)[0] == _i_plus_one()
    assert _tensor_of(inner.rhs) == "Rel"
    assert _ranks_of(inner.rhs)[0] == _i_plus_one()
    assert _tensor_of(expr.rhs) == "D"
    assert _ranks_of(expr.rhs) == [RankVariable(name="i"), RankVariable(name="v")]
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="take_left")
    assert specs[2].compute_op == UserDefinedComputeOp(name="take_right")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")
    assert specs[2].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E22_min_neighbor_label() -> None:
    """E22: ``MinNeiLbl_{i,u} = min_v NeiLbl_{i,u,v}``.

    The union merge with a min reduction is why D's empty must be +inf:
    that is min's identity.
    """
    einsum = _e(_load_program(), "E22")
    assert einsum.output_tensor == "MinNeiLbl"
    reduces = _reduce_specs(einsum)
    assert reduces[0].rank_list == ["v"]
    assert reduces[0].compute_op == UserDefinedComputeOp(name="min")
    assert reduces[0].merge_op == BuiltinMergeOp(symbol="union")


def test_max_flow_E23_new_height_is_one_above_lowest_neighbor() -> None:
    """E23: ``NewD_{i,u} = (MinNeiLbl_{i,u} + 1)_{i,u} :: AND +(intersect)``.

    Intersect, so NewD exists only for vertices that actually had a
    residual neighbour to measure against -- which is exactly the set E24
    then merges in.
    """
    einsum = _e(_load_program(), "E23")
    assert einsum.output_tensor == "NewD"
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "MinNeiLbl"
    assert _tensor_of(expr.rhs) == "One"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == BuiltinComputeOp(symbol="+")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="intersect")


def test_max_flow_E24_height_update_is_a_partial_update_over_union() -> None:
    """E24: ``D_{i+1,u} = D_{i,u} <<(union) NewD_{i,u}``.

    The partial-update idiom. D_i is the LEFT operand and NewD the right;
    the union merge keeps every vertex present, and ``update`` takes the
    right value where it exists and the left one otherwise. So a vertex
    that did not relabel keeps its height.

    Encoding this as ``take_left(intersect)`` -- value from NewD, present
    only where both are -- would retain ONLY the relabelled vertices and
    reset every other height to D's empty value. That is a silent
    correctness bug, not a stylistic difference, which is why the operand
    order and both operators are pinned here.
    """
    einsum = _e(_load_program(), "E24")
    assert einsum.output_tensor == "D"
    assert einsum.output_ranks == [_i_plus_one(), RankVariable(name="u")]
    expr = einsum.expression
    assert isinstance(expr, BinaryApp)
    assert _tensor_of(expr.lhs) == "D"
    assert _ranks_of(expr.lhs)[0] == RankVariable(name="i")
    assert _tensor_of(expr.rhs) == "NewD"
    specs = _map_specs(einsum)
    assert specs[1].compute_op == UserDefinedComputeOp(name="update")
    assert specs[1].merge_op == BuiltinMergeOp(symbol="union")


def test_max_flow_E24_is_not_encoded_as_take_left_intersect() -> None:
    """Guard against the regression E24's docstring describes.

    Kept separate so the failure message says which mistake was made.
    """
    specs = _map_specs(_e(_load_program(), "E24"))
    assert specs[1].merge_op != BuiltinMergeOp(symbol="intersect")
    assert specs[1].compute_op != UserDefinedComputeOp(name="take_left")


################################################################################
# Generational discipline
################################################################################


def test_max_flow_stateful_tensors_advance_exactly_one_generation() -> None:
    """Every iterative write lands on generation i+1, never i+2 or beyond.

    A cascade is one round; anything writing further ahead would be reading
    a generation that does not exist yet.
    """
    program = _load_program()
    iterative = {"E12": "F", "E15": "E", "E16": "R", "E19": "Act", "E24": "D"}
    for spec_id, tensor in iterative.items():
        einsum = _e(program, spec_id)
        assert einsum.output_tensor == tensor
        assert einsum.output_ranks[0] == _i_plus_one()


def test_max_flow_act_is_computed_twice_per_round() -> None:
    """Act appears at both i and i+1, and that duplication is meaningful.

    Sequential code re-reads ``e(v)`` for free after a push. Bulk-synchronous
    execution has no such ordering, so the post-push active set must be
    recomputed explicitly. E06 is the pre-push set that drives the push;
    E19 is the post-push set that drives relabel and termination.
    """
    program = _load_program()
    act_writes = _einsum_by_output(_cascade_einsums(program), "Act")
    assert len(act_writes) == 2
    generations = [e.output_ranks[0] for e in act_writes]
    assert RankVariable(name="i") in generations
    assert _i_plus_one() in generations


def test_max_flow_admissibility_is_computed_twice_per_round() -> None:
    """Admissibility likewise: from R_i before the push, from R_{i+1} after it.

    The two results land in different tensors. E09 writes ``Adm_i`` and E17
    writes ``AdmPost_{i+1}``. If both wrote Adm, the next round's E09 would
    find E17's entries already in ``Adm_{i+1}``, because an assignment never
    clears a cell. See the E17 test.
    """
    einsums = _cascade_einsums(_load_program())
    adm_writes = _einsum_by_output(einsums, "Adm")
    post_writes = _einsum_by_output(einsums, "AdmPost")
    assert [e.output_ranks[0] for e in adm_writes] == [RankVariable(name="i")]
    assert [e.output_ranks[0] for e in post_writes] == [_i_plus_one()]


################################################################################
# Stopping condition
################################################################################


def test_max_flow_stops_when_no_vertex_is_active() -> None:
    """``<> : |Act_{i+1}| == 0``.

    Occupancy -- the count of present coordinates -- not "all values are
    zero". Act's empty value is False, so an inactive vertex is absent
    rather than present-and-false, and the two readings would agree here;
    but occupancy is what the spec says and what generalises.

    The condition tests the POST-push active set, so a round that empties
    the last vertex terminates immediately rather than running once more.
    """
    program = _load_program()
    conditions = program.main_edge.cascade.stopping_conditions
    assert len(conditions) == 1
    condition = conditions[0]
    assert condition.rank_variable == "i"
    predicate = condition.predicate
    assert isinstance(predicate, Comparison)
    assert predicate.op == "=="
    assert isinstance(predicate.lhs, PropertyApp)
    assert predicate.lhs.name == "occupancy"
    assert predicate.lhs.operands[0].tensor == "Act"
    assert predicate.lhs.operands[0].ranks[0] == _i_plus_one()
    assert isinstance(predicate.rhs, TensorProjectionValue)
