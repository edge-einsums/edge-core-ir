"""Every operator a program names must be one something implements.

THE GAP THIS CLOSES
-------------------
The IR carries operator names as bare strings. `UserDefinedComputeOp.name` is
constrained by `REGEX_USER_DEFINED_NAME` and nothing else, which is right for
Layer 1: the IR should not know what is implemented. But nothing checked it
afterwards either, so a program naming an operator no registry provides passed
Layer 1 AND Layer 2, and failed for the first time inside the evaluator, at a
coordinate, partway through a cascade.

Two real mistakes came from that, a week apart, both a name that looked right:

  * `ge` / `le` / `ne` were believed missing from the builtins. They are not --
    all six comparisons ship under two spellings each. The gap was in the
    consumer, and nothing said which side was wrong.
  * `ANY` is defined as `def any_of(...)` and registered under the name `ANY`.
    A consumer keyed on the function name resolved `AND` and `OR`, then failed
    on `ANY` -- one operator in three, deep into a run.

Neither is subtle once the question is asked. Nothing was asking it.

WHY THIS TAKES A REGISTRY ARGUMENT
----------------------------------
Layer 2 is otherwise registry-independent, and that is worth keeping: a program
may legitimately be validated in one process and evaluated in another with a
different set of user functions registered. So this pass is opt-in -- pass the
registry you intend to evaluate with, and it reports what that registry cannot
provide. Validating without one stays possible, and means what it always did.

A name absent from the registry is therefore reported, not pronounced illegal.
A user-defined operator the author intends to supply at runtime is not a
mistake; it is only a mistake relative to a particular registry.

    from edge_ir.udf.builtins import default_registry
    findings = check_udf_names(program, default_registry())
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from edge_ir.ir.actions import MapSpec, PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum, flatten
from edge_ir.ir.expr import (
    AnonymousTensor,
    BinaryApp,
    RankFunction,
    UnaryApp,
)
from edge_ir.ir.op import (
    UserDefinedComputeOp,
    UserDefinedMergeOp,
    UserDefinedUnaryOp,
)
from edge_ir.ir.program import Program
from edge_ir.udf.categories import UdfCategory

#: Reported when the IR admits a name for which no category exists at all.
NO_CATEGORY = "no-category"


@dataclass(frozen=True)
class UdfUse:
    """One place a program names an operator."""

    category: UdfCategory | None
    name: str
    where: str

    def text(self) -> str:
        cat = self.category.value if self.category else "user-defined merge"
        return f"{self.where}: {cat} operator {self.name!r}"


@dataclass(frozen=True)
class UdfNameFinding:
    """A name the given registry cannot provide."""

    use: UdfUse
    reason: str
    suggestions: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.use.name

    @property
    def category(self) -> UdfCategory | None:
        return self.use.category

    def __str__(self) -> str:
        out = f"{self.use.text()} -- {self.reason}"
        if self.suggestions:
            out += f". Did you mean {' or '.join(map(repr, self.suggestions))}?"
        return out


# --------------------------------------------------------------- walking ---

def _spec_uses(spec, where: str) -> list[UdfUse]:
    uses: list[UdfUse] = []
    kind = {
        MapSpec: UdfCategory.MAP_COMPUTE,
        ReduceSpec: UdfCategory.REDUCE_COMPUTE,
        PopulateSpec: UdfCategory.POPULATE_COMPUTE,
    }.get(type(spec))

    compute = getattr(spec, "compute_op", None)
    if kind and isinstance(compute, UserDefinedComputeOp):
        uses.append(UdfUse(kind, compute.name, where))

    merge = getattr(spec, "merge_op", None)
    if isinstance(merge, UserDefinedMergeOp):
        # There is no MERGE category: merge operators are the 16 builtin truth
        # tables, and the evaluator refuses a user-defined one outright. Worth
        # reporting here rather than at the first point evaluated.
        uses.append(UdfUse(None, merge.name, where))

    coord = getattr(spec, "coord_op", None)
    if coord is not None:
        uses.append(UdfUse(UdfCategory.COORDINATE, coord.name, where))
    return uses


def _expr_uses(node, where: str) -> list[UdfUse]:
    uses: list[UdfUse] = []
    if isinstance(node, UnaryApp):
        if isinstance(node.op, UserDefinedUnaryOp):
            uses.append(UdfUse(UdfCategory.UNARY, node.op.name, where))
        uses += _expr_uses(node.operand, where)
    elif isinstance(node, BinaryApp):
        uses += _expr_uses(node.lhs, where)
        uses += _expr_uses(node.rhs, where)
    elif getattr(node, "kind", None) == "occupancy":
        for rank in node.operand.ranks:
            uses += _rank_uses(rank, where)
    elif isinstance(node, AnonymousTensor):
        uses += _expr_uses(node.expression, where)
        for rank in getattr(node, "ranks", []) or []:
            uses += _rank_uses(rank, where)

    proj = getattr(node, "proj", None)
    if proj is not None:
        for rank in getattr(proj, "ranks", []) or []:
            uses += _rank_uses(rank, where)
    return uses


def _rank_uses(rank, where: str) -> list[UdfUse]:
    """Rank expressions can name a rank-mapping function."""
    uses: list[UdfUse] = []
    if isinstance(rank, RankFunction):
        uses.append(UdfUse(UdfCategory.RANK_MAPPING, rank.func_name, where))
        for arg in getattr(rank, "args", []) or []:
            uses += _rank_uses(arg, where)
    for attr in ("lhs", "rhs", "rank", "operand"):
        inner = getattr(rank, attr, None)
        if inner is not None and not isinstance(inner, (int, str)):
            uses += _rank_uses(inner, where)
    return uses


def _einsum_uses(einsum: Einsum, where: str) -> list[UdfUse]:
    uses: list[UdfUse] = []
    for rank in einsum.output_ranks:
        uses += _rank_uses(rank, where)
    if getattr(einsum.expression, "kind", None) == "cases":
        # A case statement: each arm has its own expression and specs.
        for k, arm in enumerate(einsum.expression.arms):
            if arm.expression is not None:
                uses += _expr_uses(arm.expression, f"{where} arm {k}")
            for spec in arm.specs:
                uses += _spec_uses(spec, f"{where} arm {k}")
    else:
        uses += _expr_uses(einsum.expression, where)
    for spec in einsum.specs:
        uses += _spec_uses(spec, where)
    return uses


def udf_uses(program: Program) -> list[UdfUse]:
    """Every operator name the program uses, with where it uses it.

    Initialization order then cascade order, matching
    :func:`edge_ir.validator.single_assignment.write_sites`, so the two passes
    report locations the same way.
    """
    uses: list[UdfUse] = []
    for i, einsum in enumerate(program.initialization.einsums):
        uses += _einsum_uses(einsum, f"initialization[{i}]")
    for i, einsum in enumerate(flatten(program.main_edge.cascade)):
        uses += _einsum_uses(einsum, f"cascade[{i}]")
    return uses


# --------------------------------------------------------------- the pass ---

def check_udf_names(program: Program, registry) -> list[UdfNameFinding]:
    """Report every operator name ``registry`` cannot provide.

    An empty list means every name the program uses resolves.

    ``PropertyApp.name`` is not checked: it is a closed ``Literal`` in the IR,
    so Layer 1 has already rejected anything unknown. Builtin compute, merge
    and unary operators are not checked either, for the same reason -- their
    symbols are ``Literal`` types.
    """
    findings: list[UdfNameFinding] = []
    for use in udf_uses(program):
        if use.category is None:
            findings.append(UdfNameFinding(
                use,
                "merge operators are built-in only -- the 16 truth tables in "
                "edge_ir/runtime/op_properties.py -- so no registry can "
                "provide this",
            ))
            continue
        if registry.has(use.category, use.name):
            continue
        findings.append(UdfNameFinding(
            use,
            f"not registered under {use.category.value}",
            _suggest(use.name, registry.names(use.category)),
        ))
    return findings


def _suggest(name: str, known: list[str]) -> tuple[str, ...]:
    """Nearby registered names, matched case-insensitively.

    Case matters here more than it looks. `ANY` is registered in upper case
    while its Python function is `any_of`, and a case-sensitive comparison
    scores those far enough apart to suggest nothing -- which is exactly the
    mistake that went unreported. Folding case first brings `any_of` within
    reach of `ANY`, so the report names the spelling the registry actually
    uses.
    """
    folded = {k.lower(): k for k in known}
    hits = difflib.get_close_matches(name.lower(), list(folded), n=3, cutoff=0.6)
    out = [folded[h] for h in hits]
    # An exact match up to case is the likeliest intent, so lead with it.
    exact = folded.get(name.lower())
    if exact and exact in out:
        out.remove(exact)
    if exact:
        out.insert(0, exact)
    return tuple(out)
