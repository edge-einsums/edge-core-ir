"""Single assignment: no two Einsums may write the same tensor slice.

A cascade is single assignment. Two Einsums may not define the same slice of
the same tensor, and a slice the initialization section writes may not be
written again by the cascade.

An iterative rank makes the comparison less obvious than it looks. An Einsum
that writes ``X_{i+1}`` in round ``i`` writes the slice that an Einsum writing
``X_i`` writes in round ``i+1``, so those two Einsums conflict even though
their output ranks differ. A constant generation conflicts the same way,
because the iterative rank passes through that value.

Two writes to one tensor are reported unless some rank position proves they can
never denote the same coordinate. Only two distinct constants prove that. This
pass does not read predicates, so two writes that a predicate keeps apart are
still reported: the IR cannot see that they are disjoint.

The pass reports and does not raise, so a program that violates the rule still
loads and still runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from edge_ir.ir.einsum import Einsum
from edge_ir.ir.expr import (
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankExpression,
    RankVariable,
)
from edge_ir.ir.program import Program

_CONST = "const"
_VAR = "var"
_OPAQUE = "opaque"


@dataclass(frozen=True)
class _RankKey:
    """One output rank position, as a constant, a variable plus an offset,
    or something this pass does not read."""

    kind: str
    name: str
    offset: int

    def text(self) -> str:
        if self.kind == _CONST:
            return self.name
        if self.kind == _VAR:
            if self.offset > 0:
                return f"{self.name}+{self.offset}"
            if self.offset < 0:
                return f"{self.name}{self.offset}"
            return self.name
        return self.name


def _key(rank: RankExpression) -> _RankKey:
    if isinstance(rank, RankConstantLiteral):
        return _RankKey(_CONST, str(rank.value), 0)
    if isinstance(rank, RankConstantShapeSym):
        return _RankKey(_CONST, rank.name, 0)
    if isinstance(rank, RankVariable):
        return _RankKey(_VAR, rank.name, 0)
    if isinstance(rank, RankArith):
        base = _key(rank.lhs)
        offset = rank.rhs
        if (
            base.kind == _VAR
            and rank.op in ("+", "-")
            and isinstance(offset, RankConstantLiteral)
            and isinstance(offset.value, int)
        ):
            step = offset.value if rank.op == "+" else -offset.value
            return _RankKey(_VAR, base.name, base.offset + step)
        return _RankKey(_OPAQUE, "<arith>", 0)
    return _RankKey(_OPAQUE, "<func>", 0)


@dataclass(frozen=True)
class _Iteration:
    """The cascade's iterative rank and where it starts.

    Both have defaults: the rank variable is ``i`` and the initial value is 0,
    so a cascade that takes the defaults declares no ``IterativeRankSpec``.
    """

    variable: str
    start: int


def _iteration_of(program: Program) -> _Iteration:
    spec = program.main_edge.cascade.iterative_rank
    if spec is None:
        return _Iteration("i", 0)
    return _Iteration(spec.rank_variable, spec.initial_value)


def _can_meet(left: _RankKey, right: _RankKey, iteration: _Iteration) -> bool:
    """Whether one coordinate can satisfy both positions.

    A constant generation and the advancing rank meet only when the rank
    reaches that constant: ``X[0]`` and ``X[i+1]`` never collide when ``i``
    starts at 0, while ``X[1]`` and ``X[i+1]`` collide in the first round.
    A constant in a rank that does not advance is not a proof of anything,
    because the variable there ranges over every coordinate of the rank.
    """
    if left.kind == _CONST and right.kind == _CONST:
        return left.name == right.name
    for const, var in ((left, right), (right, left)):
        if const.kind == _CONST and var.kind == _VAR and var.name == iteration.variable:
            try:
                value = int(const.name)
            except ValueError:
                return True
            return value >= iteration.start + var.offset
    return True


@dataclass(frozen=True)
class WriteSite:
    """One Einsum's output slice, and where the Einsum sits."""

    path: str
    tensor: str
    ranks: tuple[_RankKey, ...]

    def slice_text(self) -> str:
        return f"{self.tensor}[{', '.join(k.text() for k in self.ranks)}]"


@dataclass(frozen=True)
class SingleAssignmentViolation:
    """Two Einsums that can write the same slice of one tensor."""

    tensor: str
    first: WriteSite
    second: WriteSite
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.first.path} writes {self.first.slice_text()} and "
            f"{self.second.path} writes {self.second.slice_text()}: {self.detail}"
        )


def _detail(first: WriteSite, second: WriteSite) -> str:
    for left, right in zip(first.ranks, second.ranks, strict=True):
        if left == right:
            continue
        if left.kind == _VAR and right.kind == _VAR and left.name == right.name:
            return (
                f"the rank {left.name} advances, so round {left.name} writes one "
                f"and round {left.name}+{abs(left.offset - right.offset)} the other"
            )
        if _CONST in (left.kind, right.kind):
            return "the advancing rank passes through the constant generation"
        return "the two positions can denote the same coordinate"
    return "the same slice"


def write_sites(program: Program) -> list[WriteSite]:
    """Every Einsum that writes, in initialization order then cascade order."""
    sites: list[WriteSite] = []

    def add(path: str, einsum: Einsum) -> None:
        sites.append(
            WriteSite(
                path=path,
                tensor=einsum.output_tensor,
                ranks=tuple(_key(r) for r in einsum.output_ranks),
            )
        )

    for index, einsum in enumerate(program.initialization.einsums):
        add(f"initialization[{index}]", einsum)
    for index, einsum in enumerate(program.main_edge.cascade.einsums):
        add(f"cascade[{index}]", einsum)
    return sites


def check_single_assignment(program: Program) -> list[SingleAssignmentViolation]:
    """Report every pair of Einsums that can write the same tensor slice.

    An empty list means the program is single assignment.
    """
    sites = write_sites(program)
    iteration = _iteration_of(program)
    violations: list[SingleAssignmentViolation] = []
    for i, first in enumerate(sites):
        for second in sites[i + 1 :]:
            if first.tensor != second.tensor:
                continue
            if len(first.ranks) != len(second.ranks):
                continue
            pairs = zip(first.ranks, second.ranks, strict=True)
            if all(_can_meet(a, b, iteration) for a, b in pairs):
                violations.append(
                    SingleAssignmentViolation(
                        tensor=first.tensor,
                        first=first,
                        second=second,
                        detail=_detail(first, second),
                    )
                )
    return violations
