"""Driving a cascade of Einsums.

A cascade runs its Einsums in order, once per generation of its iterative
rank, checking the stopping condition after each generation and advancing
the rank until it fires.

**Which rank is the iterative one.** In order of authority:

1. ``NestedCascade.iterative_rank`` names it outright, with an initial value
   (EDGE syntax for iterative cascades).
2. Otherwise a ``StoppingCondition`` names it, and the initial value is the
   default 0.
3. Otherwise it is derived structurally: a rank variable that some Einsum
   writes at a non-zero integer shift of itself -- `F_{i+1,d} = ... F_{i,s}`
   advances `i`. That is what makes a cascade iterative in the first place.

**When the cascade is iterative but says nothing about stopping**, the
evaluator refuses to run rather than spinning to the iteration cap: an
unbounded loop with no halt condition is exactly the under-specification
this evaluator is supposed to surface. Pass ``generations=`` to say how many
to run anyway.

**Timing of the diamond.** The predicate is evaluated with the iterative
rank bound to the generation just completed, and a true result stops there;
see ``edge_ir/evaluator/stopping.py``.

LOCAL PATCHES (edge-workbench issues 9 and 10), for review:

**The body is a tree.** An entry of ``NestedCascade.einsums`` is an Einsum, a
``ConditionalCascade`` or an inner ``NestedCascade``. A conditional cascade
evaluates its arms' conditions in order, with the enclosing iterative ranks
bound to the current generation, and runs the Einsums of the first arm whose
condition holds. An inner cascade runs to its own stopping condition once per
generation of the cascade around it, with the outer ranks still bound.

**The rank variable set is bounded by the rank.** The paper's default is
``RV_i = { i | i in [0, I) and not diamond(i) }``. When the tensor rank the
iterative variable indexes has a declared shape, a generation outside it does
not run and the cascade ends ("rank-exhausted"). An unbounded generational
rank, which is every cascade before this patch, behaves as before.

**A diamond that looks ahead of the body.** A body that writes generation
``i`` (not ``i+1``) while its diamond reads ``i+1`` -- bubble sort's
``Z_{i+1,J-1} == Z_{i,J-1}`` over a body writing ``Z_{i,j,n}`` -- cannot be
decided for the generation just completed: ``Z_{i+1}`` is not written yet.
After generation ``g`` such a diamond is evaluated at ``i = g - (r - w)``,
where ``r`` is the largest shift of ``i`` in the diamond's tensor operands
and ``w`` the largest shift the body writes; that is the latest ``i`` whose
operands exist. When ``r <= w`` (every cascade before this patch) it is
evaluated at ``g`` as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_ir.analysis.iteration_space import coord_image
from edge_ir.evaluator.einsum import EinsumResult, evaluate_einsum
from edge_ir.evaluator.environment import Environment
from edge_ir.evaluator.errors import (
    IterationLimitExceeded,
    UnderSpecifiedProgramError,
)
from edge_ir.evaluator.order import Schedule, SequentialSchedule
from edge_ir.evaluator.spaces import Coordinate
from edge_ir.evaluator.stopping import should_stop
from edge_ir.ir.einsum import (
    ConditionalCascade,
    Einsum,
    NestedCascade,
    flatten,
)
from edge_ir.ir.stopping import StoppingCondition

DEFAULT_MAX_ITERATIONS = 1_000_000


@dataclass
class CascadeTrace:
    """A record of what the cascade did, for debugging and for tests."""

    generations: int = 0
    stopped_by: str = "not-iterative"
    iterative_rank: str | None = None
    initial_generation: int = 0
    per_generation: list[list[EinsumResult]] = field(default_factory=list)


def detect_iterative_rank(cascade: NestedCascade) -> str | None:
    """The rank variable this cascade advances, or None if it does not."""
    if cascade.iterative_rank is not None:
        return cascade.iterative_rank.rank_variable
    if cascade.stopping_conditions:
        names = {sc.rank_variable for sc in cascade.stopping_conditions}
        if len(names) > 1:
            raise UnderSpecifiedProgramError(
                f"the cascade carries stopping conditions for {sorted(names)}, "
                f"i.e. more than one iterative rank. That is a nested cascade, "
                f"and NestedCascade is flat -- it cannot say which Einsums "
                f"iterate under which rank. Split it, or extend the IR."
            )
        return names.pop()
    return _structurally_advanced_rank(cascade)


def _structurally_advanced_rank(cascade: NestedCascade) -> str | None:
    """A rank variable written at a non-zero integer shift of itself."""
    advanced: set[str] = set()
    for einsum in _own_einsums(cascade.einsums):
        for subscript in einsum.output_ranks:
            image = coord_image(subscript)
            if image.is_shift and image.offset != 0 and len(image.axes) == 1:
                advanced |= set(image.axes)
    if not advanced:
        return None
    if len(advanced) > 1:
        raise UnderSpecifiedProgramError(
            f"the cascade advances more than one rank variable "
            f"({sorted(advanced)}) and does not say which one iterates it. "
            f"Name the intended one in NestedCascade.iterative_rank, and put "
            f"the Einsums that iterate under another rank in an inner "
            f"NestedCascade."
        )
    return advanced.pop()


def _own_einsums(entries: list) -> list[Einsum]:
    """The Einsums at this cascade's own level: conditional arms are part of
    it, inner cascades (which advance their own rank) are not."""
    out: list[Einsum] = []
    for entry in entries:
        if isinstance(entry, Einsum):
            out.append(entry)
        elif isinstance(entry, ConditionalCascade):
            for arm in entry.arms:
                out.extend(_own_einsums(arm.einsums))
    return out


def _rank_bound(
    cascade: NestedCascade, rank: str, env: Environment
) -> frozenset[Coordinate] | None:
    """The coordinates the iterative variable may take, or None if unbounded.

    The paper's default rank variable set is ``[0, I)``: the coordinate set
    of the generational rank. It is read off the tensor ranks the variable
    subscripts, alone and as an integer shift, in this cascade's outputs.
    """
    allowed: set[Coordinate] = set()
    bounded = False
    for einsum in flatten(cascade):
        space = env.spaces.get(einsum.output_tensor)
        if space is None:
            continue
        for position, subscript in enumerate(einsum.output_ranks):
            if position >= len(space.ranks):
                continue
            image = coord_image(subscript)
            if image.axes != frozenset({rank}) or not image.is_shift:
                continue
            rcs = space.ranks[position]
            if not rcs.bounded or rcs.coords is None:
                return None
            bounded = True
            allowed.update(
                c - image.offset
                for c in rcs.coords
                if isinstance(c, int) and not isinstance(c, bool)
            )
    return frozenset(allowed) if bounded else None


def _lookahead(cascade: NestedCascade, condition: StoppingCondition) -> int:
    """How many generations a diamond reads ahead of what the body writes."""
    rank = condition.rank_variable

    def shifts(subscripts) -> list[int]:
        found = []
        for subscript in subscripts:
            image = coord_image(subscript)
            if image.axes == frozenset({rank}) and image.is_shift:
                found.append(image.offset)
        return found

    read: list[int] = []
    predicate = condition.predicate
    for side in ("lhs", "rhs"):
        node = getattr(predicate, side, None)
        proj = getattr(node, "proj", None)
        if proj is not None:
            read += shifts(proj.ranks)
        for operand in getattr(node, "operands", None) or []:
            read += shifts(operand.ranks)
    for operand in getattr(predicate, "operands", None) or []:
        read += shifts(operand.ranks)
    written: list[int] = []
    for einsum in flatten(cascade):
        written += shifts(einsum.output_ranks)
    if not read or not written:
        return 0
    return max(0, max(read) - max(written))


def run_cascade(
    cascade: NestedCascade,
    *,
    env: Environment,
    schedule: Schedule | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    generations: int | None = None,
    pinned: dict[str, Coordinate] | None = None,
) -> CascadeTrace:
    """Run a cascade to its stopping condition (or for ``generations``).

    ``pinned`` carries the iterative ranks of the cascades around this one,
    when it is an inner cascade.
    """
    schedule = schedule or SequentialSchedule()
    outer = dict(pinned or {})
    trace = CascadeTrace()
    rank = detect_iterative_rank(cascade)
    trace.iterative_rank = rank

    def body(bound: dict[str, Coordinate]) -> list[EinsumResult]:
        return _run_body(cascade.einsums, env, schedule, bound, max_iterations)

    if rank is None:
        trace.per_generation.append(body(outer))
        trace.generations = 1
        trace.stopped_by = "not-iterative"
        return trace

    start = (
        cascade.iterative_rank.initial_value
        if cascade.iterative_rank is not None
        else 0
    )
    trace.initial_generation = start
    allowed = _rank_bound(cascade, rank, env)

    if not cascade.stopping_conditions and generations is None and allowed is None:
        raise UnderSpecifiedProgramError(
            f"the cascade advances the iterative rank {rank!r} but declares no "
            f"stopping condition, so nothing says when it ends. Add a "
            f"stopping condition to the program, or pass generations=<n> to "
            f"run a fixed number. The evaluator will not spin to the "
            f"iteration cap and call the result an answer."
        )

    ahead = {id(sc): _lookahead(cascade, sc) for sc in cascade.stopping_conditions}
    limit = generations if generations is not None else max_iterations
    for step in range(limit):
        generation = start + step
        if allowed is not None and generation not in allowed:
            trace.stopped_by = "rank-exhausted"
            return trace
        trace.per_generation.append(body({**outer, rank: generation}))
        trace.generations = step + 1
        for sc in cascade.stopping_conditions:
            at = generation - ahead[id(sc)]
            if at < start:
                continue
            if should_stop(sc, at, env=env, outer=outer):
                trace.stopped_by = "stopping-condition"
                return trace

    if generations is not None:
        trace.stopped_by = "generation-count"
        return trace

    raise IterationLimitExceeded(
        f"the cascade ran {max_iterations} generations of {rank!r} without its "
        f"stopping condition firing. Either the condition is wrong or the "
        f"program does not converge; raise max_iterations only when you know "
        f"which."
    )


def _run_body(
    entries: list,
    env: Environment,
    schedule: Schedule,
    pinned: dict[str, Coordinate],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> list[EinsumResult]:
    """Run a cascade body's entries in order; the Einsum results, flattened."""
    from edge_ir.evaluator.stopping import _comparison_empties, evaluate_predicate

    results: list[EinsumResult] = []
    for entry in entries:
        if isinstance(entry, Einsum):
            results.append(
                evaluate_einsum(entry, env=env, pinned=dict(pinned), schedule=schedule)
            )
        elif isinstance(entry, ConditionalCascade):
            # The conditions are read once, before any arm runs: an arm that
            # writes what a condition reads does not change which arm this is.
            for arm in entry.arms:
                holds = True
                if arm.when is not None:
                    empties = _comparison_empties(arm.when, env)
                    holds = evaluate_predicate(
                        arm.when,
                        dict(pinned),
                        env=env,
                        empty_lhs=empties[0],
                        empty_rhs=empties[1],
                    )
                if holds:
                    results += _run_body(
                        arm.einsums, env, schedule, pinned, max_iterations
                    )
                    break
        else:
            inner = run_cascade(
                entry,
                env=env,
                schedule=schedule,
                max_iterations=max_iterations,
                pinned=dict(pinned),
            )
            for generation in inner.per_generation:
                results += generation
    return results
