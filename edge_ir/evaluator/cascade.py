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
from edge_ir.evaluator.stopping import should_stop
from edge_ir.ir.einsum import Einsum, NestedCascade

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
    for einsum in cascade.einsums:
        for subscript in einsum.output_ranks:
            image = coord_image(subscript)
            if image.is_shift and image.offset != 0 and len(image.axes) == 1:
                advanced |= set(image.axes)
    if not advanced:
        return None
    if len(advanced) > 1:
        raise UnderSpecifiedProgramError(
            f"the cascade advances more than one rank variable "
            f"({sorted(advanced)}), i.e. it is a nested cascade, and "
            f"NestedCascade is flat -- it cannot say which Einsums iterate "
            f"under which rank. Name the intended one in "
            f"NestedCascade.iterative_rank, or extend the IR."
        )
    return advanced.pop()


def run_cascade(
    cascade: NestedCascade,
    *,
    env: Environment,
    schedule: Schedule | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    generations: int | None = None,
) -> CascadeTrace:
    """Run a cascade to its stopping condition (or for ``generations``)."""
    schedule = schedule or SequentialSchedule()
    trace = CascadeTrace()
    rank = detect_iterative_rank(cascade)
    trace.iterative_rank = rank

    if rank is None:
        trace.per_generation.append(_run_body(cascade.einsums, env, schedule, {}))
        trace.generations = 1
        trace.stopped_by = "not-iterative"
        return trace

    start = (
        cascade.iterative_rank.initial_value
        if cascade.iterative_rank is not None
        else 0
    )
    trace.initial_generation = start

    if not cascade.stopping_conditions and generations is None:
        raise UnderSpecifiedProgramError(
            f"the cascade advances the iterative rank {rank!r} but declares no "
            f"stopping condition, so nothing says when it ends. Add a "
            f"stopping condition to the program, or pass generations=<n> to "
            f"run a fixed number. The evaluator will not spin to the "
            f"iteration cap and call the result an answer."
        )

    limit = generations if generations is not None else max_iterations
    for step in range(limit):
        generation = start + step
        trace.per_generation.append(
            _run_body(cascade.einsums, env, schedule, {rank: generation})
        )
        trace.generations = step + 1
        if any(
            should_stop(sc, generation, env=env) for sc in cascade.stopping_conditions
        ):
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
    einsums: list[Einsum],
    env: Environment,
    schedule: Schedule,
    pinned: dict[str, int],
) -> list[EinsumResult]:
    return [
        evaluate_einsum(e, env=env, pinned=dict(pinned), schedule=schedule)
        for e in einsums
    ]
