"""The top-level entry point: run a whole EDGE program.

``evaluate(program, inputs=..., shape_env=..., ...)`` resolves the
declarations, runs the initialization Einsums once, drives the main cascade,
and hands back the tensors.

The determinism contract lives here too. For a given
``(Program, FunctionRegistry, TypeRegistry, inputs)`` the output is
deterministic; :func:`check_order_independence` is the sharp test of it,
running the same program under several seeded traversal orders and requiring
the outputs to agree. Per the EDGE semantics, an Einsum's result must not depend on
the order its iteration-space points are visited in, so a disagreement means
either this evaluator is relying on a traversal order or the program is not
order-independent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from edge_ir.evaluator.cascade import (
    DEFAULT_MAX_ITERATIONS,
    CascadeTrace,
    run_cascade,
)
from edge_ir.evaluator.datatype_registry import TypeRegistry
from edge_ir.evaluator.einsum import EinsumResult, evaluate_einsum
from edge_ir.evaluator.environment import Environment, build_environment
from edge_ir.evaluator.order import Schedule, SequentialSchedule, ShuffledSchedule
from edge_ir.evaluator.spaces import Coordinate, CoordSetEnv, Point, ShapeEnv
from edge_ir.evaluator.tensors import Tensor
from edge_ir.ir.program import Program
from edge_ir.udf.registry import FunctionRegistry


@dataclass
class EvaluationResult:
    """The outcome of running a program."""

    tensors: dict[str, Tensor]
    trace: CascadeTrace
    initialization: list[EinsumResult] = field(default_factory=list)

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name]

    def snapshot(self) -> dict[str, list[tuple[Point, Any]]]:
        """A comparable, order-independent view of every tensor's graph."""
        return {name: t.items() for name, t in sorted(self.tensors.items())}


def evaluate(
    program: Program,
    *,
    inputs: Mapping[str, Mapping[tuple[Coordinate, ...], Any]] | None = None,
    shape_env: ShapeEnv | None = None,
    coord_sets: CoordSetEnv | None = None,
    functions: FunctionRegistry | None = None,
    types: TypeRegistry | None = None,
    schedule: Schedule | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    generations: int | None = None,
) -> EvaluationResult:
    """Execute an EDGE program and return its tensors.

    ``generations`` overrides the stopping condition with a fixed count; it
    is also the only way to run a cascade that advances an iterative rank
    without declaring when to stop.
    """
    env: Environment = build_environment(
        program,
        inputs=inputs,
        shape_env=shape_env,
        coord_sets=coord_sets,
        functions=functions,
        types=types,
    )
    schedule = schedule or SequentialSchedule()

    init_results = [
        evaluate_einsum(e, env=env, pinned={}, schedule=schedule)
        for e in program.initialization.einsums
    ]
    trace = run_cascade(
        program.main_edge.cascade,
        env=env,
        schedule=schedule,
        max_iterations=max_iterations,
        generations=generations,
    )
    return EvaluationResult(
        tensors=env.tensors, trace=trace, initialization=init_results
    )


def check_order_independence(
    program: Program,
    *,
    seeds: Sequence[int] = (1, 2, 3, 5, 8),
    **kwargs: Any,
) -> tuple[bool, list[str]]:
    """Run under several seeded traversal orders; report any disagreement.

    Returns ``(agreed, differences)``. Each difference names the seed and the
    tensor whose graph diverged from the in-order run.
    """
    kwargs.pop("schedule", None)
    baseline = evaluate(program, schedule=SequentialSchedule(), **kwargs).snapshot()
    differences: list[str] = []
    for seed in seeds:
        other = evaluate(program, schedule=ShuffledSchedule(seed=seed), **kwargs)
        snapshot = other.snapshot()
        for name in sorted(set(baseline) | set(snapshot)):
            if baseline.get(name) != snapshot.get(name):
                differences.append(
                    f"seed {seed}: tensor {name!r} differs from the in-order run"
                )
    return (not differences, differences)
