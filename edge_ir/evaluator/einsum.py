"""Executing one Einsum end to end.

This is the glue module: it builds the iteration space, applies the Einsum
projection at each point, and then runs Map, Reduce and Populate in that
order. It contains no semantics of its own -- each action's meaning lives in
its own module, mirroring the EDGE semantics' division into Map,
Reduce and Populate.

The three phases are separated by barriers, for the reason
``edge_ir/evaluator/order.py`` sets out: Reduce queries a *completed* MapTmp,
and Populate folds over a *completed* RedTmp. Within a phase the order is
handed out by the schedule, which is the seam the randomized
iteration-order mode plugs into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edge_ir.evaluator.actions.map import map_action
from edge_ir.evaluator.actions.populate import populate_action
from edge_ir.evaluator.actions.reduce import ReductionTemporary, reduce_action
from edge_ir.evaluator.environment import Environment
from edge_ir.evaluator.errors import UnderSpecifiedProgramError
from edge_ir.evaluator.iteration import ConcreteIterationSpace, build_iteration_space
from edge_ir.evaluator.order import Phase, Schedule, SequentialSchedule
from edge_ir.evaluator.projection import einsum_projection
from edge_ir.evaluator.spaces import Coordinate, Point
from edge_ir.ir.actions import PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import Einsum
from edge_ir.ir.expr import RankVariable


@dataclass
class EinsumResult:
    """What one Einsum produced, kept for debugging and for tests."""

    iteration_space: ConcreteIterationSpace
    defined_points: list[Point]
    """Points where the Einsum projection is defined (output coord in CS^Z)."""

    map_tmp: dict[Point, Any] = field(default_factory=dict)
    red_tmp: ReductionTemporary = field(default_factory=ReductionTemporary)

    @property
    def undefined_points(self) -> int:
        return len(self.iteration_space) - len(self.defined_points)


def _single_spec(einsum: Einsum, kind: type, where: str) -> Any:
    found = [s for s in einsum.specs if isinstance(s, kind)]
    if len(found) > 1:
        raise UnderSpecifiedProgramError(
            f"{where}: {len(found)} {kind.__name__}s on one Einsum. The "
            f"reference evaluator runs a single reduction and a single "
            f"populate per Einsum; a multi-level reduce needs the Einsum "
            f"split, or the evaluator extended."
        )
    return found[0] if found else None


def mutable_positions(
    einsum: Einsum, spec: PopulateSpec, env: Environment
) -> tuple[int, ...]:
    """Rank positions of the output tensor that Populate may vary.

    ``PopulateSpec.rank_list`` names the ``*``-marked ranks, but the IR does
    not say whether an entry is a rank NAME (``"V"``, as declared on the
    tensor) or a rank VARIABLE (``"v"``, as written in the subscript). The
    corpus uses both. So both are accepted: an entry is first matched against
    the subscript positions of ``output_ranks`` -- which is where the ``*``
    is actually written -- and failing that against the declared rank names.
    """
    positions: list[int] = []
    for entry in spec.rank_list:
        by_subscript = [
            index
            for index, node in enumerate(einsum.output_ranks)
            if isinstance(node, RankVariable) and node.name == entry
        ]
        if by_subscript:
            positions.append(by_subscript[0])
            continue
        positions.extend(env.rank_positions(einsum.output_tensor, [entry]))
    return tuple(sorted(set(positions)))


def evaluate_einsum(
    einsum: Einsum,
    *,
    env: Environment,
    pinned: dict[str, Coordinate] | None = None,
    schedule: Schedule | None = None,
) -> EinsumResult:
    """Run one Einsum against the environment, updating its output tensor."""
    schedule = schedule or SequentialSchedule()
    where = f"Einsum writing {einsum.output_tensor!r}"

    output_space = env.spaces.get(einsum.output_tensor)
    if output_space is None:
        raise UnderSpecifiedProgramError(f"{where}: the output tensor is not declared.")
    output = env.tensor(einsum.output_tensor)
    output_empty = output.empty_value

    space = build_iteration_space(
        einsum,
        declarations=list(env.declarations.values()),
        spaces=env.spaces,
        shape_env=env.shape_env,
        coord_sets=env.coord_sets,
        registry=env.functions,
        pinned=pinned or {},
    )
    result = EinsumResult(iteration_space=space, defined_points=[])

    # ---- Map -----------------------------------------------------------
    output_coords: dict[Point, Point] = {}
    for point in schedule.order(list(space.points), Phase.MAP):
        bound = space.point_map(point)
        projection = einsum_projection(
            list(einsum.output_ranks),
            bound,
            point,
            output_space=output_space,
            shape_env=env.shape_env,
            registry=env.functions,
        )
        if not projection.defined:
            # Gamma^Z(is) fell outside CS^Z, so EP is undefined here: the
            # point produces no MapTmp and Reduce never observes it.
            continue
        output_coords[point] = projection.output_coord
        mapped = map_action(
            einsum.expression,
            bound,
            point,
            axes=space.axes,
            specs=list(einsum.specs),
            output_empty=output_empty,
            env=env,
            where=where,
        )
        if mapped.present:
            result.map_tmp[point] = mapped.value

    result.defined_points = sorted(output_coords)

    # ---- Reduce ---------------------------------------------------------
    reduce_spec = _single_spec(einsum, ReduceSpec, where)
    ordered = schedule.order(list(result.defined_points), Phase.REDUCE)
    result.red_tmp = reduce_action(
        reduce_spec,
        points=ordered,
        axes=space.axes,
        output_coords=output_coords,
        map_tmp=result.map_tmp,
        output_empty=output_empty,
        env=env,
        where=where,
    )

    # ---- Populate -------------------------------------------------------
    populate_spec = _single_spec(einsum, PopulateSpec, where)
    populate_order = schedule.order(sorted(result.red_tmp.values), Phase.POPULATE)
    populate_action(
        populate_spec,
        red_tmp=result.red_tmp,
        order=populate_order,
        axes=space.axes,
        output=output,
        mutable=(
            mutable_positions(einsum, populate_spec, env)
            if populate_spec is not None and populate_spec.rank_list
            else ()
        ),
        env=env,
        where=where,
    )
    return result
