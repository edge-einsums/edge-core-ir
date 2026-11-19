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

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from edge_ir.evaluator.actions.map import (
    NestedReductions,
    find_nested_reductions,
    map_action,
)
from edge_ir.evaluator.actions.populate import PopulateRound, populate_action
from edge_ir.evaluator.actions.reduce import ReductionTemporary, reduce_action
from edge_ir.evaluator.environment import Environment
from edge_ir.evaluator.errors import UnderSpecifiedProgramError
from edge_ir.evaluator.iteration import (
    ConcreteIterationSpace,
    build_iteration_space,
    predicate_holds,
)
from edge_ir.evaluator.order import Phase, Schedule, SequentialSchedule
from edge_ir.evaluator.projection import einsum_projection
from edge_ir.evaluator.spaces import Coordinate, Point
from edge_ir.ir.actions import PopulateSpec, ReduceSpec
from edge_ir.ir.einsum import CaseArm, Cases, Einsum
from edge_ir.ir.expr import RankVariable


@dataclass
class EinsumResult:
    """What one Einsum produced, kept for debugging and for tests."""

    iteration_space: ConcreteIterationSpace
    defined_points: list[Point]
    """Points where the Einsum projection is defined (output coord in CS^Z)."""

    map_tmp: dict[Point, Any] = field(default_factory=dict)
    red_tmp: ReductionTemporary = field(default_factory=ReductionTemporary)

    arms: list[EinsumResult | None] = field(default_factory=list)
    """For a case statement: one result per arm, in written order, each over
    the region of the iteration space that arm claimed (``None`` for an arm
    written ``empty``). ``red_tmp`` above is then the arms' combined result,
    and ``iteration_space`` the arms' common one (or an empty one when the
    arms do not share their axes). Empty for every other Einsum."""

    populate_rounds: list[PopulateRound] | None = None
    """One record per round of a Populate with a mutable rank, in the order
    the schedule handed the Reduce Temporary elements out. ``None`` unless
    ``evaluate_einsum(record_populate=True)`` asked for it."""

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


def _map_and_reduce(
    einsum: Einsum,
    *,
    env: Environment,
    pinned: dict[str, Coordinate],
    schedule: Schedule,
    where: str,
    keep: Callable[[dict[str, Coordinate]], bool] | None = None,
) -> EinsumResult:
    """Map then Reduce for one Einsum (or one arm of a case statement).

    ``keep`` restricts the iteration space to the points it accepts; a case
    statement uses it to give each arm its region. Nothing is written to the
    output tensor here: that is Populate, which the caller runs once.
    """
    output_space = env.spaces[einsum.output_tensor]
    output_empty = env.tensor(einsum.output_tensor).empty_value

    space = build_iteration_space(
        einsum,
        declarations=list(env.declarations.values()),
        spaces=env.spaces,
        shape_env=env.shape_env,
        coord_sets=env.coord_sets,
        registry=env.functions,
        pinned=pinned or {},
    )
    if keep is not None:
        space = replace(
            space, points=tuple(p for p in space.points if keep(space.point_map(p)))
        )
    result = EinsumResult(iteration_space=space, defined_points=[])

    # Reductions written inside parentheses (paper 7.3.2, Eq. 143) contract
    # their axes there. The Einsum then iterates the remaining axes only:
    # each outer point stands for the group of full points behind it, which
    # the anonymous tensor folds over.
    by_anon = find_nested_reductions(einsum.expression, list(einsum.specs), space.axes)
    inner_axes: set[str] = set()
    for _, dropped in by_anon.values():
        inner_axes |= dropped
    groups: dict[Point, list[Point]] = {}
    if inner_axes:
        from edge_ir.analysis.iteration_space import coord_image

        for rank in einsum.output_ranks:
            if coord_image(rank).axes & inner_axes:
                raise UnderSpecifiedProgramError(
                    f"{where}: {sorted(inner_axes)} is reduced inside "
                    f"parentheses but also appears in the output's ranks."
                )
        keep = [k for k, axis in enumerate(space.axes) if axis not in inner_axes]
        for full in space.points:
            key = tuple(full[k] for k in keep)
            groups.setdefault(key, []).append(full)
        map_points = [members[0] for members in groups.values()]
    else:
        map_points = list(space.points)

    # ---- Map -----------------------------------------------------------
    output_coords: dict[Point, Point] = {}
    for point in schedule.order(map_points, Phase.MAP):
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
        nested = None
        if inner_axes:
            key = tuple(
                point[k] for k, axis in enumerate(space.axes) if axis not in inner_axes
            )
            nested = NestedReductions(by_anon=by_anon, group=groups[key])
        mapped = map_action(
            einsum.expression,
            bound,
            point,
            axes=space.axes,
            specs=list(einsum.specs),
            output_empty=output_empty,
            env=env,
            where=where,
            nested=nested,
        )
        if mapped.present:
            result.map_tmp[point] = mapped.value

    result.defined_points = sorted(output_coords)

    # ---- Reduce ---------------------------------------------------------
    # A Reduce that ran inside parentheses is done; it is not run again here.
    consumed = {id(spec) for spec, _ in by_anon.values()}
    outer_reduces = [
        s for s in einsum.specs if isinstance(s, ReduceSpec) and id(s) not in consumed
    ]
    if len(outer_reduces) > 1:
        raise UnderSpecifiedProgramError(
            f"{where}: {len(outer_reduces)} Reduce specs apply at the output; "
            f"the reference evaluator runs one there."
        )
    reduce_spec = outer_reduces[0] if outer_reduces else None
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

    return result


def _arm_holds(arm: CaseArm, bound: dict[str, Coordinate], env: Environment) -> bool:
    """Whether a case arm's condition holds at one point."""
    from edge_ir.evaluator.stopping import _comparison_empties, evaluate_predicate

    if arm.where is not None and not predicate_holds(
        arm.where,
        bound,
        declarations=env.declarations,
        shape_env=env.shape_env,
        coord_sets=env.coord_sets,
        registry=env.functions,
    ):
        return False
    if arm.when is not None:
        empties = _comparison_empties(arm.when, env)
        if not evaluate_predicate(
            arm.when, bound, env=env, empty_lhs=empties[0], empty_rhs=empties[1]
        ):
            return False
    return True


def _evaluate_cases(
    einsum: Einsum,
    *,
    env: Environment,
    pinned: dict[str, Coordinate],
    schedule: Schedule,
    where: str,
) -> EinsumResult:
    """Map and Reduce for a case statement (paper: selective computations).

    "Each branch of a case statement is itself a valid EDGE expression, with
    its own modifier if needed. At each point in the iteration space, [it]
    selects the appropriate sub-Einsum expression." The paper's desugaring
    gives each branch its own Einsum into a temporary, restricted to the
    branch's region, and combines the temporaries with the update operator.

    That is what happens here, without naming the temporaries: every arm is
    Mapped and Reduced as its own Einsum over the points it claims -- the
    first arm, in written order, whose condition holds there -- and the
    arms' reduction temporaries are combined into one, which the Einsum's
    single Populate then writes. The output is written once.
    """
    cases = einsum.expression
    assert isinstance(cases, Cases)

    def claimed_by(index: int) -> Callable[[dict[str, Coordinate]], bool]:
        def keep(bound: dict[str, Coordinate]) -> bool:
            for earlier in cases.arms[:index]:
                if _arm_holds(earlier, bound, env):
                    return False
            return _arm_holds(cases.arms[index], bound, env)

        return keep

    combined = ReductionTemporary()
    arms: list[EinsumResult | None] = []
    for index, arm in enumerate(cases.arms):
        if arm.expression is None:
            arms.append(None)  # written `empty`: the region gets no value
            continue
        if any(isinstance(spec, PopulateSpec) for spec in arm.specs):
            raise UnderSpecifiedProgramError(
                f"{where}: arm {index} of the case statement carries a "
                f"Populate spec. An arm has its own Map and Reduce; Populate "
                f"writes the output once, so it belongs on the Einsum."
            )
        sub = Einsum.model_construct(
            output_tensor=einsum.output_tensor,
            output_ranks=einsum.output_ranks,
            expression=arm.expression,
            specs=arm.specs,
            predicates=einsum.predicates,
        )
        result = _map_and_reduce(
            sub,
            env=env,
            pinned=pinned,
            schedule=schedule,
            where=f"{where} (case arm {index})",
            keep=claimed_by(index),
        )
        for coord, value in result.red_tmp.values.items():
            if coord in combined.values:
                raise UnderSpecifiedProgramError(
                    f"{where}: two arms of the case statement produce a value "
                    f"for the output coordinate {coord}. An arm's condition "
                    f"must be decided by the output's own rank variables."
                )
            combined.values[coord] = value
            if coord in result.red_tmp.witness:
                combined.witness[coord] = result.red_tmp.witness[coord]
        arms.append(result)

    ran = [r for r in arms if r is not None]
    axes = {r.iteration_space.axes for r in ran}
    if len(axes) == 1:
        first = ran[0].iteration_space
        points = sorted({p for r in ran for p in r.iteration_space.points})
        space = replace(first, points=tuple(points))
    else:
        space = ConcreteIterationSpace(axes=(), axis_sets=(), points=(), pinned=pinned)
        if any(isinstance(s, PopulateSpec) and s.rank_list for s in einsum.specs):
            raise UnderSpecifiedProgramError(
                f"{where}: the case statement's arms range over different "
                f"rank variables, and the Einsum's Populate has a mutable "
                f"rank. The reference evaluator cannot hand Populate one "
                f"iteration-space point per element here."
            )
    out = EinsumResult(iteration_space=space, defined_points=[])
    out.defined_points = sorted({p for r in ran for p in r.defined_points})
    for r in ran:
        out.map_tmp.update(r.map_tmp)
    out.red_tmp = combined
    out.arms = arms
    return out


def evaluate_einsum(
    einsum: Einsum,
    *,
    env: Environment,
    pinned: dict[str, Coordinate] | None = None,
    schedule: Schedule | None = None,
    record_populate: bool = False,
) -> EinsumResult:
    """Run one Einsum against the environment, updating its output tensor.

    ``record_populate`` keeps what each round of Populate did in
    ``EinsumResult.populate_rounds``. Off by default; it records and changes
    nothing that is computed.
    """
    schedule = schedule or SequentialSchedule()
    where = f"Einsum writing {einsum.output_tensor!r}"

    output_space = env.spaces.get(einsum.output_tensor)
    if output_space is None:
        raise UnderSpecifiedProgramError(f"{where}: the output tensor is not declared.")
    output = env.tensor(einsum.output_tensor)

    if isinstance(einsum.expression, Cases):
        result = _evaluate_cases(
            einsum, env=env, pinned=pinned or {}, schedule=schedule, where=where
        )
    else:
        result = _map_and_reduce(
            einsum, env=env, pinned=pinned or {}, schedule=schedule, where=where
        )
    space = result.iteration_space

    # ---- Populate -------------------------------------------------------
    populate_spec = _single_spec(einsum, PopulateSpec, where)
    populate_order = schedule.order(sorted(result.red_tmp.values), Phase.POPULATE)
    rounds: list[PopulateRound] | None = [] if record_populate else None
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
        rounds=rounds,
    )
    result.populate_rounds = rounds
    return result
