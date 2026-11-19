"""Inner cascades: a cascade as one step of another.

LOCAL PATCH (edge-workbench issue 10). The paper: "A fully specified
extended Einsum consists of a cascade of Einsum expressions, and EDGE extends
this model to support nested cascades." The IR's ``NestedCascade`` was flat
and the evaluator refused a second iterative rank; here an entry of a cascade
may itself be a ``NestedCascade``.
"""

from __future__ import annotations

import itertools

import pytest

from edge_ir.evaluator import check_order_independence
from edge_ir.ir.einsum import flatten
from edge_ir.ir.program import Program
from tests._bubble_sort_program import STEPS, bubble_sort_program, run_bubble_sort


def rows(result, size):
    z = dict(result["Z"].items())
    passes = sorted({c[0] for c in z})
    return {
        i: [[z.get((i, j, n), 0) for n in range(size)] for j in range(size - 1)]
        for i in passes
    }


def test_the_tutorials_worked_example_step_by_step() -> None:
    """A = [3, 1, 4, 2]: the table in the bubble-sort tutorial."""
    result = run_bubble_sort([3, 1, 4, 2])
    assert rows(result, 4) == {
        0: [[1, 3, 0, 0], [1, 3, 4, 0], [1, 3, 2, 4]],
        1: [[1, 3, 2, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        2: [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    }
    # "Pass 2 sweeps once more and changes nothing ... the outer cascade
    # halts": the diamond Z_{i+1,J-1} == Z_{i,J-1} holds at i = 1, once pass
    # 2 exists.
    assert result.trace.stopped_by == "stopping-condition"
    assert result.trace.generations == 3


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_every_permutation_is_sorted(size: int) -> None:
    for values in itertools.permutations(range(1, size + 1)):
        result = run_bubble_sort(list(values))
        table = rows(result, size)
        assert table[max(table)][-1] == sorted(values), values


def test_a_sorted_array_stops_after_the_pass_that_changed_nothing() -> None:
    """The diamond reads Z_{i+1} while the body writes Z_i, so after pass g
    it is decided for i = g-1: passes 0 and 1 run, then it holds at i = 0."""
    result = run_bubble_sort([1, 2, 3, 4, 5])
    assert result.trace.stopped_by == "stopping-condition"
    assert result.trace.generations == 2


def test_the_pass_rank_bounds_the_outer_cascade() -> None:
    """RV_i = { i | i in [0, I) and not diamond(i) }: a reversed array needs
    every one of its I = N-1 passes, and the cascade ends with the rank."""
    result = run_bubble_sort([4, 3, 2, 1])
    assert result.trace.generations == 3
    assert rows(result, 4)[2][-1] == [1, 2, 3, 4]


def test_the_inner_cascade_runs_j_over_the_step_rank_each_pass() -> None:
    result = run_bubble_sort([3, 1, 4, 2])
    # 3 passes x 3 steps x 3 Einsums, in written order within a step.
    assert [len(g) for g in result.trace.per_generation] == [9, 9, 9]


def test_bubble_sort_does_not_depend_on_traversal_order() -> None:
    agreed, differences = check_order_independence(
        bubble_sort_program(),
        inputs={"A": {(k,): v for k, v in enumerate([5, 2, 4, 1, 3])}},
        shape_env={"|A|": 5, "N": 5, STEPS: 4},
    )
    assert agreed, differences


def test_a_nested_cascade_round_trips_and_flattens_in_written_order() -> None:
    program = bubble_sort_program()
    again = Program.model_validate_json(program.model_dump_json())
    assert again == program
    outputs = [
        [
            r.model_dump().get("name", r.model_dump().get("value"))
            for r in e.output_ranks
        ]
        for e in flatten(again.main_edge.cascade)
    ]
    assert outputs == [[0, "j", "n"], ["i", 0, "n"], ["i", "j", "n"]]
