"""Acceptance: hand-built IR artifacts against INDEPENDENT references.

Each algorithm is executed from its artifact -- the core IR the builder
produces -- and compared against a reference implementation written from the
textbook recurrence, not from the cascade. A reference derived from the
cascade could share a bug with it; these cannot.
"""

from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path

import pytest

from edge_ir.evaluator import check_order_independence, evaluate

ROOT = Path(__file__).resolve().parents[1]
for algorithm in ("bfs", "bellman-ford", "dfs", "max-flow"):
    sys.path.insert(0, str(ROOT / "examples" / "algorithms" / algorithm))

from build_bellman_ford_program import build_bellman_ford  # noqa: E402
from build_bfs_program import build_bfs  # noqa: E402
from build_dfs_program import build_dfs  # noqa: E402
from build_max_flow_program import build_max_flow  # noqa: E402
from max_flow_functions import max_flow_registry  # noqa: E402

################################################################################
# Graph corpus
################################################################################


def _adversarial_digraphs() -> list[tuple[int, list[tuple[int, int]]]]:
    return [
        (1, []),
        (2, [(0, 1)]),
        (4, [(0, 1), (1, 2), (0, 3)]),
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (5, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]),
        (6, [(0, 1), (1, 2), (2, 0), (3, 4)]),
        (6, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]),
        (4, [(0, 0), (0, 1), (1, 2)]),
    ]


def _random_digraphs(
    seed: int, count: int = 12
) -> list[tuple[int, list[tuple[int, int]]]]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        n = rng.choice([5, 8, 12, 20])
        p = rng.choice([0.15, 0.3, 0.5])
        edges = [
            (s, d) for s in range(n) for d in range(n) if s != d and rng.random() < p
        ]
        out.append((n, edges))
    return out


def _digraph_corpus() -> list[tuple[int, list[tuple[int, int]]]]:
    return _adversarial_digraphs() + _random_digraphs(20260909)


################################################################################
# BFS
################################################################################


def _reference_bfs(n: int, edges, roots) -> dict[int, int]:
    """Textbook level-synchronous BFS. Written from the definition."""
    adj: dict[int, list[int]] = {v: [] for v in range(n)}
    for s, d in edges:
        adj[s].append(d)
    dist = {r: 0 for r in roots}
    queue = deque(roots)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


@pytest.mark.parametrize("n,edges", _digraph_corpus())
def test_bfs_depths_match_an_independent_bfs(n: int, edges) -> None:
    program = build_bfs()
    result = evaluate(
        program,
        inputs={"G": {e: 1 for e in edges}},
        shape_env={"|V|": n},
        coord_sets={"id": [0]},
    )
    depths: dict[int, int] = {}
    for (_generation, vertex), value in result["F"].items():
        depths.setdefault(vertex, int(value))
    assert depths == _reference_bfs(n, edges, [0])


def test_bfs_is_order_independent() -> None:
    """The paper requires an Einsum's result not to depend on the order its
    iteration-space points are visited in. Scrambling the traversal under
    several seeds must leave every tensor byte-identical."""
    n, edges = (
        12,
        [
            (s, d)
            for s in range(12)
            for d in range(12)
            if (s * 7 + d) % 5 == 0 and s != d
        ],
    )
    agreed, differences = check_order_independence(
        build_bfs(),
        seeds=(1, 2, 3, 5, 8, 13, 21),
        inputs={"G": {e: 1 for e in edges}},
        shape_env={"|V|": n},
        coord_sets={"id": [0]},
    )
    assert agreed, differences


################################################################################
# Bellman-Ford
################################################################################


def _reference_sssp(n: int, weighted, source: int) -> dict[int, int]:
    """Textbook Bellman-Ford relaxation. Written from the definition."""
    dist: dict[int, float] = {source: 0}
    for _ in range(n):
        changed = False
        for (s, d), w in weighted.items():
            if s in dist and dist[s] + w < dist.get(d, math.inf):
                dist[d] = dist[s] + w
                changed = True
        if not changed:
            break
    return {v: int(x) for v, x in dist.items()}


def _weighted_corpus():
    rng = random.Random(4242)
    cases = [
        (2, {(0, 1): 5}, 0),
        (4, {(0, 1): 1, (1, 2): 2, (0, 2): 5, (2, 3): 1}, 0),
        (5, {(0, 1): 4, (0, 2): 1, (2, 1): 2, (1, 3): 1, (2, 3): 5, (3, 4): 3}, 0),
        (3, {(0, 1): 1, (1, 2): 1, (2, 0): 1}, 0),
        (4, {(0, 1): 1}, 0),
    ]
    for n, edges in _random_digraphs(777, 8):
        cases.append((n, {e: rng.randint(1, 9) for e in edges}, 0))
    return cases


@pytest.mark.parametrize("n,weighted,source", _weighted_corpus())
def test_bellman_ford_distances_match_an_independent_sssp(n, weighted, source) -> None:
    """This is the artifact whose UPDATE step used to encode `<<` as
    `take_right(union)` -- which drops the carried-forward distance where
    the right operand is absent (TODO.md Bugs). With `<<` shipped as the
    `update` compute op the cascade converges to the true distances."""
    program = build_bellman_ford()
    result = evaluate(
        program,
        inputs={"G": dict(weighted)},
        shape_env={"|V|": n},
        coord_sets={"root_id": [source]},
    )
    final_generation = result.trace.generations
    distances = {
        vertex: int(value)
        for (generation, vertex), value in result["D"].items()
        if generation == final_generation
    }
    assert distances == _reference_sssp(n, weighted, source)


################################################################################
# DFS pre-order
################################################################################


def _reference_preorder(n: int, edges, root: int) -> list[int]:
    """Recursive DFS descending into the MAX-coordinate undiscovered
    neighbour first, which is the sibling order the cascade's
    `select-max-val` over sigma(i, v) = i*|V| + v pins."""
    adj = {v: sorted({d for s, d in edges if s == v}) for v in range(n)}
    discovered, order = {root}, []

    def visit(v: int) -> None:
        order.append(v)
        while True:
            candidates = [u for u in adj.get(v, ()) if u not in discovered]
            if not candidates:
                return
            u = max(candidates)
            discovered.add(u)
            visit(u)

    visit(root)
    return order


def _cascade_preorder(n: int, edges, root: int) -> list[int]:
    result = evaluate(
        build_dfs(),
        inputs={"G": {e: True for e in edges}},
        shape_env={"|V|": n},
        coord_sets={"root_id": [root]},
        max_iterations=4 * n + 50,
    )
    return [vertex for (_generation, vertex), _ in sorted(result["F"].items())]


def _random_trees(seed: int, count: int = 10):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        n = rng.choice([2, 3, 4, 5, 6, 8, 10])
        out.append((n, [(rng.randrange(c), c) for c in range(1, n)]))
    return out


@pytest.mark.parametrize("n,edges", _random_trees(31337))
def test_dfs_preorder_matches_on_trees(n: int, edges) -> None:
    """The order stamp is a rank-as-value operand -- sigma(i,v) = i*|V| + v --
    which the artifact could not express at all before the `RankValue` leaf
    landed (it carried a zero-rank constant placeholder that cannot vary
    with the iteration point)."""
    assert _cascade_preorder(n, edges, 0) == _reference_preorder(n, edges, 0)


def test_dfs_preorder_diverges_from_true_dfs_on_non_tree_graphs() -> None:
    """A KNOWN cascade bug, reproduced here independently.

    VISIT marks P from S' -- the vertices just PUSHED -- rather than from F,
    the vertex just POPPED. On a tree every pushed vertex is eventually
    visited from its only parent, so the two agree; on a graph with cross
    edges a vertex marked at push time is skipped by a later, deeper path,
    and the pre-order diverges from a true DFS.

    This test pins the divergence so that fixing the cascade (VISIT from F)
    turns it into a hard failure rather than passing silently.
    """
    n = 5
    edges = [(0, 3), (0, 4), (1, 0), (2, 3), (2, 4), (4, 1), (4, 3)]
    assert _reference_preorder(n, edges, 0) == [0, 4, 3, 1]
    assert _cascade_preorder(n, edges, 0) == [0, 4, 1, 3]


################################################################################
# Max-flow (push-relabel)
################################################################################

Capacities = dict[tuple[int, int], int]


def _reference_max_flow(n: int, capacity: Capacities, source: int, sink: int) -> int:
    """Textbook Edmonds-Karp: augment along BFS-shortest residual paths.

    Written from the definition, not from the cascade. An antiparallel pair
    shares one residual pair, so both c(u,v) and c(v,u) count.
    """
    residual: dict[tuple[int, int], int] = {}
    for (u, v), c in capacity.items():
        residual[(u, v)] = residual.get((u, v), 0) + c
        residual.setdefault((v, u), 0)
    adj: dict[int, set[int]] = {u: set() for u in range(n)}
    for u, v in residual:
        adj[u].add(v)
    flow = 0
    while True:
        parent: dict[int, int] = {}
        seen = {source}
        queue = deque([source])
        while queue and sink not in seen:
            u = queue.popleft()
            for v in sorted(adj[u]):
                if v not in seen and residual[(u, v)] > 0:
                    seen.add(v)
                    parent[v] = u
                    queue.append(v)
        if sink not in seen:
            return flow
        path = []
        v = sink
        while v != source:
            path.append((parent[v], v))
            v = parent[v]
        bottleneck = min(residual[e] for e in path)
        for u, v in path:
            residual[(u, v)] -= bottleneck
            residual[(v, u)] += bottleneck
        flow += bottleneck


def _flow_networks() -> list[tuple[str, int, Capacities]]:
    """(name, n, capacities). The source is vertex 0 and the sink is n - 1."""
    adversarial: list[tuple[str, int, Capacities]] = [
        ("single-edge", 2, {(0, 1): 3}),
        ("path-source-bottleneck", 3, {(0, 1): 2, (1, 2): 5}),
        ("path-sink-bottleneck", 3, {(0, 1): 5, (1, 2): 2}),
        (
            "diamond-with-cross-edge",
            4,
            {(0, 1): 3, (0, 2): 2, (1, 3): 2, (2, 3): 3, (1, 2): 1},
        ),
        ("sink-unreachable", 4, {(0, 1): 1, (2, 3): 1}),
        (
            "antiparallel-at-source",
            4,
            {(0, 1): 4, (1, 0): 2, (1, 3): 3, (0, 2): 1, (2, 3): 5},
        ),
        (
            "antiparallel-internal",
            4,
            {(0, 1): 4, (1, 2): 4, (2, 1): 4, (2, 3): 3, (1, 3): 1},
        ),
        # Vertex 2 is a dead end, so vertex 1 must return its excess to the
        # source. This graph pushed a negative amount out of the source while
        # E17 still wrote Adm.
        ("dead-end", 4, {(0, 1): 5, (1, 2): 5, (1, 3): 1}),
        (
            "edges-into-source",
            4,
            {(0, 1): 3, (1, 0): 3, (2, 0): 4, (1, 3): 2, (2, 3): 1},
        ),
        ("internal-self-loop", 3, {(0, 1): 4, (1, 1): 2, (1, 2): 3}),
        ("no-edges", 3, {}),
    ]
    rng = random.Random(20260915)
    randomized: list[tuple[str, int, Capacities]] = []
    for k in range(12):
        n = rng.choice([4, 5, 6, 7])
        p = rng.choice([0.3, 0.5])
        capacity: Capacities = {}
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < p:
                    capacity[(u, v)] = rng.randint(1, 9)
        randomized.append((f"random-{k}", n, capacity))
    return adversarial + randomized


_MAX_FLOW_CASES = [
    pytest.param(n, capacity, id=name) for name, n, capacity in _flow_networks()
]


def _max_flow_inputs(n: int, capacity: Capacities) -> dict[str, dict]:
    return {
        "G": {e: 1 for e in capacity},
        "C": dict(capacity),
        "S": {(0,): True},
        "T": {(n - 1,): True},
        "VertexCount": {(): n},
    }


def _run_max_flow(n: int, capacity: Capacities):
    return evaluate(
        build_max_flow(),
        inputs=_max_flow_inputs(n, capacity),
        shape_env={"|V|": n},
        functions=max_flow_registry(),
        max_iterations=50 * n * n,
    )


def _generation(result, tensor: str, generation: int) -> dict:
    """One generation of an iterative tensor, keyed by its remaining ranks."""
    return {
        coord[1:]: value
        for coord, value in result[tensor].items()
        if coord[0] == generation
    }


@pytest.mark.parametrize("n,capacity", _MAX_FLOW_CASES)
def test_max_flow_matches_an_independent_edmonds_karp(
    n: int, capacity: Capacities
) -> None:
    """The final generation holds a maximum flow.

    The sink's excess is the flow value, and the source's deficit equals
    it. Every other vertex ends with zero excess, F is antisymmetric, and
    no edge carries more than its capacity.
    """
    result = _run_max_flow(n, capacity)
    final = result.trace.generations
    excess = _generation(result, "E", final)
    flow = _generation(result, "F", final)
    source, sink = 0, n - 1
    value = _reference_max_flow(n, capacity, source, sink)

    assert excess.get((sink,), 0) == value
    assert -excess.get((source,), 0) == value
    assert {u: x for (u,), x in excess.items() if u not in (source, sink)} == {}
    for (u, v), f in flow.items():
        assert flow.get((v, u), 0) == -f
    for u in range(n):
        for v in range(n):
            assert flow.get((u, v), 0) <= capacity.get((u, v), 0)


@pytest.mark.parametrize("n,capacity", _MAX_FLOW_CASES)
def test_max_flow_every_push_is_a_legal_push(n: int, capacity: Capacities) -> None:
    """Replay each generation against the push-relabel push rule.

    A push u -> v at generation g comes from an active vertex, goes along an
    admissible edge (D[u] == D[v] + 1), moves more than 0 and no more than
    min(E[u], R[u,v]), and is u's only push that generation.

    The final flow value does not catch a broken push. While E17 still
    wrote Adm, several graphs reached the right sink excess through pushes
    that break this rule, some of them negative pushes out of the source.
    """
    result = _run_max_flow(n, capacity)
    for g in range(1, result.trace.generations + 1):
        delta = _generation(result, "Delta", g)
        height = _generation(result, "D", g)
        excess = _generation(result, "E", g)
        residual = _generation(result, "R", g)
        active = _generation(result, "Act", g)
        pushers = [u for (u, _v) in delta]
        assert len(pushers) == len(set(pushers)), f"generation {g}"
        for (u, v), amount in delta.items():
            assert active.get((u,)) is True, f"generation {g}: {u}->{v}"
            assert height[(u,)] == height[(v,)] + 1, f"generation {g}: {u}->{v}"
            assert 0 < amount <= min(excess[(u,)], residual[(u, v)])


def test_max_flow_path_matches_a_hand_trace() -> None:
    """s=0 -(2)-> 1 -(5)-> t=2, traced by hand from pseudocode.md.

    Generation 0 is the preflow round. It saturates s->1, so E = {s: -2,
    1: 2}, and heights start at D_1 = {s: 3, 1: 0, t: 0}. Vertex 1 has no
    admissible edge at generation 1, so it relabels to
    1 + min(D[s], D[t]) = 1. At generation 2 it pushes min(2, 5) = 2 to t.
    No vertex is left active, so the cascade stops after three generations.
    """
    result = _run_max_flow(3, {(0, 1): 2, (1, 2): 5})

    assert result.trace.generations == 3
    assert result.trace.stopped_by == "stopping-condition"
    assert _generation(result, "D", 1) == {(0,): 3, (1,): 0, (2,): 0}
    assert _generation(result, "D", 2) == {(0,): 3, (1,): 1, (2,): 0}
    assert _generation(result, "Delta", 1) == {}
    assert _generation(result, "Delta", 2) == {(1, 2): 2}
    assert _generation(result, "E", 3) == {(0,): -2, (2,): 2}
    assert _generation(result, "F", 3) == {
        (0, 1): 2,
        (1, 0): -2,
        (1, 2): 2,
        (2, 1): -2,
    }


def test_max_flow_source_row_of_R_keeps_its_capacity() -> None:
    """A KNOWN divergence from einsum.md, pinned so a fix shows up.

    E04's source-row arm writes ``R_{1,s,v} = 0``. The evaluator drops a
    Map result equal to the output's empty value before Populate, so that
    write never happens and ``R_{1,s,v}`` keeps ``C_{s,v}`` from the
    default arm. Every Einsum that reads the source row on the way to F, E
    or D (E07, E11, E17 via Rel, E21) is gated on the source being active
    or relabelling, and it never is, so the flow is unaffected. This test
    turns into a failure once the source row is cleared.
    """
    result = _run_max_flow(3, {(0, 1): 2, (1, 2): 5})
    assert _generation(result, "R", 1)[(0, 1)] == 2


def test_max_flow_is_order_independent() -> None:
    """Vertex 1 has three admissible edges at once, so E10 must choose.

    ``select-one-admissible-v`` keeps the smallest admissible v, which does
    not depend on the order Populate visits the candidates in.
    """
    n = 6
    capacity = {
        (0, 1): 8,
        (1, 2): 3,
        (1, 3): 3,
        (1, 4): 3,
        (2, 3): 1,
        (2, 5): 2,
        (3, 5): 2,
        (4, 5): 2,
    }
    agreed, differences = check_order_independence(
        build_max_flow(),
        seeds=(1, 2, 3, 5, 8, 13, 21),
        inputs=_max_flow_inputs(n, capacity),
        shape_env={"|V|": n},
        functions=max_flow_registry(),
        max_iterations=50 * n * n,
    )
    assert agreed, differences
