# A* IR Builder Metadata

> *Created with the help of Claude Code.*

Algorithm: `astar` (single-source, single-goal shortest path, non-negative weights)

Variant: Dijkstra with a goal-directed priority. The pick orders by `f = g + h`
instead of `g`, using a precomputed Manhattan heuristic `H`; the search stops when
the goal is settled. `D` always holds true distances (`g`) -- the heuristic only
steers which vertex is picked next, never the stored distances.

Reference math: `examples/algorithms/astar/einsum.md`
Surface form:   `examples/algorithms/astar/einsum.edge`
Family writeup: `examples/algorithms/shortest-path-family.md`

Builder: none yet (see Gaps).

## Lineage (one change at a time from Dijkstra)
- `+` heuristic setup (once): `H_v = sum_xy |L_v - Lg|`, the Manhattan distance
  from each vertex's `(x, y)` to the goal's. `Lg` = `L` restricted to the goal.
- pick ordered by `f = g + h`: `PQ = DQ . H :: +`, select-min over `PQ`.
- recover true `g`: select-min over `PQ` yields a vertex carrying `f`, so
  `F = M . D` grabs its real distance back before relaxing.
- stop `||Q|| == 0`  ->  `||F . goal|| == 1` (settle the goal).

## Heuristic notes
- The heuristic is **static**: each node computes `H` once (depends only on its
  fixed `(x, y)` and the fixed goal), no `I` rank. It lives outside the loop, like `G`.
- `H = sum_xy |L_v - Lg|` is the L1 norm over the location axes. `reduce max` over
  `xy` would give Chebyshev; sum-of-squares-then-root gives Euclidean.
- Manhattan on a grid is **consistent** (and admissible), so the Dijkstra
  settle-once / no-re-open structure carries over unchanged. A merely-admissible
  (non-consistent) heuristic could require re-opening settled nodes, which this
  structure does not do.

## The goal dials the heuristic (links the whole family)
- 1 goal:        Manhattan to it          -> A* (focused, stop when settled).
- a few goals:   distance to nearest goal -> multi-goal A* (stop at first reached);
                 `goal` becomes multi-hot, the heuristic gains a reduced-away goal rank.
- every vertex:  `H = 0` everywhere       -> Dijkstra (no aim; run till queue empty).
More goals -> flatter `H` -> more Dijkstra-like.

## Verification status
- Built as named single-step diffs from the edge-expert-verified Dijkstra;
  distances/pick-order reasoned by hand. The heuristic block was reviewed for
  rank/operator correctness.
- NOT yet run through the IR builders / `make check`. No `*_program.json`. The
  argmin-vs-priority pick (select by `f`, carry `g`) in particular wants a full
  edge-expert + edge-validator pass before publishing.

## Gaps (why there is no builder yet)
- `take_left_only` is a builtin merge op (`op.py`); `select-min-s` is a
  user-defined populate coordinate op (as in DFS); `<<` and `<` are user-defined
  compute ops — same handling as Dijkstra (see `dijkstra/metadata.md`).
- `abs( . )` unary and `|a-b|` absolute-difference compute op: user-defined
  compute, no builtin symbol.
- The `(x, y)` location tensor `L` with an `XY = 2` axis rank and the
  reduce-over-axes Manhattan are new shapes not yet exercised by the IR/builders.
