# Dijkstra IR Builder Metadata

> *Created with the help of Claude Code.*

Algorithm: `dijkstra` (single-source shortest paths, non-negative weights)

Variant: priority-queue SSSP built as a minimal diff from canonical Bellman-Ford
(`examples/algorithms/bellman-ford/`). Same relax/improve/record/update body; adds a
PICK (closest queued vertex, via a populate select-min) and queue maintenance.
Cross-checked against the paper's Dijkstra (Cascade 12,
in the EDGE paper).

Reference math: `examples/algorithms/dijkstra/einsum.md`
Surface form:   `examples/algorithms/dijkstra/einsum.edge`
Family writeup: `examples/algorithms/shortest-path-family.md`

Builder: none yet (see Gaps).

## Lineage (one change at a time from Bellman-Ford)
- relax source `D -> F`: relax from the one picked vertex, not from all of `D`.
- `+` PICK on top: `DQ = Q . D`, then `F = select-min_s(DQ)` (populate).
- `+` queue upkeep below: dequeue settled (`T = Q . F :: <-(take_left_only)`), enqueue
  improved (`Q' = T . C :: OR`), reusing the improve mask `C`.
- stop `D_{i+1} == D_i`  ->  `||Q|| == 0`.

## Test vector (same graph as bellman-ford)
Directed, |V| = 4, source 0: 0->1(10), 0->2(1), 2->1(1), 1->3(1).
Settle order **0, 2, 1, 3**; final **D = {0:0, 1:2, 2:1, 3:3}**. Hand-traced;
matches the bellman-ford ground truth.

## Verification status
- Pick/queue mechanics reviewed by edge-expert against paper Cascade 12;
  distances hand-traced on the 4-node graph.
- NOT yet run through the IR builders / `make check`. No `*_program.json`.

## Gaps (why there is no builder yet)
- `take_left_only` is a builtin merge op (`op.py`, paper symbol ⊖l), so the
  dequeue is directly representable.
- `select-min-s` is a user-defined populate coordinate op, like DFS's
  `select-max-val` — representable in IR; the evaluator does not yet implement
  the min-selection.
- `<<` (update) and `<` are user-defined compute ops, the sanctioned extension
  point, handled as in `bellman-ford/metadata.md`.
- No `build_*.py` or `*_program.json` exists yet.
