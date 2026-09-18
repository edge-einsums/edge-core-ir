# Widest path / maximum-bottleneck path

> *Created with the help of Claude Code.*

Algorithm: `widest-path` (single-source maximum-bottleneck path)

Reference math: `examples/algorithms/widest-path/einsum.md`
Surface program: `examples/algorithms/widest-path/einsum.edge`
Fibertree executor: `examples/algorithms/widest-path/widest_path_fibertree.py`

## What it computes

For a directed graph whose edges carry **capacities**, the widest path from a source
`s` to a vertex `d` is the path that **maximizes the minimum edge capacity** along it
(the fattest bottleneck you can push from `s` to `d`). This is the `(max, min)` mirror
of shortest-path Bellman-Ford, which is `(min, +)`:

| | shortest path (Bellman-Ford) | widest path (this) |
|---|---|---|
| extend a path | `+` (add edge weight) | `min` (narrowest edge so far) |
| pick among routes | `min` (shortest) | `max` (widest) |
| `empty` / identity | `+inf` (unreached) | `0` (unreached) |

Same iterate-until-`stable` skeleton as Bellman-Ford; only the two operators flip.

## Origin / why this folder exists

This program was originally sitting (mislabeled) as `examples/algorithms/bellman-ford/einsum.edge`,
whose `metadata.md` + `einsum.md` actually describe min-plus shortest-path Bellman-Ford.
The two are mirror algorithms but **not** the same; this folder gives the widest-path
program its own self-consistent home. The `bellman-ford/` folder is left untouched.

## Ground-truth test vector

Directed weighted (capacity) graph on `|V| = 4`, source = vertex 0:

| edge | capacity |
|------|----------|
| 0 -> 1 | 3 |
| 0 -> 2 | 5 |
| 1 -> 3 | 3 |
| 2 -> 3 | 1 |

`G` (row = source `s`, col = dest `d`; `0` = no edge):

```
       d=0   d=1   d=2   d=3
s=0  [  0     3     5     0  ]
s=1  [  0     0     0     3  ]
s=2  [  0     0     0     1  ]
s=3  [  0     0     0     0  ]
```

Source seed: `F_0 = D_0 = [999, 0, 0, 0]` (vertex 0 has unbounded incoming bottleneck;
`999` is a finite stand-in for `+inf`).

Expected final bottlenecks **D = [999, 3, 5, 3]**:

- `d=0`: 999 (the source itself)
- `d=1`: 3 (direct edge 0->1, capacity 3)
- `d=2`: 5 (direct edge 0->2, capacity 5)
- `d=3`: 3 (route 0->1->3 gives `min(3,3)=3`, beating 0->2->3's `min(5,1)=1`)

## Line-by-line semantics

```
F[i+1, d] = G[s,d] . F[i,s]   :: map(min, intersect) reduce(max, union)
```
For each destination `d`: over every source `s`, take `min(G[s,d], F[i,s])` -- the
narrowest link on the route that reaches `s` and then crosses edge `s->d`
(`map(min, intersect)`; both operands must exist). Then `max` over all `s`
(`reduce(max, union)`) to pick the widest incoming route. So
`F[i+1,d] = max_s min(G[s,d], F[i,s])`.

```
D[i+1, d] = D[i,d] . F[i+1,d] :: map(max, union) reduce(max, union)
```
Keep the larger of the old best-known bottleneck and the new wavefront value, so `D`
grows monotonically. There is no contraction rank here (both sides are free in `d`),
so the `reduce` is a no-op / identity; the work is the elementwise `map(max, union)`.

```
stop: stable(D)
```
Iterate until `D` stops changing (fixpoint), i.e. `D_{i+1} == D_i`.

## Hand trace (matches the fibertree executor)

```
gen 0:  F0 = [999, 0, 0, 0]                     D0 = [999, 0, 0, 0]
gen 1:  F1 = [0, 3, 5, 0]   (0->1:3, 0->2:5)    D1 = [999, 3, 5, 0]
gen 2:  F2 = [0, 0, 0, 3]   (1->3: min(3,3)=3)  D2 = [999, 3, 5, 3]
gen 3:  F3 = [0, 0, 0, 0]   (node 3 has no out) D3 = [999, 3, 5, 3]  -> D3 == D2, STOP
```

Converges in 2 effective rounds; `stop` fires when `D_{i+1} == D_i`.

## Notes / caveats

- **`999` is a finite proxy for `+inf`.** A faithful "unbounded source bottleneck"
  would seed the source with `+inf` and declare `empty = +inf`; the surface program
  uses `999` and `empty = 0` so the example stays in plain finite floats. With this
  graph any real path bottleneck is `<= 5`, so `999` never interferes.
- **The `D` update's `reduce` is vacuous.** `D[i+1,d] = D[i,d] . F[i+1,d]` contracts
  no rank, so `reduce(max, union)` reduces over an empty set of contraction coords and
  collapses to the elementwise `map(max, union)`. Written out in full only to mirror
  the two-clause shape of the relaxation einsum.
- The distances above were produced by the standalone fibertree executor in this folder,
  not by executing EDGE IR.
