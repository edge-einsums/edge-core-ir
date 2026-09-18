# Bellman-Ford IR Builder Metadata

> *Created with the help of Claude Code.*

Algorithm: `bellman-ford` (single-source shortest paths)

Variant: **canonical Bellman-Ford** -- the paper's Cascade `cascade:bf`
in the EDGE paper. Relax ALL edges every
iteration; re-relax a vertex whenever a strictly shorter path is found,
gated only by the improvement check `N < D`. There is exactly ONE distance
tensor `D` (integer, empty = +inf = unreached). The `<<` update merge is
defined in the EDGE paper's Bellman-Ford appendix.

Builder: `build_bellman_ford_program.py`

Reference math: `examples/algorithms/bellman-ford/einsum.md`

## This is the CORRECTED cascade -- NOT the reviewed one

A user-supplied "Bellman-Ford" cascade was reviewed by the EDGE agents and
found to be **NOT Bellman-Ford**. That version gated the frontier with
not-visited (`\neg V`), which blocks re-relaxation: once a vertex is
visited it can never be improved again. That makes it a **single-visit
weighted BFS**, not Bellman-Ford. On the ground-truth graph below the
weighted-BFS version computes `d(3) = 11` (path 0->1->3 = 10+1), because it
locks vertex 1's distance at 10 on first visit and never relaxes it down to
2 via 0->2->1. Canonical Bellman-Ford re-relaxes vertex 1 to 2 and so gets
`d(3) = 3`. This artifact builds the canonical version from the paper, not
the reviewed cascade.

## Ground-truth test vector

Directed weighted graph on `|V| = 4`, source = vertex 0:

| edge | weight |
|------|--------|
| 0 -> 1 | 10 |
| 0 -> 2 | 1 |
| 2 -> 1 | 1 |
| 1 -> 3 | 1 |

Expected final distances **D = {0: 0, 1: 2, 2: 1, 3: 3}**.

Hand/sim trace of the canonical cascade (cross-checked against a clean
reference simulator, see "Cross-check" below):

```
i=0: N={1:10, 2:1}        NewlyRelaxed={1:10, 2:1}  D_1={0:0, 1:10, 2:1}
i=1: N={1:2, 2:1, 3:11}   NewlyRelaxed={1:2, 3:11}  D_2={0:0, 1:2, 2:1, 3:11}
i=2: N={1:2, 2:1, 3:3}    NewlyRelaxed={3:3}        D_3={0:0, 1:2, 2:1, 3:3}
i=3: N={1:2, 2:1, 3:3}    NewlyRelaxed={}           D_4 == D_3  -> STOP
```

Converges in 3 effective rounds; the stop fires when `D_{i+1} == D_i`. This
graph specifically distinguishes true BF (`d3=3`) from weighted-BFS
(`d3=11`).

## Coverage

| ID | Status | Equation / Text |
|---|---|---|
| `decl-G` | exact | G^{S ≡ \|V\|, D ≡ \|V\|} -> integer, empty = 0 |
| `decl-C` | exact | C^{I, S ≡ \|V\|} -> Boolean, empty = False (improvement mask) |
| `decl-N` | exact | N^{I, D ≡ \|V\|} -> integer, empty = ∞ (relaxed candidates) |
| `decl-NewlyRelaxed` | exact | NewlyRelaxed^{I, D ≡ \|V\|} -> integer, empty = ∞ |
| `decl-D` | exact | D^{I, S ≡ \|V\|} -> integer, empty = ∞ (the single distance tensor) |
| `decl-Lit_0` | hand-authored literal | zero-rank int `Lit_0` (value=0, empty_value=None) for the `D_0` source literal `0` |
| `init-D0` | exact | D_{0, s : s ∈ root_id} = 0 -- init Einsum, `expression=InputTensor(Lit_0)`, `predicates=[SetMembership(s ∈ root_id)]` |
| `E1 RELAX` | exact | N_{i,d} = G_{s,d} · D_{i,s} :: ⋀_s +(∩) ⋁_s min(∪) -- builtin `+`/intersect map; UDF `min`/union reduce |
| `E2 IMPROVE` | **partial (gap G2)** | C_{i,d} = N_{i,d} · D_{i,d} :: ⋀_d <(∪) -- `<` encoded as UDF compute `less_than`; merge union |
| `E3 RECORD` | exact | NewlyRelaxed_{i,d} = C_{i,d} · N_{i,d} :: ⋀_d →(∩) -- `take_right(intersect)` (keep N where C and N present) |
| `E4 UPDATE` | **approximate (gap G1)** | D_{i+1,d} = D_{i,d} · NewlyRelaxed_{i,d} :: ⋀_d <<(∪) -- encoded as `take_right(union)`; `<<` has no IR symbol |
| `stop` | exact | ⋄ : D_{i+1} ≡ D_i -- `Comparison(==)` over two `TensorProjectionValue` of `D`; the "all points satisfy" lifting is exactly the paper's `≡` (stopping.py docstring) |

## Gap Notes

Two gaps. The cascade is structurally complete, round-trips through the IR
and the JSON schema, and is mypy-clean. Both gaps are operator-symbol gaps,
not structural ones, and are faithful-for-this-input approximations; flagged
for the edge-expert.

- **G1 -- the update merge `<<` has no IR merge symbol.** Step (4) UPDATE
  uses `<<(∪)` = "take the right operand's value where right exists, else
  take the left" (EDGE paper, Bellman-Ford appendix). The 16 builtin merge
  symbols in `op.py` do not include `<<`; `docs/keywords.md` treats `<<` as
  a surface token that lowers to a Map with `<<` semantics, but there is no
  IR-level encoding. Because `NewlyRelaxed` is non-empty **only** where it
  strictly improves `D` (it was already gated by C in step 3), `<<` and
  `take_right(union)` produce identical results here, so UPDATE is encoded as
  `take_right(union)`. This is the SAME gap flagged in the DFS artifact (its
  G3). Open question for the expert / parser author: should `<<` become a
  17th builtin merge symbol, a UDF merge op, or stay a lowering-only token?

- **G2 -- the comparison `<` is not a builtin compute operator.** Step (2)
  IMPROVE is `C_{i,d} = N_{i,d} · D_{i,d} :: ⋀_d <(∪)`, where the compute op
  is the strict-less-than comparison producing a Boolean. The IR's
  `BuiltinComputeOp` symbol set is only `{+, -, *, /}` (`op.py`); comparisons
  are not builtins. It is encoded as the UDF compute op `less_than` with
  merge `union`. This is structurally faithful (UDFs are the sanctioned
  extension point for compute ops) but the paper writes `<` as if it were a
  primitive. Open question for the expert: are the relational comparison
  operators (`<`, `>`, `<=`, ...) meant to be builtin compute ops in EDGE
  (they appear inside Einsum compute positions in `cascade:bf`), or are they
  always user-defined functions named per the host? If builtin, `op.py`'s
  `BuiltinComputeOp.symbol` Literal should be widened.

## Notes

- The init block mirrors BFS/DFS: a single init einsum `D_0` with a
  `SetMembership` predicate over the host-supplied coord set `root_id` (the
  BF source vertices). `Lit_0` carries the literal `0`.
- `D`, `N`, `NewlyRelaxed` use `empty_value = float("inf")`, serialized via
  the reserved `"inf"` sentinel (tensor.py `_FLOAT_SENTINELS`) since strict
  JSON has no Infinity literal. Empty = ∞ is the "unreached / no path yet"
  marker the paper uses for SSSP.
- The paper's "Tensors" block names `D^{I, S ≡ |V|}` but the einsums project
  it as both `D_{i,s}` (RELAX, as a source) and `D_{i,d}` (IMPROVE/UPDATE, as
  a destination). It is ONE tensor; the rank variable used at each projection
  is just the loop variable for that einsum. The declaration keeps the
  paper's rank name `S`.
- The reduce compute op for RELAX is the UDF `min` (paper `\min`), paired
  with `union` -- unlike BFS's advance, which uses `ANY`/`union`. BF needs the
  numeric minimum over incoming sources, not just presence.
- The reference evaluator in `edge_ir/evaluator/` can execute this program.
  The distances above were first produced by a standalone reference
  simulator, not by executing this IR. See "Cross-check".

## Cross-check

The `{0:0, 1:2, 2:1, 3:3}` ground truth was confirmed against a clean
canonical-BF reference simulator on this exact graph (one distance tensor,
relax-all-edges, `min` reduction, `N<D` improvement gate, `<<` update,
converge on `D_{i+1} == D_i`).
