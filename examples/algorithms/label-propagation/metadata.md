# Label Propagation IR Builder Metadata

> *Toluwanimi Odemuyiwa, combined with the help of Claude Code.*

Algorithm: `label-propagation` (max-label propagation; computes connected
components)

Variant: **single-Einsum, generational-rank label propagation.** Each vertex
starts with a unique label (`id + 1`). Every generation a vertex adopts the
maximum label among its neighbours and itself. Under the reflexivity
precondition below, labels only ever increase and are bounded, so `L` converges;
vertices that end up sharing a label are in the same connected component.

Builder: `build_label_propagation_program.py`

Reference math: `examples/algorithms/label-propagation/einsum.md`

Source: an external `.edge` file (a non-repo `edgecc` surface syntax),
reconciled to the repo IR by Toluwanimi Odemuyiwa, combined with the help of Claude Code.

## The einsum

```
L_{i+1, d} = G_{s, d} . L_{i, s}  :: ⋀_s *(∩)  ⋁_s max(∪)
<> : L_{i+1} ≡ L_i
```

Structurally this is the BFS **advance** einsum
(`T_{i,d} = G_{s,d}·F_{i,s} :: ⋀_s +(∩) ⋁_s ANY(∪)`) with `+` → `*` and
`ANY` → `max`. `s` is the contraction rank (mapped + reduced, gone from the
output), `d` is preserved, `i` is the generational rank.

- The **map** `*(∩)` multiplies the source label `L[i,s]` by the edge weight
  `G[s,d]` only where **both** are present (intersect). With 1.0 edge weights
  this passes the neighbour's label through unchanged. Empty is not zero:
  `intersect` skips a point where either operand is empty, it does not
  contribute a `0`.
- The **reduce** `max(∪)` keeps the largest incoming label at each `d`.

## Input-graph contract (precondition on host-supplied `G`)

Correctness AND termination as a connected-components computation require `G` to
be a **symmetric, reflexive, 0/1 (unweighted)** adjacency matrix. These are
properties of the **data the host supplies**, not something the einsum enforces
-- the einsum is valid EDGE for any `G`; it just misbehaves if they are
violated. All three are load-bearing:

- **Reflexive (a `(d,d)` self-edge on every vertex) -- required for
  TERMINATION, not just correctness.** The self-edge is what puts vertex `d`'s
  own current label `L[i,d]` into the set of map temporaries gathered at output
  `d`. (The reduce state is seeded at the `max` *identity*, NOT at `L[i,d]` --
  so it is the **gather** that needs the self-edge, not the `max`.) With it,
  `L[i+1,d] ≥ L[i,d]`: labels are monotonically non-decreasing and bounded, so
  the stop `L_{i+1} ≡ L_i` is reachable. **Without reflexivity the iteration can
  oscillate forever** -- e.g. on the path `0-1-2` with `L_0=[1,2,3]` it cycles
  `[2,3,2] → [3,2,3] → [2,3,2] → …` and never converges.
- **Symmetric -- required because the gather is over in-edges.** Output `d`
  gathers `G_{s,d}` (edges INTO `d`), so a label flows `s → d` only if
  `(s,d) ∈ G`. Symmetry (`(s,d) ∈ G ⟺ (d,s) ∈ G`) makes a label propagate both
  ways across each undirected edge. On a directed graph this maxes over in-edges
  only and does **not** give weakly-connected components -- directed/WCC is out
  of scope.
- **0/1 weights -- required because the map is `*`.** The map multiplies edge
  weight by label, so a present entry MUST be exactly `1` (passes the label
  through unchanged). A weight `≠ 1` scales the propagated label (meaningless as
  a component id); a weight `> 1` makes labels grow unboundedly and breaks
  termination. A present entry of `0.0` is ill-formed -- it collides with the
  empty value.

(Same pattern as Bellman-Ford documenting its contract separately from the
cascade. **Open design question routed to the edge-expert:** whether the map
should be `→(∩)` -- take_right, pass the source label through unchanged --
instead of `*(∩)`, which would make "edge weights don't matter" literally true
and remove the scaling/termination landmine.)

## Ground-truth test vector

Undirected graph on `N = 5` with two components, `{0, 1, 2}` and `{3, 4}`:

```
G = [[1, 1, 0, 0, 0],
     [1, 1, 1, 0, 0],
     [0, 1, 1, 0, 0],
     [0, 0, 0, 1, 1],
     [0, 0, 0, 1, 1]]      # symmetric + reflexive, 1.0 weights

L_0 = [1, 2, 3, 4, 5]      # unique labels, label(v) = v + 1
```

Hand trace (`L[i+1, d] = max_s { G[s,d] * L[i,s] }`):

```
i=0: L_0 = [1, 2, 3, 4, 5]
i=1: L_1 = [2, 3, 3, 5, 5]
i=2: L_2 = [3, 3, 3, 5, 5]
i=3: L_3 = [3, 3, 3, 5, 5]   == L_2  ->  STOP
```

Expected final labels **L = [3, 3, 3, 5, 5]**: component `{0,1,2}` settles on
label 3, component `{3,4}` on label 5. Matches the source file's stated output
`[3.0, 3.0, 3.0, 5.0, 5.0]`. Converges in 2 effective rounds; the stop fires
when `L_{i+1} ≡ L_i`.

## Coverage

| ID | Status | Equation / Text |
|---|---|---|
| `decl-G` | exact | G^{R ≡ N, C ≡ N} -> float, empty = 0.0 |
| `decl-L` | exact | L^{I, R ≡ N} -> float, empty = 0.0 |
| `init` | exact (empty) | No init einsum. Both `G` and `L_0` are host-provided data (`<user-specified>`). See Gap Notes. |
| `E1 PROPAGATE` | exact | L_{i+1,d} = G_{s,d} · L_{i,s} :: ⋀_s *(∩) ⋁_s max(∪) -- builtin `*`/intersect map; UDF `max`/union reduce |
| `stop` | exact | ⋄ : L_{i+1} ≡ L_i -- `Comparison(==)` over two `TensorProjectionValue` of `L`; the "all points satisfy" lifting is exactly the paper's `≡` |

## Gap Notes

**No IR gaps.** Every declaration, the einsum, and the stop have an exact IR
encoding. This is the cleanest of the graph artifacts on the
initialization front -- see below.

- **Empty initialization block (first among the artifacts).** BFS / DFS /
  Bellman-Ford each carry at least one init einsum, because they seed a
  *source set* (`F_0`, `S_0`, `D_0`) from a host coord set via a
  `SetMembership` predicate. Connected-components has **no source restriction**:
  *every* vertex is initialized, and `L_0 = [1, 2, ...]` (label(v) = v + 1) is
  a fully host-specified vector. So `L_0` is **data, not computation** -- it is
  `<user-specified>`, exactly like the `G` matrix, and the `Initialization`
  block holds an empty einsum list. The concrete `L_0` vector and `G` matrix
  live in the host / test fixture, not in the program structure.

- **Deliberately avoids the D21 rank-as-value gap.** `L_{0,v} = v + 1` is a
  coordinate-as-value (the same family as the DFS `sigma(i,v) = i·|V| + v`
  stamp, which had to use a zero-rank `Stamp` placeholder because the IR
  `Expression` union has no rank-as-value leaf). Connected-components dodges
  this entirely: because `L_0` is fully host-specified with no source
  restriction, it is host data and needs no init einsum -- so there is **no
  placeholder tensor** and **no synthesized literal** (`Lit_0`, etc.) anywhere
  in this artifact. If one ever wanted the *algorithmic* `L_{0,v} = v+1` init
  form (deriving the labels rather than supplying them), D21 would resurface;
  that is the same open question DFS already routed to the edge-expert. We
  chose the host-data form precisely to stay gap-free.

## Notes

- **`max` is a user-defined compute op**, `UserDefinedComputeOp(name="max")`,
  not a builtin: the IR's `BuiltinComputeOp.symbol` set is only `{+, -, *, /}`
  (`edge_ir/ir/op.py`). This is the exact dual of Bellman-Ford's reduce-compute
  UDF `min`, and parallels BFS's reduce-compute UDF `ANY`.
- **The stop is value-convergence, not occupancy.** `L` never empties; it
  converges in value. So the stop is the Bellman-Ford form
  (`Comparison(==)` over two `TensorProjectionValue` of `L`), NOT the BFS/DFS
  occupancy form (`‖F_{i+1}‖ ≡ 0` via `PropertyApp("occupancy")`).
- **Labels MUST be strictly greater than the empty value `0.0`.** The `v+1`
  convention guarantees this. This artifact supplies `0.0` as the `max` reduce's
  identity, which here *coincides* with the empty value -- so a label equal to
  `0.0` is indistinguishable from "absent" and would be silently swallowed; a
  host supplying a `0` label (e.g. 0-indexed `label(v)=v`) is ill-formed.
  (Caveat: EDGE leaves a UDF compute op's reduce identity user-specified;
  empty = identity is a *coincidence here* for `max` over non-negative labels,
  NOT a general rule -- Bellman-Ford's `min` has identity `+∞` but empty `∞`/`0`
  depending on tensor. **Open question for the edge-expert:** should a UDF reduce
  with no registered identity fall back to the tensor's empty value?)
- **Isolated vertices / singleton components.** With reflexivity, an isolated
  vertex `v` has `N(v)={v}`, so `L[i+1,v]=L[i,v]` -- it keeps its label and is
  its own component (correct). Without reflexivity it gathers the empty set and
  its label **vanishes** (empty output) -- a second way non-reflexivity breaks
  things. The ground-truth vector has no singleton; a vertex with only a `(5,5)`
  self-loop (expect `L_final[5]=6`) would cover it.
- **Data type follows the source.** The source declares `L : float, empty =
  0.0`, so `G` and `L` are declared `float` even though the labels are whole
  numbers. (`int` would also be admissible; this honors the source verbatim.)
- The `[3,3,3,5,5]` ground truth above was produced by a hand trace
  of the einsum, not by executing this IR.
