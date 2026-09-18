# Stopping conditions — open semantic questions

> *Created with the help of Claude Code.*

Status: **open.** These came out of the EDGE-semantics
review (`edge-expert`) of the new structured `StoppingCondition` IR
(`edge_ir/ir/stopping.py`), done. None of them block the
IR layer — the structure was approved as *faithful-with-caveats*, all
the paper's stopping-condition examples are expressible, nothing is
inexpressible. They're decisions the future **validator** and
**interpreter** need pinned down, plus one paper-internal discrepancy.

The new IR shape, for reference: `StoppingCondition(rank_variable: str,
predicate: Predicate)` where `Predicate = Comparison |
DiamondBooleanApp`; `Comparison(lhs, op in {==,!=,>,<,>=,<=}, rhs)`
over `ValueExpression = TensorProjectionValue |
RankExpressionValue | PropertyApp`; `PropertyApp.name =
Literal["occupancy"]`. The diamond `<>` / `◇` is the paper's
*exclusion guard* (`RV_i = { i | … ∧ ¬◇(i) }`, paper §6 syntax), not a
`break`.

> **Post-D21 (2026-05):** `ScalarValue` was removed from `ValueExpression`. Scalars in stopping-condition comparisons are now represented as `TensorProjectionValue` references to a synthesized zero-rank `TensorDeclaration` whose `value` field carries the literal. The questions below were filed before that change; Q3(b) ("comparing a multi-element projection against a scalar literal") is now structurally a comparison against a `TensorProjectionValue`, not a `ScalarValue`. The semantic intent is unchanged — see `docs/design_decisions.md` D21 and `docs/parsing.md` for the lowering rule.

---

## Q1 — EBNF vs. examples mismatch on stopping conditions

The paper's EBNF (`tex/notes/main.ebnf-cropped.tex:40-41`) says a
stopping condition is *only* `◇_<rank> : <user-defined-function>`. But
every worked example uses comparisons (`D_{i+1} ≡ D_i`, `i ≥ 5`,
`k ≡ |V|`, `||F_{i+1}|| ≡ 0`) and the prose says "any expression that
returns True or False" (`tex/bfs-example/07-iteration.tex:55`). The new
IR follows the *examples* (adds `Comparison` + `PropertyApp`) and keeps
the EBNF's `DiamondBooleanApp` as one option. Is that the intended
resolution — i.e. is the EBNF just stale and should be updated to add a
`<comparison>` and an occupancy production? Or is there a deliberate
reason the grammar is narrower than the examples? (edge-expert
recommends routing this discrepancy to `edge-auditor` as a possible
paper bug regardless.)

## Q2 — Rank-pinning semantics for `occupancy` (and predicate-level UDFs) on multi-rank tensors

The paper writes `||F_{i+1}||` where `F^{I,S}` and clearly means
"count non-empty `s` for the fixed generation `i+1`", but it never
states a general rule for *which* ranks a property/UDF pins vs. ranges
over. The IR formalizes this as `pinned_rank_variables` (first = the
diamond's rank, rest = extra fixed ranks). Is "pin the listed ranks,
evaluate the property over the remaining ranks of each operand" the
semantics you intend? And: should the IR (or just the validator)
enforce that every name in `pinned_rank_variables` is actually a rank
of every operand, and occupies a *concrete* (non-free) slot in that
operand's projection?

## Q3 — Mismatched-shape and scalar-vs-tensor comparisons

All paper stopping-condition comparisons are
tensor-vs-identically-shaped-tensor or scalar-vs-`0`. Two corners the
IR now allows but the paper doesn't address:
- (a) comparing two projections with *different* rank sets — should
  that be rejected, or broadcast/intersected?
- (b) comparing a multi-element projection against a scalar literal —
  should that mean "every element equals the scalar" (broadcast + the
  "all points satisfy" lifting)?

And should the validator make clear that "`X ≡ 0` over all elements" is
*not* the same as "`occupancy(X) ≡ 0`" — they differ whenever `X`'s
empty value isn't `0` (as in BFS's `F` with empty `= ∞`)?

## Q4 — What is `K` in `<> : i ≥ K`, and must the rank variable be the generational one?

In the paper, rank-comparison stops always compare the *halted* rank
against a declared bound (`i ≥ 5`, `k ≡ |V|` with `◇_k`).
- (a) In your `i ≥ K`, is `K` a declared shape/bound (→
  `RankConstantShapeSym`), or a runtime tensor scalar (→ would need
  `TensorProjectionValue`)?
- (b) Should the validator require that any bare rank variable inside a
  `Comparison` predicate be either the rank named in the diamond
  subscript or another in-scope generational rank (legal in nested
  cascades, cf. `◇_j` inside an `i`-cascade in connected components)?

## Q5 — Generation-substitution / timing convention for `D_{i+1} ≡ D_i`-style predicates

The new `Comparison` node doesn't itself record "which side is the new
generation"; that's implicit in the `i+1` vs `i` rank-expressions plus
the lowering/interpreter contract. And the paper defines the diamond as
an *exclusion guard* (`RV_i = { i | … ∧ ¬◇(i) }`,
`tex/6-syntax.tex:88-91`), not a `break`. Can you confirm the intended
interpreter contract: predicate is evaluated as `Bexpr(i)` after
generation `i+1` has been produced, and a true result excludes
generation `i` (≡ "stop") — so the structural IR's job is just to carry
the predicate, and exclusion-guard semantics live entirely in the
interpreter?

---

## Supporting context from the review (not questions, just FYI)

- `Comparison` is not an invention of substance — it's the form the
  paper *actually uses* in 100% of its concrete stopping conditions
  (`tex/7-examples.tex:125,165,213,255,299,384` for `D_{i+1}≡D_i`;
  `:296,382` and `appendix/2-bf.tex:131` for `k≡|V|`;
  `tex/bfs-example/03-general.tex:108` for `i≥5`;
  `tex/bfs-example/07-iteration.tex:70,121,164` and
  `tex/7-examples.tex:461,506` for `||·||≡0`). The EBNF is just stale.
- `occupancy` is a real paper term — "the total number of non-zeros
  (nnz) in the tensor" (`tex/2-prelims.tex:48,150`), and `||·||` is its
  notation (`tex/bfs-example/07-iteration.tex:76`). It is the *only*
  tensor-→-scalar property used inside any `◇:` in the whole paper, so
  the closed `PropertyApp` set `{"occupancy"}` matches the paper
  exactly. (`|V|` in `k ≡ |V|` is a *shape*, not occupancy — different
  thing, routed through `RankConstantShapeSym`.)
- The "all points satisfy" lifting for `Comparison` is the only reading
  the paper's examples support — every multi-element comparison in a
  `◇:` is a convergence/fixpoint test ("no source can improve any
  distance", `tex/7-examples.tex:365`; "no more parents are changing",
  `:573`), which is inherently universal.
- `ComparisonOp` includes `!=, >, <, <=` even though the paper only
  ever uses `≡` (→ `==`) and `≥` (→ `>=`) in stopping conditions — a
  harmless generalization, but the validator/interpreter author should
  know the IR goes a bit beyond the paper there.
- FuseMax is not discussed anywhere in the paper; the "FuseMax-ish"
  `i ≥ K` example's only paper analogues are the iteration caps `i ≥ 5`
  and `k ≡ |V|`.
