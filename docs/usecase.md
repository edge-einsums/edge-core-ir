# Use cases

> *Created with the help of Claude Code.*

A running list of concrete example use cases for IR features. Each entry shows what the user would write at the EDGE source level and what the IR has to be able to carry. Use this when designing a new feature so the IR shape is checked against real examples, not just imagined ones.

When a feature lands, the "Lowered IR" lines below should be filled in with the actual IR snippet that gets produced.

---

## Restricted-iteration predicates (in design)

The `<rank-var> : <predicate>` syntax. Restricts which coordinates an einsum iterates over. The predicate must reference rank variables that already exist in the einsum's iteration space — the predicate doesn't introduce new variables.

### Use case 1 — BFS init: set membership

Set F at the source vertices to 0; leave the rest empty.

**EDGE source:**

```
F_{0, s : s ∈ id} = 0
```

**Meaning:** Iterate `s` only over coordinates where `s ∈ id`. `id` is a coordinate set the host supplies at runtime — same kind of thing as `CoordSetEnum` / `CoordSetInterval` today, except its contents are bound when the program runs rather than declared statically.

**Lowered IR (to be filled in when the feature lands):**
- A `SetMembership(member=RankVariable(name="s"), coord_set=CoordSetName(name="id"), negated=False)` predicate node.
- `CoordSetName` is a new variant of `CoordinateSet` (sibling of `CoordSetEnum` / `CoordSetInterval` / `CoordSetAlias`) for host-supplied sets — the IR carries the name; the runtime supplies the contents. Same model as tensor names.
- The predicate goes on `Einsum.predicates`. The parser strips it from wherever it appeared in the source.

### Use case 2 — Lower-triangular L matrix: rank-var vs rank-var

Build a matrix that's nonzero only when `s < d`.

**EDGE source:**

```
L_{s : s < d, d} = G_{s, d}
```

**Meaning:** Iterate the `(s, d)` pairs but skip the ones where `s ≥ d`. Both `s` and `d` are output rank variables and both appear in the input projection `G_{s, d}` — already in scope.

**Lowered IR (to be filled in):**
- A `PredicateComparison(lhs=RankVariable(name="s"), op="<", rhs=RankVariable(name="d"))` predicate on `Einsum.predicates`.

### Use case 3 — Set non-membership

Update tensor at every non-source vertex.

**EDGE source (hypothetical):**

```
F_{0, s : s ∉ id} = ∞
```

**Meaning:** The complement of use case 1 — every `s` that is NOT in the source set.

**Lowered IR (to be filled in):**
- `SetMembership(member=RankVariable(name="s"), coord_set=CoordSetName(name="id"), negated=True)`.

### Use case 4 — Multi-clause: AND of two conditions

Filter on two conditions at once.

**EDGE source (hypothetical, paper doesn't have a canonical short example):**

```
F_{0, s : s ∈ id ∧ s > 0} = 0
```

**Meaning:** Iterate `s` only over coordinates that are both in `id` AND strictly positive.

**Lowered IR (to be filled in):**
- A `LogicalAnd(operands=[SetMembership(member=RankVariable("s"), coord_set=CoordSetName("id")), PredicateComparison(lhs=RankVariable("s"), op=">", rhs=RankConstantLiteral(value=0))])` composition node on `Einsum.predicates`.
- Same shape for `LogicalOr`; `LogicalNot` wraps a single predicate.

### Use case 5 — REJECTED: tensor-projection LHS in a predicate comparison

Earlier draft considered allowing predicates like `T_u == True` to restrict iteration based on a tensor's value at the rank variable. **Rejected.** Predicates restrict iteration based on coordinate-valued expressions only — rank variables, rank constants, shape symbols, rank arithmetic. Using a tensor's value to restrict iteration conflates two concerns (the iteration space and the value space) that EDGE keeps separate.

If you want to restrict iteration to coordinates where `T_u == True`, declare `T` as a coordinate set (or as a bool-typed tensor and reference its T-coordinate-where-True set by name) and use `u ∈ T` as a `SetMembership` predicate instead.

### Use case 6 — Predicate referencing a rank variable that lives only in an input

A rank variable that's in an input projection but not in the output, used in a predicate.

**EDGE source:**

```
Out_u = G_{u, v : v ∈ id}
```

**Meaning:** `v` appears in the input projection `G_{u, v}` but NOT in the output `Out_u` (it gets reduced away). The predicate `v ∈ id` restricts which `v` coordinates the einsum walks.

**Lowered IR (to be filled in):**
- `SetMembership(member=RankVariable(name="v"), coord_set=CoordSetName(name="id"))` on `Einsum.predicates`.
- The IR doesn't track where in the source the predicate appeared — the parser strips it from the input-projection position and lifts it to the einsum level. Same IR shape as use case 1; only the rank variable name differs.

### Use case 7 — Range as set membership

A rank variable confined to an integer interval.

**EDGE source (hypothetical):**

```
F_{i, s : 0 ≤ s < 5} = G_{i, s}
```

**Meaning:** Iterate `s` only over the half-open interval `[0, 5)`. The range IS a coordinate set (an interval), so this is set-membership with an inline `CoordSetInterval` instead of a named set.

**Lowered IR (to be filled in):**
- `SetMembership(member=RankVariable(name="s"), coord_set=CoordSetInterval(lo=0, hi=5))`.
- Reuses the existing `CoordSetInterval` machinery in `edge_ir/ir/tensor.py`. See `docs/parsing.md` § Range syntax for how the parser converts open/closed/one-sided ranges into half-open intervals.
