# Case Statements in EDGE: Patterns and Desugarings

> *Created with the help of Claude Code.*


A reference for the different kinds of case-statement patterns and how each desugars. The central observation: case statements are not a single thing. They are a syntactic umbrella over semantically distinct patterns. A single `CaseExpression` IR node may be appropriate, but downstream lowering (or algebraic transformation) must distinguish the cases by what their predicates depend on.

This file collects the patterns to make sure each one is handled when the IR design is finalized.

**Note on the `<<` operator in the desugarings below.** Every desugaring in this document uses the `<<` shorthand notation for readability. The `<<` is itself shorthand (per Appendix B.6 of the paper) for a full Map action with `<<` as the compute operator and union as the merge operator. So `Z_{m,n} = T1_{m,n} << T2_{m,n}` further desugars to `Z_{m,n} = T1_{m,n} · T2_{m,n} :: ⋀ << (∪)` in the IR. See `parsing.md` for the full expansion rule. The desugared forms shown below are written using `<<` shorthand for clarity; the actual IR carries the fully-expanded Map form.

## Pattern 1: Predicate over rank variables only (coordinate-level)

**Paper reference**: Equation 61 in Section 5.9, desugared in Equation 155 in Section 7.3.8.

**Input form**:

```
Z_{m,n} = { A_{m,n}                                  if n ≠ m
          { B_{k,m} · C_{k,n} :: ⋀_k ×(∩) ⋁_k +(∪)   otherwise
```

**Predicate kind**: `n ≠ m`. Both `n` and `m` are iteration-space rank variables. The predicate can be evaluated at iteration-space construction time, before reading any tensor data.

**Desugaring (per the paper)**:

```
T1_{m,n}      = B_{k,m} · C_{k,n} :: ⋀_k ×(∩) ⋁_k +(∪)
T2_{m,n:n≠m}  = A_{m,n}
Z_{m,n}       = T1_{m,n} << T2_{m,n}
```

The condition `n ≠ m` becomes a *LHS coordinate restriction* on T2. T2 is only defined at points where the predicate holds; elsewhere it is empty. The final overlay (`<<`) puts T1 (the otherwise branch) underneath, with T2 on top where it is defined.

**IR feature needed**: LHS coordinate restrictions, i.e., a way to attach a rank-variable predicate to the LHS of an Einsum. This is a pending IR feature (see `session_todo.md`).

## Pattern 2: Predicate over tensor values (data-level)

**Paper reference**: Section 5.9 says "EDGE allows any Boolean predicate as conditions for each case." This pattern is implied but not given a worked example in the desugaring section.

**Input form** (illustrative):

```
Z_{m,n} = { A_{m,n}                       if X_{m,n} > 0
          { B_{m,n} · C_{m,n} :: ⋀ ×(∩)   otherwise
```

**Predicate kind**: `X_{m,n} > 0`. The truth of the predicate depends on the value stored in tensor X at coordinate `(m,n)`. This cannot be evaluated at iteration-space construction time; it requires reading X at execution time.

**Desugaring (proposed)**:

```
Mask_{m,n}  = X_{m,n} · 0 :: ⋀ >(∩)              # Mask is non-empty where X > 0
T2_{m,n}    = A_{m,n} · Mask_{m,n} :: ⋀ ←(∩)     # T2 = A wherever Mask is non-empty
T1_{m,n}    = B_{m,n} · C_{m,n} :: ⋀ ×(∩)
Z_{m,n}     = T1_{m,n} << T2_{m,n}
```

The condition `X_{m,n} > 0` desugars to a *mask tensor* (Mask) built by a compute-and-merge step. The mask is then used to gate where T2 is defined via intersection merge. No LHS restriction is used.

**IR feature needed**: `>` (and other comparison operators) as Map compute operators producing Boolean values. This is in `session_todo.md` as "Comparison and equality operators as compute ops."

## Pattern 3: Predicate combining rank variables and tensor values

**Paper reference**: not given a worked example. Implied by Section 5.9's "any Boolean predicate."

**Input form** (illustrative):

```
Z_{m,n} = { A_{m,n}                       if n > m and X_{m,n} > 0
          { B_{m,n} · C_{m,n} :: ⋀ ×(∩)   otherwise
```

**Predicate kind**: a conjunction of a coordinate-level predicate (`n > m`) and a data-level predicate (`X_{m,n} > 0`).

**Desugaring (proposed)**:

```
Mask_{m,n}      = X_{m,n} · 0 :: ⋀ >(∩)            # data-level mask
T2_{m,n:n>m}    = A_{m,n} · Mask_{m,n} :: ⋀ ←(∩)   # restricted to n>m AND masked by X>0
T1_{m,n}        = B_{m,n} · C_{m,n} :: ⋀ ×(∩)
Z_{m,n}         = T1_{m,n} << T2_{m,n}
```

The coordinate-level part of the conjunction becomes an LHS restriction; the data-level part becomes a mask gate. Disjunctions and negations would compose similarly but require some predicate-normalization (push negations inward, distribute conjunctions over disjunctions) at the desugaring step.

**IR features needed**: both LHS coordinate restrictions and comparison-as-compute-op.

## Pattern 4: Multiple branches with non-overlapping conditions

**Paper reference**: not given a worked example, but the case-statement grammar allows it.

**Input form** (illustrative):

```
Z_{m} = { 0    if m = 0
        { |V|  if m = s
        { 1    otherwise
```

This is the kind of construct that appears in the maxflow initialization: `D_{0,u} = 0`, `D_{0,u:u=s} = |V|`, with the `otherwise` being implicit (the empty value).

**Desugaring (proposed)**:

```
T1_{m}      = 1                  # the otherwise branch (constant tensor)
T2_{m:m=0}  = 0                  # branch 1: m=0
T3_{m:m=s}  = |V|                # branch 2: m=s
Tmp_{m}     = T1_{m} << T2_{m}   # overlay branch 1 onto otherwise
Z_{m}       = Tmp_{m} << T3_{m}  # overlay branch 2 onto previous result
```

Three things to notice:

1. The overlay order matters when branches *could* overlap. The desugaring fixes an order (here, later branches overwrite earlier ones). For the maxflow case the conditions `m=0` and `m=s` are mutually exclusive (assuming `s ≠ 0`), so order does not affect the result; but the IR must commit to an order somehow.
2. Constant tensors are scalars (zero-rank tensors broadcast across the iteration space) per the scalar-IR design (see `scalar_options_summary.md`).
3. Branches with `otherwise` (no predicate) form the base; each predicated branch is overlaid on top.

**IR features needed**: LHS coordinate restrictions, scalars in expressions, and a documented overlay-order convention.

## Pattern 5: Case statement on the LHS rank-variable expressions

**Paper reference**: Section 5.10 (Rank Variables as Tensors) and rank-mapping functions, Section 7.3.5.

**Input form** (illustrative, from connected components):

```
Z_f = X_a · Y_w :: ...
where f = min(a, w)
```

This is not a case statement at the surface level but is conceptually related: the output rank `f` is a *function* of two iteration variables `a` and `w`, and the function (`min`) is a conditional (`if a < w then a else w`).

**Why this matters**: a parser must not confuse a rank-mapping function with a case statement. They have similar conditional semantics but live at different levels of the IR. Rank-mapping functions appear in tensor projections (in `RankExpression`); case statements appear in Einsum expressions.

**No new desugaring**: rank-mapping functions are already supported by `RankFunction` in the IR. The case-statement framing is just an alternate way of thinking about them.

## Implications for IR design

These patterns suggest several non-negotiable IR features:

1. **LHS coordinate restrictions** (Patterns 1, 3, 4). Required for any predicate that mentions only rank variables.
2. **Comparison operators as Map compute ops** (Patterns 2, 3, 4). Required for any predicate that mentions tensor values.
3. **Scalar literals in expressions** (Pattern 4). Required for predicate-restricted constants.
4. **The `<<` update operator as a built-in compute op** (all patterns). Required for the overlay step at the end of every case-statement desugaring.

These patterns also suggest a design question:

**Should case statements be preserved in the IR or desugared at parsing?**

Arguments for preserving (a single `CaseExpression` node):

- Algebraic transformations on case statements are cleaner over the cased form than over the desugared form. A rewrite rule like "case statements with disjoint conditions can be fused" is a one-line pattern match over `CaseExpression`, but the same rule over the desugared form requires reconstructing the case structure from `<<`-chains and mask gates.
- Different backends may want different lowerings. An MLIR backend may emit a branched loop; an interpreter may evaluate predicates lazily; a TeAAL backend may want fiber-level masking. A desugared form commits to one lowering before reaching backends.
- Source-level intent is preserved. Error messages and debug output can point back to the original case structure.

Arguments for desugaring (no `CaseExpression` node):

- Smaller IR surface. Fewer node kinds for every consumer to handle.
- Forces canonicalization: there is exactly one way to encode each case statement (the desugared form). No ambiguity.
- The desugarings above are well-defined; there is no semantic information lost by performing them at parse time.

The middle path:

- The IR has a `CaseExpression` node.
- A separate canonicalization pass (a tree transformation on the IR) can desugar case statements when a downstream consumer requires it.
- Backends that prefer to work on the cased form (egglog for rewrites, MLIR for branched-loop emission) consume the IR directly.
- Backends that prefer the desugared form (the basic interpreter, possibly) run the canonicalization pass first.

This is consistent with the "core IR is the contract" framing while allowing the IR to carry richer structure than a strictly desugared form would.

## Open items

- Decide whether `CaseExpression` is a first-class IR node or a parser-only artifact.
- If first-class: design the node shape. Branches with conditions, an optional `otherwise` branch, predicate forms (rank-variable expressions, tensor-value expressions, or both).
- If not first-class: confirm the parser handles all four patterns above, with tests that verify the desugared forms match the patterns documented here.
- The "overlay order" convention for multi-branch case statements (Pattern 4) must be documented either way.