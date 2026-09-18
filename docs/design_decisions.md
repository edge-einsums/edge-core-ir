# EDGE Core IR — Design Decisions Log

> *Created with the help of Claude Code.*

A running record of design decisions made for the EDGE core IR, with the rationale for each. New decisions go at the bottom. When a decision is revisited or reversed, mark the old entry as superseded and add a new entry.

Each entry has:
- **Decision**: what was decided.
- **Rationale**: why.
- **Alternatives considered**: what else was on the table and why it was rejected.
- **Status**: active, superseded, or open.

---

## D1. ReduceSpec carries no `rank_list`

**Decision**: `MapSpec` and `ReduceSpec` do not carry a `rank_list` field. Only `PopulateSpec` does.

**Rationale**: The reduction set is determined entirely by the iteration spec: any rank variable that appears on the RHS but not the LHS is reduced over. Adding a `rank_list` to `ReduceSpec` duplicates information already present in the iteration spec. The Reduce action is one-per-Einsum and its behavior is fully specified by `(compute_op, merge_op)`; the iteration spec tells you which ranks the reduction folds over. `PopulateSpec` is different because the `rank_list` names the mutable fiber, which is not derivable from the iteration spec.

**Alternatives considered**: Keep `rank_list` on `ReduceSpec` for explicitness. Rejected because it allowed inconsistent IR where the listed ranks didn't match the iteration spec's implied reduction set.

**Status**: **superseded by D10**. The "always derivable from the iteration spec" premise does not hold when the output RVE is opaque (e.g. a user-defined `f(a, w)` whose mapping is not structurally analyzable). Map and Reduce now keep an optional `rank_list`.

---

## D2. ReduceSpec is always present in the IR

**Decision**: Every Einsum carries a `ReduceSpec` (and `PopulateSpec`) in its canonical IR form, even when the user omits them in the input. Defaults are filled by lowering. The interpreter is free to bypass execution when the spec is a no-op.

**Rationale**: Makes the IR shape uniform. Backends can rely on every Einsum having a complete `(MapSpec?, ReduceSpec, PopulateSpec?)` structure. Defaulting happens once, at lowering, and is the same for all consumers. The interpreter can recognize degenerate specs (pass-through compute, pass-through merge) and skip work.

**Alternatives considered**: Optional specs that match what the user wrote. Rejected because it forces every consumer to handle the "this spec is absent vs. defaulted" case separately.

**Status**: **open** — not currently enforced. `Einsum.specs` is `list[ComputationSpec]` ([einsum.py:84](../edge_ir/ir/einsum.py#L84)) with no structural requirement that a `ReduceSpec`/`PopulateSpec` is present. The BFS example has einsums with no explicit `PopulateSpec` (implicit default assignment). Decide whether to enforce this at lowering (and how — fill-by-default in the parser, or a Pydantic validator that rejects an Einsum without one) or drop the decision.

---

## D3. The core IR is the desugared form of EDGE

**Decision**: The IR represents the canonical, fully-desugared form of an EDGE program. Shorthand and sugar in the input form do not survive lowering. The IR is the contract between frontends (parsers, LLM tooling, embedded DSLs) and backends (interpreter, egglog, MLIR, Loops, GraphBLAST).

**Note — the core IR is NOT the AST.** A conventional AST is pre-desugaring; the core IR is post-desugaring. The parser produces an AST (pre-lowering, sugar preserved, source spans attached); lowering transforms that AST into the core IR. They are two distinct artifacts. Call these Pydantic models "the IR" or "the core IR" — not "the AST." If you need to be precise: *"the core IR is what the parser's AST is lowered to."*

**Why a separate AST and not just parse-into-IR-directly?** Four things the AST buys that a parser-with-action-rules would lose:
1. **Source-faithful error messages** — errors point at what the user typed, not at the canonical form they didn't write.
2. **IDE / LSP tooling** — completions, hover, go-to-definition need the AST (with sugar and source layout), not the desugared IR.
3. **Source-preserving formatters / refactors** — a formatter needs the AST so it doesn't normalize the user's sugar on save.
4. **Decoupled grammar evolution** — new surface sugar lowers to existing IR shape; backends and the validator don't have to know.

(Source spans themselves don't live on the IR — see D16.)

**Rationale**: Multiple consumers (interpreter, rewriter, downstream IR lowering) need a stable, unambiguous representation. If different frontends emit different IR for the same program, the contract collapses. Lowering normalizes once; everything downstream sees the same shape.

**Alternatives considered**: Preserve sugar in the IR with a `was_defaulted` or `original_form` flag. Rejected because no backend needed the information and it complicated every consumer.

**Status**: active. See `parsing.md` for the running list of shorthand-to-canonical transformations.

---

## D4. StoppingCondition is a structured tree, not an opaque name

**Decision**: `StoppingCondition` carries a `rank_variable: str` and a `predicate: Predicate`. The `Predicate` is a discriminated union of `Comparison | DiamondBooleanApp`. `Comparison` has `lhs`, `op`, `rhs` where each side is a `ValueExpression`. `ValueExpression` is a discriminated union of `TensorProjectionValue | ScalarValue | RankExpressionValue | PropertyApp`.

> **Updated by D21 (2026-05):** `ScalarValue` has been removed from `ValueExpression`. The union is now 3 variants: `TensorProjectionValue | RankExpressionValue | PropertyApp`. Scalars in comparisons are represented as `TensorProjectionValue` referencing a synthesized zero-rank `TensorDeclaration` whose `value` field carries the literal. See D21 for the full rationale.

**Rationale**: The original `StoppingCondition(rank_variable, boolean_function: str)` was a placeholder. It could not express the predicates the paper actually uses (`X_{i+1} ≡ X_i`, `occupancy(F_{i+1}) ≡ 0`, `i ≥ K`, `F_{i+3} ≡ F_i`). Structuring the predicate as a tree makes these expressible, supports algebraic manipulation by downstream tools (egglog), and gives Layer 2 validation real structure to type-check.

**Alternatives considered**:
- *Pre-registered enum of stopping condition kinds.* Rejected as inflexible; the paper's examples already use many different shapes.
- *Single function-application node serving both predicate-level and value-level roles.* Rejected because the same structural shape plays semantically different roles in the two positions (Boolean-returning at predicate level, any-returning inside a Comparison). Named them `DiamondBooleanApp` and `PropertyApp` respectively to keep the roles distinct.

**Status**: active. Implementation plan in `stopping_condition_plan.md`.

---

## D5. `PropertyApp.name` is a closed Literal type

**Decision**: `PropertyApp.name` is restricted to a closed set of built-in property names. Currently the set is `{"occupancy"}`. Extending requires adding to the Literal type definition. There are no user-defined properties.

**Rationale**: Properties of a cascade (occupancy, eventually max, min, sum) are well-defined operations the interpreter has to special-case anyway. Allowing arbitrary string names would force every consumer to handle "unknown property" cases, and most properties have well-defined semantics that should live in the IR registry rather than in user code.

**Alternatives considered**: Allow user-defined properties via a `BuiltinPropertyApp | UserDefinedPropertyApp` split. Rejected for v0; can be added later if the use case appears.

**Status**: active.

---

## D6. `DiamondBooleanApp.name` has no built-ins

**Decision**: `DiamondBooleanApp` is always a user-defined function call. There are no built-in names. The name field uses the standard user-defined-identifier pattern.

**Rationale**: A built-in `DiamondBooleanApp` would essentially be a built-in stopping condition, and the paper's stopping conditions are all expressible as `Comparison` predicates. The user-defined case is the only one that needs the `DiamondBooleanApp` variant; built-ins go through `Comparison + PropertyApp`.

**Status**: active.

---

## D7. `pinned_rank_variables` always includes the enclosing rank

**Decision**: Both `PropertyApp` and `DiamondBooleanApp` carry a `pinned_rank_variables: list[str]` field. The user supplies only the extra fixed ranks in the EDGE input (e.g., the `s` in `occupancy_s(F)`); lowering prepends the enclosing `StoppingCondition.rank_variable` to the list. The IR always carries the full set.

**Rationale**: The enclosing rank is always pinned implicitly because that is the rank the stopping condition halts. Making it explicit in the IR removes a context-dependent inference that downstream tools would otherwise have to repeat. Lowering does the prepending once.

**Alternatives considered**: Have Pydantic infer the enclosing rank at construction time. Rejected because Pydantic models do not know about their parent context, and adding parent-context inference would make the IR an active rather than passive data structure.

**Status**: active.

---

## D8. ValueExpression uses wrapper classes for discriminator dispatch

**Decision**: The `ValueExpression` union uses small wrapper classes (`TensorProjectionValue`, `ScalarValue`, `RankExpressionValue`) to give each variant a `kind` discriminator. `PropertyApp` already has its own `kind`.

> **Updated by D21 (2026-05):** `ScalarValue` is no longer one of the wrapper classes — it has been removed from the codebase. The wrapper-class pattern still applies to the remaining `ValueExpression` variants (`TensorProjectionValue`, `RankExpressionValue`) and to `InputTensor` in the `Expression` union.

**Rationale**: Matches the existing precedent in `expr.py`, where `InputTensor` is a wrapper around `TensorProjection` to make it discriminable in the `Expression` union. Avoids modifying `TensorProjection` and `RankExpression` (which are reused in other contexts) to add a discriminator they don't otherwise need.

**Alternatives considered**:
- *Smart union mode (no discriminator).* Rejected for safety: smart-union mode resolves ambiguity by "first variant that validates," which can mask bugs.
- *Add `kind` directly to `TensorProjection`.* Rejected because it would break existing IR JSON without a corresponding benefit.

**Status**: active.

---

## D9. Specs live only at the outer Einsum, not on AnonymousTensor

**Decision**: All `ComputationSpec`s live at the outer `Einsum` level. `AnonymousTensor` carries no `specs` field. Labels are flat within an Einsum: every binary in the entire expression tree (including nested anonymous tensors) has a unique label in a single global namespace. There is no label scoping.

**Rationale**: Matches the paper's Equation 41, where all specs are at the outer Einsum level and labels distinguish binaries across the whole expression. A single namespace makes label-coverage validation simple: every spec's label has exactly one binary, every binary's label has matching specs. Allowing specs on `AnonymousTensor` introduces two questions (which node holds them, what scope do their labels live in) with no benefit; the paper's model answers neither and the IR shouldn't either.

**Alternatives considered**:
- *Scoped labels (each AnonymousTensor introduces a new namespace).* Rejected because it diverges from the paper, requires parsers to track scope, and adds no expressive power.
- *Keep specs field on AnonymousTensor but require it to be empty.* Rejected because the type system cannot enforce "must be empty" cleanly, and the field becomes dead weight.

**Status**: active. Implementation plan in `anonymous_tensor_specs_fix_plan.md`.

---

## D10. Map/Reduce `rank_list` stays optional (validator hints; not always derivable)

**Decision**: `MapSpec.rank_list` and `ReduceSpec.rank_list` stay as `list[str] | None = None` — optional, can be omitted, can be empty. NOT removed. Supersedes D1.

**Rationale**: D1's premise was that the reduce set is always derivable from the iteration spec. That holds for *basic* reduces, but it does NOT always hold: an opaque user-defined output RVE (e.g. `f(a, w)` from connected-components-style mappings) can route iteration points to output coordinates in a way that's not statically derivable. In those cases the explicit `rank_list` is the authoritative source. For basic reduces, the user-supplied list is redundant with the derived set — and the Layer 2 validator can cross-check the two and hint on mismatch, catching typos and malformed reduces.

**Alternatives considered**: Remove the field entirely (D1). Rejected — see rationale above. Also tried implementing the removal and reverted after surfacing the opaque-RVE issue.

**Status**: active. `README.md` has a TODO noting the cross-check is deferred — it only applies to basic reduces; the opaque-RVE case needs handling first.

---

## D11. `PopulateSpec.rank_list` may be empty (default assignment)

**Decision**: `PopulateSpec.rank_list = Field(min_length=0)`. The field is still required (no default), but the list may be empty.

**Rationale**: An empty rank_list is the no-`*` **default-assignment** case — the output coordinate is taken from the iteration point, no mutable rank. It's the most common populate (all three BFS einsums use it implicitly). A non-empty list names the `*`-marked mutable ranks being populated.

**Alternatives considered**:
- *Keep `min_length=1`*. Rejected — default assignment is the most common populate; forcing a non-empty list is awkward.
- *Make the field fully optional (`list[str] | None`, defaulting to `None`)*. Rejected for now — requiring an explicit empty list matches the paper's surface form better.

**Status**: active.

---

## D12. `RankFunction.args` accept full rank-variable-expressions

**Decision**: `RankFunction.args` is `list[RankExpression]` — a function argument can be any RVE (variable, constant, structured arithmetic, or nested function), not just a bare rank variable.

**Rationale**: The paper's formal semantics defines an RVE as any function from the iteration space to a coordinate, so function inputs are themselves RVEs. The published EBNF lists args as a `<rank-variable-list>` (bare variables only), but it's narrower than the semantics — bare-constant subscripts (`|V|`, `0`) and arithmetic RVEs (`p+s`, `i+1`) appear in the paper's own worked examples (BFS, convolution). The IR was already extending the grammar by accepting `RankArith` in other positions; this aligns the function-arg case with that. Formal semantics canonical.

**Alternatives considered**:
- *Bare `RankVariable` only* (the grammar's literal reading). Rejected — too narrow; can't encode published examples.
- *Just `RankVariable | RankConstant`*. Rejected — inconsistent with already accepting `RankArith` in other RVE positions.

**Status**: active.

---

## D13. `TensorDeclaration.empty_value` is required but nullable

**Decision**: `empty_value` stays a required field (`empty_value: Any`, no default). It MAY be `null`/`None` (an explicit "empty is null"), but it cannot be omitted.

**Rationale**: Requiring an explicit `empty_value` matches the paper, where every declaration carries `empty=...`. `None` is a valid empty value for any datatype natively — JSON `null` ↔ Python `None` round-trips for any type, so no sentinel is needed (unlike `inf`/`-inf`/`nan`, which JSON cannot represent).

**Alternatives considered**: Make the field optional (`empty_value: Any = None`, where a missing value defaults to null). Rejected — conflates "I didn't specify one" with "the empty value is deliberately null"; the required form forces an explicit choice. The by-omission alternative is recorded as a commented-out line at the field in `edge_ir/ir/tensor.py` in case we change our minds.

**Status**: active.

---

## D14. Core IR is syntactic; lowering is per-backend, but the derivations are shared

**Decision**: The core IR (Pydantic models in `edge_ir/ir/`) is the stable contract. Turning it into something executable — resolved iteration spaces, integer coordinate maps, looked-up UDFs — is "lowering," which splits in two:

- **Derivations** (the *logic* that computes facts about an einsum: iteration space, which ranks a reduce collapses, affineness). Pinned by the paper; computed once in a backend-neutral analysis layer (`edge_ir/analysis/`) that returns plain Python values.
- **Data structures** (the actual runtime objects a backend builds). These differ per backend, so each backend owns its own. The interpreter is the first/example backend.

The analysis layer imports from `edge_ir/ir/`, never the reverse.

**Rationale**: Runtime data structures genuinely differ across backends (interpreter wants Python dicts/sets/callables; future backends would want their own shapes), so sharing them would be a premature abstraction. But the derivations do NOT differ — if two consumers each re-derive "which ranks does this reduce collapse" from prose, they can drift and silently disagree on the meaning of the same program. Cheap insurance: write each derivation once and have everyone call it. The "second consumer" trigger is already met in-repo: the Layer-2 validator's `rank_list` cross-check and the evaluator's default-resolution pass both need the same derivation. The reusable asset is the derivation logic captured as **tests** (golden tests against hand-computed paper traces), not as a prose lowering doc that can drift.

**Alternatives considered**:
- *Build a shared semantic IR now.* Rejected — premature for backends that don't exist yet and demonstrably want different shapes.
- *Each backend duplicates derivations.* Rejected — drift risk; same program, different meanings across backends.

**Status**: active.

---

## D15. Derived analysis results live in side tables, not on the core IR

**Decision**: Facts a pass *computes* about the IR (e.g. whether a `RankArith` is affine) are NOT stored on the core IR nodes. Each analysis pass returns a side table — a dict keyed by node (`id()` in-memory; a structural path if it ever has to be serialized) mapping node → result — living in `edge_ir/analysis/`. The core IR stays frozen and pure-syntax and does not know the analysis exists.

**Exception — user-declared facts stay on the node.** `RankFunction.is_affine` is *input* (the user supplies it, since the function is opaque), not derived, so it stays on the node. Only *derived* facts move to side tables. (Open: rename it to `declared_affine` to mark the difference — tracked in `edge_ir/validator/README.md`.)

**Rationale**:
- The IR is frozen, so "fill it in" really means rebuild the node and every node above it, producing a second tree you must rewire every consumer to.
- The field would otherwise ship in the serialized contract (`is_affine: null` on every un-analyzed program), and you can't tell from the JSON whether analysis ran.
- It's derived data sitting on its source — a cache that goes stale the moment the IR is rewritten (e.g. egglog). A side table is just recomputed on demand.

**Alternatives considered**:
- *Store on the node.* Rejected — see rationale.
- *Separate analyzed node type (typed-AST style).* Rejected — more machinery than the side table buys.

**Status**: active. Removed `is_affine` from `RankArith` (derived) into `edge_ir/analysis/affine.py`; kept on `RankFunction` (user input).

---

## D16. Source locations are a side map, not a field on IR nodes

**Decision**: Lowering returns `(IR, source_map: dict[id(ir_node), SourceSpan])`. The source map is consumed by the Layer-2 validator and the evaluator for error reporting. The core IR does NOT carry a `source_loc` field on any node.

**Rationale**: Same pattern as D15 (derived analysis results live in side tables). Source spans are *provenance metadata*, not part of the IR contract. Keeping them off the IR means: (a) the saved JSON stays clean (no `source_loc: null` shipping on every node before lowering); (b) two IRs that are structurally identical but came from different source positions compare equal; (c) the IR doesn't grow optional metadata fields. Synthesized nodes (nodes lowering invents — default `PopulateSpec`, default-resolved `rank_list`, flattened `NestedCascade`) simply aren't in the map. Error reporters walk up to the nearest ancestor that has an entry and report against that span.

**Alternatives considered**:
- *`source_loc: SourceLocation | None` field on `IRBase`* (previously sketched in `interpreter_future_plan.md`). Rejected — pollutes the IR contract with optional metadata that's never relevant to backends; same null-vs-real ambiguity that D15 rejected for `is_affine`.
- *Validator/evaluator errors by IR path only* ("Einsum #2, spec[1].rank_list[0]"), no source mapping. Rejected — painful UX once programs get nontrivial.

**Status**: active. Supersedes the earlier "add `source_loc` to `IRBase`" idea in the future plan.

---

## D17. Lowering errors are collectable, not fatal

**Decision**: `lower(ast)` returns `tuple[IR | None, list[LoweringError]]` — a best-effort IR (or `None` if structurally unrecoverable) plus an error list. Matches the Layer-2 validator's collectable-errors contract.

**Rationale**: Two error models — one fatal at lowering, one collectable at validation — would make it impossible for an LSP/UI to surface multiple problems in one pass. Same shape across phases keeps the error-reporting code uniform.

**Alternatives considered**: Fatal (`lower(ast) -> IR`, raises on first error). Rejected — fine for unit tests, painful for tooling.

**Status**: active.

---

## D18. AST/IR division of labor

**Decision**: A hard line between what each layer is allowed to do.
- The AST holds **only what the parser saw**. No defaults filled in. No inference. No derived fields. No name validation beyond what the parser already does.
- The AST module (`edge_ir/ast/`) does NOT import from `edge_ir/ir/`. One-way only: lowering imports both; the AST doesn't know the IR exists.
- New grammar production = new AST node + lowering rule for it, in the same change. CI should fail if a grammar production has no lowering target.

**Rationale**: AST and IR look almost the same structurally. Without an explicit rule, someone will be tempted to do "useful work" on the AST — fill in a default, compute a derived field, check a name. Each one moves a bit of IR-shaped logic into the AST. Do it enough times and the AST has quietly absorbed half the lowering pass, at which point someone proposes "let's just merge them since they're basically the same." This rule keeps the AST dumb on purpose so the separation survives.

**Alternatives considered**: No explicit policy, rely on review. Rejected — two-hierarchy designs without a written division-of-labor reliably collapse over time.

**Status**: active.

---

## D19. AST `model_dump_json()` excludes source spans by default

**Decision**: `SourceSpan` on AST nodes is excluded from `model_dump_json()` by default (`Field(exclude=True)`). Callers that want spans in the JSON pass `model_dump(exclude={})` or use a custom serializer.

**Rationale**: The AST is rarely serialized for interchange (the IR is the contract); spans are debugging metadata, not part of the AST's shape. Excluding by default keeps AST JSON small and the round-trip property clean (`parse(prettyprint_ast(ast)) == ast` doesn't need to recreate spans). Spans are also excluded from AST equality for the same reason — two ASTs with the same structure but different spans should compare equal.

**Status**: active.

---

## D20. `RankConstant` is split into Literal vs ShapeSym variants

**Decision**: Replace the single `RankConstant(value: int | str)` with two discriminated variants:

```python
class RankConstantLiteral(IRBase):
    """A literal coordinate value -- int OR a label from a declared
    char/string coord set."""

    kind: Literal["literal"] = "literal"
    value: int | str


class RankConstantShapeSym(IRBase):
    """A reference to a declared shape parameter (|V|, N, M, ...).
    The shape resolver substitutes the bound integer at
    iteration-space construction."""

    kind: Literal["shape_sym"] = "shape_sym"
    name: str
```

The `RankExpression` union expands accordingly:

```python
RankExpression = Annotated[
    RankVariable
    | RankConstantLiteral
    | RankConstantShapeSym
    | RankArith
    | RankFunction,
    Field(discriminator="kind"),
]
```

**Rationale**: The original `RankConstant(value: int | str)` overloaded two genuinely different things. A string value could be either (a) a literal label from a char-valued coordinate set (e.g. `CoordSetEnum(coords=["a","b","c"])`) — fully static, a value in the rank's coord space, OR (b) a shape-symbol reference like `|V|` that the shape resolver substitutes at iteration-space construction. A consumer reading the IR couldn't tell which. The split makes the distinction structural.

**Alternatives considered**:
- *Keep one `RankConstant` and infer from context (the rank's `CoordSetEnum` lists the label, or a shape declaration uses it).* Rejected — forces every consumer to do the context lookup, and the same string can be legal in either role.
- *Add an `is_shape_sym: bool` flag*. Rejected — same boolean-metadata-on-IR-node pattern D15 rejected.

**Future work**: Eventually we want to support user-defined `RankConstant` value types (beyond int / char-label) when a tensor's coord set uses non-builtin types. Held off for now. In the meantime, Pydantic must reject ill-typed values (e.g. a float for `RankConstantLiteral.value`) with a readable error. Verified today (a float against `int | str` produces a clear two-branch ValidationError listing both failures with the input value); a regression test pins this against future-me.

**Status**: active.

---

## D22. Runtime-vs-static categorization (when does this get resolved?)

**Decision**: Sort things in the IR into four buckets by the question *"when does this get resolved?"* These buckets are a mental model — not a runtime flag or a marker on the IR. Use them to figure out where new IR concepts belong and how analysis passes should treat them.

The buckets:

- **A. Pure structure** — known purely from the program text. Examples in the current IR: built-in operators (`BuiltinComputeOp`, `BuiltinMergeOp`, `BuiltinUnaryOp`, `BuiltinDataType`), rank variable names (`RankVariable.name`), integer literal coordinates (`RankConstantLiteral` with an integer value), the shape of a `RankArith` tree, the topology of an `Einsum`. Fully resolved at parse / lowering time.

- **B. Symbolic until iteration-space construction** — known once the user has bound their inputs, but before any iteration starts. Examples: shape symbols like `|V|`, `N`, `M` (`RankConstantShapeSym.name`, `RankDeclaration.shape="|V|"`, `CoordSetInterval.lo="|V|"`/`hi`), and `CoordSetAlias` (needs the aliased tensor's resolved rank). The shape resolver substitutes integers here, once per program run.

- **C. Reference to a runtime resource** — bound by a registry or actual data at execution setup. Examples: user-defined function names (`RankFunction.func_name`, `UserDefinedComputeOp.name`, `UserDefinedMergeOp.name`, `CoordinateOp.name`, `UserDefinedUnaryOp.name`, `UserDefinedDataType.name`), input tensor *data* (the *declaration* is bucket A; the actual values that fill it are bucket C), and named runtime sets like `id` in BFS.

- **D. Per iteration point at execution** — evaluated as the iteration walk happens. Examples: existence predicates (does this coordinate have a value in the actual data?), stopping-condition truth values, and the future restricted-iteration predicates (D23) like `s ∈ id`.

The IR does NOT mark these buckets with a flag, mixin, or wrapper class — each variant's class already partitions them. `BuiltinComputeOp` is bucket A; `UserDefinedComputeOp` is bucket C. `RankConstantLiteral` is bucket A; `RankConstantShapeSym` is bucket B. The split is structural, not metadata.

**Rationale**: The original question that motivated this was "how do I tell what's runtime vs not in the IR?" — and the original answer was sloppy because it conflated three different things into one "runtime" notion. The bucket split makes the actually-useful axis (*when does this get resolved*) explicit. Two specific observations the framework makes obvious:
- **B and C are both colloquially "runtime" but are very different phases.** B only needs the shape environment (a few integers). C needs the registry and the actual data — that's a much heavier binding step.
- **`TensorDeclaration` splits across A and C.** The declaration itself (name, ranks, dtype, empty value) is bucket A — pure structure. The actual values that fill the tensor at execution are bucket C. Don't say "tensors are runtime"; say "tensor declarations are A, tensor data is C."

**Alternatives considered**:
- *Add an `is_runtime_ref: bool` field on IR nodes*. Rejected — violates D15 (no metadata on IR nodes), ships as noise in JSON, and collapses the B/C/D distinction into a single flag that hides the actual axis (timing).
- *Three buckets instead of four (merge B and C as "before execution")*. Rejected — B is resolved by the shape resolver (a static, integer-only step); C needs a registry plus actual data (a different binding step). They run at different times and require different infrastructure.
- *Side table of runtime references (an analysis pass that enumerates them)*. Considered as a possible future extension if a second consumer needs the enumeration. Right now the consumers (validator, evaluator) can each just walk the IR — no need to factor it out yet, per the same trigger rule used in D14.

**Future**: When restricted-iteration predicates land (D23), they're bucket D. They live as their own node type (`IterationPredicate`), not as a `RankExpression` variant — predicates return true/false, rank expressions return coordinates, so they can't share a union (same shape-vs-role split as D4's `DiamondBooleanApp` vs `PropertyApp`).

**Status**: active.

---

## D21. Scalars are zero-rank tensors everywhere (no `ScalarLiteral` variant; `ScalarValue` removed)

**Decision**:

- Add `TensorDeclaration.value: Any | None = None` so a zero-rank `TensorDeclaration` carries its literal value at the declaration site. A `model_validator` enforces `value is not None ⇒ ranks == []`. `value` gets the same float-sentinel JSON encoding as `empty_value`.
- Delete `ScalarValue` from `edge_ir/ir/stopping.py`. The `ValueExpression` union shrinks from 4 → 3 variants. Comparisons reference scalars the same way expressions do — via `TensorProjectionValue` + a synthesized zero-rank `TensorDeclaration` whose `value` carries the literal.
- Relax `REGEX_TENSOR_NAME` to `^[A-Z][A-Za-z0-9_]*$` so parser-synthesized literal names (e.g. `Lit_0`, `Lit_c`) don't collide with user-declared tensor names.
- The AST→IR lowering pass is responsible for the synthesis. The parser does NOT deduplicate literal occurrences (see "No CSE at parser" below).

**Rationale**:

- One concept of "scalar" across the entire IR. Expressions, stopping conditions, and any future predicate world all reference scalars the same way — as zero-rank tensors. No special `ScalarLiteral` Expression variant; no separate `ScalarValue` in stopping.
- Matches the formal semantics: every operand in EDGE is a tensor; a scalar is the degenerate rank-0 case (paper `tex/2-prelims.tex:39`, broadcast prose at `tex/7-examples.tex:693`).
- The user-defined type (UDT) case falls out naturally — `TensorDeclaration` already carries `data_type`, so UDT scalars like `'c'` need no additional plumbing (no need to widen a `ScalarLiteral.value` field or introduce a parallel `data_type` slot).
- Keeps the parser dumb. The IR is a faithful syntactic representation; optimizations (CSE on declarations, constant folding) live in downstream passes.

**Alternatives considered**:

- *Add `ScalarLiteral` to the Expression union (Option B from the scalars design doc).* Rejected — creates a separate "scalar" concept that doesn't behave like a tensor (no iteration space; carries `value` without a `data_type`, requiring duplication of the data-type machinery `TensorDeclaration` already has). Also: `ScalarValue` in `stopping.py` was cited as a D8 precedent, but that class lives in *predicate position* (operand of a comparison) where there's no Map/Reduce/Populate semantics underneath — category error; not a real precedent for the expression world.
- *Parser synthesizes a `TensorDeclaration` plus an initialization Einsum that sets its value (Option A).* Rejected — chicken-and-egg: the init Einsum's RHS is itself a literal that needs the same lowering. The proposed UDF-returning-constant workaround requires a fake dependency on some existing tensor (because `UnaryApp.operand` is restricted to `InputTensor`), which lies about the UDF's actual semantics.
- *Overload `empty_value` to carry the scalar's value when `ranks == []` (Option E1).* Rejected. Breaks UDTs. Say you've got a char-UDT scalar literal `'c'`. Under E1 you'd set `empty_value='c'` because value rides on empty. But a normal user-declared char tensor probably already has `empty='\0'` — the UDT cares about the difference between "empty char" and "the character 'c'". E1 mashes those into one slot. Option C keeps `value` and `empty` separate, so UDT scalars and UDT tensors with real empties can both exist without colliding.

**Inheritance rule for synthesized literals** (parser-side, lives in `docs/parsing.md`):

- Source = output tensor when the literal is the RHS of an einsum assignment.
- Source = the other operand's tensor when the literal is one side of a comparison and the other side is a tensor projection.
- Source = the property's *return type* (not its underlying tensor) when the other side is a `PropertyApp`.
- `data_type` is inherited from the source.
- `empty_value` is **always `None`** for a synthesized literal. A scalar (zero-rank tensor) has no meaningful "empty" — there is one point in its coordinate space and it's either occupied (with the literal's `value`) or not; there are no unoccupied coordinates to default. The `None` applies *only* when synthesizing a literal; user-declared tensors still require an explicit `empty=`.
- Parser-synthesized names collision-detect against existing `Program.declarations`. If the candidate name collides with a user-declared tensor, the parser appends a numeric suffix (`Lit_0_1`, `Lit_0_2`, ...) until unique. The IR itself does not enforce uniqueness on `TensorDeclaration.name`; the Layer 2 validator does.

**No CSE at the parser level**: A program containing the literal `0` three times produces three synthesized declarations. The parser doesn't look across literal occurrences and doesn't merge with hand-authored constants (like max-flow's `Zero`) that happen to carry the same value. Common-subexpression elimination on tensor declarations is a downstream compiler optimization (MLIR constant-pooling, egglog equality saturation, or a future "core IR simplification" pass), and doing it at the parser would also throw away the source-position information each literal occurrence needs for error messages and tracing.

**Future work**:

- Predicates / LHS coordinate restrictions (e.g. `s : s ∈ id`) — needed to fully encode BFS's init block (`F_{0, s:s∈id} = 0`). Next on the design queue.
- Case statements (paper §`ssec:cases`) — orthogonal to predicates and scalars; on the queue after predicates.
- Runtime variable binding (the `id` set as a host-provided input) — open question; deferred.
- `tests/test_ir_stopping.py` and downstream consumers no longer dispatch on `"scalar_value"`. The JSON schema (`schemas/program.schema.json`) was regenerated to match.

**Paper-side**: requires adding `<scalar-literal>` to the `<expression>` production in the EBNF, plus a short formal-semantics paragraph about literal-as-broadcast-zero-rank-tensor.

**Status**: active.

---

## D22. `IterationPredicate` and `stopping.Predicate` are separate IR unions

**Decision**: The `IterationPredicate` union (in `edge_ir/ir/predicate.py`, used on `Einsum.predicates` to restrict iteration) and the `Predicate` union (in `edge_ir/ir/stopping.py`, used on `StoppingCondition.predicate` to halt a nested cascade) are kept as separate IR types. They are not unified into a single Boolean-predicate union.

**Rationale**: They serve different roles with different operand vocabularies, and unifying them would force one side to admit operands the other side doesn't want.

- `IterationPredicate` operates over *coordinate-valued* expressions (`RankExpression` and `CoordinateSet`). Its `PredicateComparison.lhs/rhs` are `RankExpression`. It does not admit tensor-projection LHS — a predicate restricts the iteration space, it doesn't peek at tensor values to decide. Variants: `SetMembership`, `PredicateComparison`, `LogicalAnd`, `LogicalOr`, `LogicalNot`.
- `stopping.Predicate` operates over `ValueExpression` (tensor projections, rank expressions, the built-in `PropertyApp` for things like `occupancy`). Its `Comparison.lhs/rhs` are `ValueExpression`. It admits tensor projections explicitly — stopping conditions like `‖F_{i+1}‖ ≡ 0` compare tensor properties to scalars. Variants: `Comparison`, `DiamondBooleanApp`.

A single union would either lose the structural rejection of tensor LHS in iteration predicates, or require a parameterized `BooleanCombinator[T]` whose every consumer has to special-case per `T` anyway.

**Shared piece**: both unions reuse the same `ComparisonOp` Literal (`"=="`, `"!="`, `">"`, `"<"`, `">="`, `"<="`) from `stopping.py`. That's a leaf-level reuse — the unions themselves stay separate. The comparison-operator alphabet is identical; the operand types differ.

**Alternatives considered**:
- *Single Boolean-predicate union over a generic operand type (`BooleanCombinator[T]`).* Rejected — Pydantic v2 supports generics but every downstream consumer (validator, evaluator, future MLIR backend, egglog rewriter) ends up special-casing per concrete `T` instantiation. The "DRY" win is shallow; the real shape of the work is the operand-type-specific machinery on each side.
- *Promote `LogicalAnd`/`LogicalOr`/`LogicalNot` to a shared composition layer that both unions wrap.* Rejected for v1 — the stopping side doesn't have explicit composition yet (the paper's stopping conditions are all single comparisons). If a future paper example needs `<>: P ∧ Q` as a stopping condition, that's the time to factor out the shared composition layer.

**Future work**: if stopping conditions gain composition (e.g. for paper examples not yet on the table), revisit and consider extracting `LogicalAnd`/`Or`/`Not` into a shared module that both predicate unions reuse. Until then, accept the parallel hierarchy as the cost of keeping each side's operand discipline clean.

**Status**: active.

---

## D23. Restricted-iteration predicates live on `Einsum.predicates` (with attached-at-einsum-level only)

**Decision**: An einsum can carry a predicate that restricts which iteration-space points it walks. The predicate is stored in a single slot on the einsum:

- `Einsum.predicates: list[IterationPredicate] = []` with a `model_validator(mode="after")` that rejects `len(predicates) > 1`. So the slot is "0 or 1 element" — multi-clause predicates wrap explicitly in `LogicalAnd`/`LogicalOr`/`LogicalNot`. One canonical IR shape per logical predicate.
- `IterationPredicate` is a new discriminated union in `edge_ir/ir/predicate.py` with five variants:
  - `SetMembership(member: RankExpression, coord_set: CoordinateSet, negated: bool = False)` — `s ∈ id` and `s ∉ id`.
  - `PredicateComparison(lhs: RankExpression, op: ComparisonOp, rhs: RankExpression)` — `s < d`, `s == 5`, etc. Reuses `ComparisonOp` from `stopping.py`.
  - `LogicalAnd(operands: list[IterationPredicate])` with `min_length=2`.
  - `LogicalOr(operands: list[IterationPredicate])` with `min_length=2`.
  - `LogicalNot(operand: IterationPredicate)`.
- `CoordinateSet` (in `tensor.py`) grew a fourth variant `CoordSetName(name: str)` for host-supplied coordinate sets — same convention as tensor names (the IR carries the name; the host runtime supplies the contents at execution time). Admissible in both `RankDeclaration.coord_set` and `SetMembership.coord_set`.

**Rationale**:

- **One attachment point, not per-position.** The paper's formal iteration-space constraint (`tex/semantics/2-functions.tex:451-453`) says "While such constraints may appear locally within tensor subscripts, they globally restrict the set of valid points in the iteration space." Per-position attachment would just be the same fact stored in two different IR places depending on where the user wrote it; collapsing to the einsum level keeps the representation unique. Parser lifts whatever the user wrote to `Einsum.predicates`.
- **No tensor-projection LHS in comparisons.** Predicates restrict iteration based on coordinates, not tensor values. The paper's `merge` operator (`∩`, etc.) is the canonical "filter by tensor value" mechanism — different machinery from predicates. Every paper example of a restricted-iteration predicate uses coordinate-valued operands only.
- **Predicate composition (AND/OR/NOT).** The formal section on iteration constraints is silent on composition; the prose for case statements (`tex/6-syntax.tex:415`) says "any Boolean predicate" and the cascade-rank-variable-set definition uses compound Boolean structure. Per project rule (prose stands when formal silent), AND/OR/NOT is admissible.
- **`CoordSetName` is not tagged "runtime."** Same convention the IR already uses for tensor names: declaring a tensor doesn't tag its data as "runtime"; we know it is by where it lives in the program. Same for named coord sets. No special-case machinery on the IR.

**Inheritance rules (parser-side, lives in `docs/parsing.md`)**:

- Rank variables referenced by a predicate must already appear somewhere in the einsum (output ranks or any input projection). Layer 2 validator enforces.
- Range syntax (`0 ≤ s < 5`, `s > 0`, etc.) lowers to `SetMembership(coord_set=CoordSetInterval(lo, hi))`. The parser converts all open/closed/one-sided forms to half-open `[lo, hi)` before the IR sees anything; one-sided ranges use the rank's declared shape for the implicit endpoint.

**Alternatives considered**:

- *Tensor-projection LHS in `PredicateComparison`* (e.g. `T_u == True`). Rejected — that's filter-by-tensor-value, which is what `merge` is for. Not predicate territory.
- *Per-position attachment* (predicate on output rank vs predicate on input projection). Rejected — paper says they're semantically equivalent; collapsing to one canonical slot avoids two-IR-shapes-for-one-meaning.
- *List as implicit-AND* (`Einsum.predicates = [p, q]` means `p ∧ q`). Rejected — would also admit `[LogicalAnd([p, q])]` as a second valid encoding of the same predicate. One slot, one shape.
- *General negation* of arbitrary predicates (`¬PredicateComparison(...)`). Out of scope — `!=` and `∉` cover the common cases; `LogicalNot` is available for compound predicates if needed.

**Iteration-space construction contract (interpreter-side, lives in `docs/interpreter_future_plan.md`)**: paper-canonical is "materialize the restricted set, then walk it" — the iteration space *is* the restricted set, not a filter applied during iteration. The interpreter is free to implement either materialize-then-walk or walk-and-skip as long as observable output matches; that's an interpreter performance choice, not an IR concern.

**Layer 2 validator obligations**: spec in `docs/validator_plan_predicates.md`. Eleven checks (P1–P11), covering rank-variable scope, coord-set name resolution, interval well-formedness, type compatibility, composition arity, depth bound (configurable, default 8), and the predicates-list-zero-or-one structural rule.

**Future work**:

- Case statements (paper `tex/6-syntax.tex:403-415`, `ssec:cases`). Predicates are a prerequisite (case statements lower to predicate-restricted einsums merged with `<<`); the predicate IR introduced here is the desugaring target.
- Single-input einsum semantics (`expression=InputTensor`, `specs=[]`). The paper's formal Map definition is for two-input einsums; we've quietly accepted single-input einsums in BFS init without a paper anchor. Either lower to a 2-input einsum with synthetic identity at parse time, or extend formal semantics. Pinned for the interpreter round.

**Paper-side**: requires a paper change (scalar-literal admission in stopping-condition value-expression position carries over; additionally, `<predicate>` production needs to be added to the EBNF wherever rank-variable subscripts appear).

**Status**: active.
