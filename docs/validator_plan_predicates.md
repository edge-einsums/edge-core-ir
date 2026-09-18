# Layer 2 Validation Plan — Restricted-Iteration Predicates

> *Created with the help of Claude Code.*

**Scope.** This plan specifies the Layer 2 checks the validator must run once `IterationPredicate` and `Einsum.predicates` land. It is independent of the IR implementation: it pins the contract first so the tester can write coverage in parallel with the coder. Each check is keyed by name, fires at a defined trigger, asserts a single invariant, and emits a single error class with rich location info.

**Conventions used below.**

- "Structural" = no host state needed (uses only the `Program` IR).
- "Registry" = requires an additional registry argument injected into `validate_program`.
- `ValidationError` is the collectable-error type already implied by the design-decisions entry on collectable lowering errors (`docs/design_decisions.md`) and `edge_ir/validator/README.md`. The validator returns `list[ValidationError]`, not `None` — the README's stub signature should be updated when this lands.
- All checks include enough location info to address one specific node: `(einsum_index, predicate_path)`, where `predicate_path` is a list of `.operands[i]` / `.operand` / `.lhs` / `.rhs` steps from `Einsum.predicates[k]`. The error reporter resolves this against the source map when one is available; falls back to IR path when not.

---

## Entry-point contract

The current stub:

```python
def validate_program(prog: Program, registry: TypeRegistry) -> None: ...
```

Must change to:

```python
def validate_program(
    prog: Program,
    type_registry: TypeRegistry,
    function_registry: FunctionRegistry,
    coord_set_registry: CoordSetRegistry,  # NEW — required by checks P2, P3, P8
    shape_registry: ShapeRegistry,  # NEW — required by checks P3, P5 (already implied by existing shape-symbol resolution work)
    config: ValidatorConfig
    | None = None,  # NEW — carries policy knobs like depth limit (P7)
) -> tuple[list[ValidationError], list[ValidationWarning]]: ...
```

**Return type:** the validator emits both **errors** (the IR is malformed or semantically broken; the program can't run) and **warnings** (the IR is structurally valid and would run, but the user almost certainly didn't mean it). Two separate collectable lists; same rich-location-info contract on each (D17).

- `ValidationError` — fatal in spirit. A program with any error is rejected by downstream consumers.
- `ValidationWarning` — non-fatal. Surfaced for the user (UI / CLI / debugger) to consider; downstream consumers may proceed.

- `CoordSetRegistry` is a new injected dependency. Same shape as `FunctionRegistry`: a map from `name -> CoordSetDescriptor`, where the descriptor at minimum exposes the set's element type (int vs string-label vs UDT) and, for finite sets, its contents. Whether the host resolves `CoordSetName` eagerly (full enumeration) or lazily (just promises it exists) is a host-runtime choice; the validator only needs the type-and-existence facing.
- The validator does not own the registry implementations. It accepts them by injection. The decision **not** to fold `CoordSetRegistry` into `FunctionRegistry` is deliberate: a coordinate set is data, a UDF is code; conflating them muddies the registry's contract.

---

## Predicate checks

### P1. `predicate_rank_vars_in_scope`

- **Trigger.** Walk every `Einsum.predicates[k]`, then walk the predicate tree recursively. Fire once per leaf reference to a `RankVariable` — that means every `SetMembership.rank_variable` (it is itself a bare rank-var name) AND every `RankVariable` node appearing transitively inside `PredicateComparison.lhs/rhs` (which may contain nested `RankArith`/`RankFunction`).
- **Invariant.** Every referenced rank-variable name must appear in the **scope set** of the enclosing einsum. The scope set is:
  - the set of `RankVariable.name`s in `Einsum.output_ranks` (walked recursively through `RankArith`/`RankFunction`), PLUS
  - the set of `RankVariable.name`s in any `TensorProjection.ranks` reachable from `Einsum.expression` (walked recursively through `InputTensor`, `UnaryApp`, `BinaryApp`, `AnonymousTensor`).
- **Error.** `PredicateRankVariableOutOfScope(einsum_index, predicate_path, rank_variable_name, scope_set)`. Message: `"predicate references rank variable {name!r}, but {name!r} is not in the einsum's iteration scope (scope = {scope_set}). Predicates may not introduce new rank variables."`
- **Positive fixture.** `Out_u = G_{u, v}` einsum (so `u`, `v` in scope) with `predicates=[SetMembership(rank_variable="v", coord_set=CoordSetName(name="id"))]`. Validator returns no errors for this check.
- **Negative fixture.** Same einsum, but `predicates=[SetMembership(rank_variable="w", coord_set=CoordSetName(name="id"))]`. Expect one `PredicateRankVariableOutOfScope` error naming `"w"`.
- **Notes / edge cases.**
  - Negation does not change scope semantics: a `LogicalNot(SetMembership(rank_variable="w", ...))` still fails P1.
  - A composition like `LogicalAnd([SetMembership(rank_variable="v", ...), PredicateComparison(lhs=RankVariable("w"), op="<", rhs=RankVariable("v"))])` produces TWO P1 errors when `w` is out of scope — one per leaf reference. The validator must NOT short-circuit; collectable-error contract.
  - The output-rank walk must descend into `RankArith` and `RankFunction.args`. A predicate referencing `s` is in scope when output ranks include `RankArith("+", RankVariable("s"), RankConstantLiteral(1))`.
  - `RankConstantShapeSym` references (`|V|`, `N`) are NOT rank variables; they fall under P3 instead.
- **Where it fits.** Structural. Pure IR walk. No registry needed.

---

### P2. `coord_set_name_resolves`

- **Trigger.** Per `SetMembership` node where `coord_set.kind == "named"` (the new `CoordSetName` variant). Walk every predicate tree to find these.
- **Invariant.** `CoordSetName.name` must resolve in the injected `coord_set_registry`.
- **Error.** `UnregisteredCoordSetName(einsum_index, predicate_path, name)`. Message: `"predicate references coord set {name!r}, which is not registered in the coord-set registry. Known sets: {sorted_keys}."`
- **Positive fixture.** `coord_set_registry = {"id": <descriptor>}` and `SetMembership(rank_variable="s", coord_set=CoordSetName(name="id"))`. No errors.
- **Negative fixture.** Same `SetMembership`, but `coord_set_registry = {}`. Expect one `UnregisteredCoordSetName` error.
- **Notes / edge cases.**
  - Name resolution is **case-sensitive** (matches how tensor-name resolution behaves elsewhere). The fixture should also include a case-mismatch negative ("Id" vs "id").
  - When `name` resolves but the descriptor's element type is incompatible with the rank variable's coord space, that is **not** a P2 failure — it is a separate check (P4 name-variant; see Notes under P4).
  - Pydantic enforces `CoordSetName.name: str` non-empty (pattern TBD — recommend constraining it to `REGEX_USER_DEFINED_NAME` so host registries can't smuggle in weird strings). If the pattern is enforced at the IR level, P2 only has to deal with "well-formed but unregistered."
- **Where it fits.** Registry. Requires `coord_set_registry`.

---

### P3. `coord_set_interval_endpoints_well_formed`

- **Trigger.** Per `SetMembership` node where `coord_set.kind == "interval"`. (This check is conceptually shared with the existing `CoordSetInterval` validation in tensor declarations, but the predicate-position trigger is independent — the same `CoordSetInterval` literal could be valid in one position and ill-formed in another if shape symbols differ in scope. In practice they share the same shape registry; centralize the helper.)
- **Invariant.** Both:
  1. If `lo` or `hi` is a string, it must resolve in `shape_registry` to an integer. (Matches the existing shape-symbol resolution contract.)
  2. After resolution, `lo <= hi` must hold. (Half-open `[lo, hi)` permits `lo == hi` as the empty interval.)
- **Error.** Two distinct error classes for two distinct failures:
  - `UnresolvedShapeSymbol(einsum_index, predicate_path, symbol_name, position)` where `position ∈ {"lo", "hi"}`.
  - `IllFormedInterval(einsum_index, predicate_path, lo_resolved, hi_resolved)` with message `"interval [{lo}, {hi}) is ill-formed: lo > hi after resolution"`.
- **Positive fixture.** `CoordSetInterval(lo=0, hi="|V|")` with `shape_registry={"|V|": 10}`. No errors.
- **Negative fixtures.**
  - `CoordSetInterval(lo=0, hi="|W|")` with `shape_registry={"|V|": 10}`. Expect `UnresolvedShapeSymbol`.
  - `CoordSetInterval(lo=5, hi=3)`. Expect `IllFormedInterval`.
  - `CoordSetInterval(lo="N", hi="M")` with `shape_registry={"N": 10, "M": 5}`. Expect `IllFormedInterval` after resolution.
- **Notes / edge cases.**
  - `lo == hi` is **legal** (empty iteration) but **emits a warning**: `IntervalIsEmpty(einsum_index, predicate_path, lo, hi)`. The einsum iterates over zero points — sometimes intentional (e.g. a no-op initialization path), more often a typo or a shape-symbol that resolved unexpectedly. Don't reject; do flag.
  - Negative integer endpoints: the IR allows `lo: int | str`, so `lo=-3` is structurally valid. Whether negative coords make sense is a check that belongs in P4 (type-compat with the rank's coord space), not here.
- **Where it fits.** Registry. Requires `shape_registry`.

---

### P4. `coord_set_enum_type_compatible`

- **Trigger.** Per `SetMembership` node where `coord_set.kind == "enum"`. Look up the **declared coordinate space** of the rank variable.
- **Invariant.** Every element of `CoordSetEnum.coords` must be type-compatible with the rank variable's declared coord space:
  - If the rank's coord space is integer-valued (default dense, `CoordSetInterval`, or a `CoordSetEnum` of ints), every enum element must be `int`.
  - If the rank's coord space is string/label-valued (a `CoordSetEnum` of strings), every enum element must be a string that appears in the rank's enumerated set.
  - If the rank's coord space is UDT-valued (future work), defer to the UDT registry.
- **Error.** `PredicateCoordSetTypeMismatch(einsum_index, predicate_path, rank_variable, expected_kind, offending_coord, offending_index)`.
- **Positive fixture.** Rank `S` declared with `CoordSetEnum(coords=["a","b","c"])`; rank variable `s` is iterating over `S`; predicate `SetMembership(rank_variable="s", coord_set=CoordSetEnum(coords=["a"]))`. No errors.
- **Negative fixtures.**
  - Same rank but `coord_set=CoordSetEnum(coords=[0, 1])` (ints into a string space). Expect mismatch.
  - Same rank but `coord_set=CoordSetEnum(coords=["z"])` (string not in the rank's enum). Expect mismatch with a different sub-message — `"coordinate {z!r} is not declared in the rank's coord set {known}."`
- **Notes / edge cases.**
  - The check needs to know **which rank variable maps to which rank declaration**. That's not trivial — a rank variable like `s` in `Out_s = G_{s, v}` could map to either `G.rank[0]` (and through aliases, possibly to other tensors). The validator already has to do this for general type checks; the predicate check piggybacks on the same resolver. If the resolver returns multiple candidate ranks for the same rank-variable name (e.g. it appears in two input projections with different declared coord sets), the predicate must be compatible with **all** of them; flag any mismatch as an error.
  - **Variant of P4 for `CoordSetName`:** when `SetMembership.coord_set` is `CoordSetName`, the registry descriptor's element type must also be compatible with the rank variable's coord space. Same error class; predicate path differs. Treat this as a sub-trigger under P4 rather than a separate top-level check.
  - Empty `coords=[]` is structurally valid (Pydantic doesn't enforce min_length on `CoordSetEnum.coords`). Like the empty interval in P3, this is legal and yields empty iteration; don't reject.
- **Where it fits.** Mostly structural (rank-coord-space lookup is in the IR). The `CoordSetName` sub-trigger needs the registry.

---

### P5. `predicate_comparison_operands_are_coordinate_valued`

- **Trigger.** Per `PredicateComparison` node. Examine `lhs` and `rhs`.
- **Invariant.** Both `lhs` and `rhs` must be `RankExpression` (the existing union: `RankVariable | RankConstantLiteral | RankConstantShapeSym | RankArith | RankFunction`). The Pydantic type already enforces this structurally — but the validator must **document the contract** as a check that, if the field ever widens (e.g. someone adds a `TensorProjection` variant to satisfy a different feature), this check would catch it. Until then this is a tautological no-op at runtime, but it is the spec we are pinning.
- **Additional sub-invariants the validator should enforce because Pydantic does not:**
  1. **Shape-symbol resolution.** Any `RankConstantShapeSym` referenced inside `lhs`/`rhs` must resolve in `shape_registry`. Reuses the same machinery as P3.
  2. **RankFunction type signature.** Any `RankFunction` referenced inside `lhs`/`rhs` must be registered in `function_registry` AND its declared return type must be "coordinate-valued" (integer or rank-coord-space-compatible). The existing UDF-registry check already handles registration; the predicate-position case adds the return-type constraint. If the registry doesn't carry return types yet, flag this as a follow-up and emit a `NeedsExpertReview` marker (i.e. emit the structural check now, add the return-type assertion when the registry gains it).
  3. **Operator-type compatibility.** If both sides resolve to integer-valued, all six ops in `ComparisonOp` are fine. If both sides resolve to string-label-valued (a `RankConstantLiteral` with a `str` value matched against a `RankVariable` over a char coord set), only `==` and `!=` are well-defined; `<`, `>`, `<=`, `>=` should be rejected unless the coord space carries an order (future work).
- **Errors.**
  - `PredicateComparisonNonCoordinateOperand(einsum_index, predicate_path, side, offending_node_kind)` — for the (currently structurally-impossible) widening case.
  - `PredicateComparisonOrderingUndefined(einsum_index, predicate_path, op, lhs_kind, rhs_kind)` — for the order-on-unordered-coords case.
- **Positive fixture.** `PredicateComparison(lhs=RankVariable("s"), op="<", rhs=RankVariable("d"))` where both `s` and `d` iterate over integer ranks. No errors.
- **Negative fixtures.**
  - `PredicateComparison(lhs=RankVariable("s"), op="<", rhs=RankConstantLiteral(value="a"))` where `s` iterates over an int rank: type mismatch (a separate `PredicateComparisonTypeMismatch` error; see notes).
  - `PredicateComparison(lhs=RankVariable("c"), op="<", rhs=RankConstantLiteral(value="a"))` where `c` iterates over a char-enum rank: `PredicateComparisonOrderingUndefined`.
- **Notes / edge cases.**
  - `PredicateComparisonTypeMismatch` (int vs string-label) is arguably a separate check; it's rolled under P5 because it shares the operand-walking pass. The tester should treat it as its own fixture group.
  - Document explicitly that "tensor-projection on either side" is structurally impossible (the Pydantic type rejects it); the test for this is a Pydantic-construction test, not a validator test. The validator check stays in the spec as a future-proof contract.
  - **All-constant predicate is a warning, not an error.** A `PredicateComparison` whose lhs and rhs are both `RankConstantLiteral` / `RankConstantShapeSym` (no `RankVariable` reachable on either side) evaluates to a fixed truth value at parse time. Either always restricts to nothing (`5 < 3` → vacuous einsum) or never restricts anything (`0 < 5` → equivalent to no predicate). Emit `PredicateConstantValued(einsum_index, predicate_path, evaluated_truth_value)` as a warning. The validator may compute the truth value when both sides resolve to integers and report it in the warning's diagnostic; defer the resolution if shape symbols are involved.
- **Where it fits.** Structural for the type-tautology and the all-constant warning; registry for the shape-symbol and UDF-return-type sub-invariants.

---

### P6. `composition_well_formed`

- **Trigger.** Per `LogicalAnd` / `LogicalOr` / `LogicalNot` node.
- **Invariant.** Three things:
  1. **Arity.** `LogicalAnd.operands` and `LogicalOr.operands` have `len >= 2`. (Pydantic enforces; the validator surfaces a clear error if it ever ships malformed via a bypassed validator — defense in depth.)
  2. **Trivial / suspicious patterns — WARNING-LEVEL, not error.**
     - Double negation: `LogicalNot(LogicalNot(p))` collapses to `p`. Emit a `ValidationWarning` (separate severity from `ValidationError`); do not reject.
     - Contradictory operands: `LogicalAnd([p, LogicalNot(p)])` is `False` for every iteration point. Emit a warning.
     - Tautological operands: `LogicalOr([p, LogicalNot(p)])` is `True` for every iteration point — equivalent to no predicate at all. Emit a warning.
  3. **Operand uniqueness — WARNING-LEVEL.** Duplicates inside an `And` or `Or` (`LogicalAnd([p, p])`) are redundant; warn.
- **Errors / Warnings.**
  - `LogicalCompositionArityViolation(einsum_index, predicate_path, kind, actual_arity)` — error.
  - `LogicalCompositionRedundant(einsum_index, predicate_path, pattern)` where `pattern ∈ {"double_not", "contradiction", "tautology", "duplicate"}` — warning.
- **Positive fixture.** `LogicalAnd([SetMembership(...), PredicateComparison(...)])`. No warnings, no errors.
- **Negative fixtures.**
  - `LogicalNot(LogicalNot(SetMembership(...)))`. Expect `LogicalCompositionRedundant(pattern="double_not")`.
  - `LogicalAnd([p, LogicalNot(p)])` where `p` is structurally equal on both sides. Expect `LogicalCompositionRedundant(pattern="contradiction")`.
- **Notes / edge cases.**
  - Structural equality on `IRBase` nodes is well-defined (Pydantic models compare field-by-field) — use it for the contradiction/duplicate detection.
  - **Warning channel confirmed.** The validator entry-point returns `tuple[list[ValidationError], list[ValidationWarning]]`. The four redundancy patterns above (double-not, contradiction, tautology, duplicate) are warnings.
  - `LogicalNot(LogicalAnd([p, q]))` is **not** redundant — it's a real De Morgan's-form predicate. Don't flag it.
- **Where it fits.** Structural.

---

### P7. `composition_depth_bounded`

- **Trigger.** Per `Einsum.predicates[k]` — at the root, traverse and measure max nesting depth across `LogicalAnd.operands`, `LogicalOr.operands`, `LogicalNot.operand`.
- **Invariant.** Max depth ≤ 8. (Configurable via a `ValidatorConfig` parameter so tests can drive it down to small values for negative cases.)
- **Error.** `PredicateDepthExceeded(einsum_index, predicate_index, actual_depth, limit)`.
- **Positive fixture.** Any predicate of depth ≤ 16.
- **Negative fixture.** A predicate of depth 17 (a chain of 17 nested `LogicalNot`s, easy to build programmatically).
- **Notes / edge cases.**
  - Alternative: enforce at the IR layer via a `model_validator` on `LogicalAnd`/`LogicalOr`/`LogicalNot`. Recommend **leaving it to the validator** rather than the IR — depth limits are a policy choice, not an IR invariant. The IR should permit arbitrary nesting; the validator policies the bound. This matches the "derived facts are not on the IR" principle from `docs/design_decisions.md`.
  - The bound should be a parameter to `validate_program` (or to a `ValidatorConfig` dataclass), not a hard-coded literal, so tests can drive it down for negative cases.
- **Where it fits.** Structural.

---

### P8. `determinism_under_runtime_resolution`

- **Trigger.** Per `Einsum.predicates[k]` that contains any `SetMembership` with a `CoordSetName` whose registry descriptor is **unordered** (e.g. a `set`-backed implementation in the host).
- **Invariant.** The predicate's restricted iteration space must be **order-invariant** — its truth value at any iteration point must not depend on the order in which the host enumerates the coord set's contents. Equivalent statement: the predicate's iteration-space restriction is a **set** of admitted iteration points, not a sequence.
- **Status — punt vs. structural.** Two-tier:
  - **Structural sub-check (cheap, runs here).** Reject any predicate composition that introduces order-sensitivity at the predicate level. Given the operator set in scope (`∈`, `∉`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `∧`, `∨`, `¬`), there is no operator that exposes iteration order — so the structural sub-check is currently **vacuously satisfied**, and the validator can record that fact in a comment for future-me. If a future operator is added (e.g. a "first-k coordinates" predicate operator), this check fires.
  - **Runtime sub-check (punt).** The deeper question — does the host's enumeration of `CoordSetName(name="id")` produce the same iteration space regardless of internal data-structure order — is a property-test concern. Document this as "covered by the randomized-iteration-order mode in `docs/interpreter_future_plan.md`" and do NOT try to assert it from the validator.
- **Error.** `PredicateOrderSensitiveOperator(einsum_index, predicate_path, operator_name)` — only fires when the operator alphabet grows. Today: dead code, kept as the spec.
- **Positive fixture.** Any predicate from use cases 1–7. (All current operators are order-invariant.)
- **Negative fixture.** None possible today; the spec is a placeholder for when the operator set grows.
- **Notes / edge cases.**
  - This check exists to remind future-me: when adding a new predicate-level operator (e.g. a "first 5" or "any" predicate), wire it into the order-sensitivity classifier in this check. The check is the canonical place that question gets asked.
- **Where it fits.** Structural. No registry needed today.

---

### P9. `predicate_node_is_not_an_einsum_expression_node`

- **Trigger.** Per `Einsum.predicates[k]` at the root.
- **Invariant.** The root must be an `IterationPredicate` (one of `SetMembership | PredicateComparison | LogicalAnd | LogicalOr | LogicalNot`). It must NOT be an `Expression`, `ValueExpression`, or `RankExpression` smuggled into the slot.
- **Error.** `PredicateSlotTypeError(einsum_index, predicate_index, actual_kind)`.
- **Status.** Pydantic's discriminated union enforces this at IR-construction time; the validator's job is to surface a clean error if a JSON-bypass ever ships an ill-typed predicate. Defense in depth.
- **Where it fits.** Structural.

---

### P10. `empty_predicates_list_is_legal`

- **Trigger.** Per `Einsum` whose `predicates` is `[]`.
- **Invariant.** Trivially satisfied; no check fires.
- **Notes.** Document explicitly in the test suite that `predicates=[]` is the common case (the einsum iterates over the full unrestricted space). The validator must not emit any warning or error for the empty list. Pin this with a positive fixture.

---

### P11. `predicates_list_is_zero_or_one_element`

- **Trigger.** Per `Einsum` whose `predicates` has `len >= 2`.
- **Invariant.** `Einsum.predicates` has at most one element. Multi-clause predicates must use explicit composition (`LogicalAnd` / `LogicalOr` / `LogicalNot`) inside that single element; the list slot is NOT a flat conjunction sequence.
- **Error.** `EinsumPredicatesListTooLong(einsum_index, actual_length)`. Message: `"Einsum.predicates has {actual_length} elements; expected 0 or 1. Multi-clause predicates must be wrapped in a LogicalAnd/LogicalOr/LogicalNot inside a single list element. (See docs/design_decisions.md D22 and docs/parsing.md.)"`
- **Positive fixtures.**
  - `predicates=[]` — no restriction, common case.
  - `predicates=[SetMembership(...)]` — single-clause.
  - `predicates=[LogicalAnd(operands=[SetMembership(...), PredicateComparison(...)])]` — multi-clause via explicit composition.
- **Negative fixture.** `predicates=[SetMembership(...), PredicateComparison(...)]` (two flat list elements). Expect `EinsumPredicatesListTooLong(actual_length=2)`.
- **Notes.** This is the canonical-IR-shape rule. Option (b) from the design discussion was chosen. One canonical IR per logical predicate. Downstream consumers (validator passes, egglog rewriters, MLIR backends) don't have to normalize between flat lists and explicit `LogicalAnd`s.
- **Where it fits.** Structural.

---

## Cross-check interactions with existing validator checks

These are not new checks; they are notes the implementor must heed when wiring P1–P11 into the broader validator.

- **`function_registry`** must be checked for any UDF named in a `RankFunction` inside a predicate (sub-trigger of the existing UDF-registry check). The predicate walk extends the surface area but does not change the check itself.
- **Tensor-name resolution** is unaffected — predicates do not reference tensor names. (`CoordSetAlias.tensor` does reference a tensor name, but P4's enum-type check inherits that resolution from the existing tensor-name resolver; no new logic.)
- **Affine analysis (`edge_ir/analysis/affine.py`)** is unaffected. Predicates restrict the iteration space, but affineness of `RankArith` is independent of which subset of iteration points the einsum actually walks.

---

## Test fixtures the tester should produce

A flat list, by check, of the minimum fixtures. The tester is expected to lift these into `tests/fixtures/predicates/` (or inline them as builders) so they can be reused.

| Check | Fixture | What it pins |
|---|---|---|
| P1 | BFS-init einsum + `SetMembership(rank_variable="s", ...)` where `s` is in `output_ranks` | Positive: in-scope rank var |
| P1 | Same einsum + `SetMembership(rank_variable="w", ...)` | Negative: out-of-scope rank var |
| P1 | `LogicalAnd([SetMembership(rank_variable="v", ...), PredicateComparison(lhs=RankVariable("w"), op="<", rhs=RankVariable("v"))])` with `v` in scope, `w` not | Negative: collects 1 error (not 2 — `v` is in scope; only `w` fails) |
| P2 | `CoordSetName(name="id")` with `registry={"id": ...}` | Positive |
| P2 | `CoordSetName(name="id")` with `registry={}` | Negative |
| P3 | `CoordSetInterval(lo=0, hi="|V|")` with `shape_registry={"|V|": 10}` | Positive |
| P3 | `CoordSetInterval(lo=0, hi="|W|")` with no `|W|` binding | Negative: `UnresolvedShapeSymbol` |
| P3 | `CoordSetInterval(lo=5, hi=3)` | Negative: `IllFormedInterval` |
| P3 | `CoordSetInterval(lo=0, hi=0)` | Positive (empty interval is legal) |
| P4 | Rank `S` over `CoordSetEnum(["a","b","c"])`; predicate `CoordSetEnum(["a"])` | Positive |
| P4 | Same rank; predicate `CoordSetEnum([0, 1])` | Negative: type mismatch |
| P4 | Same rank; predicate `CoordSetEnum(["z"])` | Negative: label not in declared set |
| P5 | `PredicateComparison(RankVariable("s"), "<", RankVariable("d"))` over int ranks | Positive |
| P5 | `PredicateComparison(RankVariable("c"), "<", RankConstantLiteral("a"))` over char ranks | Negative: ordering undefined |
| P5 | `PredicateComparison` with a `RankConstantShapeSym` that doesn't resolve | Negative: unresolved shape symbol |
| P6 | `LogicalAnd([p, q])` with distinct `p, q` | Positive |
| P6 | `LogicalNot(LogicalNot(p))` | Warning: double-not |
| P6 | `LogicalAnd([p, LogicalNot(p)])` | Warning: contradiction |
| P6 | `LogicalAnd([p, p])` | Warning: duplicate |
| P7 | 17-deep chain of `LogicalNot` | Negative: depth exceeded (assumes limit=16) |
| P8 | (no negative fixture today; structural placeholder) | — |
| P9 | Hand-constructed `Einsum` with garbage in `predicates[0]` (via JSON-bypass model_validate) | Negative: type error |
| P10 | Einsum with `predicates=[]` | Positive: no errors, no warnings |
| P11 | Einsum with `predicates=[p, q]` | Depends on the (a)/(b) decision |

---

## Decisions (all locked except where noted)

1. **Warning channel — LOCKED.** Validator returns `tuple[list[ValidationError], list[ValidationWarning]]`. Errors mean "the IR is broken; the program can't run." Warnings mean "structurally valid but probably a mistake." Warnings include: double-negation (P6), contradiction (P6), tautology (P6), duplicate operand (P6), empty interval `lo == hi` (P3), all-constant predicate (P5).
2. **Multi-predicate list semantics — LOCKED.** Option (b): `Einsum.predicates` has at most one element. Multi-clause predicates wrap inside `LogicalAnd`/`LogicalOr`/`LogicalNot`. Enforced by check P11.
3. **Depth limit (P7) — LOCKED at 8.** Config-driven via a `ValidatorConfig` parameter so tests can override.
4. **`CoordSetRegistry` shape.** Recommend: `element_type: Literal["int", "str", ...]`, `contents: list[int | str] | None` (None = host-resolved lazily, validator can't enumerate but can verify the name exists). Still open — settle when the registry implementation lands.
5. **`CoordSetName.name` pattern — LOCKED.** Constrained to `REGEX_USER_DEFINED_NAME`. Pattern lives on the IR field; the validator only handles "well-formed but unregistered."
6. **Return-type metadata on UDFs (P5 sub-invariant).** The current `FunctionRegistry` may not carry declared return types. If not, P5's UDF return-type check defers until the registry grows; document and add a TODO.

---

## Where this plan is silent

- It says nothing about how the parser lifts predicates into `Einsum.predicates`. That is a parser-side concern documented in `docs/parsing.md` § "Restricted-iteration predicate attachment."
- It says nothing about how the interpreter evaluates a predicate against a candidate iteration point. That is `docs/interpreter_future_plan.md`.
- It says nothing about how `CoordSetName` is bound to host data at runtime. That is a host-runtime concern; the validator only asks "does the name resolve to something the registry knows about?"
