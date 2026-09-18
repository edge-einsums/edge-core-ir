# EDGE Interpreter — Future Plan

> *Created with the help of Claude Code.*

Forward-looking roadmap for the `edge-ir-interpreter` project. The
Layer-1 IR (Pydantic shape, JSON round-trip) is complete as of
with 351 tests passing. Everything below is what comes
next, in rough dependency order.

For audit findings to action against the IR, see `debug.md` in this
same directory.

---

## Layer 2 — Semantic validator

The `edge_ir/validator/README.md` stub sketches the entry point
(`validate_program(prog: Program, registry: TypeRegistry) -> None`).
This pass runs *after* Pydantic validation and *before* execution or
compilation. Read-only; raises structured errors on cross-reference
failure.

### Cross-reference checks
- Tensor names in `TensorProjection` must resolve to a
  `TensorDeclaration` in `Program.declarations`
- Tensor names referenced in `Initialization.einsums` and
  `MainEdge.cascade.einsums` must all resolve
- Tensor names must be unique within `Program.declarations`
- `BinaryApp.label` must match exactly one `ComputationSpec.label`
  in the surrounding `Einsum.specs` (or its enclosing
  `AnonymousTensor.specs` — scoping rule depends on Finding 14)
- Rank-name strings in `MapSpec.rank_list`, `ReduceSpec.rank_list`,
  `PopulateSpec.rank_list` must reference declared ranks at the
  binary-label site
- `RankVariable.name` references inside an `Einsum` must be bindable
  by the iteration space (every variable must be bound by exactly one
  position in `output_ranks` or in the union of all input projections)
- `StoppingCondition.rank_variable` must reference a generational
  rank somewhere in scope
- User-defined function names (compute / merge / coordinate / unary /
  rank-mapping / boolean) must all be registered in a `TypeRegistry`
  or `FunctionRegistry`

### Type checks
- `TensorDeclaration.empty_value` must be a valid instance of the
  declared `data_type` (built-in: int/float/bool; user-defined:
  delegated to the registered implementation)
- Operator categories must be used in their correct slot
  (e.g., a `MapSpec.compute_op` cannot be a `CoordinateOp`)

### Affine analysis pass
- Lives at `edge_ir/analysis/affine.py` as a **side table** — does NOT
  mutate the IR. Returns a `dict[id(node), Affinity]` (AFFINE /
  NON_AFFINE / UNKNOWN) for any `RankExpression` tree. The IR stays
  frozen and pure-syntax; see `design_decisions.md`.
- `RankFunction.is_affine` (user-declared, not derived) stays on the
  node and feeds the pass for opaque functions.
- Affine rules:
  - `RankVariable` → AFFINE
  - `RankConstantLiteral` / `RankConstantShapeSym` → AFFINE
  - `RankArith{+,-}` → AFFINE iff both children AFFINE
  - `RankArith{*}` → AFFINE iff one child is a constant (`RankConstantLiteral`
    or `RankConstantShapeSym`) + other AFFINE
  - `RankArith{/,//,%}` → NON_AFFINE
  - `RankFunction` → AFFINE/NON_AFFINE per its `is_affine`, else UNKNOWN

### Default-resolution pass
- `MapSpec.rank_list = None` and `ReduceSpec.rank_list = None` must
  be resolved to a concrete list of ranks (paper Section 6.211/6.234
  bare-`\bigvee` / bare-`\bigwedge` semantics: "all iteration-space
  ranks for the corresponding action at this site")

### Error model
- Define a `ValidationError` hierarchy with rich location info
  (which Einsum, which spec, which field path)
- Errors should be collectable, not fatal — return a list so a UI
  can surface multiple problems in one pass

### Testing
- Each check gets at least one positive test (valid program passes)
  and one negative test (invalid program produces the right error
  with the right location)
- The BFS artifact at `examples/algorithms/bfs/bfs_program.json` should pass full
  validation once a default `FunctionRegistry` covers `min`, `OR`,
  `take_left`, `occupancy_zero`

---

## AST and parser (surface syntax)

The `edge_ir/ast/` directory exists as a stub.

### AST design
- AST classes mirror the grammar one-to-one (no normalization). This
  includes left-recursive list productions becoming cons-cell nodes —
  explicit choice for v0. Future versions may flatten these to `list[T]`
  at parse time if the cons-cell traversal cost shows up.
- Carry source-location info (filename, line, column, span) on every
  AST node. Spans live on the AST only; lowering converts them to a
  side map keyed by IR node id (see D16 in `design_decisions.md`).
- Pydantic-based, same `IRBase` discipline as the IR

### Parser
- Lark grammar mechanically derived from
  the paper's `main_ebnf-cropped.tex`
- ASCII-only at the parser/AST level — Unicode-to-ASCII translation
  happens here, before the IR sees anything
- Translation table lives in `edge_ir/runtime/op_properties.py`
  (already has the Unicode → ASCII direction; add the inverse)
- Source-location tracking through every rule

### AST → IR lowering
- Single-pass transform; deterministic
- Normalizes the grammar's recursive `<nested-cascade>` into the IR's
  flat `NestedCascade.einsums + .stopping_conditions` shape
- Resolves the cascade-of-cascades production
- Lowering errors include source spans

### Round-trip properties
- `lower(parse(prettyprint(ir))) == ir` for any well-formed IR
  (IR → desugared source → AST → IR is the identity)
- `lower(parse(source)) == lower(parse(prettyprint(lower(parse(source)))))`
  (idempotence on lowering)

---

## Reference evaluator

The `edge_ir/evaluator/` directory has a `datatype_registry.py`
stub defining the `TypeImpl` protocol and `TypeRegistry`.

### Iteration-space construction
- Build the iteration space for each Einsum from the union of all
  ranks appearing in input projections plus output ranks
- Resolve coordinate spaces: shape DSL → integer ranges; explicit
  enums; intervals; aliases
- Generational rank materialization (paper Section 7.3.1):
  default `i` starts at 0, advances per cascade iteration
- **Apply `Einsum.predicates`** (when the restricted-iteration-predicates
  feature lands). Predicates restrict the iteration space the einsum
  walks. The paper-canonical reading is "materialize the restricted set,
  then walk it" — i.e. the iteration space *is* the restricted set, not
  a filter applied during iteration. Worked example: for BFS init
  `F_{0, s : s ∈ id} = 0` with `id = {2, 5, 7}` and `|V| = 100`, the
  iteration space is `{(0, 2), (0, 5), (0, 7)}` — three points, not
  one hundred-with-skips. The interpreter is free to *implement* either
  strategy (materialize-then-walk OR walk-and-skip) as long as the
  observable output matches; that's an interpreter-internal performance
  choice, not an IR concern. The IR carries the predicate; the
  interpreter decides how to honor it.

### Per-Einsum execution
- For each iteration-space point:
  - Evaluate input projections (look up tensor at coordinates)
  - Apply Map/Reduce/Populate semantics per paper Section 6.4–6.6
  - Use `op_properties` registry for built-in op semantics
  - Use `FunctionRegistry` (to be built) for user-defined ops

### Cascade iteration
- Run einsums in cascade order
- Check `StoppingCondition` after each iteration
- Bump generational ranks
- Limit max iterations (configurable; default e.g. 10⁶) so a
  buggy stopping condition can't loop forever

### Function registry (sibling of `TypeRegistry`)
- Register Python implementations of user-defined compute / merge /
  coordinate / unary / rank-mapping / boolean functions
- Per-function metadata: arity, expected types, optional
  algebraic properties (similar to `op_properties`)
- A default registry seeds `min`, `max`, `OR`, `AND`, `take_left`,
  `take_right`, `occupancy_zero`, etc.

### Determinism contract
- For a given (Program, FunctionRegistry, TypeRegistry, input data),
  output is deterministic
- This is a property test target
- See "Randomized iteration-order mode" below — the strongest way to
  pressure-test this contract

### Randomized iteration-order mode
A deliberate evaluator mode that **randomly scrambles the order in
which iteration-space points are visited** (seeded, so a failing run
is reproducible). The point: per the paper, an Einsum's result must
not depend on iteration order — so if scrambling the order changes the
output, either the evaluator has a bug (it's secretly relying on a
traversal order) or the program/semantics aren't actually
order-independent. Running the same (Program, inputs) under N random
seeds and asserting all outputs are byte-equal is a sharp
determinism/order-independence test, and a great Hypothesis target.

This requires a second execution mode where **dependencies are queued
rather than satisfied implicitly by traversal order**:
- A naive in-order evaluator gets away with "read X before I've
  written it ⇒ X is still empty" by accident of iteration order. A
  scrambled evaluator can't — it needs an explicit dependency model.
- Sketch: build a dependency graph over (tensor, coordinate) cells (or
  coarser: over Einsums within a cascade, and over generational
  iterations); a point becomes *ready* only when all cells it reads
  are finalized; a worklist/queue hands out ready points in random
  order; writes mark cells finalized and may enqueue newly-ready
  points. Within an Einsum the Map/Reduce/Populate semantics already
  define the per-cell combine, so order among independent points is
  free to scramble; the queue only enforces the *real* edges.
- Open questions: granularity of the dependency graph (per-cell is
  precise but heavy; per-Einsum-per-generation is cheap but may be
  too coarse to expose bugs); how Populate (which *creates*
  coordinates) interacts with "all reads finalized"; whether reductions
  need a barrier or can stream; interaction with `StoppingCondition`
  (the cascade loop is inherently sequential across generations, so the
  scramble is *within* a generation / within an Einsum, not across
  the cascade).
- Payoff beyond testing: the dependency/worklist machinery is most of
  what a parallel or out-of-order backend would need anyway, so this
  isn't throwaway scaffolding.

---

## Verification harness

Goal: confidence that the evaluator implements the paper, and that
future compiler backends agree with the evaluator.

### Reference outputs for case studies
- BFS (already have IR; need expected outputs for sample graphs)
- SpMV, SpMM, GEMM
- Connected components (paper Section 7.3.5; tests `RankFunction`
  non-affine path)
- Single-source shortest path (Dijkstra / SPFA — both in paper
  Section 8)

### Test inputs
- Small canonical graphs (4–8 vertices) checked against hand-computed
  outputs
- Random graphs with property assertions (e.g., BFS depths are
  well-defined, monotone, bounded by graph diameter)
- Empty graph; single-vertex graph; disconnected graph

### Property tests (Hypothesis)
- IR round-trip: any constructed `Program` survives JSON dump+load
  byte-equally
- Validator monotonicity: a passing program continues to pass after
  any validity-preserving rewrite (e.g., reordering einsums in a
  cascade where order is irrelevant)
- Evaluator vs paper-formal-semantics: pen-and-paper trace matches
  evaluator trace for at least the BFS canonical example
- Iteration-order independence: same (Program, inputs) under N random
  iteration-order seeds (the "Randomized iteration-order mode" under
  Reference evaluator) ⇒ byte-equal outputs every time

### Differential testing (when backends exist)
- Same Program, multiple backends → must produce equal outputs

---

## Backend compilers (long-tail)

These are explicitly out of scope for the IR layer but the IR is
designed to feed them.

- **MLIR**: lower IR to a sparse-tensor / linalg dialect
- **TeAAL**: existing target framework; emit TeAAL IR
- **Numpy / scipy**: dense-tensor reference backend useful for
  property testing
- **egglog**: rewrite engine for algebraic simplification (the
  `ComputeOpProperties` and `MergeOpProperties` registries already
  carry the metadata egglog needs)

---

## Outstanding IR cleanups (from audit)

Tracked in `debug.md`. Summary of unfixed items in priority order:

### Paper-side decisions needed
- **Finding 4** — `REGEX_USER_DEFINED_NAME` three-way disagreement
  between EBNF (no underscore), IR (underscore), and paper examples
  (hyphens). Pick one canonical form.
- **Finding 5** — `StoppingCondition` cannot reference tensor
  arguments. Real expressivity gap; must be resolved before more
  cascade examples land.
- **Finding 8** — `take_left` as a compute operator is a
  paper-internal symbol overload (also a merge op name). The paper
  never explicitly names this UDF.
- **Finding 14** — `AnonymousTensor.specs` vs `Einsum.specs`
  scoping rule is undocumented. Pick canonical form.

### IR-side cleanups (no paper input needed)
- **Finding 6** — Update BFS script docstring citation from
  "Eq 41 / Section 7.3" to `eqn:bfsi` / `eqn:edge_bfs_full` in
  `tex/bfs-example/07-iteration.tex`. Pure doc fix.
- **Finding 7** — BFS artifact G has 2 ranks; full-spec sidebar has
  3. Either align G to the full-spec or document the deliberate
  hybrid.
- **Finding 8 (BFS evaluator-correctness side)** — F's empty value
  is `0`, which collides with valid BFS depth-0. Switch F to `float`
  with `float("inf")` or use a non-colliding sentinel.

### Smaller/style
- **Finding 9** — Switch `[]` defaults in `program.py` and
  `einsum.py` to `Field(default_factory=list)` for style.
- **Finding 10** — `RankDeclaration` allows `shape` and `coord_set`
  to coexist. Add a Pydantic validator or document precedence.
- **Finding 11** — Scalar (0-rank) tensors are paper-prose-permitted
  but EBNF-rejected. Paper-side EBNF fix for empty
  `<shape-definition>`.
- **Finding 12** — `<rank-cs-override>` semantics: open question,
  whether the tensor-name superscript is redundant or intentional.
- **Finding 13** — Superseded. `is_affine` is no longer stored on
  `RankArith` (moved to the `edge_ir/analysis/affine.py` side table).
  `RankFunction.is_affine` stays on the node as user-declared input;
  no `model_copy` rebuild needed.

---

## Syntax things

Future syntax extensions / IR shape changes that aren't bugs in the
current state but are worth tracking.

- **`PopulateSpec.rank_list` empty-list semantics** — done.
  `Field(min_length=0)`; an empty list is the no-`*` default-assignment
  case. The field is still required (no default); making it fully
  optional (`list[str] | None`) was considered and rejected for now.
  See `design_decisions.md`.
- Default-rank semantics across all three actions (Map, Reduce,
  Populate) deserves a single canonical doc — what "default" means
  for each, where the resolution happens, and what's expected of a
  validator.
- **Provenance / source locations** — handled as a side map per D16 in
  `design_decisions.md`. Lowering returns `(IR, source_map)` where
  `source_map` is keyed by IR node id; the core IR carries no
  `source_loc` field, so error reporters look up the map (missing entry
  = synthesized node, walk up to the nearest ancestor with a span).
- (Add other syntax-level ideas here as they come up.)

---

## Tooling and infrastructure

### CLI (priority: low)
- `edge validate <program.json>` — Layer 2 validator
- `edge run <program.json> --inputs <data.json>` — evaluator
- `edge schema` — emit `Program` JSON schema
- `edge parse <source.edge>` — surface → IR JSON

### Schema versioning
- The current schema at `schemas/program.schema.json` is unversioned
- Decide a versioning policy before any IR-shape change is shipped
  (semver in the schema's `$id`?)
- Add a CI check that fails if the schema diff isn't accompanied by
  a version bump

### CI / pre-commit
- Already have `make check`; currently not wired into a CI workflow
- Add GitHub Actions for: `make check`, schema-stability check,
  artifact-round-trip check
- Pre-commit hook for `make format`

### Documentation
- A user-facing tutorial: "How to encode a graph algorithm in EDGE"
- A contributor doc: "How to add a new operator category"
  (already partially in `docs/ir_file_flow.md`)

---

## Open questions for the EDGE expert / paper author

Open questions for the paper:

- Finding 4: canonical form of `<user-defined-function>` (underscore?
  hyphen? something else?)
- Finding 5: how should multi-tensor stopping conditions be encoded?
  EBNF needs an extension.
- Finding 8: should `take_left` as a compute function be formally
  named in the paper?
- Finding 11: should the EBNF accept empty `<shape-definition>` for
  scalar tensors?
- Finding 12: is the tensor-name superscript on `<rank-cs-override>`
  redundant or does it permit cross-tensor overrides?
- Finding 14: canonical form for specs on `AnonymousTensor` vs the
  enclosing `Einsum`?

---

## Out of scope

Documented here so they don't drift back in:

- Auto-differentiation through EDGE programs
- GPU codegen (defer to backends)
- Distributed execution (defer to backends)
- A graphical IR visualizer
- Anything that would require breaking the IR's frozen-data shape
