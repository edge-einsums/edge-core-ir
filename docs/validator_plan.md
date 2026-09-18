

# Layer 2 Semantic Validator — Implementation Plan

> *Created with the help of Claude Code.*

This document is both a design and a tutorial. The aim is for you to finish reading it and be able to (a) implement the validator, and (b) explain why it works the way it does.

---

## 1. Architecture overview — the "why"

### Where the validator sits

A program enters the system as JSON, becomes a `Program` via Pydantic, then heads toward execution. Three layers of checking happen along the way, and each one catches a different *kind* of mistake.

```
JSON input
   │
   ▼
[Layer 1: Pydantic]   shape checks: "is this a string where I expect a string?"
   │                  enforced automatically by Pydantic's frozen+strict+extra=forbid config.
   ▼                  produces a typed Program tree, or raises a single ValidationError.
[Layer 2: this validator]
   │                  semantic checks: "does the name F actually resolve to a
   │                  declared tensor?" "is empty_value=0 a valid int?"
   ▼                  produces a list of errors (collectable) and a side-table
[Layer 3: execution]  bundle of derived analyses.
```

Pydantic can verify the *shape* of every node — that `Einsum.output_tensor` is a string, that `BinaryApp.label` is an int. What it cannot do is reach across the tree and check that the string `output_tensor="F"` corresponds to a `TensorDeclaration(name="F")` somewhere in `Program.declarations`. Those *cross-references* are what Layer 2 is for. Layer 2 also handles type compatibility (an `empty_value=0` is fine for an `int` tensor, not for a `bool` tensor) and operator-category checks (a `CoordinateOp` cannot fill a `MapSpec.compute_op` slot).

### The two outputs

A naive validator returns "ok" or "broken." This one returns more, for two reasons.

1. **Errors are a list, not an exception.** Per D17, the validator must let an IDE or CLI show every problem at once. So the public function returns a `ValidationResult` containing `errors: list[ValidationError]`. The convention "raise on first failure" is wrong here because that forces a fix-rerun-find-next-error cycle on the user.

2. **Derived facts are a side-table bundle.** While the validator walks the IR, it ends up computing things that the evaluator and any future backend will also want: the iteration space for each Einsum (which rank variables appear, with what coord spaces); the reduce set for each `ReduceSpec` (which iteration ranks the reduce collapses); the affine table over every `RankExpression`. These are exactly the "derivations" of D14 — backend-neutral, paper-pinned, and expensive to compute twice. So the validator hands them back as an `AnalysisBundle`. Per D15, they live in side tables keyed by `id(node)`, not on the IR itself.

The implications:

- A consumer that only cares about correctness ignores `analysis` and just looks at `errors`.
- A consumer (the evaluator) that wants to start running the program receives the analysis bundle for free — it doesn't have to rederive iteration spaces.
- Two backends *cannot* drift on what "the iteration space of this Einsum" means, because they call the same shared analysis.

### Relationship to `edge_ir/analysis/`

The validator is the **orchestrator**. The shared analyses live in `edge_ir/analysis/`. The split:

- `edge_ir/analysis/`: pure derivation functions. Read the IR, return a plain Python value (often a dict keyed by `id(node)`). Don't import from the validator. Used by both validator *and* evaluator.
- `edge_ir/validator/`: cross-reference and type checks; runs the analyses, *and* produces errors. Imports from `edge_ir/analysis/`. Not used by the evaluator's execution path (only its setup phase).

The arrow points one way: validator → analysis. Never the reverse. This is the import-direction rule from D14, and it's load-bearing — if analyses imported validator code, then computing "is this affine?" would transitively pull in error-formatting machinery, which is obviously wrong.

### How registries fit in

Per D22's bucket-C distinction, "user-defined" names in the IR (a `UserDefinedComputeOp.name`, a `UserDefinedDataType.name`, a `CoordSetName.name`) are *references to runtime resources*. The IR carries the name; the *meaning* comes from a registry the host supplies at validate time.

The validator needs three registries:

- `FunctionRegistry` — the UDF registry from `docs/udf_registry_plan.md`. Per-category lookup: compute (in map/reduce/populate slots), coordinate, unary, rank-mapping, boolean. Per-UDF declaration metadata (arity, type contract, algebraic props like associativity).
- `TypeRegistry` — looks up `UserDefinedDataType` names and gives back something the validator can use to check `empty_value` (per the §10.4 narrow exception, the validator may call `DataTypeImpl.validate`).
- `CoordSetRegistry` — looks up `CoordSetName` names used in `SetMembership` predicates and in `RankDeclaration.coord_set`. Exposes element type for the type-compat check.

The validator never *runs* a UDF or queries a coordinate set's contents. It only asks the registries: "do you know this name? what's its declared shape?" This is the "decls only, not impls" rule, with the single narrow data-type exception for `empty_value` checking.

---

## 2. Directory structure

The literal tree of `edge_ir/validator/`:

```
edge_ir/validator/
├── __init__.py            re-exports validate_program, ValidationResult, the error hierarchy
├── api.py                 the public validate_program entry point; wires the passes together
├── result.py              ValidationResult, AnalysisBundle dataclasses
├── errors.py              the ValidationError hierarchy + ValidationWarning + Diagnostics accumulator
├── source_map.py          SourceSpan dataclass; helpers to resolve nodes → spans with parent fallback
├── passes/
│   ├── __init__.py        re-exports each pass function so api.py imports cleanly
│   ├── tensor_names.py    Pass 1: tensor-name resolution + uniqueness
│   ├── udf_names.py       Pass 2: UDF / type-name resolution
│   ├── labels.py          Pass 3: binary-label coverage + uniqueness within each Einsum
│   ├── rank_vars.py       Pass 4: rank-variable binding (scope)
│   ├── iteration_space.py Pass 5: orchestrates analysis/iteration_space; flags failures
│   ├── reduce_set.py      Pass 6: cross-check supplied rank_list against derived reduce set
│   ├── empty_value.py     Pass 7: empty_value matches data type
│   ├── op_categories.py   Pass 8: ops sit in the right slots
│   ├── stopping.py        Pass 9: stopping-condition checks (rank scope, types, pinned-rank contract)
│   ├── predicates.py      Pass 10: the predicate-specific checks P1–P11
│   └── affine.py          Pass 11: thin wrapper that calls analysis.affine and bundles results
├── parents.py             parent-pointer side table (walk once, then look up "what contains X")
└── README.md              (already exists; will be updated to point at this plan)
```

Two new modules under `edge_ir/analysis/` (since the validator owns the pass but the derivation is shared):

```
edge_ir/analysis/
├── affine.py              (already exists)
├── iteration_space.py     NEW — per-Einsum iteration space derivation
└── reduce_set.py          NEW — per-ReduceSpec/MapSpec derived rank-set
```

---

## 3. File-by-file breakdown

### `edge_ir/validator/__init__.py`

**Purpose:** A thin re-export layer so callers write `from edge_ir.validator import validate_program, ValidationResult` and don't need to know the internal module structure.

**Re-exports:** `validate_program`, `ValidationResult`, `AnalysisBundle`, `ValidationError` (the base), the concrete error classes, `ValidationWarning`, `SourceSpan`.

**No logic of its own.**

---

### `edge_ir/validator/api.py`

**Purpose:** The single public entry point. Sequences the passes in the correct order, collects errors and warnings into one `Diagnostics`, builds the `AnalysisBundle`.

**Public function:**

```python
def validate_program(
    program: Program,
    *,
    type_registry: TypeRegistry,
    function_registry: FunctionRegistry,
    coord_set_registry: CoordSetRegistry,
    shape_registry: ShapeRegistry,
    source_map: dict[int, SourceSpan] | None = None,
    config: ValidatorConfig | None = None,
) -> ValidationResult: ...
```

What it does, step by step:

1. Build a `ParentMap` by walking `program` once (see `parents.py`). Every IR node now has an "enclosing Einsum," "enclosing spec," etc., reachable in O(1).
2. Construct a `Diagnostics` accumulator (`errors: list[ValidationError]`, `warnings: list[ValidationWarning]`).
3. Run the passes in the order in §5 below. Each pass takes `(program, parents, registries, source_map, diagnostics, analysis_in_progress)` and either records errors or populates an analysis slot. **Passes do not raise.** A pass that depends on an earlier pass's result reads it from `analysis_in_progress`; if the earlier pass failed, the dependent pass skips the affected node(s).
4. Build `AnalysisBundle` from the populated slots.
5. Return `ValidationResult(errors=..., warnings=..., analysis=...)`.

**Internal helpers:** `_run_all_passes`, `_freeze_analysis_bundle` (so the returned bundle is read-only).

**Dependencies:** `edge_ir.ir.program.Program`, everything in `edge_ir.validator.passes`, `edge_ir.validator.parents`, `edge_ir.validator.result`, `edge_ir.validator.errors`.

---

### `edge_ir/validator/result.py`

**Purpose:** Holds the public return types.

**Public dataclasses:**

```python
@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    iteration_spaces: dict[int, IterationSpace]  # id(Einsum) → IterationSpace
    reduce_sets: dict[
        int, frozenset[str]
    ]  # id(ReduceSpec or MapSpec) → derived rank set
    affine: dict[int, Affinity]  # id(RankExpression node) → Affinity
    rank_var_scope: dict[
        int, frozenset[str]
    ]  # id(Einsum) → scope set (cached for downstream use)
    label_index: dict[int, dict[int, ComputationSpec]]  # id(Einsum) → {label: spec}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    errors: tuple[ValidationError, ...]
    warnings: tuple[ValidationWarning, ...]
    analysis: AnalysisBundle

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)
```

**Why a dataclass and not Pydantic:** `IRBase` is frozen + strict + extra=forbid because IR JSON must round-trip stably. The validator's outputs never round-trip; they're consumed immediately by the next pipeline stage. A regular dataclass is lighter, supports `dict` fields with `id(node)` keys (which Pydantic v2 dislikes), and skips serialization machinery the validator doesn't need. Make this call explicitly: **validator-owned models are dataclasses, not IRBase subclasses.**

**Dependencies:** `edge_ir.analysis.affine.Affinity`, `edge_ir.validator.passes.iteration_space.IterationSpace`, `edge_ir.ir.actions.ComputationSpec`.

---

### `edge_ir/validator/errors.py`

**Purpose:** The error hierarchy plus the `Diagnostics` accumulator.

**Public types:**

```python
@dataclass(frozen=True, slots=True)
class IRPath:
    """A symbolic path to an IR node, used when no source span is available.
    e.g. 'program.main_edge.cascade.einsums[2].specs[0].compute_op'."""

    parts: tuple[str | int, ...]

    def render(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ValidationError:
    """Base class. Subclasses add fields specific to the error category."""

    ir_path: IRPath
    source_span: SourceSpan | None  # populated when source_map has an entry
    message: str

    def format(self) -> str: ...  # human-readable: "[file.edge:12:5] message ..."


@dataclass(frozen=True, slots=True)
class ValidationWarning:
    ir_path: IRPath
    source_span: SourceSpan | None
    message: str
```

**The concrete error classes** (each a frozen dataclass extending `ValidationError`):

| Class | What it means |
|---|---|
| `UnresolvedTensorNameError(name)` | Tensor name in a projection didn't match any `TensorDeclaration` |
| `DuplicateTensorNameError(name, first_decl_path)` | Two declarations share a name |
| `UnknownUDFError(category, name)` | UDF name not in the `FunctionRegistry` for its category |
| `UnknownDataTypeError(name)` | User-defined data type name not in the `TypeRegistry` |
| `UnknownCoordSetError(name)` | `CoordSetName.name` not in the `CoordSetRegistry` |
| `UnresolvedShapeSymbolError(symbol, position)` | Shape symbol like `"|V|"` not in the `ShapeRegistry` |
| `LabelMismatchError(label, kind)` | `kind ∈ {"binary_without_spec", "spec_without_binary"}` |
| `DuplicateLabelError(label)` | Two specs/binaries share a label inside one Einsum |
| `RankVariableOutOfScopeError(name, scope)` | A `RankVariable` reference not in the scope set of its Einsum |
| `EmptyValueTypeMismatchError(declared_type, actual)` | `TensorDeclaration.empty_value` doesn't match declared `data_type` |
| `OperatorCategoryError(slot, expected, actual)` | A `CoordinateOp` showed up where a `ComputeOp` should be (or similar) |
| `ReduceRankListMismatchError(supplied, derived)` | Supplied `rank_list` doesn't match the derived reduce set |
| `RankFunctionArityError(name, expected, actual)` | Number of args differs from registered arity |
| `StoppingRankVariableNotGenerationalError(name)` | `StoppingCondition.rank_variable` doesn't name a generational rank in scope |
| `StoppingTypeMismatchError(side, lhs_type, rhs_type)` | Two sides of a stopping `Comparison` aren't type-compatible |
| `StoppingDiamondNotBooleanError(name)` | `DiamondBooleanApp` resolves to a non-Boolean-returning UDF |
| `StoppingPinnedRankContractError(rank, pinned)` | `pinned_rank_variables[0]` is not the enclosing rank |
| `PredicateRankVariableOutOfScopeError(name, scope)` | P1 |
| `IllFormedIntervalError(lo, hi)` | P3 |
| `PredicateCoordSetTypeMismatchError(...)` | P4 |
| `PredicateComparisonOrderingUndefinedError(op, kinds)` | P5 |
| `LogicalCompositionArityViolationError(kind, actual)` | P6 |
| `PredicateDepthExceededError(actual, limit)` | P7 |
| `EinsumPredicatesListTooLongError(actual)` | P11 |

And warnings: `LogicalCompositionRedundantWarning`, `IntervalIsEmptyWarning`, `PredicateConstantValuedWarning`, `SuspiciousCompositionWarning`.

**The `Diagnostics` accumulator:**

```python
@dataclass
class Diagnostics:
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    def error(self, err: ValidationError) -> None:
        self.errors.append(err)

    def warn(self, w: ValidationWarning) -> None:
        self.warnings.append(w)
```

**Why an accumulator and not a `Result[None, list[Error]]`:** A Rust-style `Result` forces every pass to thread the error list through its return type. The accumulator is a single mutable object shared by all passes — exactly what you want when passes are independent and each adds zero or more errors. Each pass returns its analysis output (or `None`), not the diagnostics; the diagnostics are a side-channel. This is simpler to read and matches how D17 talks about errors ("a list, not fatal").

**How source locations attach:** `errors.py` exposes a helper `def locate(node: Any, parents: ParentMap, source_map: dict[int, SourceSpan] | None) -> tuple[IRPath, SourceSpan | None]`. The helper:
1. Builds an `IRPath` by walking up the parent map.
2. If `source_map is None`, returns `(path, None)`.
3. Else looks up `source_map.get(id(node))`. If present, returns that. If absent (synthesized node), walks up parent links and returns the first ancestor's span. If nothing is found at all, returns `(path, None)`.

This is the D16 "synthesized nodes walk up the parent chain" contract made concrete.

**Dependencies:** `edge_ir.validator.source_map.SourceSpan`, `edge_ir.validator.parents.ParentMap`.

---

### `edge_ir/validator/source_map.py`

**Purpose:** Holds the `SourceSpan` dataclass and the resolve-with-fallback helper. Validator-internal; not part of any IR contract.

**Public types:**

```python
@dataclass(frozen=True, slots=True)
class SourceSpan:
    filename: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def render(self) -> str:  # "file.edge:12:5-12:18"
        ...
```

**Public functions:** `resolve_span(node, parents, source_map) -> SourceSpan | None`.

**Dependencies:** none (intentionally lightweight; lowering also imports `SourceSpan` from here).

---

### `edge_ir/validator/parents.py`

**Purpose:** Build a parent-pointer side table by walking the `Program` once. Lookup-only after construction.

**Public types:**

```python
@dataclass(frozen=True, slots=True)
class ParentMap:
    parent_of: dict[int, Any]  # id(child) → parent IR node
    enclosing_einsum: dict[
        int, Einsum | None
    ]  # id(any node inside an einsum) → that einsum
    enclosing_spec: dict[int, ComputationSpec | None]
    field_name_of: dict[int, str | None]  # which field of the parent this child sits in
    list_index_of: dict[int, int | None]  # which list position, when applicable
```

**Public function:** `build_parent_map(program: Program) -> ParentMap`.

**Internal helpers:** `_walk(node, parent_node, field_name, list_index, table)`.

**Why precompute:** Many passes need "what Einsum contains this node?" Without a precomputed map every pass would re-walk to find ancestors, turning the validator into O(N²). Walking once and looking up later is O(N).

**Dependencies:** `edge_ir.ir.program`, `edge_ir.ir.einsum`, `edge_ir.ir.actions` (for type annotations only).

---

### `edge_ir/validator/passes/tensor_names.py`

**Purpose:** Pass 1 — every tensor name used in the program resolves to a declared tensor, and declarations are unique.

**Public function:**

```python
def check_tensor_names(
    program: Program,
    parents: ParentMap,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> dict[str, TensorDeclaration]:
    """Build the tensor table and emit errors for every unresolved reference."""
```

What it does:
1. Build `decl_table: dict[str, TensorDeclaration]` from `program.declarations`. Emit `DuplicateTensorNameError` for each repeat.
2. Walk the IR. For every `TensorProjection.tensor`, every `Einsum.output_tensor`, every `CoordSetAlias.tensor`, every `PropertyApp.operands[i].tensor`, every `DiamondBooleanApp.operands[i].tensor`: if the name isn't in `decl_table`, emit `UnresolvedTensorNameError`.
3. Return the table for downstream passes.

**Internal helpers:** `_walk_for_tensor_refs`.

**Dependencies:** `edge_ir.ir.program`, `edge_ir.ir.tensor`, `edge_ir.ir.expr`, `edge_ir.ir.stopping`, `edge_ir.ir.predicate`.

---

### `edge_ir/validator/passes/udf_names.py`

**Purpose:** Pass 2 — every user-defined name resolves in the appropriate registry, with correct arity where applicable.

**Public function:**

```python
def check_udf_names(
    program: Program,
    parents: ParentMap,
    function_registry: FunctionRegistry,
    type_registry: TypeRegistry,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> None:
```

What it does. Walk the IR; for each user-defined name:
- `UserDefinedComputeOp.name`: look at the enclosing spec (via `parents.enclosing_spec`) to determine the category — `MAP_COMPUTE` / `REDUCE_COMPUTE` / `POPULATE_COMPUTE`. Lookup; if missing, `UnknownUDFError`.
- `UserDefinedMergeOp.name`: per the UDF plan, merge ops are built-in only. If we encounter one, emit `UnknownUDFError(category="merge", ...)` with a message hinting that merge ops cannot be user-defined. (See §14 of the UDF plan — until `UserDefinedMergeOp` is actually removed from the IR, the validator rejects it.)
- `CoordinateOp.name`: category `COORDINATE`.
- `UserDefinedUnaryOp.name`: category `UNARY`.
- `RankFunction.func_name`: category `RANK_MAPPING`. Also: check `len(node.args) == decl.arity`; emit `RankFunctionArityError` on mismatch.
- `DiamondBooleanApp.name`: category `BOOLEAN`.
- `UserDefinedDataType.name`: `type_registry.lookup(name)`; if missing, `UnknownDataTypeError`.

**Dependencies:** `edge_ir.udf` (registry), `edge_ir.ir.op`, `edge_ir.ir.tensor`, `edge_ir.ir.expr`, `edge_ir.ir.stopping`.

---

### `edge_ir/validator/passes/labels.py`

**Purpose:** Pass 3 — every `BinaryApp.label` matches exactly one `ComputationSpec.label` in the surrounding `Einsum.specs`, and vice versa; labels are unique within a single Einsum (D9 flat namespace).

**Public function:**

```python
def check_labels(
    program: Program,
    parents: ParentMap,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> dict[int, dict[int, ComputationSpec]]:
    """Returns label_index keyed by id(Einsum)."""
```

What it does. For each `Einsum` in the program (including initialization einsums and main_edge cascade einsums):
1. Build `spec_index: dict[int, ComputationSpec]` from `einsum.specs`. Emit `DuplicateLabelError` on repeats.
2. Walk `einsum.expression`, collecting `binary_labels: set[int]` from every nested `BinaryApp.label` (descending through `BinaryApp.lhs`, `BinaryApp.rhs`, `UnaryApp.operand`, `AnonymousTensor.expression`).
3. Emit `LabelMismatchError("binary_without_spec", label)` for any `label ∈ binary_labels` not in `spec_index`.
4. Emit `LabelMismatchError("spec_without_binary", label)` for any `label ∈ spec_index` not in `binary_labels`.
5. Subtle: a single label can have multiple matching specs (the BFS Einsum (a) has both a `MapSpec` and a `ReduceSpec` at label 1 — that's normal because Map and Reduce are different actions). The "duplicate" check is per-action: at most one `MapSpec` per label, at most one `ReduceSpec` per label, at most one `PopulateSpec` per label.

Return `label_index` (for the analysis bundle and for downstream passes).

**Dependencies:** `edge_ir.ir.einsum`, `edge_ir.ir.actions`, `edge_ir.ir.expr`.

---

### `edge_ir/validator/passes/rank_vars.py`

**Purpose:** Pass 4 — every `RankVariable.name` referenced inside an Einsum is bindable by the iteration space.

**Public function:**

```python
def check_rank_var_scope(
    program: Program,
    parents: ParentMap,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> dict[int, frozenset[str]]:
    """For each Einsum, build the scope set (output ranks ∪ input projection ranks).
    Emit a RankVariableOutOfScopeError for any RankVariable that isn't in scope."""
```

What it does, per `Einsum`:
1. Compute the **scope set**: collect every `RankVariable.name` reachable from `einsum.output_ranks` (descending through `RankArith` and `RankFunction.args`) and from every `TensorProjection.ranks` inside `einsum.expression`.
2. Walk the *entire* einsum's `RankExpression` content once more. For every `RankVariable` node encountered, if its name is not in the scope set, emit `RankVariableOutOfScopeError`.
3. Return `scope_table` for the analysis bundle.

Why two walks: the first one defines the scope (the union of bound names); the second one finds *references*. A name can appear in the input projections (binding) and elsewhere (reference) and we only flag references to names not bound anywhere.

**Dependencies:** `edge_ir.ir.einsum`, `edge_ir.ir.expr`.

---

### `edge_ir/validator/passes/iteration_space.py`

**Purpose:** Pass 5 — orchestrator. Calls `edge_ir.analysis.iteration_space.derive_iteration_space` per Einsum; bundles the result. If derivation fails (e.g. because earlier passes already flagged the einsum as malformed), skips the einsum gracefully.

**Public function:**

```python
def derive_iteration_spaces(
    program: Program,
    scope_table: dict[int, frozenset[str]],
    shape_registry: ShapeRegistry,
    diagnostics: Diagnostics,
) -> dict[int, IterationSpace]:
```

The actual derivation lives in `edge_ir/analysis/iteration_space.py`. This pass is just the orchestration boundary.

**Dependencies:** `edge_ir.analysis.iteration_space`.

---

### `edge_ir/validator/passes/reduce_set.py`

**Purpose:** Pass 6 — for each `MapSpec`/`ReduceSpec` with a supplied `rank_list`, derive the set from the iteration space and warn on mismatch. **Skip** when the einsum's output RVE contains an opaque `RankFunction` (D10: derivation is unsound there).

**Public function:**

```python
def cross_check_reduce_sets(
    program: Program,
    iteration_spaces: dict[int, IterationSpace],
    diagnostics: Diagnostics,
    parents: ParentMap,
) -> dict[int, frozenset[str]]:
    """Returns derived reduce sets keyed by id(spec)."""
```

What it does, per `Einsum`, per `MapSpec`/`ReduceSpec`:
1. If the einsum's output ranks contain a `RankFunction`, skip — D10 says derivation is unsound. (Record this fact in the analysis bundle as `derived = None` for that spec, so the evaluator knows the user-supplied list is authoritative.)
2. Otherwise compute `derived = iteration_space.ranks - output_rank_names`.
3. If the spec's `rank_list` is `None`, no error; record `derived` as the answer.
4. If supplied, normalize both to sets and compare. On mismatch, emit `ReduceRankListMismatchError(supplied, derived)`.

The reduce-set derivation itself lives in `edge_ir/analysis/reduce_set.py` as a pure function `derive_reduce_set(einsum, iteration_space) -> frozenset[str] | None`.

**Dependencies:** `edge_ir.analysis.reduce_set`, `edge_ir.ir.actions`, `edge_ir.ir.einsum`.

---

### `edge_ir/validator/passes/empty_value.py`

**Purpose:** Pass 7 — `TensorDeclaration.empty_value` matches the declared `data_type`. Same check for the new `value` field on zero-rank tensors.

**Public function:**

```python
def check_empty_values(
    program: Program,
    type_registry: TypeRegistry,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> None:
```

What it does, per `TensorDeclaration`:
1. If `data_type` is `BuiltinDataType(name="int")`: `empty_value` must be `int` or `None` (D13). Note: Python `bool` is a subclass of `int`, but `True/False` for an int-typed tensor is suspect — emit a warning, not an error.
2. If `name="float"`: must be `float`, `int` (widening is fine), or `None`. The sentinel decoding in `tensor.py` already turned `"inf"` → `math.inf` before this pass sees it.
3. If `name="bool"`: must be `bool` or `None`.
4. If `data_type` is `UserDefinedDataType(name=...)`: look up in `type_registry`. Per UDF plan §10.4, the validator may call exactly `DataTypeImpl.validate(empty_value)`. On failure, emit `EmptyValueTypeMismatchError`.

Also check `value` (the new D21 field) the same way for the zero-rank case.

**Dependencies:** `edge_ir.ir.tensor`, `edge_ir.udf` (TypeRegistry).

---

### `edge_ir/validator/passes/op_categories.py`

**Purpose:** Pass 8 — operators are in the right slots.

**Public function:**

```python
def check_op_categories(
    program: Program,
    parents: ParentMap,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> None:
```

This is mostly enforced by Pydantic's discriminated unions already (a `MapSpec.compute_op` is typed `ComputeOp`, which doesn't include `CoordinateOp`). But the validator's job is *defense in depth* in case a JSON-bypass ever ships an ill-typed instance, plus the cross-category UDF-name issues that Pydantic can't catch (e.g. a UDF named "min" registered as a unary op but used in a `MapSpec.compute_op`). The actual cross-category check lives in `udf_names.py` already (because that's where category context is established). So `op_categories.py` is small — it just walks for `isinstance` defense and emits `OperatorCategoryError` if something's wrong.

**Dependencies:** `edge_ir.ir.actions`, `edge_ir.ir.op`.

---

### `edge_ir/validator/passes/stopping.py`

**Purpose:** Pass 9 — stopping-condition checks.

**Public function:**

```python
def check_stopping_conditions(
    program: Program,
    parents: ParentMap,
    scope_table: dict[int, frozenset[str]],
    function_registry: FunctionRegistry,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> None:
```

What it does, per `StoppingCondition` in `program.main_edge.cascade.stopping_conditions`:
1. `rank_variable` must refer to a generational rank. A "generational rank" is one used as the leading rank of some Einsum's output in this cascade with successive values across iterations (typically `i`, `i+1`). The check is structural: the named variable must appear as a leading `RankVariable` or `RankArith(+, RankVariable, RankConstantLiteral)` in at least one cascade einsum's output ranks. If not, emit `StoppingRankVariableNotGenerationalError`.
2. For each `Comparison`: walk `lhs` and `rhs`. Compute their value types:
   - `TensorProjectionValue` → type of the referenced tensor declaration's `data_type`.
   - `RankExpressionValue` → coordinate type (integer by default).
   - `PropertyApp(name="occupancy", ...)` → returns `int`.
3. If the types aren't compatible for the comparison op (e.g. comparing a bool tensor projection against an int property), emit `StoppingTypeMismatchError`.
4. For each `DiamondBooleanApp.name`: look up in `function_registry` under category `BOOLEAN`. If found, check the declared return type is Boolean; if not Boolean, emit `StoppingDiamondNotBooleanError`. (If the registry doesn't carry return types yet — open question in the UDF plan — defer with a TODO.)
5. The pinned-rank lowering contract: `PropertyApp.pinned_rank_variables[0]` and `DiamondBooleanApp.pinned_rank_variables[0]` must equal the enclosing `StoppingCondition.rank_variable`. If not, emit `StoppingPinnedRankContractError`.

**Dependencies:** `edge_ir.ir.stopping`, `edge_ir.ir.tensor`, `edge_ir.udf`.

---

### `edge_ir/validator/passes/predicates.py`

**Purpose:** Pass 10 — the eleven predicate-specific checks P1–P11 from `docs/validator_plan_predicates.md`.

**Public function:**

```python
def check_predicates(
    program: Program,
    parents: ParentMap,
    scope_table: dict[int, frozenset[str]],
    coord_set_registry: CoordSetRegistry,
    shape_registry: ShapeRegistry,
    function_registry: FunctionRegistry,
    config: ValidatorConfig,
    diagnostics: Diagnostics,
    source_map: dict[int, SourceSpan] | None,
) -> None:
```

What it does (one helper per check):
- **P1** `_check_rank_vars_in_scope(predicate, scope_set, ...)` — walk every leaf `RankVariable`; emit `PredicateRankVariableOutOfScopeError` for each name not in scope. Don't short-circuit; collect every leaf.
- **P2** `_check_coord_set_name_resolves(set_membership, coord_set_registry, ...)` — for `coord_set.kind == "named"`, lookup name; emit `UnknownCoordSetError` if missing.
- **P3** `_check_interval_well_formed(set_membership, shape_registry, ...)` — for `coord_set.kind == "interval"`, resolve `lo` and `hi` if string; check `lo <= hi`. Errors and warnings per the plan.
- **P4** `_check_enum_type_compatible(set_membership, parents, ...)` — find the rank declaration for the `member`'s rank variable; ensure every enum element is type-compatible.
- **P5** `_check_predicate_comparison_operands(pc, shape_registry, function_registry, ...)` — shape-symbol resolution, UDF return type, operator-on-coord-space compatibility, all-constant warning.
- **P6** `_check_composition_well_formed(predicate, ...)` — pattern-match on `LogicalNot(LogicalNot(...))`, `LogicalAnd([p, LogicalNot(p)])`, `LogicalOr([p, LogicalNot(p)])`, duplicates inside `LogicalAnd`/`LogicalOr`; emit warnings.
- **P7** `_check_depth_bounded(predicate, config.predicate_depth_limit, ...)` — recursive depth count; emit `PredicateDepthExceededError`.
- **P8** *vacuous today.* Leave a `# P8: see validator_plan_predicates.md` comment.
- **P9** structural type assertion; trivially passes for Pydantic-constructed nodes, but kept as defense-in-depth.
- **P10** the empty-list case; no work needed.
- **P11** check `len(einsum.predicates) <= 1` (Pydantic also enforces this with a `model_validator`; defense in depth).

**Dependencies:** `edge_ir.ir.predicate`, `edge_ir.ir.tensor`, `edge_ir.ir.expr`, `edge_ir.udf`.

---

### `edge_ir/validator/passes/affine.py`

**Purpose:** Pass 11 — wrapper that calls `edge_ir.analysis.affine.affinity_table` for every `RankExpression` node reachable in the program, and assembles the result into a single dict for the bundle.

**Public function:**

```python
def derive_affine_table(program: Program) -> dict[int, Affinity]:
```

Walks the program to find every `RankExpression` root (every `Einsum.output_ranks[i]`, every `TensorProjection.ranks[i]`, every `PredicateComparison.lhs/rhs`, every `RankExpressionValue.expr`, every `SetMembership.member`, every `PropertyApp.operands[*].ranks[i]`, every `DiamondBooleanApp.operands[*].ranks[i]`). Calls `affinity_table(root)` on each, merges results.

Affine analysis cannot fail — it only ever returns AFFINE/NON_AFFINE/UNKNOWN — so no errors are produced.

**Dependencies:** `edge_ir.analysis.affine`, `edge_ir.ir.expr`.

---

### `edge_ir/analysis/iteration_space.py` (NEW)

**Purpose:** Pure derivation. For one `Einsum`, given a `ShapeRegistry`, compute its iteration space.

**Public types:**

```python
@dataclass(frozen=True, slots=True)
class IterationSpace:
    ranks: frozenset[str]  # the set of rank-variable names that iterate
    coord_spaces: dict[str, CoordSpace]  # rank name → coord space (resolved)
    restricted_by: tuple[
        IterationPredicate, ...
    ]  # the einsum's predicates, attached for record
```

**Public function:**

```python
def derive_iteration_space(
    einsum: Einsum,
    shape_registry: ShapeRegistry,
) -> IterationSpace:
    """Build the iteration space per paper Section 6 (canonical formal semantics)."""
```

What it does:
1. Collect rank-variable names from output ranks and every input `TensorProjection.ranks`. The union is the iteration ranks (D14 derivation).
2. For each rank-variable name, resolve its coord space: walk to the declared `TensorDeclaration.ranks[k]` it binds against, take that `RankDeclaration.shape` or `coord_set`, resolve any shape symbols through the `shape_registry`.
3. If the einsum has a predicate, attach it (the actual restriction-application is the interpreter's job; the validator only records).

**Dependencies:** `edge_ir.ir.einsum`, `edge_ir.ir.expr`, `edge_ir.ir.tensor`.

---

### `edge_ir/analysis/reduce_set.py` (NEW)

**Purpose:** Pure derivation. Compute the reduce set for one `ReduceSpec` or `MapSpec` from the iteration space.

**Public function:**

```python
def derive_reduce_set(
    einsum: Einsum,
    iteration_space: IterationSpace,
) -> frozenset[str] | None:
    """Returns the set of rank variables that the reduce collapses (iteration
    ranks minus output ranks). Returns None if the einsum's output RVE
    contains an opaque RankFunction — D10 says derivation is unsound there."""
```

**Dependencies:** `edge_ir.ir.einsum`, `edge_ir.ir.expr`, `edge_ir.analysis.iteration_space`.

---

## 4. The error model

### The `ValidationError` hierarchy

Pinned in §3 above under `errors.py`. The base is a frozen dataclass. Each concrete subclass adds its identifying fields. Pattern matching works (`match err: case UnresolvedTensorNameError(name=n): ...`).

Why a class hierarchy and not an enum + payload dict: the dataclass form gives static typing, IDE autocomplete, and exhaustiveness checking. An enum-of-categories with a `dict[str, Any]` payload is the same information but loses every type guarantee. The cost of writing N small dataclasses is paid once; the cost of `details["name"]` lookups is paid forever.

### How errors are collected

The `Diagnostics` accumulator (see §3, `errors.py`). It's a single mutable object shared by all passes. Each pass appends; nothing raises.

The alternative (`Result[None, list[Error]]` Rust-style) was rejected because every pass would have to return either its analysis output *or* the error list, which forces sum-typed returns that Python doesn't model cleanly. The accumulator is simpler: passes return their analysis output, errors get accumulated on the side.

### How source locations attach

The validator's entry point takes `source_map: dict[int, SourceSpan] | None = None` per D16. The map is built by the AST → IR lowering pass (when one exists) and keyed by `id(ir_node)`.

The `locate(node, parents, source_map)` helper:
1. Computes an `IRPath` by walking parents (always available, e.g. `program.main_edge.cascade.einsums[2].specs[0].compute_op`).
2. If `source_map is None`, returns `(path, None)`.
3. Else `source_map.get(id(node))`. If present, that's the span.
4. If absent (a synthesized node, like a default-filled `PopulateSpec`), walks up `parents.parent_of[id(node)]` repeatedly until finding an ancestor with a `source_map` entry. Returns that ancestor's span.
5. If no ancestor has a span either, returns `None`.

Errors carry `source_span: SourceSpan | None`. The IR path is always present so headless tools (CI, schema validators) still get a stable address.

### Error formatting

`ValidationError.format()` is a separate concern (a presentation detail), so it lives on the base class. The default format:

```
[file.edge:12:5-12:18] UnresolvedTensorNameError: tensor 'Q' is not declared
  at program.main_edge.cascade.einsums[2].expression.lhs.proj.tensor
```

When `source_span is None`, the first line falls back to just the error type and message. The IR path always appears. A consumer that wants different formatting overrides `format()` per subclass or wraps the error list themselves.

---

## 5. The validation passes (in execution order)

Pass order matters because later passes consume earlier ones' analyses. The order I'm proposing — and why each comes when it does — is:

### Pass 1: Tensor-name resolution

**Purpose:** Every tensor reference resolves to a declaration; declarations are unique.
**Walks:** `Program.declarations`, every `TensorProjection`, every `Einsum.output_tensor`, every `CoordSetAlias.tensor`, every `PropertyApp.operands[*].tensor`, every `DiamondBooleanApp.operands[*].tensor`.
**Checks:** Build a name → declaration table. Each reference looks up; missing → `UnresolvedTensorNameError`. Duplicate declarations → `DuplicateTensorNameError`.
**Produces:** A `dict[str, TensorDeclaration]` used by later passes (especially the iteration-space and stopping-condition passes).
**Depends on:** Nothing — runs first.
**Example error:**
```
Input snippet: an einsum projecting from tensor "Q" where Q is not declared:
  Einsum(output_tensor="T", expression=InputTensor(proj=TensorProjection(tensor="Q", ranks=[...])))
Diagnostic:
  [program.main_edge.cascade.einsums[0].expression.proj.tensor]
  UnresolvedTensorNameError: tensor 'Q' is not declared
  Did you mean: G, F, P, T?
```

### Pass 2: UDF and type-name resolution

**Purpose:** Every user-defined name (compute, coord, unary, rank-mapping, boolean, data type) resolves in its registry with correct arity.
**Walks:** Every `UserDefinedComputeOp.name` (categorized by enclosing spec), every `CoordinateOp.name`, every `UserDefinedUnaryOp.name`, every `RankFunction.func_name`, every `DiamondBooleanApp.name`, every `UserDefinedDataType.name`. Also flags `UserDefinedMergeOp` instances as unsupported.
**Checks:** Existence + arity (for rank functions). Builds no shared table.
**Produces:** Errors only.
**Depends on:** Pass 1, only weakly — runs after tensor names so error messages for "Q's data_type uses an unknown data type" can still mention Q. In practice they're independent.
**Example error:**
```
Input snippet: a MapSpec whose compute_op is an unregistered UDF:
  MapSpec(label=1, compute_op=UserDefinedComputeOp(name="saturating_add"), ...)
  with function_registry that doesn't contain ("MAP_COMPUTE", "saturating_add")
Diagnostic:
  [...einsums[0].specs[0].compute_op]
  UnknownUDFError: compute operator 'saturating_add' is not registered (category: MAP_COMPUTE)
  Known map-compute ops: +, -, *, /, take_left, take_right
```

### Pass 3: Label coverage

**Purpose:** Every `BinaryApp.label` has matching `ComputationSpec`(s); every spec's label has a matching binary; labels are unique per (action category, einsum) under the D9 flat-namespace rule.
**Walks:** Each `Einsum`'s `specs` list and `expression` tree (descending into nested `AnonymousTensor`s and `UnaryApp`s).
**Checks:** Build per-einsum spec index; build per-einsum binary-label set; cross-check.
**Produces:** A `dict[int, dict[int, ComputationSpec]]` keyed by `id(einsum)` for the bundle (the evaluator needs this index).
**Depends on:** Nothing.
**Example error:**
```
Input snippet: an einsum with BinaryApp(label=2, ...) but no spec at label 2:
  Einsum(specs=[MapSpec(label=1, ...)], expression=BinaryApp(label=2, ...))
Diagnostic:
  [...einsums[0].expression.label]
  LabelMismatchError: BinaryApp has label 2, but no ComputationSpec with label 2
  exists in this Einsum. Existing labels: [1].
```

### Pass 4: Rank-variable binding

**Purpose:** Every `RankVariable.name` in an Einsum is bindable by the iteration space.
**Walks:** Per einsum: (1) collect scope set from output and input projection rank lists; (2) walk all `RankVariable` references; emit error for each unbound reference.
**Checks:** Membership in scope set.
**Produces:** `dict[int, frozenset[str]]` keyed by `id(einsum)` — the scope set — used by predicate checks (P1) and iteration-space derivation.
**Depends on:** Nothing.
**Example error:**
```
Input snippet:
  Einsum(output_tensor="T", output_ranks=[RankVariable("i"), RankVariable("d")],
         expression=InputTensor(proj=TensorProjection(tensor="G",
             ranks=[RankVariable("s"), RankVariable("d")])),
         specs=[ReduceSpec(label=1, rank_list=["q"], ...)])  # 'q' never appears
Diagnostic:
  [...einsums[0].specs[0].rank_list[0]]
  RankVariableOutOfScopeError: rank variable 'q' is not in this einsum's scope
  scope = {i, d, s}
```

### Pass 5: Iteration-space construction

**Purpose:** For each einsum, compute its iteration space — the rank variables that iterate, with their coord spaces.
**Walks:** Each einsum.
**Checks:** Resolves shape symbols via `shape_registry`; emits `UnresolvedShapeSymbolError` on failure.
**Produces:** `dict[int, IterationSpace]` for the bundle.
**Depends on:** Pass 1 (tensor table — needed to look up rank declarations) and Pass 4 (scope set).
**Example error:**
```
Input snippet:
  TensorDeclaration(name="G", ranks=[RankDeclaration(name="S", shape="|W|"), ...])
  with shape_registry that has |V| but not |W|.
Diagnostic:
  [program.declarations[0].ranks[0].shape]
  UnresolvedShapeSymbolError: shape symbol '|W|' is not in the shape registry
  Known shapes: |V|, N, M
```

### Pass 6: Reduce-set cross-check

**Purpose:** When the user supplied `MapSpec.rank_list` or `ReduceSpec.rank_list`, verify it matches the derived set, and surface useful hints when it doesn't.
**Walks:** Each `Einsum`'s `specs`; per `MapSpec`/`ReduceSpec` with non-`None` `rank_list`.

**Governing rules from `README.md` (Hints/Error checking section), quoted verbatim so the implementation traces back to the source of truth:**

> 1. We should probably find a way to provide hints to the user if it seems like they want to reduce in some way, but their Einsum is malformed. Like adding v to the rank list, but then v appears in the output --> that's malformed.
>
> 2. TODO (ignoring for now): the reduce-set derivation (reduced ranks = iteration ranks minus output ranks) is only valid for BASIC reduce ops. It's not always derivable -- an opaque output RVE like a user function f(a,w) can map iteration points to output coords in a way we can't figure out from structure. So the rank_list cross-check / derived-set hint above only applies to basic reduces. Skipping this validation for now; revisit when we handle opaque output RVEs.

**Checks (mapped to those rules):**

1. **Opaque-RVE skip** (rule #2). If any output rank expression of the Einsum is a `RankFunction` (or transitively contains one), the reduce-set isn't statically derivable — skip the cross-check entirely for this spec. Record `None` in the side table so downstream consumers can tell "we couldn't decide" apart from "we decided it was empty." Revisit when we handle opaque output RVEs.

2. **Mismatched `rank_list` — general case** (rule #1, default shape). Derive `iteration_ranks - output_ranks`. If the supplied `rank_list` doesn't match the derived set, raise `ReduceRankListMismatchError` with the supplied list, the derived set, and the difference both ways.

3. **`v` in `rank_list` AND `v` in output — named malformed case** (rule #1, specific). A named sub-case of the mismatch check: when a rank variable `v` appears both in the supplied `rank_list` AND in any output rank expression, the user is asking to reduce over a rank they're also keeping in the output. That's structurally impossible — there's only one copy of `v` and they want it both collapsed and broadcast. Raise `ReduceRankInOutputError` instead of the generic mismatch, with a hint pointing at the contradiction. This is the malformed shape the README calls out by name, and it's the one users most often actually write by accident.

**Produces:** `dict[int, frozenset[str] | None]` keyed by `id(spec)`. Set value = derived reduce-set. `None` = skipped per opaque-RVE rule #2.
**Depends on:** Pass 5 (iteration space).

**Example error — generic mismatch:**
```
Input snippet (BFS Einsum (a) but mistyped):
  ReduceSpec(label=1, rank_list=["d"], ...)   # should be ["s"] for BFS
Diagnostic:
  [...einsums[0].specs[1].rank_list]
  ReduceRankListMismatchError: supplied rank_list ['d'] does not match the
  derived reduce set {'s'} (iteration ranks {i, d, s} minus output ranks {i, d}).
  Either the rank_list is wrong, or the expression is missing the rank you intended to reduce.
```

**Example error — named malformed case (`v` in `rank_list` overlaps output):**
```
Input snippet (hypothetical malformed BFS Einsum):
  output_ranks=[RankVariable("i"), RankVariable("d")]   # output mentions 'd'
  ReduceSpec(label=1, rank_list=["d"], ...)             # rank_list ALSO mentions 'd'
Diagnostic:
  [...einsums[0].specs[1].rank_list]
  ReduceRankInOutputError: rank_list ['d'] includes 'd', which also appears in the
  einsum's output ranks. You can't reduce over a rank you're also keeping in the
  output -- there's only one copy of 'd'. Either drop 'd' from the rank_list, or
  remove it from the output ranks.
```

**Example skip (opaque output RVE — rule #2):**
```
Input snippet:
  output_ranks=[RankFunction(func_name="cluster_id",
                              args=[RankVariable("a"), RankVariable("w")])]
  ReduceSpec(label=1, rank_list=["a"], ...)
No diagnostic raised; result side table records:
  derived_reduce_sets[id(spec)] = None   # skipped per README rule #2
A future pass (once opaque-RVE handling lands) will revisit this.
```

### Pass 7: Empty-value type check

**Purpose:** `TensorDeclaration.empty_value` is a valid instance of `data_type`. Same for `value` on zero-rank tensors.
**Walks:** `Program.declarations`.
**Checks:** Builtin types matched directly; user-defined types via the type registry's `validate(...)` method (the §10.4 narrow exception).
**Produces:** Errors only.
**Depends on:** Pass 2 (so we know the UDT name resolved before trying to call `.validate`).
**Example error:**
```
Input snippet:
  TensorDeclaration(name="G", data_type=BuiltinDataType(name="int"), empty_value="hello")
Diagnostic:
  [program.declarations[0].empty_value]
  EmptyValueTypeMismatchError: empty_value 'hello' is not a valid 'int' value
  Declared data_type: int; got: str
```

### Pass 8: Operator-category check

**Purpose:** Operators sit in their right slots.
**Walks:** Every `MapSpec`/`ReduceSpec`/`PopulateSpec`; check the type of each operator field.
**Checks:** Discriminated unions in Pydantic already enforce this structurally; the validator pass is defense in depth. Also handles cross-category UDF misuse caught by `udf_names.py`.
**Produces:** Errors only.
**Depends on:** Pass 2.
**Example error:** (Rare, since Pydantic catches most. Possible when JSON has been hand-crafted.)
```
Diagnostic:
  [...einsums[0].specs[0].compute_op]
  OperatorCategoryError: expected ComputeOp in MapSpec.compute_op slot,
  found CoordinateOp
```

### Pass 9: Stopping-condition checks

**Purpose:** Stopping conditions are well-typed and respect the pinned-rank contract (D7).
**Walks:** `program.main_edge.cascade.stopping_conditions`.
**Checks:** (a) `rank_variable` is a generational rank used in some cascade einsum; (b) `Comparison.lhs/rhs` are type-compatible; (c) `DiamondBooleanApp.name` is registered and returns Boolean; (d) `pinned_rank_variables[0]` == enclosing `StoppingCondition.rank_variable`.
**Produces:** Errors only.
**Depends on:** Passes 1 (tensor names), 2 (UDF names), 4 (scope sets), 5 (iteration spaces).
**Example error:**
```
Input snippet:
  StoppingCondition(rank_variable="i", predicate=Comparison(
      lhs=PropertyApp(name="occupancy", pinned_rank_variables=["j"], operands=[...]),
      ...))
  # pinned[0] is 'j', should be 'i'
Diagnostic:
  [...stopping_conditions[0].predicate.lhs.pinned_rank_variables[0]]
  StoppingPinnedRankContractError: pinned_rank_variables[0]='j' must equal
  the enclosing StoppingCondition.rank_variable='i'
```

### Pass 10: Predicate-specific checks

**Purpose:** P1–P11 from `docs/validator_plan_predicates.md`.
**Walks:** Each `Einsum.predicates[0]` (when present); recursively for compositions.
**Checks:** All eleven; uses scope sets (Pass 4), coord set registry, shape registry.
**Produces:** Errors and warnings.
**Depends on:** Pass 4 (scope sets).
**Example error:** (P1)
```
Input snippet (BFS init F_0 but mistyped):
  Einsum(output_ranks=[RankConstantLiteral(0), RankVariable("s")],
         predicates=[SetMembership(member=RankVariable("w"), coord_set=CoordSetName("id"))])
Diagnostic:
  [...initialization.einsums[0].predicates[0].member]
  PredicateRankVariableOutOfScopeError: predicate references rank variable 'w',
  but 'w' is not in the einsum's iteration scope (scope = {s}).
  Predicates may not introduce new rank variables.
```

### Pass 11: Affine analysis

**Purpose:** Compute affinity for every `RankExpression` node; bundle for the evaluator and any future MLIR-affine backend.
**Walks:** Every `RankExpression` root in the program.
**Checks:** No errors — `analyze_affinity` always returns one of AFFINE/NON_AFFINE/UNKNOWN.
**Produces:** `dict[int, Affinity]` for the bundle.
**Depends on:** Nothing — could run first, but logically it belongs after the structural passes.

---

## 6. Worked example — validating the BFS program step by step

This is the part you'll learn the most from. The validator visits the BFS IR pass by pass. I'll narrate what it sees and what it produces.

**Input:** `examples/algorithms/bfs/bfs_program.json` — six tensor declarations (`G`, `F`, `P`, `T`, `Lit_0`, `Lit_True`), two initialization einsums (`F_0`, `P_0`), three main-cascade einsums (a/b/c), one stopping condition.

**Registries:**
- `function_registry`: contains `ANY` (reduce compute), `take_left` (map compute, the UDF flavor), `OR` (map compute), and `occupancy` is built-in (it's a `PropertyApp`, not a UDF). All four BFS UDFs covered.
- `type_registry`: empty — BFS uses only built-in `int`/`bool`.
- `coord_set_registry`: contains `"id"` with descriptor `CoordSetDescriptor(element_type="int", contents=None)` (the host promises to supply it; the validator only checks existence).
- `shape_registry`: contains `"|V|"` bound to some integer like `100`.

### Pass 1: Tensor-name resolution

The validator builds the declaration table:
```
decl_table = {
    "G": <TensorDeclaration G>,
    "F": <TensorDeclaration F>,
    "P": <TensorDeclaration P>,
    "T": <TensorDeclaration T>,
    "Lit_0": <TensorDeclaration Lit_0>,
    "Lit_True": <TensorDeclaration Lit_True>,
}
```

It then walks every `TensorProjection`. Each one (the `Lit_0` in `F_0`'s expression, the `Lit_True` in `P_0`'s expression, `G_{s,d}` in einsum (a), `F_{i,s}` in (a), `T_{i,d}` in (b), `P_{i,d}` in (b), `F_{i+1,d}` in (c), `P_{i,d}` in (c), `F_{i+1}` in the stopping condition, `Lit_0` in the stopping condition) resolves. No errors.

It also checks `Einsum.output_tensor` for each einsum: `F`, `P`, `T`, `F`, `P` — all in the table.

**Result:** zero errors. `decl_table` returned for downstream passes.

### Pass 2: UDF / type name resolution

Walking the IR, it finds:
- `ReduceSpec.compute_op=UserDefinedComputeOp(name="ANY")` in einsum (a). Lookup `("REDUCE_COMPUTE", "ANY")`. Found.
- `MapSpec.compute_op=UserDefinedComputeOp(name="take_left")` in einsum (b). Lookup `("MAP_COMPUTE", "take_left")`. Found.
- `MapSpec.compute_op=UserDefinedComputeOp(name="OR")` in einsum (c). Lookup `("MAP_COMPUTE", "OR")`. Found.

No `UserDefinedDataType` references in BFS (everything is `int` or `bool`). No `RankFunction` (all rank expressions are bare variables, literals, or `RankArith("+", i, 1)`). No `DiamondBooleanApp` (the stopping condition uses `PropertyApp("occupancy", ...)` instead).

**Result:** zero errors.

### Pass 3: Label coverage

For each Einsum:

- `F_0` (init): `specs=[]`, `expression=InputTensor(proj=Lit_0)`. No binaries reachable, no specs. **Trivially passes.**
- `P_0` (init): same shape. Passes.
- Einsum (a): `specs=[MapSpec(label=1), ReduceSpec(label=1)]`, expression is one `BinaryApp(label=1)`. Spec index: `{1: [MapSpec, ReduceSpec]}` (each category present at most once at label 1 — fine). Binary labels: `{1}`. Match. Passes.
- Einsum (b): `specs=[MapSpec(label=1)]`, expression `BinaryApp(label=1, lhs=InputTensor(T), rhs=UnaryApp(not, InputTensor(P)))`. Binary labels: `{1}`. Match. Passes.
- Einsum (c): `specs=[MapSpec(label=1)]`, expression `BinaryApp(label=1, lhs=InputTensor(P), rhs=InputTensor(F))`. Labels: `{1}`. Match.

**Result:** zero errors. `label_index = {id(einsum_a): {1: [MapSpec, ReduceSpec]}, id(einsum_b): {1: [MapSpec]}, id(einsum_c): {1: [MapSpec]}}`.

### Pass 4: Rank-variable binding

For each einsum, build the scope set, then walk for references:

- **`F_0`:** output ranks = `[RankConstantLiteral(0), RankVariable("s")]`. Input projection (`Lit_0`) has no ranks. Scope = `{s}`. Walk all `RankVariable` nodes; only `s` appears. Passes. (Note: the literal `0` is a `RankConstantLiteral`, not a variable.)
- **`P_0`:** scope = `{d}`. Passes.
- **Einsum (a):** output ranks `[i, d]` → `{i, d}`. Input projections: `G_{s,d}` adds `{s, d}`; `F_{i, s}` adds `{i, s}`. Scope = `{i, s, d}`. References walked: `s, d, i, s` — all in scope. The `MapSpec.rank_list=["s"]` and `ReduceSpec.rank_list=["s"]` — both refer to `s`, in scope.
- **Einsum (b):** output ranks `[RankArith("+", i, 1), d]` → `{i, d}` (descended through RankArith). Input projections: `T_{i,d}` → `{i, d}`, `P_{i,d}` → `{i, d}`. Scope = `{i, d}`. References: `i, d, i, d`. Passes. The `rank_list=["d"]` checks out.
- **Einsum (c):** scope = `{i, d}`. Output ranks `[i+1, d]`. References: `i, d, i, d` (and in `F_{i+1, d}`: `i, d`). Passes.

**Result:** zero errors. Returns scope_table.

### Pass 5: Iteration-space construction

Per einsum, derive iteration space. For Einsum (a):
- ranks = `{i, s, d}` (the scope set).
- coord_spaces:
  - `i`: bound by `F_{i,s}`'s declaration `F`, rank `I`, no shape — generational, unbounded.
  - `s`: bound by `G`'s rank `S`, shape `|V|` → 100.
  - `d`: bound by `G`'s rank `D`, shape `|V|` → 100.
- restricted_by = `()` (no predicates).

For `F_0`:
- ranks = `{s}` (plus the literal `0`, which is not a variable so not iterated). Actually, the iteration space is more subtle: the output is `F_{0, s}`, so the einsum walks one value of the leading rank (`0`) and all values of `s`. We record `ranks={s}` with `coord_spaces={s: shape "|V|" = 100}`, and `restricted_by=(SetMembership(s, CoordSetName("id")),)`.

For the others, similar.

**Result:** iteration spaces populated. No errors (shape `|V|` resolves).

### Pass 6: Reduce-set cross-check

For Einsum (a)'s `ReduceSpec(label=1, rank_list=["s"], compute_op=ANY, merge_op=union)`:
- Output ranks = `{i, d}`. Iteration ranks = `{i, s, d}`. Derived reduce set = `{s}`.
- Supplied = `{s}`. **Match.** No error.

For Einsum (a)'s `MapSpec(label=1, rank_list=["s"])`:
- Same derivation. Supplied = `{s}`. Match.

Einsum (b) `MapSpec(label=1, rank_list=["d"])`: iteration = `{i, d}`, output = `{i, d}`, derived = `{}`. Supplied = `{d}`. **Mismatch!**

Wait — let me reread. For BFS (b), the iteration ranks should be derived from output + input projection ranks. Output is `[i+1, d]` → variables `{i, d}`. Inputs are `T_{i,d}` → `{i, d}`, `P_{i,d}` → `{i, d}`. So iteration = `{i, d}`. Derived reduce set = iteration − output = `{}`. Supplied `["d"]` does not match `{}`.

This is interesting. The `rank_list=["d"]` is on a `MapSpec`, and Map's `rank_list` semantics (per `actions.py` docstring) is different from Reduce's. For Map, the docstring says: "when omitted, map ranges over the full iteration space; per the canonical formal semantics, the merge operator selects the effectual points." So `rank_list` on Map is the supplied set of ranks the map is over, which by convention equals the iteration space (or a documented subset).

The cross-check rule should be Map-vs-Reduce aware. For Map, the supplied `rank_list` is the iteration space (or a subset), not the *reduce set*. For Reduce, it's the reduce set.

**This is a real find from the worked example.** The plan should pin this distinction: `reduce_set.py` only cross-checks `ReduceSpec.rank_list` against `derived_reduce_set`. For `MapSpec.rank_list`, the cross-check is against the iteration ranks (the supplied list must be a subset of the iteration ranks). I'll update the pass description: rename `reduce_set.py` to `rank_list_check.py` to cover both, with different rules per spec kind.

Continuing: BFS (b)'s `MapSpec(rank_list=["d"])` is a subset of iteration `{i, d}` — passes. Einsum (c) similarly.

**Result:** zero errors.

### Pass 7: Empty-value type check

- `G`: data_type=int, empty_value=0. `isinstance(0, int)`. Passes.
- `F`: data_type=int, empty_value=`math.inf` (decoded from `"inf"`). `isinstance(math.inf, float)`, not int. **Warning or error?** Per D13 plus the float-sentinel decoding, the project treats `float("inf")` as a legal int-empty in practice (BFS does exactly this). The cleanest call: the validator treats this as a warning ("empty_value uses float infinity for an int tensor — fine but unusual"), not an error.

This is another real finding: the worked example surfaces an existing convention that the plan should call out. The empty-value pass should:
- Allow `int | float` for int tensors when the float is `math.inf` or `-math.inf` (BFS pattern).
- Reject anything else.

- `P`: bool, False. Passes.
- `T`: int, `math.inf`. Same warning as F.
- `Lit_0`: int, empty_value=None (allowed by D13), value=0. Passes; value=0 matches data_type int.
- `Lit_True`: bool, empty_value=None, value=True. Passes.

**Result:** zero errors. Possibly two warnings (for F and T using inf as int empty).

### Pass 8: Operator-category check

Walks specs. All `compute_op`s are `BuiltinComputeOp` or `UserDefinedComputeOp`. All `merge_op`s are `BuiltinMergeOp`. No `PopulateSpec` in BFS, so no `CoordinateOp`. No miscategorizations. **Zero errors.**

### Pass 9: Stopping-condition checks

The one stopping condition:
```
StoppingCondition(rank_variable="i",
    predicate=Comparison(
        lhs=PropertyApp(name="occupancy", pinned_rank_variables=["i"], operands=[F_{i+1}]),
        op="==",
        rhs=TensorProjectionValue(proj=Lit_0_proj))
)
```

Checks:
- `rank_variable="i"`: does `i` appear as a generational rank in cascade einsums? Yes — Einsums (a), (b), (c) all use `i` (or `i+1`) as their leading output rank. Passes.
- Comparison types: `PropertyApp(occupancy)` returns `int`. `Lit_0` is declared `data_type=int`. Both ints — `==` is compatible. Passes.
- Pinned-rank contract: `pinned_rank_variables=["i"]`, enclosing `rank_variable="i"`. First element is `"i"`. Passes.

**Result:** zero errors.

### Pass 10: Predicate-specific checks

Two predicates in BFS (both in initialization):

For `F_0`'s predicate `SetMembership(member=RankVariable("s"), coord_set=CoordSetName("id"), negated=False)`:
- **P1** (rank var in scope): `s` is in `F_0`'s scope `{s}`. Passes.
- **P2** (coord set name resolves): `id` is in `coord_set_registry`. Passes.
- **P3** (interval well-formed): n/a, coord_set is `named`, not `interval`.
- **P4** (enum type compat): n/a for `named`. (Sub-trigger: the descriptor's `element_type` is `int`, and `s` is a rank variable over rank `S` of `F` with shape `|V|` (int). Compatible.)
- **P5** (comparison operands): n/a, no `PredicateComparison`.
- **P6** (composition): n/a, no `Logical*`.
- **P7** (depth bound): depth = 1, well under 8. Passes.
- **P11** (predicates list length): `len(predicates)=1`. Passes.

For `P_0`'s predicate, same shape, same outcome.

**Result:** zero errors, zero warnings.

### Pass 11: Affine analysis

Walks every `RankExpression`. For BFS, every one is either a `RankVariable`, a `RankConstantLiteral`, or a `RankArith("+", RankVariable, RankConstantLiteral)`. All AFFINE.

Returns `{id(node): Affinity.AFFINE for every RVE node}`.

### Final result for BFS

```
ValidationResult(
    errors=(),
    warnings=(...maybe two warnings about inf-as-int-empty for F and T...),
    analysis=AnalysisBundle(
        iteration_spaces={
            id(einsum_a): IterationSpace(ranks={i, s, d}, coord_spaces={...}, restricted_by=()),
            id(einsum_b): IterationSpace(ranks={i, d}, ...),
            id(einsum_c): IterationSpace(ranks={i, d}, ...),
            id(F_0): IterationSpace(ranks={s}, ..., restricted_by=(SetMembership(s, "id"),)),
            id(P_0): IterationSpace(ranks={d}, ..., restricted_by=(SetMembership(d, "id"),)),
        },
        reduce_sets={
            id(einsum_a_reduce): frozenset({"s"}),
            id(einsum_a_map): frozenset({"i", "s", "d"}),    # Map case: iteration ranks
            ...
        },
        affine={id(node): Affinity.AFFINE for each RVE},
        rank_var_scope={id(einsum_a): frozenset({"i","s","d"}), ...},
        label_index={id(einsum_a): {1: [MapSpec, ReduceSpec]}, ...},
    )
)
```

`result.has_errors` is `False`. The evaluator receives the analysis bundle and can start setting up its iteration walks immediately.

---

## 7. Worked example — validating the max-flow program step by step

Max-flow is a heavier program: 30 tensor declarations (including the synthesized literals `Zero`, `One`, `VertexCount`, `FalseConst`, `MinIdentity`, `Lit_0`), 25 cascade einsums (`E01` through `E25`), four initialization einsums, one stopping condition. It introduces `PopulateSpec` (BFS doesn't have any), heavy use of `AnonymousTensor` for sub-expressions, and a populate that names mutable ranks (`E11`'s `rank_list=["v"]`).

**Registries needed:**
- `function_registry`: must include `take_right`, `take_left`, `AND`, `OR`, `gt`, `eq`, `min`, `pick-admissible-edge` (populate compute), and `select-one-admissible-v` (coordinate op). Plus the built-in `+`, `-`, `*`. (`take_right` and `take_left` here are user-defined compute ops, not the merge-op flavor.)
- `coord_set_registry`: empty for max-flow (it uses no `CoordSetName`).
- `shape_registry`: `|V|` bound to some integer.

### Pass 1: Tensor-name resolution

30 declarations: `G, C, F, R, E, D, Act, S, T, In, Out, NST, ActR, Lbl, Adm, PushCand, Delta, InPush, OutPush, HasAdm, Rel, NeiLbl, MinNeiLbl, NewD, Zero, One, VertexCount, FalseConst, MinIdentity, Lit_0`. All distinct. **No errors.**

Walking the IR finds many `TensorProjection`s. Spot-check a couple:
- `init` einsum `D_{0,u} = S_u * VertexCount` references `S` and `VertexCount` — both declared. Passes.
- `E07`: `Act = NST .^1 (E .^2 Zero)`. References `NST`, `E`, `Zero` — all declared.
- `E11`: `PushCand = Adm .^1 One`. References `Adm` and `One`. Passes.
- Stopping condition: `occupancy(Act_{i+1}) == Lit_0`. Both declared. Passes.

**Zero errors.**

### Pass 2: UDF / type name resolution

Walks UDF references. `UserDefinedComputeOp(name="take_right")` appears many times (in init `D_{0,u}`, `E18`); `take_left` similarly. `AND`, `OR`, `gt`, `eq`, `min` all appear as compute UDFs. `pick-admissible-edge` is a populate compute. `CoordinateOp(name="select-one-admissible-v")` is the populate's coord op.

All must be in the registry. **Assuming the registry has them, zero errors.**

This is exactly the kind of case where the validator earns its keep: if `gt` is missing from the registry, the validator reports it immediately with the path to its first occurrence (e.g. `...einsums[6].specs[1].compute_op`), and continues walking — finds every other `gt` use and reports each one. The user sees, in one CLI run, every place they need to add a registration.

### Pass 3: Label coverage

Most einsums have either one label (`{1}`) or two (`{1, 2}` for the einsums with anonymous tensors like `E07`, `E09`, `E12`, `E13`, `E16`, `E17`, `E20`, `E22`). Spot-check:

- `E07`: `Act = NST .^1 (E .^2 Zero)`. Specs: `MapSpec(label=1, "AND", "intersect")`, `MapSpec(label=2, "gt", "intersect")`. Binaries: outer `BinaryApp(label=1)`, inner `BinaryApp(label=2)` inside the `AnonymousTensor`. Per D9, labels live in one flat namespace inside the Einsum, including binaries inside nested AnonymousTensors. Both labels match.
- `E11`: `PushCand = Adm .^1 One` with `PopulateSpec(label=1, rank_list=["v"], compute=pick-admissible-edge, coord=select-one-admissible-v)`. Binary `{1}`, spec `{1}`. Match.

**Zero errors.**

### Pass 4: Rank-variable binding

Spot check `E11` (PopulateSpec, the interesting one):
- Output: `[i, u, v]` → `{i, u, v}`.
- Inputs: `Adm_{i,u,v}` → `{i, u, v}`, `One` → `{}`.
- Scope = `{i, u, v}`.
- `PopulateSpec.rank_list=["v"]` — `v` in scope. Passes.

Spot check `E13` (the AnonymousTensor case):
- Output: `[i+1, u, v]` → `{i, u, v}`.
- Inputs (descending through anonymous tensor): `F_{i,u,v}`, `Delta_{i,u,v}`, `Delta_{i,v,u}` — all use `{i, u, v}`.
- Scope = `{i, u, v}`. Passes.

**Zero errors.** Returns scope_table.

### Pass 5: Iteration-space construction

For each einsum, resolves shape symbols. `|V|` resolves. All scope variables get coord spaces. `E11`'s iteration space:
```
IterationSpace(
    ranks={i, u, v},
    coord_spaces={u: <int range |V|=100>, v: <int range |V|=100>, i: <generational, unbounded>},
    restricted_by=(),
)
```

**Zero errors.**

### Pass 6: Reduce-set / rank-list cross-check

The most interesting case is `PopulateSpec.rank_list=["v"]` on `E11`. PopulateSpec's `rank_list` per D11 names the `*`-marked mutable ranks. The validator's cross-check is *different* here — it doesn't derive from iteration-output; it just checks the listed ranks are members of the iteration space's ranks. `v ∈ {i, u, v}`. Passes.

The `unary_reduce` helpers (`E02`, `E03`, `E14`, `E15`, `E19`, `E23`) produce `ReduceSpec(rank_list=...)`. For `E02`:
- `In_{1,v} = F_{1,u,v} . Zero`. Output ranks = `{v}` (the literal `1` is not a variable). Iteration ranks = `{u, v}`. Derived reduce set = `{u}`. Supplied = `["u"]`. **Match.**

`E14`: `InPush_{i,u} = Delta_{i,v,u} . Zero`. Output = `{i, u}`. Iteration = `{i, u, v}`. Derived = `{v}`. Supplied = `["v"]`. Match.

**Zero errors.**

### Pass 7: Empty-value type check

`Zero(int, 0)`, `One(int, 1)`, `VertexCount(int, 0)`, `FalseConst(bool, False)`, `MinIdentity(int, 0)`, `Lit_0(int, None, value=0)`. Each declared empty value matches. Plus the larger tensors: `G(int, 0)`, `S(bool, False)`, etc. All compatible.

**Zero errors.**

### Pass 8: Operator-category check

PopulateSpec on `E11`: `compute_op` is `UserDefinedComputeOp(pick-admissible-edge)`, `coord_op` is `CoordinateOp(select-one-admissible-v)`. Both fit their slots.

**Zero errors.**

### Pass 9: Stopping conditions

```
StoppingCondition(rank_variable="i",
    predicate=Comparison(
        lhs=PropertyApp(name="occupancy", pinned_rank_variables=["i"], operands=[Act_{i+1}]),
        op="==",
        rhs=TensorProjectionValue(proj=Lit_0)))
```

- `i` is a generational rank? Many cascade einsums (`E07, E08, E09, ...`) use `i` or `i+1` as their leading output rank. Yes.
- Types: occupancy returns int; Lit_0 is int. Compatible.
- Pinned[0]=`"i"` matches enclosing `"i"`. Passes.

**Zero errors.**

### Pass 10: Predicate-specific checks

Max-flow has **no predicates** anywhere (every einsum has `predicates=[]`). All P1–P11 trivially pass.

**Zero errors, zero warnings.**

### Pass 11: Affine analysis

Walks every RVE. Max-flow uses `RankArith("+", i, 1)` (affine), bare variables (affine), bare literals (affine). No `RankFunction`. All AFFINE.

### Final result for max-flow

```
ValidationResult(
    errors=(),
    warnings=(),
    analysis=AnalysisBundle(<25 cascade einsums + 4 init einsums of iteration spaces>, ...),
)
```

### What's different vs BFS

1. **PopulateSpec.** Max-flow's `E11` is the first time the validator sees a `PopulateSpec` with non-empty `rank_list=["v"]` (a `*`-marked mutable rank). The reduce-set pass has to special-case populate: the rank list is not "the reduce set" but "the populated ranks," and the check is just "every listed rank is in the iteration ranks." BFS has no populate, so this rule never fires there. The plan must spell this out.

2. **AnonymousTensors.** BFS uses no `AnonymousTensor`; max-flow uses many (`E07`, `E09`, `E12`, `E13`, `E16`, `E17`, `E20`, `E22`). The label-coverage walker must descend through `AnonymousTensor.expression` to find every `BinaryApp.label`. Per D9, labels are flat across the whole expression tree of one Einsum, including labels inside nested AnonymousTensors. The plan's `labels.py` pass description must call this out explicitly.

3. **Unary reduce idiom.** The `unary_reduce` helper synthesizes a `BinaryApp(label=1, lhs=<real_input>, rhs=<scalar identity>)` to give `ReduceSpec` a label site. The validator sees these as ordinary `BinaryApp`s and accepts them. (The gap note in metadata.md captures that this is a workaround.)

4. **No predicates, no `CoordSetName`.** Max-flow doesn't need the coord-set or shape registries the way BFS init does. So the predicate pass is silent and `coord_set_registry` is unused.

5. **Many synthesized scalar literals (`Zero`, `One`, etc.).** These are zero-rank tensors declared by the user (not parser-synthesized like `Lit_0`); they have `empty_value=0` and the new `value=None` (because they're "container" scalars where data is supplied at runtime, not the literal case). Important call-out: the empty-value pass treats hand-authored zero-rank tensors the same way as ranked ones — `empty_value=0` is valid for `int`. The `value` field is only checked when present.

---

## 8. How the validator integrates with `edge_ir/analysis/`

### Which analyses are pure derivations vs validator-owned

| Concept | Lives in | Returned as | Why |
|---|---|---|---|
| Affine analysis | `edge_ir/analysis/affine.py` | dict[id(RVE), Affinity] | Pure derivation; backend-neutral; future MLIR affine dialect wants this same answer. |
| Iteration-space construction | `edge_ir/analysis/iteration_space.py` (new) | `IterationSpace` per einsum | Pure derivation; the evaluator's walk-the-iteration-space loop reads exactly this. |
| Reduce-set derivation | `edge_ir/analysis/reduce_set.py` (new) | frozenset[str] per spec | Pure derivation; depends on iteration space, hence the import direction. |
| Tensor-name resolution | `edge_ir/validator/passes/tensor_names.py` | dict + errors | Not a "fact about the program"; it's a *check*. No backend wants this as a side table. |
| UDF-name resolution | `edge_ir/validator/passes/udf_names.py` | errors only | Same: a check, not a derivation. |
| Label coverage | `edge_ir/validator/passes/labels.py` | label_index + errors | Mostly errors. The label_index is a small caching convenience for the evaluator, but it could rebuild it trivially; keeping it in the bundle saves one walk. |
| Predicate checks (P1–P11) | `edge_ir/validator/passes/predicates.py` | errors + warnings | Pure check; nothing to share. |

The rule: **derivations** (answers to "what is X for this program?" — iteration space, affinity, reduce set) live in `analysis/`. **Checks** (answers to "is X well-formed?" — cross-references, type matches) live in `validator/`. A derivation can be wrong (it produces an answer that is then used to flag errors), but the derivation itself doesn't know about errors.

### Import direction

`validator/` imports from `analysis/`. `analysis/` does **not** import from `validator/`. This is enforceable as a CI lint check (`ruff` rule disallowing `from edge_ir.validator` inside `edge_ir/analysis/`).

Why: per D14, analyses must be usable by the evaluator and any future backend without dragging in the validator's error-handling machinery. If `analysis/iteration_space.py` ever called `Diagnostics.error(...)`, then every evaluator setup path would either need its own `Diagnostics` or would have to ignore the errors — both wrong.

When an analysis encounters something it can't compute (e.g. an unresolved shape symbol), it returns a sentinel: an `IterationSpace` with the offending rank's `coord_space=None`, or raises a narrow `AnalysisError` that the validator pass catches and turns into a `ValidationError`. The validator does the error wrapping; the analysis does the math.

---

## 9. Public API and call site

A consumer (an evaluator, a future MLIR backend, the CLI) sees this:

```python
from edge_ir.validator import validate_program, ValidationResult
from edge_ir.ir.program import Program
from edge_ir.udf import default_registry  # FunctionRegistry, TypeRegistry
from edge_ir.validator.source_map import SourceSpan

# ... build registries and source map ...

program: Program = Program.model_validate_json(open("bfs_program.json").read())
function_registry = default_registry()       # min, OR, take_left, occupancy_zero, ANY, ...
type_registry = function_registry.types      # same registry exposes a types view
coord_set_registry = MyCoordSetRegistry({"id": <descriptor>})
shape_registry = MyShapeRegistry({"|V|": 100})
source_map: dict[int, SourceSpan] | None = ...   # None when IR was built directly, populated when AST was lowered

result: ValidationResult = validate_program(
    program,
    type_registry=type_registry,
    function_registry=function_registry,
    coord_set_registry=coord_set_registry,
    shape_registry=shape_registry,
    source_map=source_map,
)

if result.has_errors:
    for err in result.errors:
        print(err.format())
    raise SystemExit(1)

# Warnings are non-fatal; surface them but proceed.
for w in result.warnings:
    print(w.message)

# Now use the analysis bundle to set up the evaluator.
analysis = result.analysis
for einsum_id, ispace in analysis.iteration_spaces.items():
    ...  # set up the walk
```

The return type:

```python
@dataclass(frozen=True, slots=True)
class ValidationResult:
    errors: tuple[ValidationError, ...]
    warnings: tuple[ValidationWarning, ...]
    analysis: AnalysisBundle

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)
```

And the bundle:

```python
@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    iteration_spaces: dict[int, IterationSpace]
    reduce_sets: dict[int, frozenset[str] | None]  # None when D10 opaque-RVE case
    affine: dict[int, Affinity]
    rank_var_scope: dict[int, frozenset[str]]
    label_index: dict[int, dict[int, list[ComputationSpec]]]
```

The dicts are keyed by `id(node)`. They're valid as long as the `program` is alive. Per `analyze_affinity`'s docstring, you must not serialize them — recompute after deserialization. The plan inherits that contract.

---

## 10. Test plan

The validator is testable from outside (run `validate_program` on a fixture, assert on the result) and that's the test style I'd push toward — internal unit tests of each pass should exist but only when the pass has nontrivial branching that black-box tests can't exercise reliably.

### Per-pass positive test

For each pass, one positive fixture that exercises the pass and confirms no error is emitted. Examples:

- `test_tensor_names_positive`: a 1-Einsum program with one tensor, used in projection and as output. Assert `result.errors == ()`.
- `test_udf_names_positive`: tiny program using one user-defined compute op; registry contains it. Assert no errors.
- `test_labels_positive`: 1 binary, 1 matching MapSpec. No errors.
- ... and so on for all 11 passes.

### Per-pass negative test

For each pass, one negative fixture that produces exactly the expected error class with the expected `IRPath` and (when relevant) `source_span`.

- `test_tensor_names_negative_unresolved`: builds a `TensorProjection(tensor="Q", ...)` with no `Q` declared. Assert `result.errors` contains exactly one `UnresolvedTensorNameError(name="Q")` with path ending in `expression.proj.tensor`.
- `test_labels_negative_binary_without_spec`: assert `LabelMismatchError(label=2, kind="binary_without_spec")`.
- For predicate checks, lean on the fixtures matrix in `docs/validator_plan_predicates.md` — they're already enumerated per P1–P11.

### Algorithm-level tests

Two big end-to-end tests:

- `test_bfs_validates_clean`: load `bfs_program.json`; build the BFS registry (`min`, `OR`, `take_left`, `ANY`, `occupancy_zero` — and the coord_set_registry with `"id"`); call `validate_program`; assert `result.has_errors == False`. Optionally assert specific contents of `analysis.iteration_spaces[id(einsum_a)].ranks == frozenset({"i", "s", "d"})`.
- `test_max_flow_validates_clean`: same shape; build the max-flow registry (all the UDFs from §3.E01–E25 above); validate; assert clean.

### Multi-error collection test

One pathological program with several distinct errors:

```python
def test_multi_error_collection():
    # Build a program with:
    #   - an unresolved tensor name
    #   - a binary label without a matching spec
    #   - a rank variable out of scope
    #   - an empty_value that doesn't match data_type
    program = ...
    result = validate_program(program, ...)
    error_types = {type(e).__name__ for e in result.errors}
    assert "UnresolvedTensorNameError" in error_types
    assert "LabelMismatchError" in error_types
    assert "RankVariableOutOfScopeError" in error_types
    assert "EmptyValueTypeMismatchError" in error_types
    assert len(result.errors) >= 4
```

This is the test that catches "the validator is short-circuiting somewhere it shouldn't" — the most important behavioral property of the whole system per D17.

### Property tests (optional, hypothesis)

- `validator_idempotence`: a passing program continues to pass after `Program.model_validate(program.model_dump_json())` round-trip.
- `bundle_keys_are_node_ids`: every key in `analysis.iteration_spaces` corresponds to an actual Einsum reachable from `program`.

### Source-span propagation tests

- A program where `source_map` contains an entry for the offending node; assert `result.errors[0].source_span` matches.
- A program where the offending node has no entry but its parent does; assert `result.errors[0].source_span` matches the parent's span.
- A program where `source_map=None`; assert `result.errors[0].source_span is None` and the path is still present.

---

## Closing notes (and the one rule I want to call out)

A theme runs through all of this: **the validator is a coordinator, not a thinker.** The math (iteration spaces, affinity, reduce sets) lives in `analysis/` where every backend can call it. The checks live in `validator/passes/` as small files, each with one job. The errors are a list, collected, with rich location info, never raised. The bundle hands the evaluator everything it needs to start running. Nothing in the validator changes the IR, ever — that's why `IRBase` can stay frozen.

The one rule worth pinning above all others: **passes do not short-circuit.** If pass 4 finds three out-of-scope rank variables, it reports all three, not just the first. If pass 1 finds two unresolved tensor names, it reports both, then pass 2 runs even if pass 1 had errors. The only time a pass skips work is when an earlier pass's analysis result is missing for a specific node (e.g. pass 5 didn't produce an iteration space for einsum X because pass 4 flagged it as malformed — pass 6 then skips X's reduce-set check). The skip is per-node and never per-program. This is what gives a user the one-shot "fix all my problems" experience the design decision (D17) was after.

---

### Critical files for implementation

- edge_ir/validator/api.py
- edge_ir/validator/errors.py
- edge_ir/validator/result.py
- edge_ir/analysis/iteration_space.py
- edge_ir/validator/passes/predicates.py