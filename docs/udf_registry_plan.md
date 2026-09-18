# User-Defined Function (UDF) Registry — Design Plan

> *Created with the help of Claude Code.*

Status: **draft for discussion.** This is the working document for the
"UDF registry" feature. It is written to be picked up cold in a longer
design conversation, so it spells out terminology, the full category
list, the user-facing layout, concrete examples, how the IR layer is
(un)affected, how the validator and the evaluator hook in, and the open
design questions that still need a decision.

Author: drafted by Claude (CLI).

---

## Review notes (first pass)

Decisions/feedback from the first read of this draft. **These override
the body where they conflict; the body has been partly updated to
match but treat this section as authoritative.**

- **Merge operators: BUILT-IN ONLY — not a UDF category.** Drop the
  `merge/` folder, the `@merge_op` decorator, and merge from the
  registry's category axis. Open follow-up (a *code* change, not just
  this doc): should `UserDefinedMergeOp` be removed from `edge_ir/ir/op.py`
  too? Flagged in §14.
- **Unary operators: confirmed** as a UDF category. Keep as-is.
- **Rank-mapping functions: confirmed** as a UDF category (incl. the
  affine-flag handling in §10.3). Keep as-is.
- **Stopping condition / boolean functions: NEEDS WORK — to
  revisit.** Treat §4.7 and §10.5 as provisional / placeholder; do not
  build against them yet. A separate design pass is coming.
- (Compute map/reduce/populate, coordinate, datatype: no objection
  raised yet — still open per §10.)

---

## 0. TL;DR

- A UDF is anything an EDGE program references **by name** that the
  IR itself does not define the meaning of: compute operators,
  coordinate operators, unary operators, rank-mapping functions,
  stopping-condition (boolean) functions [*see review notes — being
  revisited*], and user-defined **data types**. (Merge operators are
  **built-in only** — not a UDF category; see review notes.) The IR
  only ever stores the *name string*; the
  meaning lives outside the IR.
- The user authors UDFs as decorated Python functions/classes in a
  **directory with a fixed sub-folder layout** (one folder per
  category; compute ops split into `map/`, `reduce/`, `populate/`).
- A loader walks that directory, runs the decorators, and produces a
  `UdfRegistry` value.
- That registry is passed explicitly to `validate_program(prog,
  registry)` (Layer 2 validator) and to `evaluate(prog, inputs,
  registry)` (the reference evaluator). The validator uses the
  **declaration** half (names, arities, type contracts, algebraic
  props); the evaluator uses the **implementation** half (the
  callables).
- The IR (Pydantic models in `edge_ir/ir/`) does **not change** for
  this feature — it already carries everything it needs (the name).
  This document explains why, and where the registry instead plugs in.

---

## 1. Terminology: evaluator vs interpreter vs compiler backend

These terms get used loosely in the existing notes; pin them down here.

- **Evaluator** (a.k.a. **interpreter** — same thing in this project):
  the component that takes an EDGE `Program` plus concrete input
  tensor data plus a `UdfRegistry`, and **directly computes** the
  output tensor data by walking the IR tree. It is the *reference
  semantics*: if the paper and the evaluator disagree, the evaluator
  is wrong. It is a tree-walking interpreter — correctness over speed.
  Lives in `edge_ir/evaluator/`. (`docs/interpreter_future_plan.md`
  calls this "the interpreter"; pick one term project-wide — suggested:
  use **"evaluator"** for the noun/module/class, and "run"/"interpret"
  as informal verbs. The point for a reader: *it is one thing, not
  two.*)

- **Compiler backend** (the contrast class): also produces outputs,
  but **indirectly** — it *translates* the IR into another
  representation (MLIR sparse-tensor dialect, TeAAL IR, numpy/scipy
  code, egglog rules) and *that artifact* computes the result. Fast,
  but you must trust the translation, which is why backends are
  differential-tested against the evaluator. Out of scope here; named
  only to explain a layering decision below.

- **Why this matters for the registry's location:** *both* the
  evaluator and any future compiler backend consume the same IR *and
  the same UDF registry* (a backend needs to know `min` is
  commutative+associative to emit a parallel reduction; the evaluator
  needs the callable). The validator *also* needs the registry, and it
  runs *before* either. So the registry must **not** live inside
  `edge_ir/evaluator/`. It gets its own top-level package,
  `edge_ir/udf/`, that `validator/`, `evaluator/`, and future
  `backends/` all import from. (The existing
  `edge_ir/evaluator/datatype_registry.py` stub folds into this.)

---

## 2. The full set of UDF categories ("anything else that is udf?")

Yes — beyond the three you listed (map/reduce/populate compute,
coordinate, datatype) there are **three more** that are user-definable:
unary operators, rank-mapping functions, and boolean (stopping)
functions [the last one **being revisited** — see review notes].
**Merge operators are built-in only** (the 16 in `op_properties.py`)
and are **not** in this registry. Here is the table, cross-referenced
to the IR nodes that name each one.

| # | Category | IR node that carries the name | Where it appears in a program | Python contract (Python-only-for-now) | Extra metadata the *declaration* needs |
|---|---|---|---|---|---|
| 1 | **Map compute op** | `UserDefinedComputeOp` (in `op.py`) | `MapSpec.compute_op` | `(left: V, right: V) -> V` — pointwise combine of two operand *values* | — |
| 2 | **Reduce compute op** | `UserDefinedComputeOp` | `ReduceSpec.compute_op` | `(acc: V, val: V) -> V` — folded over a rank | identity element; is it associative? commutative? (needed for reduction well-definedness *and* for egglog/backends) |
| 3 | **Populate compute op** | `UserDefinedComputeOp` | `PopulateSpec.compute_op` | `(left: V, right: V) -> V` — value then handed to the coordinate op | — |
| 4 | **Coordinate op** | `CoordinateOp` (in `op.py`; *user-defined only*, no builtins) | `PopulateSpec.coord_op` | `(value: V, output_fiber: Fiber) -> Iterable[Coord]` — given the computed value and the current state of the output fiber being populated, yield the coordinate(s) to write at | — |
| 5 | **Unary op** | `UserDefinedUnaryOp` (in `op.py`; `not` builtin + UDF) | `UnaryApp.op` | `(value: V) -> V` — transform one element | — |
| 6 | **Rank-mapping function** | `RankFunction.func_name` (in `expr.py`) | `RankFunction` (inside a `TensorProjection`'s rank list) | `(*rank_vars: int) -> int` — maps iteration-space integer coords to a coordinate (the canonical case is `min(a, w)` from connected components, non-affine) | `is_affine: bool` (the validator can't derive this for an opaque fn; see §10.3) |
| 7 | **Boolean / stopping function** *(provisional — being revisited)* | `StoppingCondition.boolean_function` (in `einsum.py`) | `StoppingCondition` (in a `NestedCascade`) | `(generational_fiber) -> bool` — true ⇒ stop iterating (canonical case: `occupancy_zero`) | — (note the **expressivity gap**, audit Finding 5: today it can only see the generational fiber, not arbitrary tensor args; flagged in §10.5) |
| 8 | **User-defined data type** | `UserDefinedDataType` (in `tensor.py`) | `TensorDeclaration.data_type` | *not a function* — a small class: `validate(v) -> bool`, `equal(a, b) -> bool`, `empty() -> V`, optionally `to_json(v)` / `from_json(j)` for serialization | — |

**Not a UDF category — merge operators.** The IR's `op.py` currently
also defines `UserDefinedMergeOp`, but per the review notes merge ops
are built-in only (the 16 truth tables in `op_properties.py`). So the
registry has no merge slot, there is no `merge/` folder, and there is
no `@merge_op` decorator. Whether `UserDefinedMergeOp` should be
*removed from the IR* is a separate code-change question — see §14.

Notes on the table:

- **Compute ops are one IR category, but you want three folders.**
  The IR's `UserDefinedComputeOp` does *not* record "I am a map
  compute op" — the *spec it sits in* does (`MapSpec.compute_op` vs
  `ReduceSpec.compute_op` vs `PopulateSpec.compute_op`). So at the IR
  level there is one "compute op" namespace. You want the *user-facing*
  folders split because the **contracts differ** (a reduce compute op
  is folded, so it had better be associative; a populate compute op's
  result is consumed by a coordinate op). That split can be enforced
  to varying degrees — see the design question in §10.1.
- **`take_left` ambiguity** (already documented in `op.py`): the paper
  symbol `take_left` is both one of the 16 built-in *merge* ops *and*
  a plausible user-defined *compute* op (a function returning its left
  argument value). Different categories; the IR tells them apart by
  field (`merge_op` vs `compute_op`), and the registry must too —
  which is why category is part of the lookup key (see §7). (This
  remains true even though merge ops aren't in the registry: the
  *built-in* `take_left` merge op lives in `op_properties.py`, the
  *user-defined* `take_left` compute op would live in the registry.)

---

## 3. User-facing directory layout

A UDF set is a **directory** with this fixed shape. The user creates
this once for their project (or per program). The loader (see §7)
discovers every `.py` file under it.

```
my_udfs/                         # ← user picks the name; passed to load_udfs()
├── compute/
│   ├── map/                     # @map_compute_op    — (left, right) -> value
│   │   ├── saturating_add.py
│   │   └── ...
│   ├── reduce/                  # @reduce_compute_op — (acc, val) -> value  (must be assoc.)
│   │   ├── min.py
│   │   ├── logical_or.py
│   │   └── ...
│   └── populate/                # @populate_compute_op — (left, right) -> value
│       └── ...
├── coordinate/                  # @coordinate_op     — (value, output_fiber) -> Iterable[coord]
│   └── frontier_targets.py
│                                # (no merge/ folder — merge ops are built-in only; see review notes)
├── unary/                       # @unary_op          — (value) -> value
│   └── ...
├── rank_mapping/                # @rank_mapping_fn   — (*ints) -> int  ; carries is_affine
│   └── min_label.py
├── boolean/                     # @boolean_fn        — (generational_fiber) -> bool   [PROVISIONAL — being revisited]
│   └── occupancy_zero.py
└── datatypes/                   # @datatype          — class with validate/equal/empty/...
    └── complex64.py
```

Design choices baked into this layout (each revisited in §10):

- **One file per UDF** is the recommended convention (greppable,
  small, easy to point a teammate at), but the loader does not enforce
  it — a file may declare several. The *folder* a file lives in is the
  organizational signal; the *decorator* is the authoritative
  category tag (explicit beats magic — see §10.2).
- **Built-in defaults ship in the repo following the exact same
  layout** under `edge_ir/udf/builtins/` (so `min`, `max`, `logical_or`,
  `logical_and`, `take_left`, `take_right`, `occupancy_zero`, … are
  literally a worked example the user can copy from). `default_registry()`
  loads that directory.
- **Multi-language future:** today every file is Python. When other
  languages land, a sibling `udf_manifest.{json,toml}` in the
  directory root maps `name → {language, location, entrypoint,
  contract}`, and the loader dispatches to a language-specific
  `UdfImpl` subclass. The *declaration* the validator sees is identical
  regardless of impl language. Not built now; the seam is the
  `UdfImpl` ABC in §7.

---

## 4. What the user writes — examples per category

These are illustrative; exact decorator names/kwargs are part of the
discussion. `V` = "whatever Python value the relevant tensor's data
type uses" (int/float/bool for builtins, or the user's datatype class
instances).

### 4.1 Map compute op

```python
# my_udfs/compute/map/saturating_add.py
from edge_ir.udf import map_compute_op


@map_compute_op("saturating_add")
def saturating_add(left: int, right: int) -> int:
    """Add, clamped at 255. Used as a Map compute op."""
    return min(left + right, 255)
```

### 4.2 Reduce compute op (note the extra metadata)

```python
# my_udfs/compute/reduce/min.py
import math
from edge_ir.udf import reduce_compute_op


@reduce_compute_op("min", identity=math.inf, associative=True, commutative=True)
def min_reduce(acc: float, val: float) -> float:
    """Reduction by minimum. identity is +inf so an empty reduction = +inf."""
    return val if val < acc else acc
```

The `identity` / `associative` / `commutative` kwargs go into the
*declaration*. The evaluator uses `identity` to seed an empty
reduction; a future MLIR/egglog backend uses `associative` /
`commutative` to legally parallelize/reorder. The validator may
*warn* if a reduce compute op is declared non-associative (the
paper's reduction semantics assume associativity) — exact policy is a
discussion point.

### 4.3 Populate compute op + coordinate op (they come as a pair)

```python
# my_udfs/compute/populate/identity_value.py
from edge_ir.udf import populate_compute_op


@populate_compute_op("identity_value")
def identity_value(left, right):
    """The 'value' a Populate computes; its coordinate op decides *where* it lands."""
    return left  # whatever the algorithm needs as the populated value
```

```python
# my_udfs/coordinate/frontier_targets.py
from edge_ir.udf import coordinate_op


@coordinate_op("frontier_targets")
def frontier_targets(value, output_fiber):
    """Given a computed value and the current output fiber, yield the
    coordinate(s) to populate. (Sketch — real signature for Fiber TBD
    when the evaluator's tensor representation is designed.)"""
    if value:  # value is "present/non-empty"
        yield from output_fiber.absent_coords()
```

### 4.4 Merge op — *removed*

Merge operators are built-in only (the 16 truth tables in
`op_properties.py`); there is no `@merge_op` decorator and no `merge/`
folder. (Slot kept so the other §4.x numbers don't shift.) See review
notes.

### 4.5 Unary op

```python
# my_udfs/unary/clamp_nonneg.py
from edge_ir.udf import unary_op


@unary_op("clamp_nonneg")
def clamp_nonneg(value: float) -> float:
    return value if value > 0.0 else 0.0
```

### 4.6 Rank-mapping function (carries the affine flag)

```python
# my_udfs/rank_mapping/min_label.py
from edge_ir.udf import rank_mapping_fn


@rank_mapping_fn("min_label", affine=False)  # min(a, w) is non-affine
def min_label(a: int, w: int) -> int:
    return a if a < w else w
```

The `affine=` kwarg is what the validator can't otherwise know for an
opaque function (see §10.3 for whether this lives here, on the IR
node, or both).

### 4.7 Boolean / stopping function — *PROVISIONAL, being revisited*

A separate design pass on stopping conditions is pending; the
shape below is a placeholder, not a commitment. Do not build against
it yet.

```python
# my_udfs/boolean/occupancy_zero.py   (PROVISIONAL)
from edge_ir.udf import boolean_fn


@boolean_fn("occupancy_zero")
def occupancy_zero(generational_fiber) -> bool:
    """Stop when the latest frontier generation has no present coords.
    (Today the fn only sees the generational fiber — audit Finding 5
    is the gap that this should eventually take tensor args.)"""
    return generational_fiber.nnz() == 0
```

### 4.8 User-defined data type

```python
# my_udfs/datatypes/complex64.py
from dataclasses import dataclass
from edge_ir.udf import datatype


@datatype("complex64")
@dataclass
class Complex64:
    re: float
    im: float

    @staticmethod
    def validate(v: object) -> bool:
        return isinstance(v, Complex64)

    @staticmethod
    def equal(a: "Complex64", b: "Complex64") -> bool:
        return a.re == b.re and a.im == b.im

    @staticmethod
    def empty() -> "Complex64":
        return Complex64(0.0, 0.0)

    # optional, for JSON-serialized input/output tensor data:
    @staticmethod
    def to_json(v: "Complex64"):
        return {"re": v.re, "im": v.im}

    @staticmethod
    def from_json(j):
        return Complex64(j["re"], j["im"])
```

The `@datatype("complex64")` registration lets a tensor declare
`data_type = UserDefinedDataType(name="complex64")`, and lets the
validator check that a `TensorDeclaration.empty_value` is a valid
instance via `Complex64.validate(...)` (this is the
`_check_empty_values_match_data_types` step in the validator README).

---

## 5. How the Pydantic IR deals with all of this — *it doesn't change*

This is the important property and worth stating loudly:

- Every UDF reference in the IR is **already** just a name string
  validated against `patterns.REGEX_USER_DEFINED_NAME`
  (`^[A-Za-z_][A-Za-z0-9_-]*$`):
  - `UserDefinedComputeOp.name`
  - `UserDefinedMergeOp.name` *(present in the IR today, but the
    registry does not use it — merge ops are built-in only; see review
    notes / §14 for whether to remove this from the IR)*
  - `CoordinateOp.name`
  - `UserDefinedUnaryOp.name`
  - `RankFunction.func_name` (+ its own `is_affine: bool | None`)
  - `StoppingCondition.boolean_function`
  - `UserDefinedDataType.name`
- The IR has **no field for an implementation**, and should not get
  one — the IR is "strictly data" (see `IRBase`: `frozen`, `strict`,
  `extra="forbid"`) and stays JSON-round-trippable. A `Program` is
  portable; the registry that gives its UDF names meaning is supplied
  separately at validate/evaluate time.
- So adding the UDF registry feature touches **zero files in
  `edge_ir/ir/`**. (Possible *optional* exception: the affine
  annotation story in §10.3 — but even that is "already there" via
  `RankFunction.is_affine` on the IR plus the side-table pass at
  `edge_ir/analysis/affine.py` for `RankArith` affineness.)
- Concretely: when the parser/lowering eventually turns surface
  syntax into IR, a `min(a, w)` rank mapping becomes
  `RankFunction(func_name="min", args=[RankVariable(name="a"),
  RankVariable(name="w")], is_affine=None)` — *no reference to any
  Python object*. Resolving `"min"` to a callable is 100% the
  evaluator's job; checking `"min"` is *registered with the right
  arity* is 100% the validator's job.

What *is* new code, all **outside** `edge_ir/ir/`:

- `edge_ir/udf/` — the registry, decl/impl types, decorators, loader,
  built-in defaults (see §7).
- `edge_ir/validator/` — the Layer 2 checks that consult the registry
  (see §8). (Currently a README stub.)
- `edge_ir/evaluator/` — uses the registry to actually run UDFs (see
  §9). (Currently a `datatype_registry.py` stub that this subsumes.)

---

## 6. The declaration / implementation split

Each registered UDF has two halves:

- **Declaration** (`UdfDecl`, pure data — a frozen pydantic
  `IRBase`-style model, no callables): `name`, `category`
  (the enum), `arity` (or "variadic"), parameter/return type contract
  (as far as it can be expressed — at minimum "value", "presence",
  "coord", "int"), and category-specific extras (`identity` +
  `associative` + `commutative` for reduce compute; `is_affine` for
  rank mapping). This is *all the validator needs* — it never calls a
  UDF.
- **Implementation** (`UdfImpl`, an ABC): wraps however the function
  actually runs. `PythonUdfImpl` holds a Python callable. Future:
  `SubprocessUdfImpl`, `WasmUdfImpl`, `CtypesUdfImpl`, … For data
  types, the analogous pair is `DataTypeDecl` (just the name +
  whatever metadata) and `DataTypeImpl` (`validate`/`equal`/`empty`/
  optional json hooks); `PythonDataTypeImpl` wraps the user's class.

Why split: the validator is supposed to give a downstream consumer
(another evaluator, a compiler) confidence the program is well-formed
*without needing to execute anything* — so it must be able to do its
job from declarations alone. It also means a `Program` + the *decls*
(which are themselves serializable) is enough to typecheck a program
on a machine that can't run the impls (different language, sandbox,
CI lint job).

---

## 7. The `UdfRegistry` and how UDFs get loaded

### 7.1 The registry object

```python
# sketch — edge_ir/udf/registry.py
class UdfRegistry:
    # internally: dict[(UdfCategory, str), tuple[UdfDecl, UdfImpl]]
    #            + dict[str, tuple[DataTypeDecl, DataTypeImpl]]   (datatypes)
    def decl(self, category: UdfCategory, name: str) -> UdfDecl: ...
    def impl(self, category: UdfCategory, name: str) -> UdfImpl: ...
    def datatype_decl(self, name: str) -> DataTypeDecl: ...
    def datatype_impl(self, name: str) -> DataTypeImpl: ...

    # lookups raise a structured UdfLookupError (collected, not fatal,
    # so the validator can report all missing UDFs at once)
```

The lookup key is `(category, name)` — this is what lets the
map/reduce/populate-compute split be real if we want it (§10.1), keeps
a `coordinate` UDF named `foo` from colliding with an `unary` UDF
named `foo`, and lets a user-defined `take_left` *compute* op coexist
with the built-in `take_left` *merge* op (which lives in
`op_properties.py`, not here).

### 7.2 Loading

```python
# sketch — edge_ir/udf/__init__.py
def load_udfs(path: str | Path) -> UdfRegistry:
    """Walk `path` (a directory with the §3 layout), import every .py
    file so its decorators fire, return the populated registry."""


def default_registry() -> UdfRegistry:
    """Load the built-ins shipped under edge_ir/udf/builtins/."""


def load_udfs(path, *, base: UdfRegistry | None = None) -> UdfRegistry:
    """...starting from `base` (typically default_registry()) so users
    only have to author the *extra* UDFs, not re-declare min/max/etc."""
```

Mechanism for "the decorator registers somewhere": during a
`load_udfs` call, a context-local (`contextvars`) "currently-loading
registry" is set; `@map_compute_op(...)` etc. append to it. **Outside**
a load, a decorated function is just a normal function (no global
mutation) — so importing a UDF module in a unit test doesn't pollute
anything. This gives both ergonomics (just write decorated functions in
the right folder) and purity (the registry is a value, threaded
explicitly downstream).

### 7.3 Where the registry gets handed off

```python
validate_program(prog: Program, registry: UdfRegistry) -> list[ValidationError]
evaluate(prog: Program, inputs: TensorData, registry: UdfRegistry) -> TensorData
# (and future) compile_to_mlir(prog: Program, registry: UdfRegistry) -> MlirModule
```

This matches the signature already sketched in
`edge_ir/validator/README.md` (`validate_program(prog, registry)`).

---

## 8. How the **validator** hooks in

The Layer 2 validator (see `docs/interpreter_future_plan.md` §"Layer 2"
and `edge_ir/validator/README.md`) consults **only the declarations**:

- **`_check_all_udf_names_registered`** — walk the IR; for every
  `UserDefinedComputeOp` collect `(compute_category, name)` where
  `compute_category` is `map`/`reduce`/`populate` *depending on which
  spec it's inside* (the IR walk knows this structurally); for every
  `CoordinateOp`, `UserDefinedUnaryOp`, `RankFunction`, and
  `UserDefinedDataType` → look up `registry.decl(category, name)`.
  (`StoppingCondition.boolean_function` *will* be in this set once the
  stopping-condition redesign lands — §10.5 — but is provisional for
  now. `UserDefinedMergeOp` is *not* checked against the registry —
  merge ops are built-in only; if an IR program somehow contains a
  `UserDefinedMergeOp`, that's either an error to flag or a reason to
  remove the node from the IR — see §14.) Missing → a collected
  `ValidationError` with the location (which Einsum, which spec, which
  field path).
- **Arity / contract checks** — `RankFunction.args` length must match
  the rank-mapping decl's arity; a binary compute op decl must be
  arity-2; a unary op decl arity-1; etc.
- **Operator-category-in-the-right-slot** — already on the validator's
  list ("a `MapSpec.compute_op` cannot be a `CoordinateOp`"); the
  registry's category axis is exactly what makes this checkable.
- **`_check_empty_values_match_data_types`** — for a builtin
  `data_type`, check `empty_value` is a Python `int`/`float`/`bool`
  (with the "inf"/"-inf"/"nan" sentinel decoding already handled in
  `tensor.py`); for a `UserDefinedDataType`, call the registry's
  `datatype_decl(name)` and (if we decide the validator may touch
  impls for this one case) `datatype_impl(name).validate(empty_value)`.
  — *Decision point:* does the validator stay impl-free and only
  check the *name* resolves, leaving `empty_value` validation to the
  evaluator? Cleaner layering vs. catching the error earlier. (See
  §10.4.)
- **Affine-analysis pass** — when it hits a `RankFunction`, it pulls
  `is_affine` from the rank-mapping decl (the `affine=` kwarg) rather
  than guessing. (See §10.3.)
- **`_check_reduce_ops_associative`** (new, optional) — warn/error if
  a `ReduceSpec.compute_op` resolves to a reduce compute decl marked
  `associative=False`.

All errors are *collected*, not raised eagerly, so a UI gets every
problem in one pass (already the validator's stated contract).

---

## 9. How the **evaluator** (interpreter) hooks in

The evaluator consults **the implementations**:

- **Iteration-space construction** — when resolving a `RankFunction`
  in a `TensorProjection`, it calls `registry.impl(RANK_MAPPING,
  func_name).call(*coord_ints)` to get the coordinate.
- **Per-Einsum execution** — for each binary label in the expression
  it finds the matching `ComputationSpec`:
  - `MapSpec` → `compute = registry.impl(MAP_COMPUTE,
    compute_op.name)` (or the builtin from `op_properties` if it's a
    `BuiltinComputeOp`); the merge op is **always built-in** — look up
    its truth table in `op_properties.py` by `merge_op.symbol` (no
    registry call). For each iteration point: the merge truth table on
    the presence flags decides effectfulness; `compute` on the values
    decides the result.
  - `ReduceSpec` → fold `registry.impl(REDUCE_COMPUTE, ...).call` over
    the listed ranks, seeded with the decl's `identity`; the built-in
    merge truth table (from `op_properties.py`) handles empty cases.
  - `PopulateSpec` → `compute = registry.impl(POPULATE_COMPUTE, ...)`
    produces the value; `coord = registry.impl(COORDINATE,
    coord_op.name)` consumes `(value, output_fiber)` to decide which
    coordinates to write.
  - `UnaryApp` → `registry.impl(UNARY, op.name).call(value)` (or
    builtin `not`).
- **Cascade iteration** — after each pass, evaluate each
  `StoppingCondition` via `registry.impl(BOOLEAN,
  boolean_function).call(generational_fiber)`; stop when true; bump
  the generational rank; enforce a max-iteration cap.
- **Data types** — when reading/writing tensor elements of a
  `UserDefinedDataType`, use `registry.datatype_impl(name)` for
  `equal` (e.g. to decide if a value equals the empty value),
  `empty()`, and json hooks for serialized input/output data.

The evaluator never imports user code directly — it only ever goes
through `UdfRegistry`. (Same for any future compiler backend, which
uses the *decls'* algebraic-property metadata to choose legal
rewrites/parallelizations.)

---

## 10. Open design questions (the stuff to actually discuss)

### 10.1 Are map/reduce/populate compute ops one namespace or three?

The IR has *one* `UserDefinedComputeOp` category; the action it serves
is determined by the enclosing spec. But you want three *folders* and
noted "different signature requirements." Options:

- **A — one namespace, folders are cosmetic.** Register `min` once;
  it's usable as any compute op. The folder split is documentation
  only; the contract differences ("reduce compute must be
  associative") are conventions the validator *can't* enforce per-use.
  Simplest. Matches the IR.
- **B — three namespaces keyed by `(action, name)`.** You must
  `@reduce_compute_op("min")` to use `min` in a `ReduceSpec`. The
  validator checks a `MapSpec.compute_op.name` resolves *specifically*
  in the map-compute namespace. Strongest guarantees; matches your
  mental model and the folder layout 1:1. Costs: more registration
  boilerplate; the registry's category enum gains three compute
  variants; a name reused across actions needs three decorators (or
  see C).
- **C — one decl, multi-action opt-in.** `@compute_op("min",
  actions=["map", "reduce"], identity=..., associative=True)`. The
  folder a file sits in just sets the *default* `actions`. The decl
  records which actions it's valid for; the validator checks the use
  site is in that set. Middle ground.

Recommendation: **C** (or B if you want the folder layout to be
load-bearing rather than advisory). Worth deciding early — it shapes
the `UdfCategory` enum and every decorator name.

### 10.2 Does the folder imply the category, or does the decorator?

Recommendation: **the decorator is authoritative**; the folder is an
organizational convention the loader does *not* parse. (Folder-implies-
category is cute but breaks the moment someone reorganizes, vendors a
file, or wants two categories in one file. Explicit > implicit.) Open
to the inverse if you'd rather the folders be enforced structure.

### 10.3 Where does the affine flag for a rank-mapping fn live?

Three candidates: (a) the `@rank_mapping_fn(..., affine=False)` kwarg
on the *decl* (single source of truth, but then the IR's
`RankFunction.is_affine` is redundant); (b) the IR node
`RankFunction.is_affine` (set by the author of the program, but they
might not know); (c) both, with the validator's affine pass copying
decl→IR and erroring on conflict. Recommendation: **(a) is the source
of truth; the validator's affine pass populates `RankFunction.is_affine`
from it** (so backends that only see the IR still get the answer), and
flags a conflict if the program author also set it inconsistently.

### 10.4 May the validator touch *implementations* at all?

The clean rule is "validator: decls only; evaluator: impls." The one
place it's tempting to break that is validating `empty_value` against
a user-defined data type (you'd want `Complex64.validate(empty_value)`
at *validate* time, not blow up later in the evaluator). Options:
keep the validator pure and accept the error surfaces at evaluate
time; OR allow the validator to call *only* `DataTypeImpl.validate`
(a deliberately narrow exception); OR require user data types to also
provide a *pure-data* description of valid `empty_value`s (overkill).
Recommendation: **narrow exception** — the validator may call
`DataTypeImpl.validate`, nothing else.

### 10.5 The stopping-function expressivity gap (audit Finding 5) — *a dedicated pass on this is pending; treat as provisional*

Today `StoppingCondition` carries only a `rank_variable` and a
`boolean_function` name — the function can't be handed arbitrary
tensor arguments, so conditions like "stop when `‖F_{i+1}‖ ≡ 0`" work
(it's *the* generational fiber) but "stop when `A` and `B` agree"
don't. The boolean-fn contract in §4.7 reflects today's reality
(`(generational_fiber) -> bool`). If/when the IR grows the ability to
pass tensor args to a stopping condition, this contract widens to
`(*tensor_snapshots) -> bool`. Flag for the paper-side discussion;
the registry design should *not* over-fit to the narrow form (keep
the impl signature open-ended / pass a small context object rather
than a bare fiber).

### 10.6 Registry composition / overriding

`load_udfs(path, base=default_registry())` lets a user add to the
built-ins. Should a user be allowed to *override* a built-in name
(e.g. redefine `min`)? Recommendation: **no, by default** — collision
raises (like `TypeRegistry.register` already does); add an explicit
`allow_override=True` escape hatch if a real need shows up. Keeps
programs portable (a `Program` referencing `min` means the same thing
everywhere).

### 10.7 How are *input/output tensor data* represented & serialized?

Out of scope for the registry per se, but the coordinate-op and
data-type contracts (`output_fiber`, `to_json`/`from_json`) can't be
finalized until the evaluator's tensor/fiber representation is
designed. Track jointly. (The §4 examples deliberately keep `Fiber`
abstract.)

---

## 11. Built-in defaults to ship (following the §3 layout)

Under `edge_ir/udf/builtins/` so `default_registry()` loads them and
they double as worked examples:

- `compute/reduce/`: `min`, `max`, `logical_or`, `logical_and`,
  `add` (the BFS `+` reduction), …
- `compute/map/`: `take_left`, `take_right` (the *compute-op* flavor,
  distinct from the built-in merge ops of the same paper symbol), …
- `coordinate/`: whatever the BFS `ANY`/populate examples need.
- (no `merge/` — merge ops are built-in only, in `op_properties.py`.)
- `unary/`: example only (`not` is builtin).
- `rank_mapping/`: `min_label` for connected components (non-affine).
- `boolean/`: `occupancy_zero` for BFS — *provisional, gated on the
  stopping-condition redesign*.
- `datatypes/`: an example (e.g. a `fraction` type) for docs.

Acceptance check: with `default_registry()`, the BFS artifact at
`examples/algorithms/bfs/bfs_program.json` passes `validate_program` *and*
`evaluate` on a small hand-checked graph (this is already the stated
milestone in `docs/interpreter_future_plan.md`; the registry is the
missing piece). Note BFS now uses `F → float, empty = inf`.

---

## 12. Testing strategy

- **Loader**: a fixture UDF directory under `tests/fixtures/udfs/`;
  assert `load_udfs` discovers each category, that decorators outside
  a load don't mutate global state, that collisions raise, that
  `base=` composition works.
- **Decls**: each category — a positive registration and a malformed
  one (wrong arity, missing `identity` on a reduce compute, bad name
  vs. `REGEX_USER_DEFINED_NAME`) → right error.
- **Validator × registry**: program references an unregistered UDF →
  collected `ValidationError` at the right path; wrong-category use
  (`CoordinateOp` in a `MapSpec` slot) → error; arity mismatch on a
  `RankFunction` → error; `empty_value` invalid for a user datatype →
  error (per §10.4 decision).
- **Evaluator × registry**: hand-checked small graphs; UDF actually
  invoked (not just name-resolved); reduce uses `identity` for the
  empty case; stopping fn terminates the cascade.
- **Determinism**: `(Program, inputs, registry)` → same output bytes
  every run (a Hypothesis target from the future plan).

---

## 13. Out of scope (so it doesn't drift back in)

- Actually implementing non-Python `UdfImpl` backends (only the ABC
  seam + the manifest sketch are in scope).
- Sandboxing user UDF code.
- Hot-reloading UDFs.
- A package/registry-server distribution story for sharing UDF sets.
- Changing the IR (`edge_ir/ir/`) — this feature must not require it
  (see §5).

---

## 14. Decisions needed before coding (checklist for the next conversation)

**Settled in the review** (see review notes at top):
- Merge ops are **built-in only** — no `merge/` folder, no `@merge_op`,
  no merge slot in the registry. ✅
- Unary ops are a UDF category. ✅
- Rank-mapping functions are a UDF category. ✅

**Still open:**

1. §10.1 — one compute namespace, three, or multi-action opt-in? (Drives
   the `UdfCategory` enum and the decorator names.)
2. §10.2 — decorator-authoritative vs folder-authoritative for category?
3. §10.3 — affine flag on the decl, the IR node, or both-with-reconcile?
4. §10.4 — may the validator call `DataTypeImpl.validate`, or stay
   strictly impl-free?
5. §10.6 — allow overriding built-in UDF names? (default: no)
6. Naming: package name `edge_ir/udf/` vs `edge_ir/registry/` vs
   `edge_ir/user_defined/`; and "evaluator" vs "interpreter" as the
   project-wide term for the execution engine (§1).
7. §10.7 — is the tensor/fiber data representation far enough along to
   nail down the `coordinate_op` and `datatype` json contracts, or do
   we stub `Fiber` for now?
8. **New, from the review:** should `UserDefinedMergeOp` be **removed
   from `edge_ir/ir/op.py`** (since merge ops are built-in only)? If
   removed: a code change to `op.py` + the `MergeOp` union shrinks to
   just `BuiltinMergeOp` + update `op.py`/`actions.py` docstrings + a
   test pass. If kept: the validator should treat any
   `UserDefinedMergeOp` instance as a validation error. (This feature
   per §13 "must not require an IR change" — so removing it is a
   *separate* decision/PR, not part of the registry work.)
9. **Pending:** the stopping-condition / boolean-function
   redesign (§4.7, §10.5). Until that lands, `boolean` stays out of
   the first implementation slice.
