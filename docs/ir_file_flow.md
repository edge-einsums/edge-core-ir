# EDGE IR — file flow and module map

> *Created with the help of Claude Code.*

This document describes the layout of `edge_ir/` as of the IR build-out
completion (Step 7 of the implementation plan). It is a guide to
"what does this file do, and what depends on it." For semantics
questions, the EDGE Notation Paper is authoritative.

This document does NOT describe the AST, validator, evaluator, or
verification harness. Those subpackages exist as stubs but are out of
scope for the current IR layer.

---

## 1. The four layers (current state vs future state)

```
                     ┌─────────────────────────┐
                     │   EDGE source text       │   .edge files (future)
                     └────────────┬─────────────┘
                                  │ parser (NOT BUILT)
                                  ▼
                     ┌─────────────────────────┐
                     │   AST (edge_ir/ast/)    │   stub directory
                     └────────────┬─────────────┘
                                  │ AST→IR lowering (NOT BUILT)
                                  ▼
                     ┌─────────────────────────┐
                     │   IR (edge_ir/ir/)      │   ◄── THIS LAYER IS DONE
                     └────────────┬─────────────┘
                                  │ Layer 2 validator pass (NOT BUILT)
                                  │   • cross-reference resolution
                                  │   • affine analysis on RankArith
                                  ▼
                     ┌─────────────────────────┐
                     │  Validated IR (same     │
                     │  shape, with metadata)  │
                     └────────────┬─────────────┘
                                  │ evaluator / backend (NOT BUILT)
                                  ▼
                     ┌─────────────────────────┐
                     │   Reference output      │
                     │   or compiled artifact  │
                     └─────────────────────────┘
```

The IR layer (`edge_ir/ir/`) is pure data — no methods on IR nodes, no
evaluation logic. It is JSON-serializable end to end. The validator,
evaluator, and any compiler backend are downstream consumers that read
the IR but do not modify it.

The IR is intentionally permissive about cross-references: a
`TensorProjection` names its tensor by string, but the IR does not
verify that the string resolves to a declared `TensorDeclaration`.
Resolution is the validator's job (Layer 2). This separation is what
lets the IR round-trip through JSON without losing information.

---

## 2. Top-level repo layout

```
edge-ir-interpreter/
├── edge_ir/                    # the package
│   ├── __init__.py             # empty
│   ├── ir/                     # ◄── IR layer (done)
│   ├── runtime/                # ◄── operator-properties registry (done)
│   ├── ast/                    # stub
│   ├── validator/              # stub (has a README sketch)
│   ├── evaluator/              # stub (has a datatype_registry sketch)
│   └── verify/                 # stub
├── tests/                      # pytest, mirrors the IR layout
│   ├── test_ir_*.py            # one per IR module
│   ├── test_op_properties.py
│   ├── test_bfs_artifact.py
│   └── ...
├── examples/algorithms/            # per-algorithm IR builders + artifacts
│   ├── bfs/                    # depth-tracking BFS
│   │   ├── einsum.md           #   paper math (LaTeX)
│   │   ├── build_bfs_program.py#   IR builder script
│   │   └── bfs_program.json    #   generated IR artifact
│   └── max-flow/               # full-edge push-relabel
│       ├── einsum.md
│       ├── pseudocode.md
│       ├── metadata.md         #   coverage + gap notes
│       └── build_max_flow_program.py
├── schemas/
│   └── program.schema.json     # JSON Schema generated from Program
├── docs/                       # design notes
│   ├── ir_file_flow.md          #   (this file)
│   ├── debug.md                 #   audit findings + fix log
│   ├── interpreter_future_plan.md  # long-term roadmap
│   └── ...
├── Makefile                    # `make check`, `make format`, etc.
├── pyproject.toml              # ruff + mypy + pytest config
└── README.md
```

---

## 3. The IR modules — what each one defines

### `edge_ir/ir/base.py`
Defines `IRBase`, the parent class of every IR node. Configures
`frozen=True`, `strict=True`, `extra="forbid"` once, so individual
nodes don't have to. A node that inherits from `IRBase` is immutable
after construction, rejects unknown fields, and does not silently
coerce types.

**Depends on:** Pydantic only.
**Imported by:** every other IR module.

### `edge_ir/ir/patterns.py`
Regex patterns for the lexical classes from the EDGE grammar:
`REGEX_RANK_NAME`, `REGEX_TENSOR_NAME`, `REGEX_RANK_VARIABLE`,
`REGEX_USER_DEFINED_NAME`, `REGEX_BINARY_LABEL`, `REGEX_COORDINATE`.
This is the single source of truth for naming rules; when the grammar
changes, this file changes first.

**Depends on:** nothing.
**Imported by:** `tensor.py`, `expr.py`, `op.py`, `einsum.py`.

### `edge_ir/ir/tensor.py`
Tensor declarations (the left-hand side of the EDGE program's
`<declarations>` section). Defines:
- `BuiltinDataType` / `UserDefinedDataType` + `DataType` discriminated union
- `CoordSetEnum` / `CoordSetInterval` / `CoordSetAlias` + `CoordinateSet`
  discriminated union (for explicit coordinate sets — sparse subsets,
  alias-to-other-tensor's-rank, or interval ranges)
- `RankDeclaration` — one rank with `name`, optional `shape`, optional
  `coord_set`. Generational ranks (no shape known up front, like the
  `i` in BFS) leave `shape=None`.
- `TensorDeclaration` — name + ranks + data type + empty value

**Depends on:** `base.py`, `patterns.py`.
**Imported by:** `program.py`, `examples/algorithms/bfs/build_bfs_program.py`.

### `edge_ir/ir/expr.py`
The right-hand side of an Einsum, plus rank-expression machinery for
tensor accesses. Defines:

**Rank expressions** (what appears in tensor subscripts):
- `RankVariable` — bare iteration variable, e.g. `m`, `sm`
- `RankConstantLiteral` — a literal coordinate value: an integer, or a
  label from a declared char/string coord set
- `RankConstantShapeSym` — a reference to a declared shape parameter
  (`|V|`, `N`, `M`, ...); the shape resolver substitutes the bound
  integer at iteration-space construction
- `RankArith` — `m + 1`, `k * |V|`, `m % N`, etc. Affineness of an
  arith tree is computed by `edge_ir/analysis/affine.py` as a side
  table; it's NOT stored on the node.
- `RankFunction` — opaque user-defined mapping like `min(a, w)` for
  non-affine rank expressions
- `RankExpression` — discriminated union over the four

`TensorProjection` — a tensor name plus a list of rank expressions.
This is the form `A_{m, k}` from the paper.

**Expression tree** (the body of an Einsum):
- `InputTensor` — wraps a `TensorProjection`; the leaf case
- `UnaryApp` — a unary operator applied to an `InputTensor`
  (operand is restricted to `InputTensor`, not arbitrary `Expression`,
  per the grammar — paper Section 5.5)
- `BinaryApp` — a binary `.` between two Expressions, tagged with a
  `label: int` that links it to a matching `ComputationSpec`
- `AnonymousTensor` — a parenthesized sub-Einsum used as an operand;
  carries its own expression, output ranks, and computation specs
- `Expression` — discriminated union over the four. Recursive: a
  `BinaryApp` contains two more `Expression`s, an `AnonymousTensor`
  contains an `Expression`. Pydantic v2 + `from __future__ import
  annotations` handles the recursion automatically.

**Depends on:** `base.py`, `patterns.py`, `actions.py` (for
`ComputationSpec` inside `AnonymousTensor`), `op.py` (for `UnaryOp`).
**Imported by:** `einsum.py`, `examples/algorithms/bfs/build_bfs_program.py`.

### `edge_ir/ir/op.py`
The four operator categories the paper distinguishes. ASCII names
only — Unicode-to-ASCII translation is the parser's job; the IR sees
ASCII. Each category has a built-in variant (with a fixed `Literal`
symbol) and a user-defined variant (with a name).

- `BuiltinComputeOp(symbol ∈ {"+", "-", "*", "/"})` and
  `UserDefinedComputeOp(name)` → `ComputeOp` union
- `BuiltinMergeOp(symbol ∈ {16 names from Appendix A})` and
  `UserDefinedMergeOp(name)` → `MergeOp` union
- `CoordinateOp(name)` — single class, no union (no built-in
  coordinate operators per the grammar; only user-defined)
- `BuiltinUnaryOp(symbol ∈ {"not"})` and `UserDefinedUnaryOp(name)`
  → `UnaryOp` union

The 16 merge-op ASCII names map back to paper Unicode via the
registry in `runtime/op_properties.py`. Operator algebraic
properties are NOT stored on the operator nodes — they live in the
registry.

**Depends on:** `base.py`, `patterns.py`.
**Imported by:** `actions.py`, `expr.py`, `examples/algorithms/bfs/build_bfs_program.py`.

### `edge_ir/ir/actions.py`
The three EDGE actions the paper formalizes in Sections 6.4–6.6:
- `MapSpec(label, rank_list?, compute_op, merge_op)` —
  pointwise compute with merge-op gating
- `ReduceSpec(label, rank_list, compute_op, merge_op)` —
  collapse the listed ranks
- `PopulateSpec(label, rank_list, compute_op, coord_op)` —
  produce a coordinate set from a value
- `ComputationSpec` — discriminated union over the three

A `ComputationSpec.label` matches a `BinaryApp.label` in the
expression tree (this matching is a Layer 2 concern; the IR does not
enforce it).

**Depends on:** `base.py`, `op.py`.
**Imported by:** `expr.py` (for `AnonymousTensor.specs`),
`einsum.py` (for `Einsum.specs`),
`examples/algorithms/bfs/build_bfs_program.py`.

### `edge_ir/ir/einsum.py`
The middle-level structural shapes per paper Section 7.3:
- `Einsum(output_tensor, output_ranks, expression, specs)` — the
  workhorse: an output assignment plus the computation specs that
  apply to its binary operations
- `StoppingCondition(rank_variable, boolean_function)` — the
  `<>:` clause that stops a cascade's iteration
- `NestedCascade(einsums, stopping_conditions)` — the top-level
  cascade structure; flat list of einsums + flat list of stop
  conditions. The grammar's recursive `<nested-cascade>` is
  normalized to this flat form by the future lowering pass.

**Depends on:** `base.py`, `patterns.py`, `actions.py`, `expr.py`.
**Imported by:** `program.py`, `examples/algorithms/bfs/build_bfs_program.py`.

### `edge_ir/ir/program.py`
The top of the IR tree — a complete EDGE program:
- `Initialization(einsums)` — the `<initializations>` section,
  a list of Einsums (may be empty)
- `MainEdge(cascade)` — wraps the `NestedCascade` for the
  `<main-edge>` section
- `Program(declarations, initialization, main_edge)` — the three
  sections of the EDGE grammar, glued together

`Program` is the JSON schema's root type and the inter-layer contract
that any future parser must produce.

**Depends on:** `base.py`, `tensor.py`, `einsum.py`.
**Imported by:** `examples/algorithms/bfs/build_bfs_program.py`,
`tests/test_bfs_artifact.py`.

### `edge_ir/ir/__init__.py`
Currently empty. Re-exports could be added later but the IR is
intentionally opt-in: each consumer imports directly from the
relevant submodule.

---

## 4. The runtime registry — `edge_ir/runtime/op_properties.py`

This is **not** an IR module. It is a sibling of `edge_ir.ir`. The
IR does not import it, and it does not import the IR. This separation
is enforced by a test that runs in a subprocess
(`tests/test_op_properties.py`).

It holds:

1. **Unicode aliases** (`COMPUTE_OP_UNICODE`, `MERGE_OP_UNICODE`,
   `UNARY_OP_UNICODE`) — for any consumer that wants to render the
   ASCII IR names back into the paper's mathematical notation.

2. **Compute operator algebraic properties** — `ComputeOpProperties`
   frozen dataclass with `commutative`, `associative`, `identity`,
   `left_absorbing`, `right_absorbing`. The `COMPUTE_OP_PROPERTIES`
   dict has 4 entries (one per built-in).

3. **Merge operator semantics** — `MergeOpProperties` frozen
   dataclass with the 4-entry truth table, `commutative`, and
   `includes_neither` (whether the operator outputs True when both
   inputs are absent — affects whether compute is invoked with empty
   values per paper Section 6.4). The `MERGE_OP_PROPERTIES` dict
   has 16 entries.

4. **Lookup helpers** — `get_compute_props`, `get_merge_props`,
   `render_compute_unicode`, `render_merge_unicode`.

The evaluator and any future egglog rewriter consult this registry.
The IR is the discriminator (the ASCII symbol on a
`BuiltinComputeOp`); the registry is the lookup table for what that
symbol means.

---

## 5. Dependency graph (concise)

Within `edge_ir/ir/`, imports always flow from less-derived to
more-derived modules — there are no cycles:

```
                    base.py
                  ┌────┴────┐
                  ▼         ▼
            patterns.py   (used by all)
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
    tensor.py  op.py    (no other peers)
                  │
                  ▼
            actions.py
                  │
                  ▼  (also imports tensor's RankExpression… wait, it doesn't)
              expr.py  (imports actions.py + op.py)
                  │
                  ▼
            einsum.py  (imports actions.py + expr.py + patterns.py)
                  │
                  ▼
            program.py  (imports tensor.py + einsum.py)
```

Outside the IR layer:

```
runtime/op_properties.py  ─── independent (no IR imports)

examples/algorithms/bfs/build_bfs_program.py  ─── imports
    edge_ir.ir.{tensor, op, actions, expr, einsum, program}
    Constructs a Program in code, dumps it as indented JSON.

examples/algorithms/bfs/bfs_program.json  ─── output of the script.
                                Read by tests/test_bfs_artifact.py.

schemas/program.schema.json  ─── output of
    Program.model_json_schema(), generated post-build.
    The inter-layer contract for any future parser.
```

---

## 6. End-to-end flow: how a BFS Program lives

This is the round-trip flow for the only Program currently in the
artifact suite. It exercises every IR module.

```
1. examples/algorithms/bfs/build_bfs_program.py
   │
   │ in code, build:
   │   • 4 TensorDeclarations  (G, F, P, T) → tensor.py
   │   • 3 Einsums               (a, b, c)   → einsum.py
   │     └── each contains:
   │         • output_tensor + output_ranks (RankVariable, RankArith)
   │         • Expression tree (BinaryApp, InputTensor, UnaryApp,
   │           TensorProjection, RankExpression)
   │         • specs list (MapSpec / ReduceSpec / PopulateSpec)
   │     └── operators (BuiltinComputeOp, BuiltinMergeOp,
   │         BuiltinUnaryOp, UserDefinedComputeOp)
   │   • 1 StoppingCondition                  → einsum.py
   │   • NestedCascade wrapping einsums + stops → einsum.py
   │   • MainEdge wrapping cascade            → program.py
   │   • Initialization (empty)               → program.py
   │   • Program(declarations, initialization, main_edge) → program.py
   │
   ▼
2. program.model_dump_json(indent=2)
   │  Pydantic walks the tree, serializes each discriminated union
   │  variant with its "kind" tag.
   ▼
3. examples/algorithms/bfs/bfs_program.json   (~7.7 KB, 345 lines)
   │
   │ (this file is committed to the repo as the integration artifact)
   │
   ▼
4. tests/test_bfs_artifact.py
   │  Reads the JSON.
   │  Calls Program.model_validate_json(raw)
   │  → Pydantic re-builds the tree, dispatching each "kind" through
   │     its discriminator.
   │  Asserts:
   │    • round-trip JSON equality  (load → dump → equal)
   │    • declarations == {G, F, P, T}
   │    • cascade has 3 einsums
   │    • stopping_condition.rank_variable == "i"
   │    • stopping_condition.boolean_function == "occupancy_zero"
```

The same flow generalizes to any future Program. The artifact's job is
to be a fixed point: any change to the IR's serialization or
validation that breaks this round-trip is caught immediately.

---

## 7. The test surface

Each IR module has a paired test file in `tests/`:

| Module                        | Test file                  | Focus                                      |
| ----------------------------- | -------------------------- | ------------------------------------------ |
| `ir/base.py`                  | `test_ir_base.py`          | frozen / strict / extra=forbid contract    |
| `ir/patterns.py`              | `test_ir_patterns.py`      | regex acceptance/rejection cases           |
| `ir/tensor.py`                | `test_ir_tensor.py`        | tensor declarations + coord sets           |
| `ir/expr.py`                  | `test_ir_expr.py`          | rank exprs + Expression tree + recursion   |
| `ir/op.py`                    | `test_ir_op.py`            | all 4 operator categories + 16 merge names |
| `ir/actions.py`               | `test_ir_actions.py`       | Map / Reduce / Populate + union dispatch   |
| `ir/einsum.py`                | `test_ir_einsum.py`        | Einsum / NestedCascade / StoppingCondition |
| `ir/program.py`               | `test_ir_program.py`       | Initialization / MainEdge / Program        |
| `runtime/op_properties.py`    | `test_op_properties.py`    | 16 truth tables + Unicode + isolation      |
| `examples/algorithms/bfs/build_bfs_program.py`| `test_bfs_artifact.py`     | round-trip + structural checks             |

Tests exercise actual semantics, not just API shape:
- Each discriminated union is validated via `pydantic.TypeAdapter`
  reading raw JSON, not by direct class construction
- All 16 merge-op truth tables are walked cell-by-cell against the
  paper's Appendix A
- Recursion is tested via deeply nested expression trees
- Round-trips assert structural equality, not just non-None

`make check` runs ruff lint + ruff format-check + mypy strict + pytest.
Current count: 349 passed, 2 pre-existing skips, all checks green.

---

## 8. What is NOT here yet (and where it will live)

| Future work                       | Will live in                  |
| --------------------------------- | ----------------------------- |
| Parser (.edge → AST)              | `edge_ir/ast/` + a new module |
| AST → IR lowering                 | `edge_ir/ast/` (or sibling)   |
| Layer 2 validator (cross-refs,    | `edge_ir/validator/`          |
| affine analysis, label resolution)|                               |
| Reference evaluator               | `edge_ir/evaluator/`          |
| Verification harness              | `edge_ir/verify/`             |
| Compiler backends (MLIR, TeAAL)   | new top-level subpackages     |

The validator stub at `edge_ir/validator/README.md` sketches what
that pass needs to do; the evaluator stub at
`edge_ir/evaluator/datatype_registry.py` sketches the host-language
type-binding mechanism. Neither is implemented.

---

## 9. Where to start reading, by goal

- **"How do I add a new operator category?"** Start with `op.py`,
  then add a corresponding entry to `runtime/op_properties.py`,
  then wire it into `actions.py` if it's used in a spec.
- **"How is the BFS algorithm encoded?"** Read
  `examples/algorithms/bfs/build_bfs_program.py` top to bottom; it is heavily
  commented and references each paper equation it implements.
- **"What does the JSON look like?"** Open
  `examples/algorithms/bfs/bfs_program.json`. Cross-reference with
  `schemas/program.schema.json` for the type contract.
- **"What are the IR's design constraints?"** Read the EDGE Notation
  Paper (Sections 6 and 7) — that's the authoritative source.
- **"What is the long-term plan?"** Read
  `docs/interpreter_future_plan.md`.
- **"What audit findings are still open?"** Read `docs/debug.md`.
- **"Why does this regex / this Literal / this discriminator look the
  way it does?"** Cross-reference
  the EDGE paper, Section 6
  (semantics), Section 7 (grammar), Appendix A (merge operators), and
  the EBNF at the paper's `main_ebnf-cropped.tex`.
