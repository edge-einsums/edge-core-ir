# Reserved Keywords

> *Created with the help of Claude Code.*

This file enumerates the strings the EDGE IR treats as reserved. User-declared names (tensor names, UDT type names, UDF compute / merge / coordinate / unary / rank-mapping / boolean function names, rank names, rank variables) **must not** shadow any of these. If a user writes one of these strings in a position the IR interprets it specially, the IR takes the special meaning, not the user-defined one.

If a UDT genuinely needs a value space that overlaps with these strings, that's a design-level limitation we'd have to address (e.g. by switching to a different sentinel encoding for floats).

---

## Float JSON sentinels

Used on `TensorDeclaration.empty_value` and `TensorDeclaration.value`. Standard JSON has no infinity or NaN literal, so non-finite Python floats are encoded as reserved strings on serialization and decoded back to floats on load.

| Sentinel string | Decoded Python value |
|---|---|
| `"inf"` | `math.inf` |
| `"-inf"` | `-math.inf` |
| `"nan"` | `math.nan` |
| `"infinity"` | `math.inf` (alias for `"inf"`) |
| `"-infinity"` | `-math.inf` (alias for `"-inf"`) |

The decoder fires whenever a string in this set appears in the `empty_value` or `value` slot. The result: a UDT whose legitimate value space includes any of these exact strings cannot represent that value losslessly as it would round-trip through JSON as the corresponding float. The strings are therefore reserved: **users may not declare a UDT whose value space overlaps with this set**.

Defined in `edge_ir/ir/tensor.py` as `_FLOAT_SENTINELS`.

---

## Built-in compute operators (`BuiltinComputeOp.symbol`)

```
"+"   "-"   "*"   "/"
```

(From `edge_ir/ir/op.py:52`.) The Unicode multiplication and division glyphs in the paper are normalized to `*` and `/` by the parser before reaching the IR. Any other compute operator name (e.g. `min`, `OR`, `take_left`, `ANY`) is a `UserDefinedComputeOp` and must match `REGEX_USER_DEFINED_NAME`.

A user-defined function whose name collides with one of these four symbols is rejected at IR-construction time (the field validator pattern is `REGEX_USER_DEFINED_NAME`, which doesn't admit `+`, `-`, `*`, `/`).

---

## Built-in merge operators (`BuiltinMergeOp.symbol`) — 16 names from paper Appendix A

```
"pass_through"      "no_pass"           "intersect"
"take_left_only"    "take_left"         "take_right_only"
"take_right"        "xor"               "union"
"nor"               "xnor"              "not_right"
"not_left"          "implies_left"      "implies_right"
"nand"
```

(From `edge_ir/ir/op.py:110-127`.) These are the 16 truth-table outcomes on two Boolean presence flags. The Unicode-to-ASCII translation table lives in `edge_ir/runtime/op_properties.py`.

Naming overlap to be aware of: `take_left` and `take_right` are **also** common names for user-defined *compute* operators (`UserDefinedComputeOp(name="take_left")` etc.). These are distinct operator categories despite sharing the name; the IR distinguishes them by which field they appear in (`merge_op` vs `compute_op`).

A user-defined merge operator must match `REGEX_USER_DEFINED_NAME` and is allowed to overlap names with these 16 (it lives in a separate field). But the surface-syntax parser is expected to resolve ambiguity by looking at the operator's *position* — these 16 ASCII names ARE the merge-op alphabet when seen in the merge-op slot.

---

## Built-in unary operators (`BuiltinUnaryOp.symbol`)

```
"not"
```

(From `edge_ir/ir/op.py:178`.) Maps to the paper's Unicode negation symbol. Any other unary operator name is `UserDefinedUnaryOp` and must match `REGEX_USER_DEFINED_NAME`.

---

## Coordinate operators (`CoordinateOp`)

**No built-ins.** All coordinate operators are user-defined (used only inside `PopulateSpec`). See `edge_ir/ir/op.py:148-159`. The user-defined name must match `REGEX_USER_DEFINED_NAME`.

---

## Built-in data types (`BuiltinDataType.name`)

```
"int"   "float"   "bool"
```

(From `edge_ir/ir/tensor.py`'s `BuiltinDataType` definition.) Any other type name is a `UserDefinedDataType` (e.g. `"char"`, `"VertexLabel"`) and must match `REGEX_USER_DEFINED_NAME`.

---

## Built-in properties for `PropertyApp.name`

```
"occupancy"
```

(From `edge_ir/ir/stopping.py:140`.) The single built-in tensor-property extractor. Returns a non-negative integer count of non-empty coordinates in the operand's iteration space. To add a future property, extend the `Literal` in `PropertyApp.name`.

---

## Reserved comparison operators (`Comparison.op`)

```
"=="    "!="    ">"    "<"    ">="    "<="
```

(From `edge_ir/ir/stopping.py:163`.) The paper uses the corresponding Unicode glyphs; the parser rewrites them to these ASCII forms before the IR sees them. The same six operators are expected to be reused for restricted-iteration predicates when that feature lands.

---

## Reserved set-membership operators (restricted-iteration predicates)

```
"in"        "not_in"
```

ASCII forms for the Unicode `∈` and `∉`. Used in restricted-iteration predicates like `s : s ∈ id` (BFS initialization). User-defined function names cannot shadow these — they live in the predicate operator slot, which the parser recognizes as a closed set.

The IR home for these is the future restricted-iteration-predicate union (not built yet — see the TODO entry "Restricted-iteration predicates"). Listed here so the keyword catalog stays ahead of the IR change.

---

## Surface-syntax reserved tokens (parser/lowering, not stored in the IR)

These live at the EDGE source-text / parser level. The IR doesn't see them as raw strings (they're translated to structural IR nodes during lowering), but the parser MUST recognize them and refuse to interpret them as user-defined names:

- `"True"`, `"False"` — Boolean literals (Path A in `docs/parsing.md`).
- `<<` — update merge operator. Lowered to a Map action with `<<` in the compute slot (see `docs/parsing.md` §`The << update shorthand`).
- `||...||` — occupancy property shorthand. Lowered to `PropertyApp(name="occupancy", ...)`.
- `⋄` — stopping-condition diamond. Lowered to a `StoppingCondition` (see `docs/parsing.md` §Stopping conditions).

These aren't enforced in the current IR (no parser exists yet), but when the parser is ready it must treat these tokens specially.

---

## How to use this file

When adding a new built-in operator, data type, or property to the IR, update this file at the same time. When the JSON serialization sentinel set changes (`_FLOAT_SENTINELS`), update this file too. The lists above are the canonical reserved set; tests and documentation that reference "all built-in X" should cross-reference here.

Source files this index mirrors:
- [edge_ir/ir/op.py](../edge_ir/ir/op.py) — compute, merge, unary, coordinate ops
- [edge_ir/ir/tensor.py](../edge_ir/ir/tensor.py) — data types + float sentinels
- [edge_ir/ir/stopping.py](../edge_ir/ir/stopping.py) — `PropertyApp.name`, `ComparisonOp`
- [edge_ir/ir/patterns.py](../edge_ir/ir/patterns.py) — regex patterns for user-defined names
