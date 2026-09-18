# Parsing notes

> *Created with the help of Claude Code.*


Transformations the parser/lowerer applies when translating EDGE input into the core IR.

The general rule: shorthand forms in the input always lower to the full canonical EDGE form before becoming IR. The IR carries only the canonical form. Any time the input grammar allows an abbreviated or sugared form, this file records what the full form is and how to expand to it.

## Shorthand expansions

### Coordinate-set overrides

The EBNF for a coordinate-set override is written as `<rank-name>^<tensor-name> = ⟨...⟩`, with the tensor superscript on the left of the equals sign. In practice, the superscript is always the tensor currently being declared (there is no cross-tensor override case in the language). The superscript is redundant in input and is dropped during lowering: the IR places `coord_set` directly on the relevant `RankDeclaration` of the tensor being declared. The `^<tensor-name>` annotation never appears in the IR.

### The `<<` update shorthand

Per Appendix B.6 of the EDGE paper, the `<<` operator is shorthand for a full Map action with `<<` as the compute operator and union as the merge operator.

**Input form**:

```
P_{i+1,d} << F_{i+1,d}
```

**Desugared form** (what the IR carries):

```
P_{i+1,d} = P_{i,d} · F_{i+1,d} :: ⋀_d << (∪)
```

Lowering fills in three things the shorthand drops:

1. The previous-iteration version of the LHS tensor (`P_{i,d}` here) as the first factor on the RHS. The shorthand `P_{i+1,d} << F_{i+1,d}` implicitly carries forward the previous iteration's values.
2. The binary `·` operator joining the two factors.
3. The Map spec with `<<` as the compute op and union as the merge op.

The IR sees only the full Map form. `<<` lives in the IR as a built-in compute operator (`BuiltinComputeOp(symbol="<<")`) with fixed semantics per Appendix B.6:

- both operands empty → empty
- left operand empty, or both non-empty → return right value
- right operand empty → return left value

### Scalar literals

A scalar literal appearing in the EDGE input (e.g. `0`, `5`, `True`, `'c'`) lowers to a synthesized zero-rank `TensorDeclaration` carrying the literal as its `value` field (per the Option C decision for scalars: a scalar IS a tensor — a zero-rank tensor). The literal does not survive lowering as a leaf in the expression tree; it is replaced by an `InputTensor` referring to the synthesized declaration.

**Two distinct paths depending on the literal's surface form:**

### Path A — builtin literals (int, float, bool): direct recognition

The parser recognizes integer, float, and boolean literals from their surface syntax alone, with no context lookup. These are the three `BuiltinDataType` cases:

| Surface form | Synthesized declaration |
|---|---|
| `5`, `-3`, `0` | `TensorDeclaration(ranks=[], data_type=BuiltinDataType(name="int"), empty_value=None, value=<the int>)` |
| `5.0`, `-3.14`, `0.5` | `TensorDeclaration(ranks=[], data_type=BuiltinDataType(name="float"), empty_value=None, value=<the float>)` |
| `True`, `False` | `TensorDeclaration(ranks=[], data_type=BuiltinDataType(name="bool"), empty_value=None, value=<the bool>)` |

This works **anywhere** a literal can appear — single-input einsum, multi-input einsum body, comparison rhs, stopping condition. No surrounding tensor required.

### Path B — UDT literals (single-quoted opaque tokens like `'c'`, or anything not recognizable as int/float/bool): context-required

UDT literals are opaque to the parser — `'c'` could be a char, a one-letter token, a label from any UDT. The parser CANNOT recognize them on their own; it needs a typed receiver.

**Single-input einsum (assignment where the literal IS the whole RHS):** the parser inherits `data_type` from the output tensor.

```
▷ Tensors
T^{S ≡ |V|} → char, empty = '\0'

▷ Initialization
T_s = 'c'                       # single-input: parser inherits data_type from T (char)
```

**Multi-input contexts (binary operand, comparison rhs against a UDT-typed operand, etc.):** the parser CANNOT infer the data_type — there's no single output to inherit from, and the other operand may not be of the same UDT. The user MUST pre-declare a named zero-rank tensor of the intended type, and reference it by name:

```
▷ Tensors
T^{S ≡ |V|} → char, empty = '\0'
CharC^{}    → char, empty = '\0'       # user-declared constant tensor

▷ Initialization
CharC = 'c'                            # single-input: parser inherits from CharC
                                        # (only place a bare UDT literal is allowed)

▷ Extended Einsum
Result_s = T_s ·² CharC                # multi-input: reference the named constant
```

If the parser sees a UDT literal in a multi-input position with no typed receiver to inherit from, it **rejects** the program with: `"UDT scalar literal at <location>: cannot be used as a binary operand. Declare a named zero-rank tensor of the intended type and reference it by name."`

### Why the asymmetry between builtin and UDT?

Builtin int/float/bool have unambiguous surface forms. `5` is an int regardless of where it appears. The parser doesn't need context.

UDT literals are opaque (the parser doesn't know the value-space of `char` vs `token` vs `label` vs any other UDT a user might define). So the parser refuses to guess — when a UDT literal appears in a multi-input position, the user must spell out which UDT they meant by pre-declaring a tensor.

### `empty_value=None` always, for synthesized literals

A scalar (zero-rank tensor) has no meaningful "empty" — there's a single point in its coordinate space and it's either occupied (with the literal's `value`) or not. There are no unoccupied coordinates to default. `None` is the direct, literal expression of "this tensor has no empty value."

Applies to both Path A and Path B synthesis. User-declared tensors still require an explicit `empty=` in the surface syntax; the always-`None` rule applies only to parser-synthesized `Lit_X` declarations during lowering. (`data_type` is required for both user-declared and synthesized declarations — the IR rejects construction without it.)

#### Worked examples

##### 1. Stopping-condition scalar — the BFS case

Input (snippet):

```
▷ Tensors
F^{I, S ≡ |V|} → integer, empty = ∞

▷ Extended Einsum
⋄ : ||F_{i+1}|| ≡ 0      # occupancy(F_{i+1}) == 0
```

Lowering for the `0`:

```python
# Other operand is a PropertyApp (occupancy). Inherit data_type from the
# property's RETURN type (int), not from F. empty_value is None because
# a synthesized literal is a scalar (zero-rank tensor) with no empty.

TensorDeclaration(
    name="Lit_0",                                  # fresh synthesized name
    ranks=[],                                      # zero-rank
    data_type=BuiltinDataType(name="int"),         # = occupancy.return_type
    empty_value=None,                              # scalars have no empty
    value=0,                                       # the literal
)

# Comparison's rhs becomes a TensorProjectionValue referring to Lit_0:
Comparison(
    lhs=PropertyApp(name="occupancy", pinned_rank_variables=["i"], operands=[...F's projection...]),
    op="==",
    rhs=TensorProjectionValue(proj=TensorProjection(tensor="Lit_0", ranks=[])),
)
```

##### 2. UDT scalar in initialization

Input:

```
▷ Tensors
T^{S ≡ |V|} → char, empty = '\0'

▷ Initialization
T_s = 'c'
```

Lowering:

```python
# Contextual tensor: T (the output of the init einsum) — provides data_type.
# empty_value is always None for a synthesized literal (a scalar has no
# meaningful empty; see the always-None rule above).
TensorDeclaration(
    name="Lit_c",
    ranks=[],
    data_type=UserDefinedDataType(name="char"),  # = T.data_type
    empty_value=None,  # scalars have no empty
    value="c",  # the literal
)

# Init einsum becomes a broadcast of Lit_c across s:
Einsum(
    output_tensor="T",
    output_ranks=[RankVariable("s")],
    expression=InputTensor(TensorProjection(tensor="Lit_c", ranks=[])),
    specs=[...],  # broadcast Map spec
)
```

##### 3. Scalar as a binary operand in an expression body

Input (from max-flow's height-rule einsum):

```
▷ Tensors
D^{I, U ≡ |V|, V ≡ |V|} → integer, empty = 0

▷ Extended Einsum
Lbl_{i,u,v} = D_{i,u} · (D_{i,v} ·² 1)_{i,v} :: ⋀¹ ≡(∪) ⋀² +(∩)
```

Lowering for the `1` inside the anonymous tensor:

```python
# Contextual tensor: Lbl (the enclosing einsum's output tensor) — provides
# data_type. empty_value is always None for a synthesized literal.
TensorDeclaration(
    name="Lit_1",
    ranks=[],
    data_type=BuiltinDataType(name="int"),  # = Lbl.data_type
    empty_value=None,  # scalars have no empty
    value=1,  # the literal
)

# The `1` in the body is replaced by an InputTensor to Lit_1:
BinaryApp(
    label=2,
    lhs=InputTensor(
        TensorProjection(tensor="D", ranks=[RankVariable("i"), RankVariable("v")])
    ),
    rhs=InputTensor(TensorProjection(tensor="Lit_1", ranks=[])),
)
```

#### Notes

**Naming convention for synthesized declarations:** TBD at parser time. Candidates — `Lit_<value>` (e.g. `Lit_5`, `Lit_c`), `Lit_<counter>` (e.g. `Lit_0`, `Lit_1`), `_lit_<sha>`. The IR doesn't care; pick one that round-trips cleanly. For the hand-written BFS builder, we use `Lit_<value>` (so the BFS stopping condition's `0` becomes `Lit_0`).

**Collision detection at synthesis time:** The relaxed `REGEX_TENSOR_NAME` (`^[A-Z][A-Za-z0-9_]*$`) admits underscores, so a user can legally declare a tensor named `Lit_0`. When a parser-synthesized name would collide with an existing user-declared tensor name, the parser detects the collision (by looking up the candidate name in the `Program.declarations` accumulated so far) and appends a numeric suffix until unique: `Lit_0` → `Lit_0_1` → `Lit_0_2` → etc. The collision-handling rule is a parser-implementation contract; the IR itself does not enforce uniqueness on `TensorDeclaration.name` (that's deferred to the Layer 2 validator).

**No deduplication at the parser level.** A program containing the literal `0` three times produces **three** synthesized declarations. The parser does not look across literal occurrences and does not check whether an existing zero-rank tensor in the program (e.g. a hand-authored `Zero` constant) already carries the same `(data_type, value)`. Merging equivalent constants is **common subexpression elimination on tensor declarations** — a classic compiler optimization that belongs in a downstream pass (MLIR's constant-pooling, egglog equality saturation, a future "core IR simplification pass"). Doing CSE at the parser would also throw away source-position information for each literal occurrence, which we need for source-mapped error messages and tracing. Names must still be unique per `REGEX_TENSOR_NAME` (TensorDeclaration.name is the IR's discriminator); the parser uses a collision-avoiding scheme — a monotonic counter, a value+suffix (`Lit_0`, `Lit_0_1`, `Lit_0_2`), or a source-position suffix. Choice TBD at parser-implementation time.

**Same rule for stopping conditions:** The unified design (`ScalarValue` removed from `stopping.py`; comparisons reference scalars via `TensorProjectionValue` + a synthesized zero-rank `TensorDeclaration`) means the inference rule above applies uniformly across expressions and stopping conditions. There is no separate scalar mechanism in the predicate world.

**Wrapping convention:** A lowered literal always appears as `InputTensor(TensorProjection(tensor="Lit_X", ranks=[]))` — the leaf form. Never wrapped in `AnonymousTensor`. `AnonymousTensor` exists for parenthesized sub-expressions that have their own output ranks; a literal has no sub-expression to parenthesize.

### Range syntax in predicates

When the parser sees a range constraint on a rank variable inside a predicate — `s : 0 ≤ s < 5`, `s : 0 ≤ s ≤ 5`, `s : s > 0`, etc. — it normalizes the surface form to a half-open interval and emits `SetMembership(member=RankVariable(name="s"), coord_set=CoordSetInterval(lo=<resolved>, hi=<resolved>))`. The IR carries only half-open `[lo, hi)`; the surface form's open-vs-closed-vs-one-sided shape is the parser's problem.

**Conversion rules (parser-side, before the IR sees anything):**

| Surface form | Lowered `CoordSetInterval` |
|---|---|
| `lo ≤ s < hi` | `(lo=lo, hi=hi)` — direct |
| `lo < s < hi` | `(lo=lo+1, hi=hi)` — shift lo |
| `lo ≤ s ≤ hi` | `(lo=lo, hi=hi+1)` — shift hi |
| `lo < s ≤ hi` | `(lo=lo+1, hi=hi+1)` — shift both |
| `s > lo` (no upper) | `(lo=lo+1, hi=<rank's shape>)` — parser fills upper from the rank's declared shape |
| `s ≥ lo` (no upper) | `(lo=lo, hi=<rank's shape>)` |
| `s < hi` (no lower) | `(lo=0, hi=hi)` — assumes 0-based ranks (the default) |
| `s ≤ hi` (no lower) | `(lo=0, hi=hi+1)` |

For one-sided ranges, the parser uses the rank's declared shape from its `RankDeclaration` as the implicit endpoint. If the rank has no declared shape (e.g. a generational rank with `shape=None`), the parser **cannot** lower a one-sided range to `CoordSetInterval` — there's no concrete upper bound. In that case it falls back to emitting a `PredicateComparison` (e.g. `PredicateComparison(lhs=RankVariable("i"), op=">", rhs=RankConstantLiteral(0))`) and the Layer 2 validator may surface a warning that the range couldn't be canonicalized.

**Shape-symbol endpoints survive.** When `hi` is a shape symbol like `|V|`, the lowered `CoordSetInterval` carries the symbol unchanged (`CoordSetInterval(lo=0, hi="|V|")`). The evaluator resolves it at iteration-space construction time, same as everywhere else.

**The IR doesn't see the user's surface form.** Once the parser does the conversion, the artifact reads `[lo, hi)` regardless of whether the user wrote `0 ≤ s ≤ 4`, `s ∈ [0, 5)`, or `0 ≤ s < 5`. Same IR for all three.

### Restricted-iteration predicate attachment

The `<rank-var> : <predicate>` syntax can appear in the source at an einsum's output rank position or inside an input projection. The IR represents the predicate at the **einsum level only** — there is one slot, `Einsum.predicates`, that holds every predicate restricting that einsum's iteration. There is no per-output-rank or per-input-projection attachment in the IR.

**Why one slot:** The predicate's job is to restrict the iteration space the einsum walks over. The iteration space belongs to the einsum, not to any particular rank position or input projection. Per-position attachment would just mean the same fact gets recorded in different places depending on how the user wrote the source; collapsing everything to the einsum level makes the IR representation unique.

**Parser-side normalization:** the parser strips the predicate from wherever the user wrote it in the source (output rank position, input projection, or anywhere else the surface syntax admits) and lifts it into the einsum's predicates list. Surface readability ("attach to output where possible") is a parser concern. The IR doesn't see that distinction.

**Constraint (checked at Layer 2):** every rank variable referenced inside a predicate must already appear somewhere in the einsum — in the output rank list, in some input projection, or both. A predicate cannot introduce a new rank variable.

**Worked example — all three source positions lower to the same IR:**

Source 1: `F_{0, s : s ∈ id} = 0` (predicate on output rank).
Source 2: `L_{s, d : s < d} = G_{s, d}` (predicate on output rank, both vars in output).
Source 3: `Out_u = G_{u, v : v ∈ id}` (predicate on input projection, `v` is input-only).

All three lower to `Einsum.predicates = [<the predicate>]`. The parser doesn't need to record where the user wrote it; the IR doesn't need to track it.

## Stopping conditions

- `||X_{<index>}||` in the input is a syntactic shorthand for the built-in unary operator `occupancy(X_{<index>})`. The parser must rewrite `||·||` to `occupancy(·)` during lowering. The IR carries `occupancy` as a built-in unary operator; the `||·||` notation never appears in the IR.
- The diamond's optional rank-variable subscript (`⋄_k`, `⋄_i`, plain `⋄`) names the generational rank the condition halts. Plain `⋄` is shorthand for "the cascade's only generational rank"; lowering should resolve it to an explicit rank name.
- `PropertyApp` and `DiamondBooleanApp` each carry a `pinned_rank_variables: list[str]` field. The user supplies only the *extra* pinned ranks in the EDGE input (e.g., the `s` in `occupancy_s(F)`); lowering prepends the enclosing `StoppingCondition`'s generational rank variable so the IR always carries the full set.