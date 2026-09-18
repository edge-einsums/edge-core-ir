# Core IR + Registries + Validator/Analysis — Decisions Pass

> *Created with the help of Claude Code.*

The agenda we resolve **before** drafting the design doc(s). Goal: get the
**core IR**, the **registries**, and the **validator + analysis** "up to par"
— *then* build the actual interpreter (which already has a forward plan in
[interpreter_future_plan.md](interpreter_future_plan.md)).

Each decision is tagged by **who decides**:

- **[SEMANTICS]** — an EDGE-language correctness call. Routed through
  edge-expert / edge-auditor before we lock it. Do not settle solo.
- **[ENGINEERING]** — an IR/implementation design call. We settle together.
- **[PAPER]** — needs the paper author to make a judgment call
  (often an EBNF or naming canonicalization).
- **[PARKED]** — explicitly deferred to a separate pass.

Status legend: ⬜ open · 🔵 in flight (agents) · ✅ decided · ⏭️ deferred.

Once these are resolved we pick the doc shape (one combined design doc vs.
three) based on how entangled they turn out to be.

---

## Part A — Core IR

| ID | Tag | Decision | Current lean | Blocks |
|----|-----|----------|--------------|--------|
| **CIR-1** | [SEMANTICS] ✅ | **Rank-as-value leaf.** Add an Expression leaf so a rank expression can sit in *value* position (`Z_{m,n} = A_{m,n} · m`). Sub-q's: allowed positions; arbitrary arith/functions (DFS `Stamp = (i×\|V\|)+v`, `F·i`) vs bare var; data type when cast; presence flag in the merge. | Add `RankExpressionValue` wrapping a `RankExpression`; allow full arith; integer value from the coord; always-present in merge. Confirm w/ expert. | DFS bug fix (2 xfails); case stmts |
| **CIR-2** | [SEMANTICS] ✅ | **Must every Einsum carry an explicit ReduceSpec + PopulateSpec?** Or are they defaultable (no-reduce + default-assignment populate), with lowering synthesizing the default? `Einsum.specs` is currently a bare list with no rule. (TODO "Small"; design_decisions D2.) | Defaultable; lowering fills the canonical default; absence is *not* meaningful. Confirm paper default w/ expert. | Validator default-resolution pass; analysis |
| **CIR-3** | [ENGINEERING] ⬜ | **Rename `RankFunction.is_affine` → `declared_affine`.** It's user-declared input (function is a black box), not the analysis result we just moved to `analysis/affine.py`. Pure clarity rename. | Do it. Coordinate with REG-3 (affine flag location). | REG-3 |
| **CIR-4** | [SEMANTICS] ✅ | **Remove `UserDefinedMergeOp` from the IR?** There are exactly 16 two-input boolean functions and all 16 are built-in — is a user-defined merge op even admissible? (udf_registry_plan §14 item 8.) | Remove (merge space is closed at 16); shrink `MergeOp` union to `BuiltinMergeOp`. Confirm closure w/ expert. | Registry merge-category (already dropped) |
| **CIR-5** | [SEMANTICS+SYNTAX] ⬜ | **Case statements.** Surface syntax + lowering. Paper: each branch is a valid EDGE expr; lowers to predicate-restricted einsums merged with `<<`. (TODO "Language features"; D23 predicates are the desugar target.) | Likely **defer** out of this pass — predicates (done) + `<<` (CIR/REG) are the prerequisites; tackle after they land. | — (downstream) |
| **CIR-6** | [ENGINEERING] ⬜ | **`RankDeclaration` lets `shape` and `coord_set` coexist** (audit Finding 10). Add a Pydantic validator or document precedence. | Add a validator rejecting both, or document precedence. Small. | — |
| **CIR-7** | [PAPER] ⬜ | **Scalar (0-rank) tensors** are prose-permitted but EBNF-rejected (Finding 11). Paper-side EBNF fix for empty `<shape-definition>`. | Open question: accept empty shape in EBNF? IR already admits via `Lit_*`. | — |
| **CIR-8** | [PAPER] ⬜ | **`REGEX_USER_DEFINED_NAME` canonical form** (Finding 4): EBNF (no underscore) vs IR (underscore) vs paper examples (hyphens) disagree. Pick one. | Open question: which canonical form? IR currently `^[A-Za-z_][A-Za-z0-9_-]*$`. | UDF naming (REG) |

---

## Part B — Registries

Context: the **built-in op-properties registry** ([op_properties.py](../edge_ir/runtime/op_properties.py))
already exists (16 merge truth tables, 4 compute ops, `not`, Unicode↔ASCII).
The **UDF registry** does NOT — only a detailed draft
([udf_registry_plan.md](udf_registry_plan.md)). The review already
settled: merge ops are built-in only (no UDF merge category); unary +
rank-mapping are UDF categories. Open items below.

| ID | Tag | Decision | Current lean | Blocks |
|----|-----|----------|--------------|--------|
| **REG-1** | [ENGINEERING] ⬜ | **Compute-op namespace** (plan §10.1): one namespace, three (map/reduce/populate), or multi-action opt-in `@compute_op(actions=[...])`? Shapes the whole category enum + decorator names. | **C — multi-action opt-in** (plan's rec), or B if folders should be load-bearing. | Loader, validator category checks |
| **REG-2** | [ENGINEERING] ⬜ | **Decorator vs folder authoritative** for category (§10.2). | Decorator authoritative; folder is convention. | Loader |
| **REG-3** | [ENGINEERING] ⬜ | **Affine flag location** (§10.3): on the decl, the IR node, or both-with-reconcile? | Decl is source of truth; validator's affine pass copies decl→IR (`RankFunction.declared_affine`), flags conflicts. | CIR-3; analysis |
| **REG-4** | [ENGINEERING] ⬜ | **May the validator call `DataTypeImpl.validate`** to check `empty_value`, or stay strictly impl-free? (§10.4) | Narrow exception: validator may call `validate` only. | Validator type checks |
| **REG-5** | [ENGINEERING] ⬜ | **Allow overriding built-in UDF names?** (§10.6) | No by default; `allow_override=True` escape hatch. | Loader |
| **REG-6** | [ENGINEERING] ⬜ | **Naming**: package `edge_ir/udf/` (vs `registry`/`user_defined`); and "evaluator" vs "interpreter" as the project-wide term. | `edge_ir/udf/`; **"evaluator"** the noun, "interpret/run" informal verbs. | Everything downstream |
| **REG-7** | [SEMANTICS] ✅ | **`<<` default compute op definition** — the BF/DFS bug fix. Right operand where present, else carry the left forward; paired with union. Confirm exact def from `appendix/1-misc.tex:76-81`. | Ship in default registry per expert's exact wording. | BF (1 xfail) + DFS `<<` (1 xfail) |
| **REG-8** | [PARKED] ⏸️ | **Stopping / boolean-function category** redesign (§4.7, §10.5). Separate design pass. Stays OUT of the first registry slice. | Defer; keep registry impl signature open-ended (context obj, not bare fiber). | Stopping-condition validator (VAL-7) |
| **REG-9** | [ENGINEERING] ⬜ | **Fiber/tensor data representation** readiness (§10.7) — needed to finalize coordinate-op and datatype JSON contracts. | Likely **defer** to the interpreter phase; stub `Fiber` for now. | Coordinate-op + datatype impl contracts |
| **REG-10** | [ENGINEERING] ⬜ | **Fold the `datatype_registry.py` stub** into the UDF registry (`edge_ir/udf/`). | Do it as part of building the package. | — |

---

## Part C — Validator + Analysis

Context: Layer 2 validator is **entirely unbuilt** (only [validator/README.md](../edge_ir/validator/README.md)
+ the predicate-check plan in TODO). Analysis has only [affine.py](../edge_ir/analysis/affine.py);
iteration-space construction + reduce-set derivation are unbuilt. Most of these
**consume** the registry decisions, so they come after Part B.

| ID | Tag | Decision | Current lean | Blocks |
|----|-----|----------|--------------|--------|
| **VAL-1** | [ENGINEERING] ⬜ | **Error model**: `ValidationError` hierarchy w/ rich location (which Einsum/spec/field path), **collectable not fatal** (return a list). | Build it; mirror the lowering-error shape (D17). | All checks |
| **VAL-2** | [ENGINEERING] ⬜ | **Cross-reference checks** set: tensor names resolve + unique; UDF names registered (per category); `BinaryApp.label` ↔ exactly one `ComputationSpec.label`; rank-name strings reference declared ranks; rank-var bindability; stopping `rank_variable` is generational. | Implement the README list; depends on REG-1 (category keys). | — |
| **VAL-3** | [SEMANTICS] ✅ | **Reduce-set derivation + default-rank resolution** (bare ⋀/⋁, bare populate): is reduced-set = (iteration ranks − output ranks)? Caveat for opaque output RVEs (`f(a,w)`). Confirm paper rule (§6.211/6.234). | Derive for basic reduces; skip/just-warn for opaque RVEs (README note). | CIR-2; analysis module |
| **VAL-4** | [ENGINEERING] ⬜ | **Predicate-specific checks** — the 5 bullets in TODO "Layer 2 validator" (rank var must pre-exist; `CoordSetName` resolves; interval/enum type-compat; comparison sides are coord-valued; composition arity). | Implement per [validator_plan_predicates.md](validator_plan_predicates.md). | — |
| **VAL-5** | [ENGINEERING] ⬜ | **Type checks**: `empty_value` instance-of `data_type` (built-in vs UDF via REG-4); operator-category-in-correct-slot (`MapSpec.compute_op` ≠ a `CoordinateOp`). | Implement; depends on REG-1/REG-4. | — |
| **VAL-6** | [ENGINEERING] ⬜ | **Analysis module layout**: iteration-space construction + reduce-set live in `edge_ir/analysis/` (pure functions, side tables), shared by validator AND evaluator (D14/D15). | Confirm shape; affine.py is the prototype. | Evaluator (later) |
| **VAL-7** | [PAPER] ⏸️ | **Finding 5** — `StoppingCondition` can't reference tensor args (expressivity gap). Tied to REG-8 (the stopping-condition pass). | Defer with REG-8. | — |
| **VAL-8** | [PAPER] ⬜ | **Finding 14** — `AnonymousTensor.specs` vs `Einsum.specs` scoping rule undocumented. Pick canonical form (affects the binary-label↔spec check VAL-2). | Open question: which scope owns the specs? | VAL-2 |

---

## Part D — Sequencing & open framing

**Dependency order** (why "IR → registries → validator/analysis → interpreter"):

1. **IR forks first** (CIR-1, CIR-2, CIR-4) — they change the IR shape every
   other layer reads. CIR-1 (rank-as-value) also unblocks DFS xfails.
2. **Registry decisions** (REG-1…REG-7) — the validator's category/arity
   checks and the evaluator both key off the registry's category axis.
   REG-7 (`<<`) unblocks the BF/DFS `<<` xfails *once there's an evaluator
   to confirm the fix*.
3. **Validator + analysis** (VAL-*) — consume the IR + registry decisions;
   produce the structured checks + shared iteration-space/reduce-set
   functions the evaluator will call.
4. **Interpreter** — out of scope for THIS pass; see interpreter_future_plan.md.

**Note on the xfails:** the `<<` (BF, DFS) and max-flow empty-value bugs can't
be *confirmed fixed* until the evaluator exists — fixing the encoding (REG-7,
CIR-1) flips the *encoding* xfails, but "does it compute the right answer on a
graph" needs execution. So the bug list is partly a Part-B/Part-A job and
partly waits on the interpreter.

**Doc-shape decision (deferred until this pass is filled in):** one combined
design doc vs. three. Lean: if Parts A/B/C stay as entangled as they look
(registry ⇄ validator, IR ⇄ both), one combined doc + one plan doc.

---

## Resolved this pass

Settled by edge-expert (paper-cited,). Detail graduates to
design_decisions.md when the design doc is drafted.

- **CIR-1 — rank-as-value: ADD the leaf (`RankExpressionValue`).** Admissible
  per `09-rves.tex:101-111`. Spec:
  1. May appear **anywhere a value operand can**, incl. nested in a BinaryApp
     (it's a projection of the iteration point `is`; compute already takes
     `is` as input, `semantics/3-map.tex:173-179`). Forbidden in
     subscript/coordinate position (that's the value-in-subscript ban).
  2. Expression may be **arbitrary rank arithmetic + rank functions**, not just
     a bare var — so DFS `Stamp = (i×|V|)+v` and `F·i` are legal. Constraint:
     every symbol must be an **iteration-space rank variable or declared
     constant**; no tensor value inside.
  3. Cast value = the rank **coordinate** from that rank's coordinate set;
     integer coord space → integer value. (Char/label coord space → see
     Q-A below.)
  4. **Always present** — contributes a constant `True` presence to whatever
     merge pairs it (so `A_{m,n}·m :: ⋀ ×(∩)` reduces to "present iff A
     present"). Never carries an `Exists` predicate.

- **CIR-2 — default specs: DEFAULTABLE on the surface; lowering ALWAYS
  synthesizes the explicit default; absence is never meaningful.** The
  *lowered core IR* carries explicit specs. Canonical defaults
  (`6-syntax.tex:220-244`): default reduce `⋁ +(𝟙)` — a **no-op reduce with
  an empty reduce-set** when all ranks are preserved (it's not "no reduce," it's
  `+`/pass-through contracting zero ranks); default populate `⋘ 𝟙(𝟙)`
  (assignment, mutable-rank set ∅); default merge pass-through `𝟙`. → Engineering
  follow-up: enforce "exactly one ReduceSpec + one PopulateSpec per Einsum" in
  the IR/lowering (Pydantic), and drop any None/absent marker.

- **CIR-4 — REMOVE `UserDefinedMergeOp`.** Merge space is **closed at 16**
  (`appendix/0-merge.tex:4-18`, `semantics/3-map.tex:97-99`): exactly
  2^(2^2)=16 boolean fns on two inputs, all built-in. No admissible
  user-defined merge. Keep user-defined **compute** and **coordinate** (those
  *are* open, `appendix/1-misc.tex:51-57`). Shrink `MergeOp` union to
  `BuiltinMergeOp`.

- **REG-7 — `<<` is a COMPUTE op (not merge), paired with union (∪).** Exact
  def (`appendix/1-misc.tex:76-81`), `compute(left, right)`:
  both empty → empty; left empty → right; both non-empty → right; **right empty
  → left** (carry the previous value forward). Lowering:
  `P_{i+1,d} = P_{i,d} · F_{i+1,d} :: ⋀ <<(∪)` (left = carried-forward, right =
  new). The current `take_right(union)` encoding is wrong on exactly the
  (left present, right empty) row — `take_right` returns empty there and drops
  the carried value; `<<` keeps it. Ship `<<` as a default registry compute op.

- **VAL-3 — reduce-set = (iteration ranks − output ranks)** for structurally
  derivable outputs (bare rank var / invertible RVE). For an **opaque output
  RVE** `f(a,w)` (non-invertible), do NOT emit a concrete reduced-rank list —
  emit "reduce required, rank-set non-derivable" and conservatively treat the
  RVE's input ranks as contracted-via-aliasing (`6-syntax.tex:314-316`,
  `appendix/1-misc.tex:41-45`). Default **populate** mutable-rank set = the
  `*`-marked ranks, default ∅; never derived by subtraction.

### Open questions (edge-expert flagged these — paper genuinely silent)

- **Q-A (relates to CIR-1c):** when a rank-as-value is drawn from a *char/label/
  enum* rank coordinate set (e.g. `{a,b,c,d,e}`) and then fed to an **arithmetic**
  compute (`×`, `+`), the paper defines the value as "the coordinate" but never
  defines a label→numeric coercion. Is that **ill-typed (reject)**, or is there
  an **implicit ordinal coercion** (label → its index in the rank coordinate
  set)? — Does NOT block the integer case (DFS, the common case); only the
  char-coordinate corner.
- **Q-B (relates to CIR-1 formal grounding):** rank-as-value has **no
  set-theoretic denotation** in the formal semantics — it lives only in prose
  (`09-rves.tex`) + the `IS` argument to compute. The "appears anywhere a value
  can" + "always-present in merge" rulings are the consistent prose reading, but
  the formal section never states them. Confirm they're intended so they can be
  promoted into the formal text (and the IR leaf is grounded in formal, not just
  an example).
