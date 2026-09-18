# EDGE IR — outstanding findings (audit)

> *Created with the help of Claude Code.*

Independent audit of the completed Pydantic IR. Source: edge-auditor
report run after Step 7 of the implementation plan, with the IR build
at 349/349 tests passing.

The report flagged 15 findings: 1 BUG, 5 INCONSISTENCIES, 6 AMBIGUITIES,
3 NITS. None block commit; several block downstream work (parser,
evaluator, validator).

---

## Top findings

#### Fixed:
1. **BUG** — wrong Unicode glyphs for `take_left`/`take_right` merge ops
   (`edge_ir/runtime/op_properties.py:41,43`). IR uses `◁`/`▷` but
   paper uses `←`/`→`. Pretty-printers will produce wrong output.
2. **INCONSISTENCY** — wrong Unicode for `not_left`/`not_right`
   (`edge_ir/runtime/op_properties.py:48,49`). Paper Appendix A has
   been updated to `↛`/`↚` (per a REVIEW comment), but EBNF still says
   `¬r`/`¬l` and the IR matches the EBNF. Paper-internal disagreement.
3. **INCONSISTENCY** — `ReduceSpec.rank_list` requires non-empty, but
   the EBNF AND paper examples (Section 6, lines 211 and 234) both
   allow bare `\bigvee` with no rank list. Will reject valid programs.
4. **INCONSISTENCY** — `REGEX_USER_DEFINED_NAME` includes underscores
   that the EBNF doesn't, but paper's own examples use hyphens
   (`select-any-s`, `user-rand-func`) which neither EBNF nor IR allow.
   Three-way disagreement.
5. **AMBIGUITY** — `StoppingCondition` can't reference which tensor the
   boolean function tests. Paper has `\diamond: D_{i+1} \equiv D_i` and
   `\diamond: F_{i+3} \equiv F_i` etc. — IR collapses these into a
   function name with no args. Real expressivity hole.
8. **NIT/SEMANTIC** — F's `empty_value=0` in the artifact. Paper uses
   `\infty`. With `0` as the sentinel, depth-0 vertices collide with
   empty — any future evaluator will produce wrong BFS results.
#### Won't Fix:
6. **INCONSISTENCY** — BFS script citation says "Eq 41, Section 7.3"
   but Eq 41 is actually the single-einsum push BFS using AND/OR,
   NOT the 3-einsum cascade in the artifact. The artifact actually
   corresponds to `eqn:bfsi`/`eqn:edge_bfs_full` in
   `tex/bfs-example/07-iteration.tex:67-70`.
#### Not Fixed:







**Other findings:** the AnonymousTensor-vs-Einsum specs scoping rule
isn't documented (Finding 14), `RankDeclaration` allows shape and
coord_set to coexist with no precedence rule (Finding 10), `is_affine`
is a frozen-model field that says "filled by validator" but can't be
mutated in place (Finding 13), and a few NITs. Full list in the
appendix below.

---

## Action ordering

- **Findings 1 and 6 are pure doc/string fixes — safe to do now.**
- **Findings 2, 4, 5, 14** hinge on paper-side decisions that should
  go to the EDGE expert / paper author before resolving.
- **Finding 3** (relaxing `ReduceSpec.rank_list`) is a clean fix once
  you've decided.
- **Finding 8 (the F empty value)** is a BFS-artifact-level decision.

---

## Appendix — complete findings list

The numbering matches the auditor's original report.

### Finding 1 — BUG ✅ FIXED
Wrong Unicode glyphs for take-left and take-right merge ops.
`edge_ir/runtime/op_properties.py:41,43`. IR had `◁` (U+25C1) and `▷`
(U+25B7); paper uses `←` (U+2190) and `→` (U+2192). Cross-references:
`tex/appendix/0-merge.tex:91, 151`, `tex/bfs-example/07-iteration.tex:68,119,162`,
`tex/7-examples.tex:455-456,501-502`, `tex/bfs-example/05-merge-compute.tex:150,164`.
**Resolution:** updated `MERGE_OP_UNICODE["take_left"]` and
`MERGE_OP_UNICODE["take_right"]` to `"←"` / `"→"`. 47/47
`test_op_properties.py` still pass.

### Finding 2 — INCONSISTENCY ✅ FIXED
Wrong Unicode for not-left/not-right. `op_properties.py:48,49` had
`"¬ r"` / `"¬ l"` (ASCII-ish strings). Paper Appendix A REVIEW
comments at `tex/appendix/0-merge.tex:306-308,364-366` updated symbols
to `\nrightarrow` (↛, U+219B) and `\nleftarrow` (↚, U+219A). EBNF at
`tex/notes/main.ebnf-cropped.tex:60` still said `\neg_r` / `\neg_l`.
Three-way disagreement among Appendix, EBNF, and IR. **Resolution:**
(a) Updated EBNF at
`tex/notes/main.ebnf-cropped.tex:60-61` to
use `\nrightarrow` / `\nleftarrow` matching the Appendix.
(b) Updated `MERGE_OP_UNICODE["not_right"]` / `["not_left"]` to
`"↛"` / `"↚"`. EBNF, Appendix, and IR now agree.

### Finding 3 — INCONSISTENCY ✅ FIXED
`ReduceSpec.rank_list` mandated `Field(min_length=1)` at
`edge_ir/ir/actions.py:64`. The grammar at
`tex/notes/main.ebnf-cropped.tex:52-53` permits omission; paper
examples at `tex/6-syntax.tex:211,234` use bare `\bigvee` with no
rank list. The docstring at `actions.py:18-19` claimed "reduce always
requires a rank list" — contradicted by the paper.
**Resolution:**
(a) Changed `ReduceSpec.rank_list` to `list[str] | None = None`,
mirroring `MapSpec.rank_list` exactly. `PopulateSpec.rank_list`
remains mandatory (paper requires the mutable rank).
(b) Rewrote the module docstring and the `ReduceSpec` class docstring
to describe `rank_list = None` as "all iteration-space ranks for the
corresponding action at this site, resolved by the validator/
evaluator," with paper Section 6.211/6.234 citation for the bare
`\bigvee` case.
(c) Tests: removed `test_reduce_spec_requires_rank_list` and
`test_reduce_spec_rejects_empty_rank_list` (encoded the obsolete
invariant); added 4 new tests covering bare-reduce construction,
empty `[]`, None round-trip, and discriminated-union dispatch on
omitted `rank_list`; extended the heterogeneous-list union round-trip
to include a bare reduce. Net test delta +2 (351 total). Full
`make check` green.

**Downstream task** (not this fix): the validator/evaluator must
resolve `ReduceSpec.rank_list is None` to "all reducible ranks at
this binary-label site" before such a program can run end-to-end.

### Finding 4 — INCONSISTENCY ✅ FIXED
`REGEX_USER_DEFINED_NAME` at `edge_ir/ir/patterns.py:30` was
`r"^[A-Za-z_][A-Za-z0-9_]*$"` — included underscores. EBNF at
`tex/notes/main.ebnf-cropped.tex:66` said `/[A-Za-z][A-Za-z0-9]*/`
(no underscore, no hyphen). Paper examples at
`tex/7-examples.tex:439,493,704` use hyphens (`select-any-s`,
`select-min-s`, `user-rand-func`) — neither EBNF nor IR accepted them.
Three-way disagreement.
**Resolution:**
(a) IR: relaxed `REGEX_USER_DEFINED_NAME` to
`r"^[A-Za-z_][A-Za-z0-9_-]*$"`. Body now accepts letters, digits,
underscores, and hyphens. First char unchanged (letter or `_`).
(b) EBNF: updated `<user-defined-function>` to
`/[A-Za-z][A-Za-z0-9_-]*/` — added both `_` and `-` to body. EBNF
first char remains stricter (letter only) per paper convention; the
IR's leading-`_` permissiveness is documented as deliberate drift.
(c) Tests: 1 inverted (`test_user_defined_compute_op_rejects_hyphen`
→ `_accepts_hyphen`); 9 added in `test_ir_op.py` and
`test_ir_patterns.py` covering all 4 UDF op categories + body
edge cases (`foo--bar`, `foo-`) and leading-hyphen rejection;
3 stale tests in other consumers caught by the validator and fixed:
`test_stopping_condition_rejects_hyphen_in_boolean_function` →
`_accepts_hyphen` in `test_ir_einsum.py`; `test_user_defined_rejects_special_chars`
split into `test_user_defined_data_type_accepts_hyphen` +
`test_user_defined_data_type_rejects_space` in `test_ir_tensor.py`;
fixture `tests/fixtures/user_defined_data_type/invalid/special_chars.json`
swapped from `My-Type` to `My Type` (still illegal). Net test
delta +10 (361 total). Full `make check` green.

**Drift note:** the IR is intentionally one bit more permissive than
the EBNF (allows leading underscore). Documented in the comment block
above the regex at `edge_ir/ir/patterns.py:26-32`.

### Finding 5 — AMBIGUITY
`StoppingCondition` at `edge_ir/ir/einsum.py:75-92` is just
`(rank_variable, boolean_function)`. Paper stopping conditions
reference specific tensors and offsets:
- `\diamond: ||F_{i+1}|| \equiv 0`
- `\diamond: D_{i+1} \equiv D_i`
- `\diamond: F_{i+3} \equiv F_i` (`tex/7-examples.tex:618`)
- `\diamond_j: P_{j+1} \equiv P_j` (`tex/7-examples.tex:559,561`)

The IR provides only a function name with no way to encode tensor
arguments. The paper's grammar (EBNF line 40) is silent on this — IR
is faithful to EBNF, but EBNF is incomplete vs. paper usage. Real
expressivity gap. Two reasonable resolutions: extend
`StoppingCondition` with `args: list[TensorProjection]`, or treat
boolean-function as an opaque host-language closure. Decision needed
before more programs are added to the artifact suite.

### Finding 6 — INCONSISTENCY
BFS artifact's G declared with 2 ranks at
`examples/algorithms/bfs/build_bfs_program.py:43-51` and `examples/algorithms/bfs/bfs_program.json:3-22`.
Paper's full-spec sidebar at `tex/bfs-example/07-iteration.tex:152`
declares `G^{I, S\equiv|V|, D\equiv|V|}` (3 ranks including
generational I). Paper's early walkthrough at
`tex/bfs-example/02-declarations.tex:43` uses 2 ranks. Artifact uses
the early-walkthrough G with the full-spec iterative einsums — a
hybrid no single paper section contains verbatim. Suggested fix:
either update declarations to match `eqn:edge_bfs_full` (3-rank G
broadcast over I) or note in script docstring that this artifact
combines two paper sources.

### Finding 7 — INCONSISTENCY
`examples/algorithms/bfs/build_bfs_program.py:4` cites "paper Equation 41, Section 7.3."
Eq 41 in the paper is actually `eqn:rbfs` at `tex/7-examples.tex:41`
— the SINGLE-EINSUM "push BFS" with an inner anonymous tensor, using
AND/OR Boolean ops and only TWO einsums. Section 7 is "EDGE Syntax"
(`tex/6-syntax.tex`); BFS case studies are in Section 8.1.1
(`ssec:rrbfs` in `tex/7-examples.tex`). The artifact actually
corresponds to `eqn:bfsi` / `eqn:edge_bfs_full` at
`tex/bfs-example/07-iteration.tex:67-70`. Suggested fix: replace the
citation with `Reference: paper Equation eqn:bfsi (and full-spec
eqn:edge_bfs_full), tex/bfs-example/07-iteration.tex`.

### Finding 8 — AMBIGUITY (paper-side gap)
The compute operator `take_left` in einsum (b) at
`examples/algorithms/bfs/build_bfs_program.py:172-178` resolves a paper-internal
overload. Paper writes `:: \bigwedge_{d} \leftarrow(\cap)` where the
position before the parenthesis is the `<compute-operator>` slot. EBNF
has no built-in compute symbol `←`. The same `←` is the formal
take-left MERGE operator in Appendix A item 4. So `←` is overloaded:
take-left merge AND a take-left compute function the paper never
explicitly names. Artifact's choice
`UserDefinedComputeOp(name="take_left")` is pragmatic but invented.
Name collision with `BuiltinMergeOp(symbol="take_left")` is acceptable
(different op categories) but confusing to readers. Suggested fix:
either request paper-side patch naming this UDF, or document the
convention in `op.py` docstring.


TODO: fix this!
### Finding 9 — NIT
Mutable default values: `einsums: list[Einsum] = []` at
`edge_ir/ir/program.py:39`, `stopping_conditions: list[StoppingCondition] = []`
at `edge_ir/ir/einsum.py:128`. Pydantic v2 deep-copies defaults and
the models are frozen, so this is safe — but it's inconsistent with
`Field(default_factory=list)` and trips ruff if anyone copies the
pattern into a non-frozen model. Suggested fix:
`Field(default_factory=list)` for both. Or leave as-is and accept the
style inconsistency.

### Finding 10 — AMBIGUITY
`RankDeclaration` at `edge_ir/ir/tensor.py:129-152` allows `shape` and
`coord_set` to coexist without a consistency check. A declaration
like `RankDeclaration(name="S", shape=5, coord_set=CoordSetEnum(coords=["a","b","c"]))`
is accepted, with the shape (5) and the enum (3 elements) disagreeing.
The docstring doesn't say which wins when both are present. Suggested
fix: add a Pydantic validator rejecting inconsistent combinations,
document precedence, or make them mutually exclusive at the type
level (`shape_or_coords: int | str | CoordinateSet | None`).

### Finding 11 — AMBIGUITY (paper-side)
Scalar (0-rank) tensors. IR's `TensorDeclaration.ranks` at
`edge_ir/ir/tensor.py:179` allows `[]` (consistent with paper prose at
`tex/2-prelims.tex:39`). EBNF at `tex/notes/main.ebnf-cropped.tex:28-29`
requires non-empty `<shape-definition>` — surface syntax can't express
scalar declarations. Paper-prose-vs-EBNF disagreement. IR is fine;
flag for paper-side EBNF fix.

### Finding 12 — AMBIGUITY
`<rank-cs-override>` interpretation. EBNF at
`tex/notes/main.ebnf-cropped.tex:18-23` puts `<rank-name>` AND
`^<tensor-name>` on the LEFT of `=`. The tensor superscript is
either redundant (always the tensor being declared) or intentional
(allows cross-tensor overrides). IR moves `coord_set` INSIDE each
`RankDeclaration` (`edge_ir/ir/tensor.py:129-152`), forcing the
"this tensor's rank" reading — strictly less expressive than literal
EBNF. Cannot tell from EBNF or paper text which is intended. Open question.

### Finding 13 — NIT
`is_affine: bool | None = None` on `RankArith` at
`edge_ir/ir/expr.py:127` and `RankFunction` at `edge_ir/ir/expr.py:163`,
documented as "filled out by the validator." But the model is
`frozen=True`, so the validator cannot mutate the field — must use
`model_copy(update=...)` and rebuild the tree. Docstring doesn't
acknowledge this. Suggested fix: update docstring to clarify the
rebuild requirement, or move `is_affine` to a separate lookup table
outside the frozen tree (analogous to `op_properties` registry
pattern).

### Finding 14 — AMBIGUITY
Both `AnonymousTensor` (`edge_ir/ir/expr.py:258-279`) and `Einsum`
(`edge_ir/ir/einsum.py:43-68`) have `specs: list[ComputationSpec]`.
Paper's Eq 41 puts ALL specs at the outer einsum level with labels
distinguishing the binaries (including labels that cross the
parenthesis boundary). IR allows specs in either or both locations —
not pinned down which is canonical. Different parsers may make
different choices. Suggested fix: document the relationship — pick
one of: (a) AnonymousTensor specs cover only its internal binaries
with scoped labels; (b) AnonymousTensor specs must be empty; (c)
specs may be on either, with documented label scoping.

### Finding 15 — NIT (acknowledged in script)
F's `empty_value=0` in the artifact at
`examples/algorithms/bfs/build_bfs_program.py:8-9,54-60`. Paper at
`tex/bfs-example/07-iteration.tex:153` uses `\infty`. Script docstring
acknowledges the divergence as a sentinel choice. But `0` collides
with valid BFS depth-0 values — any future evaluator will produce
wrong BFS results for the source vertex. Suggested fix: change F's
data type to `float` so we can use `float("inf")`, or use a
non-colliding sentinel like `-1`. Option (a) is closer to the paper.

---

## Verification results from the audit

Confirmed during the audit (no findings):
- All 16 merge-op truth tables in `op_properties.py` match Appendix A
  cell-by-cell. Zero issues here.
- The `commutative` flag on each merge op is correctly derived from
  the truth table: `commutative` iff `tt[(F,T)] == tt[(T,F)]`.
- The `includes_neither` flag is correctly derived: `includes_neither`
  iff `tt[(F,F)] == True`.
- Compute op Unicode mappings are correct (`+`, `-`, `×`, `÷`).
- Unary op Unicode mapping is correct (`¬`).
- Discriminator dispatch is correct for all unions; no collision risks
  among `kind` values.
- JSON schema at `schemas/program.schema.json` is comprehensive — all
  union variants present and cross-referenced via `$defs`.
- Recursion in `Expression` resolves via `from __future__ import
  annotations`; no missing `model_rebuild()` calls.
- `op_properties.py` has no IR coupling (verified by subprocess test
  in `tests/test_op_properties.py`).
