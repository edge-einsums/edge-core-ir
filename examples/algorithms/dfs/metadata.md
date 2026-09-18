# DFS IR Builder Metadata

> *Created with the help of Claude Code.*

Algorithm: `dfs`

Variant: sequential depth-first search via the **order-stamped value
encoding** (stack payload = stamp `sigma(i, v) = i*|V| + v`; PEEK = populate
argmax; pop = masked removal; push = update merge). Full spec and rationale:
`einsum.md` (this folder).

Builder: `build_dfs_program.py`

Reference math: `examples/algorithms/dfs/einsum.md`

## Ground-truth test vector

Directed graph on `|V| = 4`: edges `0->1, 0->2, 1->3, 2->3`. Root = vertex 0.

Hand-traced **pop / discovery order = [0, 2, 3, 1]**. Siblings are visited in
descending vertex id because `sigma` gives higher-id vertices higher stamps
(so 2 is popped before 1). Final `P = {0, 1, 2, 3}` (all visited). The stack
empties one step after vertex 1 is popped, satisfying the stopping condition
`||S_{i+1}|| == 0`.

## Coverage

| ID | Status | Equation / Text |
|---|---|---|
| `decl-G` | exact | G^{S ≡ \|V\|, D ≡ \|V\|} -> Boolean, empty = False |
| `decl-S` | exact | S^{I, V ≡ \|V\|} -> integer, empty = -1 (stack; payload = stamp) |
| `decl-P` | exact | P^{I, V ≡ \|V\|} -> Boolean, empty = False (visited) |
| `decl-F` | exact | F^{I, V ≡ \|V\|} -> integer, empty = -1 (one-hot top) |
| `decl-N` | exact | N^{I, D ≡ \|V\|} -> Boolean, empty = False (neighbors) |
| `decl-U` | exact | U^{I, D ≡ \|V\|} -> Boolean, empty = False (undiscovered neighbors) |
| `decl-T` | exact | T^{I, V ≡ \|V\|} -> integer, empty = -1 (stack after pop) |
| `decl-Sprime` | exact | Sprime^{I, V ≡ \|V\|} -> integer, empty = -1 (S'; newly stamped) |
| `decl-One` | hand-authored literal | zero-rank int `One` (value=1); the populate binary's right operand, mirroring max-flow's `PushCand = Adm . One` |
| `decl-Stamp` | **PLACEHOLDER (gap)** | zero-rank int `Stamp` (value=None); stand-in for the per-point rank-as-value stamp `sigma`. NOT the real operand -- see Gap Notes G1 |
| `decl-Lit_0` | hand-authored literal | zero-rank int `Lit_0` (value=0) for the stopping condition literal `0` |
| `decl-Lit_True` | hand-authored literal | zero-rank bool `Lit_True` (value=True) for `P_0`'s True |
| `init-S0` | **partial (gap G1)** | S_{0, v : v ∈ root_id} = sigma(0,v) = v -- restricted-iteration predicate exact; RHS uses the `Stamp` placeholder instead of the per-point value `v` |
| `init-P0` | exact | P_{0, v : v ∈ root_id} = True |
| `E1 PEEK` | **partial (gap G2)** | F_{i,v*} = S_{i,v} :: ⋘_v 𝟙(select-max-val) -- PopulateSpec(rank_list=["v"], compute=take_left, coord=select-max-val); `select-max-val` is a UDF coord op (G2) |
| `E2 ADVANCE` | exact | N_{i,d} = G_{s,d} · F_{i,s} :: ⋀_s ←(∩) ⋁_s ANY(∪) |
| `E3 MASK` | exact | U_{i,d} = N_{i,d} · ¬P_{i,d} :: ⋀_d ←(∩) |
| `E4a POP` | exact | T_{i,v} = S_{i,v} · ¬F_{i,v} :: ⋀_v ←(∩) (masked removal; no `−`/`>>`) |
| `E4b STAMP` | **partial (gap G1)** | Sprime_{i,v} = U_{i,v} · sigma(i+1,v) :: ⋀_v →(∩) -- map shape exact; RHS uses the `Stamp` placeholder instead of the rank-as-value `(i+1)*\|V\|+v` |
| `E4c PUSH` | **approximate (gap G3)** | S_{i+1,v} = T_{i,v} · Sprime_{i,v} :: ⋀_v <<(∪) -- encoded as `take_right(union)`; `<<` has no IR symbol (G3) |
| `E5 VISIT` | exact | P_{i+1,v} = P_{i,v} · Sprime_{i,v} :: ⋀_v OR(∪) |
| `stop` | exact | ⋄ : ‖S_{i+1}‖ ≡ 0 |

## Gap Notes

Three gaps. The cascade is structurally complete and round-trips through the
IR + JSON schema, but three pieces are not faithfully expressible in the
current IR and are flagged for the edge-expert.

- **G1 -- the rank-as-value stamp `sigma` is not expressible in the
  `Expression` union.** Step (4b) `Sprime = U · sigma(i+1, v)` and the
  initialization `S_{0,v} = sigma(0, v) = v` both put a per-iteration-point
  rank-variable-as-value (`(i+1)*|V| + v`, resp. `v`) in a value-operand
  position. The paper sanctions rank-as-value (the EDGE paper's rank-variables-as-tensors section),
  but the IR `Expression` union is `InputTensor | UnaryApp |
  BinaryApp | AnonymousTensor` with **no rank-as-value leaf**. D21 deliberately
  keeps scalars as zero-rank broadcast tensors -- those are *constants* and
  cannot depend on `(i, v)`. The builder uses a zero-rank `Stamp` placeholder
  operand to keep the cascade shape complete; the placeholder is explicitly
  **not** the real value. This is the central open question for the expert:
  does the IR need a `RankExpressionValue`-style leaf in `Expression` (and a
  matching paper-side `<expression>` production), or is there a sanctioned
  lowering of a rank-as-value operand to existing nodes?

- **G2 -- `select-max-val` is a user-defined coordinate operator.** PEEK is a
  populate argmax over the stamp. It is encoded exactly like max-flow's
  `PushCand = Adm . One :: ⋘_v 𝟙(pick-admissible-edge)`:
  `PopulateSpec(rank_list=["v"], compute_op=take_left, coord_op=
  CoordinateOp("select-max-val"))`. `select-max-val` resolves through the UDF
  registry (per `docs/udf_registry_plan.md`), which is still a plan, not yet
  implemented. No IR change is needed -- coordinate operators are user-defined
  by definition (`keywords.md`: "No built-ins" for `CoordinateOp`) -- but the
  evaluator will need the impl. Open question for the expert: is
  "select-extremum-over-a-stored-stamp" already covered by the paper's
  argmin/argmax populate examples, or does it want a named EDGE extension?

- **G3 -- the update merge `<<` has no IR symbol.** Step (4c) PUSH uses
  `<<(∪)` ("take right where right exists, else take left",
  the EDGE paper's `<<` definition). The 16 builtin merge symbols in
  `op.py`/`op_properties.py` do not include `<<`; `docs/keywords.md` lists
  `<<` as a surface-syntax token that lowers to a Map with `<<` semantics, but
  no IR-level encoding exists yet. Because T and Sprime have disjoint supports
  in this algorithm, `<<` and plain `union` coincide here, so PUSH is encoded
  as `take_right(union)` as a faithful-for-this-input approximation. Open
  question for the expert / parser author: should `<<` become a 17th builtin
  merge symbol, a UDF merge op, or stay a lowering-only token?

## Notes

- The DFS init block mirrors BFS: two einsums (`S_0`, `P_0`), each with a
  `SetMembership` predicate over the host-supplied coord set `root_id` (the
  DFS root vertices). `Lit_True` carries the literal `True` for `P_0`.
- `S`'s `empty_value` is `-1`, chosen outside the stamp range (stamps start at
  `sigma(0,0) = 0`), so no stamp collides with it.
- The `Stamp` placeholder (G1) is the only declaration whose `value` is
  `None` and whose meaning is *not* what the IR literally encodes; it exists
  solely to keep the cascade structurally complete pending the expert's
  ruling on rank-as-value operands.
- `Sprime` is the IR-legal spelling of the math tensor `S'` (`REGEX_TENSOR_NAME`
  forbids the apostrophe).
- The reference evaluator in `edge_ir/evaluator/` can execute this program.
  The
  `[0, 2, 3, 1]` ground truth above is hand-traced.
