# BFS IR Builder Metadata

> *Created with the help of Claude Code.*

Algorithm: `bfs`

Variant: depth-tracking BFS (full-edge form; paper `eqn:bfsi` / `eqn:edge_bfs_full` in the EDGE paper's BFS example)

Builder: `build_bfs_program.py`

Reference math: `examples/algorithms/bfs/einsum.md` (mirrors the EDGE paper's BFS sidebar)

## Coverage

| ID | Status | Equation / Text |
|---|---|---|
| `decl-G` | exact | G^{S ≡ \|V\|, D ≡ \|V\|} → integer, empty = 0 |
| `decl-F` | exact | F^{I, S ≡ \|V\|} → integer, empty = ∞ |
| `decl-T` | exact | T^{I, D ≡ \|V\|} → integer, empty = ∞ |
| `decl-P` | exact | P^{I, D ≡ \|V\|} → Boolean, empty = False |
| `decl-Lit_0` | hand-authored (proxy for parser-synthesized literal) | zero-rank int Lit_0 with value=0, empty_value=None (for the stopping condition's literal `0`; also reused as the RHS of init-F0) |
| `decl-Lit_True` | hand-authored (proxy for parser-synthesized literal) | zero-rank bool Lit_True with value=True, empty_value=None (for the RHS of init-P0) |
| `init-F0` | exact | F_{0, s : s ∈ id} = 0 — encoded as an init Einsum with `expression=InputTensor(Lit_0)` and `predicates=[SetMembership(member=RankVariable("s"), coord_set=CoordSetName("id"))]` |
| `init-P0` | exact | P_{0, d : d ∈ id} = True — encoded as an init Einsum with `expression=InputTensor(Lit_True)` and `predicates=[SetMembership(member=RankVariable("d"), coord_set=CoordSetName("id"))]` |
| `E01` | exact | T_{i,d} = G_{s,d} · F_{i,s} :: ⋀_s +(∩) ⋁_s ANY(∪) |
| `E02` | exact | F_{i+1,d} = T_{i,d} · ¬P_{i,d} :: ⋀_d ←(∩) |
| `E03` | exact | P_{i+1,d} = P_{i,d} · F_{i+1,d} :: ⋀_d OR(∪) |
| `stop` | exact | ⋄ : ‖F_{i+1}‖ ≡ 0 |

## Gap Notes

No remaining IR gaps for BFS. The previously-flagged `init-F0` / `init-P0` predicate gap was closed in the 2026-05 restricted-iteration-predicates round (D23). Every einsum and declaration in `einsum.md` now has an exact IR encoding.

## Notes

- **The BFS init block is now fully encoded.** `init.einsums` has two einsums (F_0 and P_0); each carries a `SetMembership` predicate referencing the host-supplied coord set `id` (BFS source vertices). The synthesized `Lit_0` and `Lit_True` zero-rank tensors carry the literal RHS values.
- The `Lit_0` declaration is **hand-authored** in the builder script. A future parser would synthesize it during AST→IR lowering using the rules in `docs/parsing.md`; until that parser exists, the BFS artifact serves as a fixture mirroring what the parser will eventually produce.
- This builder was reconciled with `einsum.md` in the 2026-05 scalar-literal round (see `docs/design_decisions.md` D21):
  - F's `data_type` flipped from `float` to `int`.
  - T's `empty_value` flipped from `0` to `float("inf")` (serializes as the `"inf"` sentinel).
  - Einsum (a)'s reduce compute_op flipped from `UserDefinedComputeOp(name="min")` to `UserDefinedComputeOp(name="ANY")`.
- The stopping condition's literal `0` was previously encoded as `ScalarValue(value=0)` in `edge_ir/ir/stopping.py`. `ScalarValue` has been deleted; the literal is now a synthesized zero-rank `TensorDeclaration` named `Lit_0` (`data_type=int`, `empty_value=None`, `value=0`), referenced via `TensorProjectionValue` per D21. `empty_value=None` because the contextual source is `occupancy(F)`, a `PropertyApp` whose return type is `int` with algorithm-dependent empty.
