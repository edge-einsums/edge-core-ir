# Max-flow IR Builder Metadata

Algorithm: `max-flow`

Variant: `Full-Edge / push-relabel`

Builder: `build_max_flow_program.py`

Artifact: `max_flow_program.json`

Source of truth: `Push_Relabel_Max_Flow.ipynb` (EDGE tutorial zoo). Einsum IDs
E01–E24 match the tutorial and the step-through visualizer. See `CHANGES.md`
for the diff against the previous version of this artifact.

## Status vocabulary

- **exact** — the IR encodes the equation with no loss.
- **partial** — the IR encodes the equation's meaning, but through a
  workaround rather than a matching construct. Every `partial` row has a
  corresponding entry in Gap Notes.

## Coverage

| ID | Status | Equation / Text |
|---|---|---|
| `decl-G` | exact | G^{U ≡ \|V\|, V ≡ \|V\|} → Integer, empty = 0 |
| `decl-C` | exact | C^{U ≡ \|V\|, V ≡ \|V\|} → Integer, empty = 0 |
| `decl-F` | exact | F^{I, U ≡ \|V\|, V ≡ \|V\|} → Integer, empty = 0 |
| `decl-R` | exact | R^{I, U ≡ \|V\|, V ≡ \|V\|} → Integer, empty = 0 |
| `decl-E` | exact | E^{I, U ≡ \|V\|} → Integer, empty = 0 |
| `decl-D` | exact | D^{I, U ≡ \|V\|} → Integer, empty = +∞ |
| `decl-Act` | exact | Act^{I, U ≡ \|V\|} → Boolean, empty = false |
| `decl-S` | exact | S^{U ≡ \|V\|} → Boolean, empty = false |
| `decl-T` | exact | T^{U ≡ \|V\|} → Boolean, empty = false |
| `decl-FS` | exact | FS^{I, U ≡ \|V\|, V ≡ \|V\|} → Integer, empty = 0 (staging tensor for E01/E02) |
| `decl-NeiLbl` | exact | NeiLbl^{I, U, V} → Integer, empty = +∞ (holds heights; see below) |
| `decl-MinNeiLbl` | exact | MinNeiLbl^{I, U} → Integer, empty = +∞ |
| `decl-NewD` | exact | NewD^{I, U} → Integer, empty = +∞ |
| `init-F0` | partial | F_{0,u,v} = 0 |
| `init-E0` | partial | E_{0,u} = 0 |
| `init-D1` | partial | D_{1,u} = 0 |
| `init-D1-source` | partial | D_{1,u:u=s} = \|V\| |
| `init-S` | partial | S_{u:u=s} = true |
| `init-T` | partial | T_{u:u=t} = true |
| `E01` | exact | FS_{1,u,v} = S_u ⋅ C_{u,v} :: ⋀ *(∩) |
| `E02` | exact | F_{1,u,v} = FS_{1,u,v} ⋅ FS_{1,v,u} :: ⋀ -(∪) |
| `E03` | partial | E_{1,v} = F_{1,u,v} :: ⋁ +(∪) |
| `E04` | partial | R_{1,u,v} = case {0 if u=s; C_{v,u} if v=s ∧ u≠s; C_{u,v} otherwise} |
| `E05` | exact | NST_u = ¬S_u ⋅ ¬T_u :: ⋀ AND(∩) |
| `E06` | exact | Act_{i,u} = NST_u ⋅^1 (E_{i,u} ⋅^2 0)_{i,u} :: ⋀^1 ←(∩) ⋀^2 >(∩) |
| `E07` | exact | ActR_{i,u,v} = Act_{i,u} ⋅ R_{i,u,v} :: ⋀ ←(∩) |
| `E08` | exact | Lbl_{i,u,v} = D_{i,u} ⋅ (D_{i,v} + 1)_{i,v} :: ⋀ ≡(∩) |
| `E09` | exact | Adm_{i,u,v} = ActR_{i,u,v} ⋅ Lbl_{i,u,v} :: ⋀ AND(∩) |
| `E10` | partial | PushCand_{i,u,v*} = Adm_{i,u,v} ⋘_{v*} 𝟙(pick-admissible-edge) |
| `E11` | exact | delta_{i,u,v} = (E_{i,u} ⋅^1 R_{i,u,v})_{i,u,v} ⋅^2 PushCand_{i,u,v} :: ⋀^1 min(∩) ⋀^2 ←(∩) |
| `E12` | exact | F_{i+1,u,v} = (F_{i,u,v} ⋅^1 delta_{i,u,v})_{i,u,v} ⋅^2 delta_{i,v,u} :: ⋀^1 +(∪) ⋀^2 -(∪) |
| `E13` | partial | InPush_{i,v} = delta_{i,u,v} :: ⋁ +(∪) |
| `E14` | partial | OutPush_{i,u} = delta_{i,u,v} :: ⋁ +(∪) |
| `E15` | exact | E_{i+1,u} = (E_{i,u} ⋅^1 InPush_{i,u})_{i,u} ⋅^2 OutPush_{i,u} :: ⋀^1 +(∪) ⋀^2 -(∪) |
| `E16` | exact | R_{i+1,u,v} = (R_{i,u,v} ⋅^1 delta_{i,u,v})_{i,u,v} ⋅^2 delta_{i,v,u} :: ⋀^1 -(∪) ⋀^2 +(∪) |
| `E17` | exact | Adm_{i+1,u,v} = R_{i+1,u,v} ⋅ Lbl_{i,u,v} :: ⋀ →(∩) |
| `E18` | partial | HasAdm_{i+1,u} = Adm_{i+1,u,v} :: ⋁ OR(∪) |
| `E19` | exact | Act_{i+1,u} = NST_u ⋅^1 (E_{i+1,u} ⋅^2 0)_{i+1,u} :: ⋀^1 ←(∩) ⋀^2 >(∩) |
| `E20` | exact | Rel_{i+1,u} = Act_{i+1,u} ⋅ ¬HasAdm_{i+1,u} :: ⋀ AND(∩) |
| `E21` | exact | NeiLbl_{i,u,v} = (R_{i+1,u,v} ⋅^1 Rel_{i+1,u})_{i,u,v} ⋅^2 D_{i,v} :: ⋀^1 ←(∩) ⋀^2 →(∩) |
| `E22` | partial | MinNeiLbl_{i,u} = NeiLbl_{i,u,v} :: ⋁ min(∪) |
| `E23` | exact | NewD_{i,u} = (MinNeiLbl_{i,u} ⋅ 1)_{i,u} :: ⋀ +(∩) |
| `E24` | partial | D_{i+1,u} = D_{i,u} ⋅ NewD_{i,u} :: ⋀ <<(∪) |
| `stop` | exact | ◇ : \|Act_{i+1}\| ≡ 0 |

## Naming divergences

| Spec | IR | Why |
|---|---|---|
| `delta` | `Delta` | IR tensor names must match `^[A-Z][A-Za-z0-9_]*$`. |
| `<<` | compute op `update` | Operator names must match `^[A-Za-z_][A-Za-z0-9_-]*$`, which rejects `<<`. |

## Gap Notes

| Equation | Unsupported syntax construct | Why current IR cannot represent it | Closest workaround | Recommended IR extension | Priority |
|---|---|---|---|---|---|
| init-F0 / F_{0,u,v} = 0; init-E0 / E_{0,u} = 0; init-D1 / D_{1,u} = 0; E06/E19 comparisons to 0; E08/E23 additions by 1 | Scalar value literals in expression position. | **ADDRESSED 2026-05.** `TensorDeclaration` gained `value: Any \| None` for zero-rank tensors; scalars are zero-rank tensors everywhere (see `docs/design_decisions.md` D21). The parser will synthesize a zero-rank `TensorDeclaration` per literal occurrence. | Builder still uses host-named constants (`Zero`, `One`, `VertexCount`, `FalseConst`, `MinIdentity`) for these init / comparison sites because they predate the IR change and were hand-authored. The synthesized `Lit_0` for the stopping condition coexists with `Zero` — CSE between them is a downstream optimizer's job (see D21 "No CSE at the parser level"). | Implemented (D21). | done |
| init-D1-source / D_{1,u:u=s} = \|V\|; init-S / S_{u:u=s} = true; init-T / T_{u:u=t} = true | Predicate-constrained rank variable expression u:u=s or u:u=t. | RankExpression variants are bare variables, constants, arithmetic, or opaque functions; none can restrict a rank variable to coordinates satisfying a predicate. | Treat S and T as sparse Boolean input masks supplied by the host, and use S_u with VertexCount to derive the source height. | Add predicate-constrained rank expressions or guarded tensor projections, plus validation that singleton source/sink constraints select exactly one coordinate. | high |
| E04 / R_{1,u,v} = case {0 if u=s; C_{v,u} if v=s ∧ u≠s; C_{u,v} otherwise} | Case statement with ordered predicates over rank coordinates. | The current Expression union has no Case/Where/Select node and ComputationSpec only describes binary actions by label. | Emit three sequential R initialization einsums in REVERSE priority order — default C_{u,v}, then source column through S_v, then source row zero through S_u — so that later-overwrites-earlier reproduces the written arm priority, including at the (s,s) cell. This preserves the components but not a first-class mutually-exclusive case expression, and the ordering discipline is invisible to the IR. | Add CaseExpression with predicate arms over rank variables/tensor masks, explicit priority/otherwise semantics, and validation for overlapping arms. | high |
| E03; E13; E14; E18; E22 | Reduce action attached to a single input tensor with no BinaryApp label. | ReduceSpec requires a label that links to a BinaryApp; InputTensor-only expressions have no label site. | Introduce a scalar identity tensor and a synthetic BinaryApp so the ReduceSpec has a label. | Allow ReduceSpec to target a unary/input expression directly, or add an explicit reduction expression node independent of binary labels. | high |
| E10 / PushCand_{i,u,v*} = Adm_{i,u,v} ⋘_{v*} 𝟙(pick-admissible-edge) | Populate/selector shorthand over a single tensor and nondeterministic/algorithmic choice. | PopulateSpec can be represented only as a spec linked to a BinaryApp, and CoordinateOp is just an opaque name; there is no first-class 'choose exactly one v per u' invariant. | Create a synthetic BinaryApp Adm ⋅ One and attach PopulateSpec(label=1, rank_list=['v'], compute_op='pick-admissible-edge', coord_op='select-one-admissible-v'). | Add selector/populate expression syntax with cardinality constraints and deterministic/nondeterministic choice annotations. | high |
| E22 / MinNeiLbl_{i,u} = NeiLbl_{i,u,v} :: ⋁ min(∪) | Unary min-reduction identity / empty-neighborhood behavior. | Current ReduceSpec needs a binary label and TensorDeclaration has no type-aware infinity/max-int literal for int min identity. | Use a host-provided MinIdentity scalar in a synthetic binary min reduction. | Add reduction identities to op registry or specs, and validate empty reductions by data type. | high |
| E24 / D_{i+1,u} = D_{i,u} ⋅ NewD_{i,u} :: ⋀ <<(∪) | The partial-update compute operator `<<`. | `BuiltinComputeOp.symbol` is limited to `+ - * /`, and the user-defined operator-name pattern rejects `<<` as a name. | Emit a `UserDefinedComputeOp(name="update")`, whose meaning — take the right value where present, else the left — lives outside the IR. | Add `<<` as a builtin compute operator. The EDGE paper treats it as standard (eqs. 197–198, Cascades 4/5/6), and leaving it user-defined means every consumer must independently agree on what it does. | high |

## Evaluator contract (not expressible in the IR)

The cascade assumes **a stored value equal to a tensor's declared `empty` is
treated as absent**. This is load-bearing in at least two places:

- E07's intersect drops saturated residual edges (`R = 0 = empty`) with no
  explicit positivity guard on R.
- E06/E19's `> 0` test relies on a vertex with zero excess being absent
  rather than present-and-false.

Nothing in the IR records this convention, so two conforming evaluators could
disagree. It belongs in the semantic model alongside the merge truth tables.

Because of it, **an intermediate's empty value is a correctness decision, not
bookkeeping**. `NeiLbl`, `MinNeiLbl` and `NewD` hold heights and must use
`+inf`; with `0` a height-0 neighbour is dropped before `E22`'s `min` and the
vertex never relabels. The tutorial declares only the nine named tensors, so
these were chosen here and are written down in `einsum.md` rather than left
implicit in the builder.

A second unrecorded convention: **writing a cell's empty value clears it.**
E04's lowered source-row arm writes `0` to `R[s,*]`, and `0` is `R`'s empty
value — if that did not clear the cell, the default arm's capacity would
survive and the source would still have outgoing residual.
