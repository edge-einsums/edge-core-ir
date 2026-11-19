# Max-flow artifact — what changed and why

The max-flow spec in this directory was the draft I handed over. It has since
been corrected in the EDGE tutorial zoo
(`Push_Relabel_Max_Flow.ipynb`),
validated against a working evaluator, and used to drive a step-through
visualizer. This rewrite brings the artifact in line with that corrected
version.

An earlier review raised six open questions. Items 1, 2, 4, 5 and 6 are
resolved below; item 3 is unchanged. Five further divergences were found that
were not on that list, one of which changes the computed max flow.

---

## Numbering

Einsum IDs are now **E01–E24**, matching the tutorial and the visualizer, so a
row in `metadata.md`, a line in `einsum.md` and a step in the animation all
refer to the same Einsum. The old numbering ran E01–E25 over a different
cascade and no longer applies.

The IR cascade has 26 entries for 24 spec Einsums, because E04's three-arm
`cases` is lowered to three sequential writes to `R`.

---

## Corrections

### 1. The preflow is now antisymmetric  (E01, E02 — was E01)

Previously one Einsum:

```
F_{1,u,v} = S_u . C_{u,v} :: /\ *(∩)
```

Flow is antisymmetric — `f(u,v) = -f(v,u)` — but this stored only the forward
half, which made `F` the odd one out among the tensors. It is now staged
through `FS` and antisymmetrized:

```
FS_{1,u,v} = S_u . C_{u,v}          :: /\ *(∩)
F_{1,u,v}  = FS_{1,u,v} . FS_{1,v,u} :: /\ -(∪)
```

A tensor cannot be subtracted from its own transpose in a single Einsum, hence
the staging tensor. `FS` is declared alongside the rest.

### 2. Excess is one reduction, not three Einsums  (E03 — was E02–E04)

Because `F` is now antisymmetric, outflow is already stored as negative
entries on the reverse cells, so summing the column nets it off:

```
E_{1,v} = F_{1,u,v} :: \/ +(∪)
```

`In` and `Out` existed only to compute in-minus-out on a non-antisymmetric
`F`. They are removed from the declarations — leaving them would invite an
evaluator to keep computing dead tensors.

### 3. D's empty value is +inf, not 0  (resolves open question 1)

`D` was declared `Integer, empty` and the builder filled `0`. Height 0 is a
real and common height — every non-source vertex starts there — so it cannot
double as the absent marker. Both `E08` (the height rule) and `E21` (the
neighbour-height gather) read `D` through an **intersect**; with `empty = 0` a
height-0 neighbour is dropped before reaching `E22`'s `min`, and the vertex
relabels above its true lowest neighbour.

`+inf` is also the correct identity for that `min`.

The earlier review observed that `D_{0,u} = 0` "does nothing when 0 is the empty
value" and inferred D wanted a non-zero sentinel. That inference was right.

### 4. D's initialization moved to generation 1

`D_{0,u} = 0` → `D_{1,u} = 0`. The preflow Einsums write generation 1, and the
iterative body reads `D_{i}` from `i = 1` onward, so generation 0 was never
read.

### 5. The height rule merges over intersect, not union  (E08)

```
Lbl_{i,u,v} = D_{i,u} . (D_{i,v} + 1)_{i,v} :: /\ ==(∩)     # was ==(∪)
```

Under a union merge a vertex with no height is compared as though it had one
and can satisfy the height rule spuriously. A height should only participate
when it exists.

### 6. The height update is a partial update  (E24) — **this one changed the answer**

Previously:

```
D_{i+1,u} = NewD_{i,u} . D_{i,u} :: /\ <-(∩)
```

and the builder emitted `take_left(take_left)`, which does not match the
surface form either — worth noting on its own, since with no parser nothing
checks that `einsum.edge` and the builder agree.

Both encodings are wrong in the same way. `take_left` is present-iff-**left**-
present and the left operand is `NewD`, which exists only for vertices that
relabel. So `D_{i+1}` retained *only* the relabelled vertices and every other
height silently reverted to `D`'s empty value — which, per item 3, was `0`.
Heights reset to zero every round.

Now:

```
D_{i+1,u} = D_{i,u} . NewD_{i,u} :: /\ <<(∪)
```

The union merge keeps every vertex present; `<<` takes the right value where
it exists and the left one otherwise. Vertices that did not relabel keep their
height. This is the standard partial-update idiom in the EDGE paper
(eqs. 197–198, used in Cascades 4/5/6).

Encoding note: `<<` is emitted as `UserDefinedComputeOp(name="update")`.
Compute builtins are limited to `+ - * /`, and the operator-name pattern
rejects `<<` outright. Given how standard the operator is, it probably belongs
in `BuiltinComputeOp` — flagged in `metadata.md` Gap Notes.

### 7. Case arms are emitted in reverse priority  (E04 — resolves open question 2)

The written order is `0 if u=s`, `C_{v,u} if v=s ∧ u≠s`, `C_{u,v} otherwise`,
with the `u=s` arm highest priority. Sequential lowering means the *last*
write wins, so the arms are now emitted in reverse priority: otherwise, then
source column, then source row.

Previously the source-column arm came last and overwrote the source-row arm at
the `(s,s)` cell — the reverse of the stated order. Rather than assume
"no self-loops" as an invariant, the ordering now reproduces the written
semantics whatever the graph contains.

The spec also gained the explicit `∧ u ≠ s` guard on the second arm, so the
arms are mutually exclusive as written rather than relying on ordering alone.

### 8. Act's second copy binds at the right generation  (E19)

```
Act_{i+1,u} = NST_u .^1 (E_{i+1,u} .^2 0)_{i+1,u}    # binding was _{i,u}
```

The parenthesised subexpression reads `E_{i+1}` but was annotated at
generation `i`.

### 9. Rel is written at i+1  (E20, E21)

`Rel_{i,u}` → `Rel_{i+1,u}`, and `E21` reads it there. `Rel` is derived from
`Act_{i+1}` and `HasAdm_{i+1}`, so labelling it generation `i` misdescribed
which state it belongs to.

### 10. Act uses take-left, not AND  (E06, E19)

```
Act_{i,u} = NST_u .^1 (...) :: /\^1 <-(∩)      # was AND(∩)
```

Both operands are Booleans so the result is the same here; `<-` says what is
meant — the mask's truth is carried, the excess comparison only gates.

### 11. Height-valued intermediates inherit D's empty value  (found by replay)

`NeiLbl`, `MinNeiLbl` and `NewD` were declared `Integer, empty = 0`. They hold
**heights**, so the argument from item 3 applies to them unchanged: a
height-0 neighbour gathered into `NeiLbl` is treated as absent and never
reaches `E22`'s `min`; a `MinNeiLbl` of 0 is likewise dropped, so `E23`
produces no `NewD` and the vertex does not relabel at all.

Fixing `D` but leaving these at 0 reintroduces the same bug one step later in
the cascade. All three are now `empty = +inf`, and `MinIdentity` — the
identity operand for `E22`'s `min` — is `+inf` rather than `0` for the same
reason.

The tutorial's Tensors block declares only the nine named tensors, so these
values had to be chosen here rather than transcribed. They are now written
down explicitly in `einsum.md` and `einsum.edge` instead of living only in the
builder.

**How it was found.** The tutorial's reference trace records what every
einsum produced. Replaying the IR through a JS interpreter and diffing
against that trace flagged `E21`/`E22`/`E23` immediately. That harness lives
in the tutorial repo as `viz/verify_interp.js`.

### 12. Literal scalars carry their value

`Zero`, `One`, `FalseConst` and `MinIdentity` were declared with an
`empty_value` but `value: null`, so a consumer reading the JSON had no way to
learn that `One` is 1. They now carry `value`. `VertexCount` still does not:
`|V|` is graph data a host supplies alongside `G` and `C`.

---

### 13. Inflow reductions read their operand in declared order  (E03, E13)

Both inflow sums were written with the operand's ranks swapped:

```
E_{1,u}      = F_{1,v,u}      :: \/ +(∪)
InPush_{i,u} = delta_{i,v,u}  :: \/ +(∪)
```

They now read:

```
E_{1,v}      = F_{1,u,v}      :: \/ +(∪)
InPush_{i,v} = delta_{i,u,v}  :: \/ +(∪)
```

**This computes exactly the same numbers** — it is a renaming of the two rank
variables, and the interpreter replay across all 222 cascade steps is
unchanged. What it fixes is the reading: inflow to a vertex is now expressed
as "reduce over the source end of every edge into it", with `F` and `delta`
indexed in their declared `(U, V)` order, rather than as an access that looks
like a transpose but is not one. E14 (outflow) already read `delta_{i,u,v}`,
so the three reductions now share one convention.

The cost is a local naming clash: the vertex being summed into is named `v`
in these two Einsums and `u` in most others. That is only a variable name —
`E` and `InPush` are still declared with a single rank `U`, and every other
Einsum still indexes them with `u`. Renaming the *rank* would move the clash
rather than remove it, since `E` is read as `E_{i,u}` in E06, E11, E15 and E19.

---

## Also addressed

- **Open question 5** — `max_flow_program.json` is now committed, matching the
  bfs / dfs / bellman-ford convention. A test asserts it does not drift from
  the builder.
- **Open question 4** — the `decl-*` rows are now genuinely `exact`: the spec
  states explicit empty values, so there is something exact to match.
- **Open question 6** — the over-broad strict-xfail on all six integer tensors
  is gone. It is replaced by two positive tests: `empty = 0` for
  `G/C/F/R/E` (where 0 and absent genuinely coincide), and `empty = +inf` for
  `D` specifically.
- **Open question 3** — unchanged. `Lit_0.empty_value` is still `None`,
  matching the BFS convention.

---

## Still not expressible in the IR

Two things this cascade depends on that the IR does not record:

1. **`value == empty ⟹ absent`.** E07's intersect drops saturated residual
   edges without a positivity guard purely because `R = 0` is `R`'s empty
   value. An earlier draft carried an explicit `PosR` guard; it was dropped
   once this convention was settled. Two conforming evaluators could disagree
   about it today.

2. **Sequential-overwrite semantics for the lowered case arms.** That later
   Einsums overwrite earlier ones is what makes E04 correct, and it lives in
   list order rather than in any node.

Both belong in the semantic model alongside the merge truth tables.

---

### 14. `AdmPost` is a second quantity, and the spec now names it  (E17, E18)

The builder wrote E17's output to `AdmPost` while `einsum.md` still wrote
`Adm_{i+1}`. That disagreement is resolved in the spec's favour of the
builder: `AdmPost` is declared, and E17 and E18 name it.

It was introduced as a workaround — E17 wrote `Adm_{i+1}`, the next round's
E09 wrote the same slice, and because an assignment only adds values, E17's
entries survived into that round's push. They included inactive vertices, the
source among them, and edges to vertices that had just relabelled, so
vertices pushed along edges the height rule no longer admitted.

**But it is not a workaround, and that is why the spec adopts it rather than
the builder reverting.** The two Einsums compute different quantities:

```
E09:  Adm_{i,u,v}      = ActR_{i,u,v} . Lbl_{i,u,v}  :: /\ AND(∩)
E17:  AdmPost_{i+1,u,v} = R_{i+1,u,v} . Lbl_{i,u,v}  :: /\ ->(∩)
```

E09 is *admissible edges out of an ACTIVE vertex, at the current heights*.
E17 is *edges still admissible by residual and height alone*, with no `Act`
factor — and, decisively, it reads `Lbl_i`, the heights BEFORE this round's
relabel, where next round's E09 reads `Lbl_{i+1}`, the heights after.

Measured on the CLRS instance: E17's output and the next round's E09 output
differ in **11 of 11** comparable rounds, because `Lbl_i` differs from
`Lbl_{i+1}` in 9 of them (E24 relabels in 8). Contrast `Act`, where E19's
output equals the next round's E06 output in **all 11** rounds — there the
two really are redundant.

And E17 cannot be made to read `Lbl_{i+1}`:

    E17 -> E18 (HasAdm) -> E20 (Rel) -> E21..E23 -> E24 (D_{i+1}) -> Lbl_{i+1}

`Lbl_{i+1}` needs `D_{i+1}`, which needs `Rel`, which needs E17. A cycle. Nor
is this an encoding accident: the relabel test asks "at my CURRENT height, is
there anywhere left to push? If not, raise it". Asked with post-relabel
heights it is a different question.

One thing that IS free: adding the `Act_{i+1}` factor to E17, to make it read
in the same shape as E09, changes `Rel` in 0 of 11 rounds, because E20
already intersects with `Act_{i+1}` and absorbs it. Not done — it would add a
dependency and force a reorder for no change in meaning.

So `Adm` and `AdmPost` are two tensors because they hold two things. The
remaining single-assignment violations (`D`, `F`, `E`, `R`, `Act`) are
unaffected by this entry.

---

### 15. `Act` is seeded once instead of recomputed every round  (E06)

E06 was `Act_{i,u}`, in the loop. It is now `Act_{1,u}`, run once.

E19 already computes `Act_{i+1,u}` from the excess the push produced, with the
same formula. So from round 2 onward, E06 recomputed a value E19 had written
the round before — verified **identical on all 11 comparable rounds** of the
CLRS instance. It was also a single-assignment conflict: E19's `Act_{i+1}` and
the next round's E06 `Act_{i+1}` are the same slice, with neither being a
seed.

Pinned to generation 1, E06 seeds the loop: round 1 reads it, later rounds
read the previous round's E19. `E07` and the stopping condition are unaffected
— they read `Act` at whatever generation is current.

Effect on the driven run: **235 steps to 224**, 233 Einsum executions to 222.
Same 12 generations, same maximum flow of 23, same 18 answer checks.

**The validator finding did not disappear — it changed class.** Before:

    cascade[7] writes Act[i, u] and cascade[20] writes Act[i+1, u]:
      the rank i advances, so round i writes one and round i+1 the other

After:

    cascade[7] writes Act[1, u] and cascade[20] writes Act[i+1, u]:
      the advancing rank passes through the constant generation

That is the same pattern as `F`, `E`, `R` and `D`: a seed written at a literal
generation inside the cascade, which the loop later writes at `i+1`. All five
resolve together when initialization moves out of the cascade or writes
generation 0, which is the remaining item. `Act` is no longer a separate
problem.

Contrast E17/E09 (entry 14): there the two writes compute genuinely different
quantities, so neither is redundant and both stay in the loop.
