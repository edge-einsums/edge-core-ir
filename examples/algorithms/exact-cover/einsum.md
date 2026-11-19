# Exact Cover (Knuth's Algorithm X) Einsum

Source: `_algorithms/set_cover.md` in `edge-algorithms`, branch
`add-set-cover-algorithm`, with the corrections listed below. The ASCII form
that is checked against the IR is `edge-workbench/examples/set-cover/einsum.edge`.

Semantic assumption used throughout: **a stored value equal to a tensor's
`empty` value is treated as absent.**

Changes relative to the tutorial:

1. **E04 split into E04 + E05.** The tutorial's
   `TPath = Path · d* :: ⋀ <(∩)` turns path depths into `True`; run as
   written, the cascade reports non-covers as solutions. `Keep` selects the
   entries shallower than `d*`, and `TPath` takes `Path`'s depths at those
   entries. Every later Einsum is renumbered (20 in all).
2. **Every tensor is declared** (EDGE paper §7.1): `Keep`, plus `F`, `d*`,
   `TPath` and `NewPathEntry`, which the tutorial used without declaring.
3. **`Col`, `Row` and `InitS` are user-specified inputs** (§7.2). `Col` and
   `Row` are read by E10 and E12 and written by nothing. `InitS` is the stack
   seed the tutorial states only in prose.
4. **E17's `select-min-val` breaks ties by first arrival**, as `select-any`
   does: any column with the fewest active rows is a correct branch. Under
   sequential order this still picks the lowest column, so the tutorial's
   trace is unchanged.

Not changed: every other Einsum is exactly as the tutorial writes it, and the
IR lowers each one to ONE Einsum -- no helper tensors. Where the written form
is shorthand, the IR carries it as the paper's parenthesized, labelled form,
and the walkthrough shows those labels step by step:

- E02 `⌊F/|R|⌋`: the floor is a unary on the quotient, desugared as the paper
  defines a unary (§5.5): `((F ·¹ |R|) ·² 1) :: ⋀¹ /(∩) ⋀² ⌊⌋(←) ⋁²_r max(∪)`.
- E12: `(Row ·¹ ¬ConflictRow) ·² ¬SelRow :: ⋀¹ ←(∩) ⋀² ←(∩)`.
- E13 `‖ActiveCol‖ ≡ 0`: one Einsum, using the occupancy expression added to
  the IR for this (`edge_ir.ir.expr.Occupancy`, a local change).
- E18: `(A ·¹ PickedCol) ·² ActiveRow :: ⋀¹ ←(∩) ⋁¹_c ANY(∪) ⋀² ←(∩)`.
- E19 `σ(d*+1, r)`: `Candidates ·⁴ (((d* ·¹ 1) ·² |R|) ·³ r) :: ⋀¹ +(∩) ⋀² *(∩) ⋀³ +(∩) ⋀⁴ →(∩)`.

```tex
\begin{align}
▹ \text{Tensors} \\
A^{R, C} &\to \text{Boolean},\ \text{empty} = \text{false} \\
S^{I, R} &\to \text{Integer},\ \text{empty} = -1 \\
Path^{I, R} &\to \text{Integer},\ \text{empty} = -1 \\
SelRow^{I, R} &\to \text{Boolean},\ \text{empty} = \text{false} \\
CovCol^{I, C} &\to \text{Boolean},\ \text{empty} = \text{false} \\
ActiveCol^{I, C} &\to \text{Boolean},\ \text{empty} = \text{false} \\
ConflictRow^{I, R} &\to \text{Boolean},\ \text{empty} = \text{false} \\
ActiveRow^{I, R} &\to \text{Boolean},\ \text{empty} = \text{false} \\
Success^{I} &\to \text{Boolean},\ \text{empty} = \text{false} \\
Sol^{I, R} &\to \text{Boolean},\ \text{empty} = \text{false} \\
RowCounts^{I, C} &\to \text{Integer},\ \text{empty} = 0 \\
ActiveCounts^{I, C} &\to \text{Integer},\ \text{empty} = 0 \\
PickedCol^{I, C} &\to \text{Boolean},\ \text{empty} = \text{false} \\
Candidates^{I, R} &\to \text{Boolean},\ \text{empty} = \text{false} \\
NewChoices^{I, R} &\to \text{Integer},\ \text{empty} = -1 \\
T^{I, R} &\to \text{Integer},\ \text{empty} = -1 \\
Keep^{I, R} &\to \text{Boolean},\ \text{empty} = \text{false} \\
\\
▹ \text{Not declared in the tutorial} \\
F^{I, R},\ TPath^{I, R},\ NewPathEntry^{I, R} &\to \text{Integer},\ \text{empty} = -1 \\
d^\ast{}^{I} &\to \text{Integer},\ \text{empty} = 0 \\
\\
▹ \text{User-specified inputs} \\
Col^{C},\ Row^{R} &\to \text{Boolean},\ \text{empty} = \text{false}
&& \text{...true at every column / row: the rank universes} \\
InitS^{R} &\to \text{Integer},\ \text{empty} = -1
&& \text{...the stack seed} \\
\\
▹ \text{Stamp function} \\
\sigma(\text{depth}, r) &= \text{depth} \cdot |R| + r \\
\\
▹ \text{Initializations} \\
S_{0,r} &= InitS_r
&& \text{...rows covering the column with fewest rows, stamped } \sigma(1, r) \\
\\
▹ \text{Extended Einsums} \\
▹ \textbf{peek and pop} \\
F_{i,r^\ast} &= S_{i,r} :: \lll_{r^\ast} \mathbf{1}(\text{select-max-val})
&& \text{E01 ...the top of the stack: the largest stamp} \\
d^\ast_i &= \lfloor F_{i,r} / |R| \rfloor :: \bigvee \max(\cup)
&& \text{E02 ...its depth} \\
T_{i,r} &= S_{i,r} \cdot \neg F_{i,r} :: \bigwedge \leftarrow(\cap)
&& \text{E03 ...pop it} \\
▹ \textbf{update path} \\
Keep_{i,r} &= Path_{i,r} \cdot d^\ast_i :: \bigwedge <(\cap)
&& \text{E04 ...path entries shallower than the popped depth} \\
TPath_{i,r} &= Path_{i,r} \cdot Keep_{i,r} :: \bigwedge \leftarrow(\cap)
&& \text{E05 ...keep those entries, with their depths} \\
NewPathEntry_{i,r} &= F_{i,r} \cdot d^\ast_i :: \bigwedge \rightarrow(\cap)
&& \text{E06 ...the popped row at its depth} \\
Path_{i+1,r} &= TPath_{i,r} \cdot NewPathEntry_{i,r} :: \bigwedge <\!\!<(\cup)
&& \text{E07 ...the new path} \\
▹ \textbf{active subset computation} \\
SelRow_{i+1,r} &= Path_{i+1,r} \cdot 0 :: \bigwedge \ge(\cup)
&& \text{E08 ...rows on the path} \\
CovCol_{i+1,c} &= A_{r,c} \cdot SelRow_{i+1,r} :: \bigwedge \leftarrow(\cap)\ \bigvee ANY(\cup)
&& \text{E09 ...columns they cover} \\
ActiveCol_{i+1,c} &= Col_c \cdot \neg CovCol_{i+1,c} :: \bigwedge \leftarrow(\cap)
&& \text{E10 ...columns still to cover} \\
ConflictRow_{i+1,r} &= A_{r,c} \cdot CovCol_{i+1,c} :: \bigwedge \leftarrow(\cap)\ \bigvee ANY(\cup)
&& \text{E11 ...rows that would cover a covered column again} \\
ActiveRow_{i+1,r} &= Row_r \cdot \neg ConflictRow_{i+1,r} \cdot \neg SelRow_{i+1,r} :: \bigwedge \leftarrow(\cap)
&& \text{E12 ...rows still usable} \\
▹ \textbf{success evaluation} \\
Success_{i+1} &= \lVert ActiveCol_{i+1} \rVert \equiv 0
&& \text{E13 ...every column covered} \\
Sol_{i+1,r} &= SelRow_{i+1,r} \cdot Success_{i+1} :: \bigwedge \leftarrow(\cap)
&& \text{E14 ...record the solution} \\
▹ \textbf{branch and push} \\
RowCounts_{i+1,c} &= A_{r,c} \cdot ActiveRow_{i+1,r} :: \bigwedge \leftarrow(\cap)\ \bigvee +(\cup)
&& \text{E15 ...active rows per column} \\
ActiveCounts_{i+1,c} &= RowCounts_{i+1,c} \cdot ActiveCol_{i+1,c} :: \bigwedge \leftarrow(\cap)
&& \text{E16 ...restricted to active columns} \\
PickedCol_{i+1,c^\ast} &= ActiveCounts_{i+1,c} :: \lll_{c^\ast} \mathbf{1}(\text{select-min-val})
&& \text{E17 ...fewest active rows; ties to the first arrival} \\
Candidates_{i+1,r} &= (A_{r,c} \cdot PickedCol_{i+1,c} :: \bigwedge \leftarrow(\cap)\ \bigvee ANY(\cup)) \cdot ActiveRow_{i+1,r} :: \bigwedge \leftarrow(\cap)
&& \text{E18 ...active rows covering it} \\
NewChoices_{i+1,r} &= Candidates_{i+1,r} \cdot \sigma(d^\ast_i + 1, r) :: \bigwedge \rightarrow(\cap)
&& \text{E19 ...stamp them one level deeper} \\
S_{i+1,r} &= T_{i,r} \cdot NewChoices_{i+1,r} :: \bigwedge <\!\!<(\cup)
&& \text{E20 ...push} \\
\\
◇ &: \lVert S_{i+1} \rVert \equiv 0
&& \text{...stop when the stack is empty}
\end{align}
```
