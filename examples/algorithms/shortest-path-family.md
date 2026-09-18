# Shortest-path family: BFS → Bellman-Ford → Dijkstra → A\*

> *Created with the help of Claude Code.*

One EDGE skeleton, four algorithms. Each is a small, **named** diff from the one
before it, and the distance tensor `D` is the constant spine the whole way. The
pick gets smarter at each step: relax from everyone → from the closest → from the
closest by `g + h`.

Artifacts:
- BFS:           `examples/algorithms/bfs/`
- Bellman-Ford:  `examples/algorithms/bellman-ford/`
- Dijkstra:      `examples/algorithms/dijkstra/`
- A\*:           `examples/algorithms/astar/`

## The through-line

- **`D` = distances** (the answer) is the same tensor in BF, Dijkstra, and A\*.
- **`F` = frontier** = the vertices you relax from this round. BFS: the whole
  wavefront. Bellman-Ford: *nobody* — it relaxes from all of `D`, so it has no
  frontier. Dijkstra: the single closest unsettled vertex. A\*: same single
  vertex, chosen by `g + h`.
- **Bellman-Ford → Dijkstra:** relax from `F` instead of all of `D`, add a
  priority-queue pick on top and queue maintenance below. The
  improve / record / update body is unchanged.
- **Dijkstra → A\*:** order the pick by `f = g + h` instead of `g`. Add a static
  heuristic `H`; stop when the goal is settled instead of when the queue empties.
- **The goal dials the heuristic.** One goal → focused A\*. A few goals → A\* aimed
  at the nearest. Every vertex a goal → `H = 0` everywhere → Dijkstra. More goals
  flatten `H`, so A\* slides continuously back into Dijkstra.

## Mechanics (how they're written)

```latex
% requires: amsmath, booktabs
\newcommand{\map}{\bigwedge}\newcommand{\red}{\bigvee}\newcommand{\pop}{\lll}

\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}llll@{}}
\toprule
 & \textbf{Bellman--Ford} & \textbf{Dijkstra} & \textbf{A\textsuperscript{*}} \\
\midrule
heuristic & --- & --- & $H_{v}=\sum_{xy}\lvert L_{v,xy}-Lg_{xy}\rvert$\; (once) \\
\addlinespace
pick & \emph{--- (uses all of $D$)} & $F=\operatorname*{argmin}_{s}(Q\!\cdot\!D)$ & $F=\operatorname*{argmin}_{s}(Q\!\cdot\!D+H)$ \\
\addlinespace
relax & \multicolumn{3}{l}{$N_{i,d}=G_{s,d}\,X_{i,s}\ ::\ \map{+}(\cap)\;\red\min(\cup)$\qquad with $X{=}\mathbf{D}$ (BF),\; $X{=}\mathbf{F}$ (Dijkstra, A\textsuperscript{*})} \\
\addlinespace
improve & \multicolumn{3}{l}{$C_{i,d}=N_{i,d}\,D_{i,d}\ ::\ \map{<}(\cup)$} \\
record  & \multicolumn{3}{l}{$\mathsf{NewlyRelaxed}_{i,d}=C_{i,d}\,N_{i,d}\ ::\ \map{\rightarrow}(\cap)$} \\
update  & \multicolumn{3}{l}{$D_{i+1,d}=D_{i,d}\,\mathsf{NewlyRelaxed}_{i,d}\ ::\ \map{\ll}(\cup)$} \\
\addlinespace
dequeue & --- & \multicolumn{2}{l}{$T_{i,d}=Q_{i,d}\,F_{i,d}\ ::\ \map{\leftarrow}(\text{take\_left\_only})$} \\
enqueue & --- & \multicolumn{2}{l}{$Q_{i+1,d}=T_{i,d}\,C_{i,d}\ ::\ \map\mathrm{OR}(\cup)$} \\
\addlinespace
stop & $D_{i+1}\equiv D_{i}$ & $\lVert Q_{i+1}\rVert\equiv 0$ & $\lVert F_{i}\!\cdot\!\text{goal}\rVert\equiv 1$ \\
\bottomrule
\end{tabular}
\caption{The Bellman--Ford / Dijkstra / A\textsuperscript{*} family as one EDGE skeleton. \emph{improve}, \emph{record}, and \emph{update} are identical across all three (spanned rows); the queue maintenance (\emph{dequeue}, \emph{enqueue}) is shared by Dijkstra and A\textsuperscript{*}. They differ only in: what they relax from ($X{=}D$, everyone, vs.\ $X{=}F$, the picked frontier); how the frontier is picked (by $g$ vs.\ $g{+}h$); and when they stop. Setting $H\equiv 0$ turns A\textsuperscript{*}'s pick into Dijkstra's.}
\label{tab:sssp-family}
\end{table}
```

## Behaviour (how they act)

```latex
% requires: amssymb (\checkmark), booktabs
\begin{table}[t]
\centering\footnotesize
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}lccc@{}}
\toprule
 & \textbf{Bellman--Ford} & \textbf{Dijkstra} & \textbf{A\textsuperscript{*}} \\
\midrule
handles negative weights      & \checkmark & --- & --- \\
re-relaxes settled vertices   & \checkmark\ (every round) & --- (settle-once) & --- (settle-once) \\
goal-directed                 & --- & --- & \checkmark\ (via $H$) \\
relaxes from                  & all vertices & closest unsettled & closest by $g{+}h$ \\
priority key                  & --- & $g$ & $g+h$ \\
stops when                    & $D$ stable & queue empty & goal settled \\
needs admissible heuristic    & --- & --- & \checkmark \\
\bottomrule
\end{tabular}
\caption{Behavioural comparison. The three share the EDGE skeleton of Table~\ref{tab:sssp-family} but differ in what they assume and how they explore.}
\label{tab:sssp-properties}
\end{table}
```

## Status

The mechanics and surface forms are hand-traced and (for Dijkstra) cross-checked
against the paper's Cascade 12. None of Dijkstra / A\* has an IR builder or
`*_program.json` yet — the populate `select-min` and the `<<` update use
user-defined ops, and the `(x, y)` / Manhattan reduction is a new tensor shape the
builders don't emit yet (see each `metadata.md`).
