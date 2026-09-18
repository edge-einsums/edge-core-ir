# Dijkstra (single-source shortest paths)

> *Created with the help of Claude Code.*

Built as a minimal diff from Bellman-Ford: same relax / improve / record / update
body, plus a priority-queue PICK on top and queue maintenance below. See
`examples/algorithms/shortest-path-family.md` for the BFS -> BF -> Dijkstra -> A\*
progression and the side-by-side comparison.

$$
\begin{align}
\newcommand{\map}{\bigwedge}\newcommand{\red}{\bigvee}\newcommand{\pop}{\lll}
G^{S=|V|,\, D=|V|} &\;\to\; \text{integer},\ \text{empty}=\infty \\
N^{I,\, D=|V|}      &\;\to\; \text{integer},\ \text{empty}=\infty \\
C^{I,\, D=|V|}      &\;\to\; \text{boolean},\ \text{empty}=\text{False} \\
\mathsf{NewlyRelaxed}^{I,\, D=|V|} &\;\to\; \text{integer},\ \text{empty}=\infty \\
D^{I,\, S=|V|}      &\;\to\; \text{integer},\ \text{empty}=\infty \\
Q^{I,\, S=|V|}      &\;\to\; \text{boolean},\ \text{empty}=\text{False} \\
DQ^{I,\, S=|V|}     &\;\to\; \text{integer},\ \text{empty}=\infty \\
F^{I,\, S=|V|}      &\;\to\; \text{integer},\ \text{empty}=\infty \\
T^{I,\, D=|V|}      &\;\to\; \text{boolean},\ \text{empty}=\text{False} \\
\\
&\triangleright\ \text{init: source at distance } 0 \text{ and in the queue}\\
D_{0,\,s\,:\,s\in\text{root}} &= 0\\
Q_{0,\,s\,:\,s\in\text{root}} &= \text{True}\\
&\triangleright\ \text{pick: distances of queued vertices, then the closest}\\
DQ_{i,s} &= Q_{i,s}\cdot D_{i,s} &&\;::\; \map \rightarrow(\cap)\\
F_{i,\,s^*} &= DQ_{i,s} &&\;::\; \pop_{s^*}\, \mathbb{1}(\text{select-min-s})\\
&\triangleright\ \text{relax from } F\\
N_{i,d} &= G_{s,d}\cdot F_{i,s} &&\;::\; \map +(\cap)\ \red \min(\cup)\\
&\triangleright\ \text{improve: which } d \text{ got strictly shorter?}\\
C_{i,d} &= N_{i,d}\cdot D_{i,d} &&\;::\; \map <(\cup)\\
&\triangleright\ \text{record the improved candidates}\\
\mathsf{NewlyRelaxed}_{i,d} &= C_{i,d}\cdot N_{i,d} &&\;::\; \map \rightarrow(\cap)\\
&\triangleright\ \text{update distances}\\
D_{i+1,d} &= D_{i,d}\cdot \mathsf{NewlyRelaxed}_{i,d} &&\;::\; \map \ll(\cup)\\
&\triangleright\ \text{dequeue the settled vertex}\\
T_{i,d} &= Q_{i,d}\cdot F_{i,d} &&\;::\; \map \leftarrow(\text{take\_left\_only})\\
&\triangleright\ \text{enqueue whoever improved}\\
Q_{i+1,d} &= T_{i,d}\cdot C_{i,d} &&\;::\; \map \text{OR}(\cup)\\
&\triangleright\ \text{stop when the queue is empty}\\
&\diamond_i:\ \lVert Q_{i+1}\rVert \equiv 0
\end{align}
$$
