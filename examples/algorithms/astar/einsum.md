# A\* search (single-source, single-goal shortest path)

> *Created with the help of Claude Code.*

Dijkstra with the pick ordered by `f = g + h` instead of `g`, a precomputed
Manhattan heuristic `H`, and a goal-settled stop. Setting `H` to 0 everywhere
collapses the pick back to Dijkstra's. The distance tensor `D` always holds true
distances (`g`); the heuristic only steers the pick. See
`examples/algorithms/shortest-path-family.md` for the full progression and comparison.

$$
\begin{align}
\newcommand{\map}{\bigwedge}\newcommand{\red}{\bigvee}\newcommand{\pop}{\lll}
&\triangleright\ \textbf{inputs}\\
G^{S=|V|,\, D=|V|} &\;\to\; \text{integer},\ \text{empty}=\infty \\
L^{S=|V|,\, XY=2}   &\;\to\; \text{integer},\ \text{empty}=\infty && \triangleright\ L_{v,x},\,L_{v,y}:\ \text{each vertex's } (x,y)\\
\text{goal}^{S=|V|} &\;\to\; \text{boolean},\ \text{empty}=\text{False} && \triangleright\ \text{one-hot target vertex}\\
&\triangleright\ \textbf{heuristic (computed once, no } I \text{ rank)}\\
Lg^{XY=2}           &\;\to\; \text{integer},\ \text{empty}=\infty \\
AD^{S=|V|,\, XY=2}  &\;\to\; \text{integer},\ \text{empty}=\infty \\
H^{S=|V|}          &\;\to\; \text{integer},\ \text{empty}=\infty \\
&\triangleright\ \textbf{search state (per round } i\text{)}\\
D^{I,\, S=|V|}      &\;\to\; \text{integer},\ \text{empty}=\infty \\
N^{I,\, D=|V|}      &\;\to\; \text{integer},\ \text{empty}=\infty \\
C^{I,\, D=|V|}      &\;\to\; \text{boolean},\ \text{empty}=\text{False} \\
\mathsf{NewlyRelaxed}^{I,\, D=|V|} &\;\to\; \text{integer},\ \text{empty}=\infty \\
Q^{I,\, S=|V|}      &\;\to\; \text{boolean},\ \text{empty}=\text{False} \\
DQ^{I,\, S=|V|}     &\;\to\; \text{integer},\ \text{empty}=\infty \\
PQ^{I,\, S=|V|}     &\;\to\; \text{integer},\ \text{empty}=\infty \\
M^{I,\, S=|V|}      &\;\to\; \text{integer},\ \text{empty}=\infty \\
F^{I,\, S=|V|}      &\;\to\; \text{integer},\ \text{empty}=\infty \\
T^{I,\, D=|V|}      &\;\to\; \text{boolean},\ \text{empty}=\text{False} \\
\\
&\triangleright\ \text{heuristic setup: every node's Manhattan distance to the goal (once)}\\
&\triangleright\ \text{grab the goal's own } (x,y)\\
Lg_{xy}    &= L_{s,xy}\cdot \text{goal}_{s}  &&\;::\; \map \leftarrow(\cap)\ \red \text{ANY}(\cup)\\
&\triangleright\ \text{per-axis absolute difference } \lvert L_{v}-Lg\rvert\\
AD_{v,xy} &= \text{abs}\!\left( L_{v,xy}\cdot Lg_{xy}\right) &&\;::\; \map -(\cap)\\
&\triangleright\ \text{sum over the axes } = \text{ Manhattan}\\
H_{v}      &= AD_{v, xy}                     &&\;::\; \red +(\cup)\\
\\
&\triangleright\ \text{init: source at distance } 0 \text{ and in the queue}\\
D_{0,\,s\,:\,s\in\text{root}} &= 0\\
Q_{0,\,s\,:\,s\in\text{root}} &= \text{True}\\
&\triangleright\ \text{distances of queued vertices } (= g)\\
DQ_{i,s} &= Q_{i,s}\cdot D_{i,s} &&\;::\; \map \rightarrow(\cap)\\
&\triangleright\ \text{priority } f = g + h\\
PQ_{i,s} &= DQ_{i,s}\cdot H_{s} &&\;::\; \map +(\cap)\\
&\triangleright\ \text{pick the min-} f \text{ vertex}\\
M_{i,\,s^*} &= PQ_{i,s} &&\;::\; \pop_{s^*}\, \mathbb{1}(\text{select-min-s})\\
&\triangleright\ \text{frontier carries the TRUE distance } g \text{, not } f\\
F_{i,s} &= M_{i,s}\cdot D_{i,s} &&\;::\; \map \rightarrow(\cap)\\
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
&\triangleright\ \text{stop when the goal is settled (} F \text{ and goal are both one-hot)}\\
&\diamond_i:\ \lVert F_{i,s}\cdot \text{goal}_{s}\rVert \equiv 1
\end{align}
$$
