$$
\triangleright \text{Tensors} \\
G^{S \equiv |V|, D \equiv |V|} \to \text{integer}, \text{ empty} = 0 \\
C^{I, S \equiv |V|} \to \text{Boolean}, \text{ empty} = \text{False} \\
N^{I, D \equiv |V|} \to \text{integer}, \text{ empty} = \infty \\
\mathsf{NewlyRelaxed}^{I, D \equiv |V|} \to \text{integer}, \text{ empty} = \infty \\
D^{I, S \equiv |V|} \to \text{integer}, \text{ empty} = \infty \\
\\
\triangleright \text{Initialization (source = vertex } 0\text{)} \\
G \to \langle \text{user-specified} \rangle \\
\mathit{root\_id} \to \langle \text{user-specified} \rangle, \text{a coordinate set over the rank } S \\
D_{0, s : s \in root\_id} = 0 \\
\\
\triangleright \text{Extended Einsum (one full relaxation round per iteration } i\text{)} \\
\text{(1) RELAX:   } N_{i, d} = G_{s, d} \cdot D_{i, s} :: \bigwedge_s +(\cap)\ \bigvee_s \min(\cup) \\
\text{(2) IMPROVE: } C_{i, d} = N_{i, d} \cdot D_{i, d} :: \bigwedge_d <(\cup) \\
\text{(3) RECORD:  } \mathsf{NewlyRelaxed}_{i, d} = C_{i, d} \cdot N_{i, d} :: \bigwedge_d \rightarrow(\cap) \\
\text{(4) UPDATE:  } D_{i+1, d} = D_{i, d} \cdot \mathsf{NewlyRelaxed}_{i, d} :: \bigwedge_d \texttt{<<}(\cup) \\
\diamond : D_{i+1} \equiv D_i
$$
