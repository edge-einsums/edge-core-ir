$$
\triangleright \text{Tensors} \\
G^{S \equiv |V|, D \equiv |V|} \to \text{Boolean}, \text{ empty} = \text{False} \\
S^{I, V \equiv |V|} \to \text{integer}, \text{ empty} = -1 \\
P^{I, V \equiv |V|} \to \text{Boolean}, \text{ empty} = \text{False} \\
F^{I, V \equiv |V|} \to \text{integer}, \text{ empty} = -1 \\
N^{I, D \equiv |V|} \to \text{Boolean}, \text{ empty} = \text{False} \\
U^{I, D \equiv |V|} \to \text{Boolean}, \text{ empty} = \text{False} \\
T^{I, V \equiv |V|} \to \text{integer}, \text{ empty} = -1 \\
S'^{I, V \equiv |V|} \to \text{integer}, \text{ empty} = -1 \\
\\
\triangleright \text{Stamp (order-stamped value encoding)} \\
\sigma(i, v) = i \cdot |V| + v \quad\text{(stateless lexicographic, a function of the iteration point)} \\
\\
\triangleright \text{Initialization (root = vertex } 0\text{)} \\
G \to \langle \text{user-specified} \rangle \\
\mathit{root\_id} \to \langle \text{user-specified} \rangle, \text{a coordinate set over the rank } V \\
S_{0, v : v \in root\_id} = \sigma(0, v) \\
P_{0, v : v \in root\_id} = \text{True} \\
\\
\triangleright \text{Extended Einsum (one pop+expand per iteration } i\text{)} \\
\text{(1) PEEK:  } F_{i, v*} = S_{i, v} :: \lll_{v*} \mathbf{1}(\text{select-max-val}) \\
\text{(2) ADV:   } N_{i, d} = G_{s, d} \cdot F_{i, s} :: \bigwedge \leftarrow(\cap)\ \bigvee \text{ANY}(\cup) \\
\text{(3) MASK:  } U_{i, d} = N_{i, d} \cdot \neg P_{i, d} :: \bigwedge \leftarrow(\cap) \\
\text{(4a) POP:  } T_{i, v} = S_{i, v} \cdot \neg F_{i, v} :: \bigwedge \leftarrow(\cap) \\
\text{(4b) STAMP: } S'_{i, v} = U_{i, v} \cdot \sigma(i+1, v) :: \bigwedge \rightarrow(\cap) \\
\text{(4c) PUSH: } S_{i+1, v} = T_{i, v} \cdot S'_{i, v} :: \bigwedge \texttt{<<}(\cup) \\
\text{(5) VISIT: } P_{i+1, v} = P_{i, v} \cdot S'_{i, v} :: \bigwedge \text{OR}(\cup) \\
\diamond : \| S_{i+1} \| \equiv 0
$$
