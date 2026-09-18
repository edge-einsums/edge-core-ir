$$
\triangleright \text{Tensors} \\
G^{S \equiv |V|, D \equiv |V|} \to \text{integer}, \text{ empty} = 0 \\
F^{I, S \equiv |V|} \to \text{integer}, \text{ empty} = \infty \\
T^{I, D \equiv |V|} \to \text{integer}, \text{ empty} = \infty \\
P^{I, D \equiv |V|} \to \text{Boolean}, \text{ empty} = \text{False} \\
\\
\triangleright \text{Initialization} \\
G \to \langle \text{user-specified} \rangle \\
\mathit{id} \to \langle \text{user-specified} \rangle, \text{a coordinate set of the same type as the rank } S \text{ in } F \\
F_{0, s : s \in id} = 0 \\
P_{0, d : d \in id} = \text{True} \\
\\
\triangleright \text{Extended Einsum} \\
T_{i, d} = G_{s, d} \cdot F_{i, s} :: \bigwedge_s +(\cap)\ \bigvee_s \text{ANY}(\cup) \\
F_{i+1, d} = T_{i, d} \cdot \neg P_{i, d} :: \bigwedge_d \leftarrow(\cap) \\
P_{i+1, d} = P_{i, d} \cdot F_{i+1, d} :: \bigwedge_d \text{OR}(\cup) \\
\diamond : \| F_{i+1} \| \equiv 0
$$
