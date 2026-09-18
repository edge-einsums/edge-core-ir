> *Toluwanimi Odemuyiwa, combined with the help of Claude Code.*

$$
\triangleright \text{Tensors} \\
G^{R \equiv N, C \equiv N} \to \text{float}, \text{ empty} = 0.0 \\
L^{I, R \equiv N} \to \text{float}, \text{ empty} = 0.0 \\
\\
\triangleright \text{Initialization} \\
G \to \langle \text{user-specified} \rangle, \text{ symmetric reflexive adjacency} \\
L_0 \to \langle \text{user-specified} \rangle, \text{ unique per-vertex labels } (\text{label}(v) = v + 1) \\
\\
\triangleright \text{Extended Einsum} \\
L_{i+1, d} = G_{s, d} \cdot L_{i, s} :: \bigwedge_s \times(\cap)\ \bigvee_s \max(\cup) \\
\diamond : L_{i+1} \equiv L_i
$$
