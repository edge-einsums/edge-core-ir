# Widest path / maximum-bottleneck path -- surface math

> *Created with the help of Claude Code.*

$$
\triangleright \text{Tensors} \\
G^{R \equiv |V|, C \equiv |V|} \to \text{float}, \text{ empty} = 0 \\
F^{I, R \equiv |V|} \to \text{float}, \text{ empty} = 0 \\
D^{I, R \equiv |V|} \to \text{float}, \text{ empty} = 0 \\
\\
\triangleright \text{Initialization (source = vertex } 0\text{)} \\
G \to \langle \text{user-specified edge capacities} \rangle \\
F_0 = D_0 = \langle 999,\ 0,\ 0,\ 0 \rangle \quad (\text{source bottleneck} \approx +\infty) \\
\\
\triangleright \text{Extended Einsum (one ``widen'' round per iteration } i\text{)} \\
F_{i+1, d} = G_{s, d} \cdot F_{i, s} :: \bigwedge \min(\cap)\ \bigvee \max(\cup) \\
D_{i+1, d} = D_{i, d} \cdot F_{i+1, d} :: \bigwedge \max(\cup)\ \bigvee \max(\cup) \\
\diamond : D_{i+1} \equiv D_i
$$

This is the **max-min** (widest-path) mirror of the **min-plus** Bellman-Ford SSSP in
`examples/algorithms/bellman-ford/`. Same iterate-until-stable skeleton; the semiring is
flipped: instead of `min` over `sum`s it takes `max` over `min`s.
