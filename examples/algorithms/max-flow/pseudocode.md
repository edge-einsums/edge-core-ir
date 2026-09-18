# Parallel Push-Relabel Pseudocode

This is the algorithmic reference for `einsum.md`. It is written as a compact
bulk-synchronous push-relabel algorithm using the same tensor names as the EDGE
expression where possible.

## Tensors

Inputs:

- `G[u,v]`: graph adjacency / edge structure.
- `C[u,v]`: capacity of edge `(u, v)`.
- `s`: source vertex.
- `t`: sink vertex.

Output:

- `F[i,u,v]`: flow after the final generation `i`.

State tensors:

- `F[i,u,v]`: flow at generation `i`.
- `R[i,u,v]`: residual capacity at generation `i`.
- `E[i,u]`: excess at vertex `u` at generation `i`.
- `D[i,u]`: height/label of vertex `u` at generation `i`.
- `S[u]`: source mask, true iff `u = s`.
- `T[u]`: sink mask, true iff `u = t`.

Intermediate tensors:

- `NST[u]`: true iff `u` is neither source nor sink.
- `Act[i,u]`: true iff `u` is an active internal vertex.
- `ActR[i,u,v]`: active vertex `u` with positive residual capacity to `v`.
- `Lbl[i,u,v]`: true iff `D[i,u] = D[i,v] + 1`.
- `Adm[i,u,v]`: admissible residual edge mask.
- `PushCand[i,u,v]`: selected push edge mask.
- `Delta[i,u,v]`: amount pushed from `u` to `v`.
- `InPush[i,u]`: total flow pushed into `u` in generation `i`.
- `OutPush[i,u]`: total flow pushed out of `u` in generation `i`.
- `HasAdm[i,u]`: true iff `u` has an admissible outgoing residual edge.
- `Rel[i,u]`: true iff `u` should be relabeled.
- `MinNeiLbl[i,u]`: minimum label of a positive-residual neighbor.
- `NewD[i,u]`: newly computed height for relabeled vertices.

```text
Procedure ParallelPushRelabel(G, C, s, t):

    <<initialize>>
    parallel for each edge (u, v) in G do
        F[0,u,v] <- 0
        R[0,u,v] <- C[u,v]
    end

    parallel for each vertex u in G do
        D[0,u] <- 0
        E[0,u] <- 0
        S[u] <- (u = s)
        T[u] <- (u = t)
        NST[u] <- not S[u] and not T[u]
    end

    D[0,s] <- |V|

    <<preflow from source>>
    parallel for all u, v in G do
        F[1,u,v] <- F[0,u,v]
        R[1,u,v] <- R[0,u,v]
    end

    parallel for all u in G do
        E[1,u] <- E[0,u]
    end

    parallel for each neighbor v of s do
        F[1,s,v] <- C[s,v]
        F[1,v,s] <- -C[s,v]

        R[1,s,v] <- 0
        R[1,v,s] <- C[s,v]

        E[1,v] <- E[0,v] + C[s,v]
    end

    E[1,s] <- E[0,s] - sum over neighbors w of s of C[s,w]

    i <- 1

    while there exists an active vertex u with NST[u] and E[i,u] > 0 do

        <<find active vertices>>
        parallel for each vertex u do
            Act[i,u] <- NST[u] and E[i,u] > 0
        end

        <<find admissible edges from active vertices>>
        parallel for each edge (u, v) do
            ActR[i,u,v] <- Act[i,u] and R[i,u,v] > 0
            Lbl[i,u,v] <- D[i,u] = D[i,v] + 1
            Adm[i,u,v] <- ActR[i,u,v] and Lbl[i,u,v]
        end

        <<push phase>>
        parallel for each active vertex u do
            PushCand[i,u,w] <- false for all w
            if there exists v with Adm[i,u,v] then
                choose one such v using pick_admissible_edge
                PushCand[i,u,v] <- true
            end
        end

        parallel for each edge (u, v) do
            if PushCand[i,u,v] then
                Delta[i,u,v] <- min(E[i,u], R[i,u,v])
        end

        <<accumulated flow, residual, and excess updates>>
        parallel for each edge (u, v) do
            F[i+1,u,v] <- F[i,u,v] + Delta[i,u,v] - Delta[i,v,u]
            R[i+1,u,v] <- R[i,u,v] - Delta[i,u,v] + Delta[i,v,u]
        end

        parallel for each vertex u do
            InPush[i,u] <- sum over v of Delta[i,v,u]
            OutPush[i,u] <- sum over v of Delta[i,u,v]
            E[i+1,u] <- E[i,u] + InPush[i,u] - OutPush[i,u]
        end

        <<relabel phase, using updated residual and excess>>
        parallel for each vertex u do
            Act[i+1,u] <- NST[u] and E[i+1,u] > 0
            HasAdm[i+1,u] <- exists v such that R[i+1,u,v] > 0
                              and D[i,u] = D[i,v] + 1
            Rel[i,u] <- Act[i+1,u] and not HasAdm[i+1,u]
        end

        parallel for each vertex u do
            if Rel[i,u] then
                MinNeiLbl[i,u] <- min { D[i,v] : R[i+1,u,v] > 0 }
                NewD[i,u] <- MinNeiLbl[i,u] + 1
                D[i+1,u] <- NewD[i,u]
            else
                D[i+1,u] <- D[i,u]
            end
        end

        i <- i + 1
    end

    return F[i]
end
```
