"""Fibertree executor for the EDGE widest-path (maximum-bottleneck) algorithm.

ENVIRONMENT
-----------
This script needs the ``fibertree`` library, which is not a dependency
of this package. Install it separately.

Run (cwd = repo root):

    python examples/algorithms/widest-path/widest_path_fibertree.py

WHAT THIS IS
------------
A standalone fibertree emulation of the EDGE program in this folder's
``einsum.edge``. It is the (max, min) mirror of (min, +) Bellman-Ford on a
capacity-weighted directed graph. It walks the iteration space of the two
einsums explicitly (Fiber co-iteration / intersection for the relax map, an
explicit max-reduction over the contraction rank) so the einsum structure is
visible, rather than collapsing into a one-line numpy call.

EDGE program emulated (verbatim from einsum.edge)::

    F[i+1, d] = G[s,d] . F[i,s]   :: map(min, intersect) reduce(max, union);
    D[i+1, d] = D[i,d] . F[i+1,d] :: map(max, union) reduce(max, union);
    stop: stable(D);

  relax  : F[i+1, d] = max_s  min( G[s,d], F[i,s] )
           map = min over the INTERSECT of supports (both G[s,d] and F[i,s]
           must be present, i.e. non-empty / non-zero); reduce = max over the
           contraction rank s with UNION support.
  update : D[i+1, d] = max( D[i,d], F[i+1,d] ) elementwise over UNION support.
           This einsum has no contraction rank, so reduce(max, union) is a
           no-op identity; the work is the elementwise map(max, union).
  stop   : iterate until D is stable (D[i+1] == D[i]).

Acceptance test: final D MUST equal the ground truth [999, 3, 5, 3].
The script prints a per-generation trace of F and D, then asserts the final D
and prints PASS. It exits 0 on success.
"""

import sys

from fibertree import Fiber, Payload, Tensor

# --------------------------------------------------------------------------
# Problem definition (from einsum.edge / metadata.md)
# --------------------------------------------------------------------------
N = 4

# G[R=s, C=d] : capacity adjacency matrix. 0 == "no edge" (treated as empty).
G_INIT = [
    [0.0, 3.0, 5.0, 0.0],
    [0.0, 0.0, 0.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
]

# Source seed: vertex 0 has an unbounded incoming bottleneck (999 ~ +inf).
F0_INIT = [999.0, 0.0, 0.0, 0.0]
D0_INIT = [999.0, 0.0, 0.0, 0.0]

GROUND_TRUTH_D = [999.0, 3.0, 5.0, 3.0]

MAX_GENS = 16  # sane iteration cap; this graph converges in <= 3 effective rounds


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fiber_to_dense(fiber, n=N):
    """Render a rank-1 Fiber (over coords 0..n-1) as a dense Python list.

    Empty (absent) coordinates read back as 0.0, matching ``empty = 0.0``.
    """
    dense = [0.0] * n
    for c, p in fiber:
        dense[c] = float(Payload.get(p))
    return dense


def fmt(vec):
    """Compact, readable dense-vector formatting for the trace."""
    parts = []
    for v in vec:
        if float(v).is_integer():
            parts.append(f"{int(v):>3d}")
        else:
            parts.append(f"{v:>3g}")
    return "[" + ", ".join(parts) + "]"


# --------------------------------------------------------------------------
# RELAX einsum:  F_next[d] = max_s min( G[s,d], F_cur[s] )
#
# Iteration-space walk (the contraction rank is s):
#   for each source s present in F_cur (the wavefront support):
#       intersect F_cur's payload at s  with  G's row s (the out-edges of s)
#       -> this is map(min, intersect): both G[s,d] and F[i,s] must exist.
#       for each surviving dest d:  contribute min(G[s,d], F[i,s])
#   reduce(max, union): accumulate those contributions into F_next[d] by max.
# --------------------------------------------------------------------------
def relax(g_tensor, f_cur, verbose=True):
    g_root = g_tensor.getRoot()  # rank-2 fiber: s -> (d -> capacity)
    f_next = Fiber(default=0.0)  # output wavefront over dest d

    # Walk the contraction rank s as the *intersection* of F_cur's support
    # with G's source support: only sources that both (a) have a known
    # bottleneck and (b) have outgoing edges can relax anything.
    for s, (f_s, g_row) in f_cur & g_root:
        f_s_v = float(Payload.get(f_s))
        # f_s = F[i,s] (the bottleneck reaching s); g_row = fiber d -> G[s,d].
        for d, cap in g_row:
            cap_v = float(Payload.get(cap))
            # map(min, intersect): edge exists (g_row only holds non-empty
            # caps) AND f_s is present (guaranteed by the s-intersection).
            contrib = min(cap_v, f_s_v)
            if verbose:
                print(
                    f"      relax  s={s} d={d}: "
                    f"min(G[{s},{d}]={cap_v:g}, F[{s}]={f_s_v:g}) "
                    f"= {contrib:g}"
                )
            # reduce(max, union): scatter-max into the output coordinate d.
            existing = float(
                Payload.get(f_next.getPayload(d, default=0.0, allocate=False))
            )
            if contrib > existing:
                ref = f_next.getPayloadRef(d)
                ref <<= contrib

    return f_next


# --------------------------------------------------------------------------
# UPDATE einsum:  D_next[d] = max( D_cur[d], F_next[d] )  over UNION support.
# No contraction rank -> reduce(max, union) is identity; the work is the
# elementwise map(max, union). We co-iterate the UNION of the two supports.
# --------------------------------------------------------------------------
def update(d_cur, f_next, verbose=True):
    d_next = Fiber(default=0.0)

    for d, (mask, dv, fv) in d_cur | f_next:
        # union yields a presence mask plus each operand's payload (0.0 if
        # absent, matching empty = 0.0). map(max, union): take whichever
        # exists, max if both do.
        dv_v = float(Payload.get(dv))
        fv_v = float(Payload.get(fv))
        best = max(dv_v, fv_v)
        if verbose:
            print(
                f"      update d={d}: "
                f"max(D[{d}]={dv_v:g}, F[{d}]={fv_v:g}) = {best:g} "
                f"[{mask}]"
            )
        if best != 0.0:
            ref = d_next.getPayloadRef(d)
            ref <<= best

    return d_next


# --------------------------------------------------------------------------
# Driver: iterate generations until stable(D).
# --------------------------------------------------------------------------
def run(verbose=True):
    G = Tensor.fromUncompressed(rank_ids=["R", "C"], root=G_INIT, name="G")
    F = Tensor.fromUncompressed(rank_ids=["R"], root=F0_INIT, name="F")
    D = Tensor.fromUncompressed(rank_ids=["R"], root=D0_INIT, name="D")

    f_cur = F.getRoot()
    d_cur = D.getRoot()

    print("=" * 64)
    print("EDGE widest-path (maximum-bottleneck) -- fibertree executor")
    print("=" * 64)
    print(
        f"gen 0:  F0 = {fmt(fiber_to_dense(f_cur))}   D0 = {fmt(fiber_to_dense(d_cur))}"
    )

    final_dense = fiber_to_dense(d_cur)

    for i in range(MAX_GENS):
        if verbose:
            print(f"\n  -- generation {i} -> {i + 1} --")
            print("    RELAX  F[i+1,d] = max_s min(G[s,d], F[i,s]):")
        f_next = relax(G, f_cur, verbose=verbose)

        if verbose:
            print("    UPDATE D[i+1,d] = max(D[i,d], F[i+1,d]):")
        d_next = update(d_cur, f_next, verbose=verbose)

        f_dense = fiber_to_dense(f_next)
        d_prev_dense = fiber_to_dense(d_cur)
        d_dense = fiber_to_dense(d_next)

        stable = d_dense == d_prev_dense
        tag = "  -> stable, STOP" if stable else ""
        print(
            f"gen {i + 1}:  F{i + 1} = {fmt(f_dense)}   D{i + 1} = {fmt(d_dense)}{tag}"
        )

        f_cur = f_next
        d_cur = d_next
        final_dense = d_dense

        # stop: stable(D)  -- D reached its fixpoint.
        if stable:
            break
    else:
        raise RuntimeError(f"widest-path did not stabilize within MAX_GENS={MAX_GENS}")

    return final_dense


# --------------------------------------------------------------------------
# OPTIONAL animation (nice-to-have; never allowed to fail the executor).
#
# Replays the SAME iteration-space walk on persistent G/F/D tensors, adding
# one canvas frame per iteration-space point so the per-coordinate map/reduce
# activity is visible:
#   * relax point (s, d): highlight read G[s,d] + read F[s], then the write
#     F[d] (reduce(max) landing on the output coordinate).
#   * update point d: highlight read D[d] + read F[d], then the write D[d].
# Rendered to widest_path.mp4 in this folder via the cv2-backed MovieCanvas
# (no ffmpeg dependency). Wrapped in try/except: any failure prints a note and
# returns without touching the verified result above.
# --------------------------------------------------------------------------
def render_animation(out_path):
    from fibertree import TensorCanvas  # local import: only needed here

    G = Tensor.fromUncompressed(rank_ids=["R", "C"], root=G_INIT, name="G")
    F = Tensor.fromUncompressed(rank_ids=["R"], root=F0_INIT, name="F")
    D = Tensor.fromUncompressed(rank_ids=["R"], root=D0_INIT, name="D")

    canvas = TensorCanvas(G, F, D)
    canvas.addActivity((), (), (), worker="seed")  # gen 0 initial state

    g_root = G.getRoot()
    f_root = F.getRoot()
    d_root = D.getRoot()

    for _ in range(MAX_GENS):
        d_before = fiber_to_dense(d_root)

        # ---- RELAX: build the next wavefront into a fresh fiber, animating
        #      each (s, d) read and the scatter-max write into F[d]. ----
        f_next_vals = [0.0] * N
        for s, (f_s, g_row) in f_root & g_root:
            f_s_v = float(Payload.get(f_s))
            for d, cap in g_row:
                contrib = min(float(Payload.get(cap)), f_s_v)
                # read G[s,d] and F[s] (map(min, intersect))
                canvas.addActivity((s, d), (s,), (), worker="relax")
                if contrib > f_next_vals[d]:
                    f_next_vals[d] = contrib

        # Commit F = f_next in place, animating each populated/updated coord
        # (reduce(max, union) landing on the output coordinate d).
        for d in range(N):
            ref = f_root.getPayloadRef(d)
            ref <<= f_next_vals[d]
        for d in range(N):
            if f_next_vals[d] != 0.0:
                canvas.addActivity((), (d,), (), worker="F[d] write")

        # ---- UPDATE: D[d] = max(D[d], F[d]) elementwise over the union. ----
        for d in range(N):
            new_d = max(
                float(Payload.get(d_root.getPayload(d, default=0.0, allocate=False))),
                f_next_vals[d],
            )
            ref = d_root.getPayloadRef(d)
            ref <<= new_d
            if new_d != 0.0:
                canvas.addActivity((), (d,), (d,), worker="D[d] update")

        if fiber_to_dense(d_root) == d_before:
            break

    canvas.addActivity((), (), (), worker="stable/STOP")
    canvas.saveMovie(out_path)
    return out_path


def main():
    final_d = run(verbose=True)

    print("\n" + "=" * 64)
    print(f"final D = {fmt(final_d)}")
    print(f"expected = {fmt(GROUND_TRUTH_D)}")

    assert final_d == GROUND_TRUTH_D, (
        f"widest-path mismatch: got {final_d}, expected {GROUND_TRUTH_D}"
    )
    print("PASS: final D matches ground truth [999, 3, 5, 3]")

    # Optional animation -- never allowed to affect the verified result.
    import os

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "widest_path.mp4"
    )
    try:
        render_animation(out_path)
        size = os.path.getsize(out_path)
        print(f"animation: wrote {out_path} ({size} bytes)")
    except Exception as exc:  # noqa: BLE001 - animation is strictly optional
        print(f"animation: skipped (render failed: {exc!r})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
