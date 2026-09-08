"""The 2D HP lattice model: energy, move operators, and the SA/REMC kernels.

A conformation is an ``(n, 2)`` int32 array of lattice coordinates, paired with
an occupancy grid ``G`` of shape ``(2n+5, 2n+5)`` offset by ``off = n+2`` so that
any reachable coordinate indexes into it. Keeping the grid in sync with the
coordinates is what makes the pivot move cheap; every operator below either
maintains it or leaves the occupied set untouched.

Energy is ``E = -(number of non-sequential H-H lattice contacts)``, so lower is
better and the ground state is the maximum-contact fold.

Six operators share the move budget:

===============  ======  =========================================
Operator         Budget  Cost
===============  ======  =========================================
Corner flip       12%    O(n) via incremental delta-E
End flip           8%    O(n) via incremental delta-E
Simple pull        8%    O(n) via incremental delta-E
Crankshaft        15%    O(n^2) full recompute, grid unchanged
Pivot rotation    35%    O(s) collision + O(n^2) recompute
Bond rebridging   22%    O(n^2) full recompute, grid unchanged
===============  ======  =========================================

Pivot, crankshaft and rebridging together take 72% of the budget. That skew
toward large-scale structural change is what lets the search escape the deep
local minima that trap local-move-only solvers past ~40 residues; the ablation
in the paper measures the effect.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - exercised only without numba
    raise SystemExit(
        "numba is required to run the solver.\n"
        "Install the pinned dependencies with:  pip install -r requirements.txt"
    ) from exc

#: Unit steps on the square lattice, indexed 0-3.
DX = np.array([1, -1, 0, 0], dtype=np.int32)
DY = np.array([0, 0, 1, -1], dtype=np.int32)


# --------------------------------------------------------------------------- #
# Energy and geometry
# --------------------------------------------------------------------------- #

@njit(cache=False)
def energy_full(coords, seq, n):
    """Total energy, i.e. minus the number of non-sequential H-H contacts."""
    e = 0
    for i in range(n):
        if seq[i] != 1:
            continue
        xi = coords[i, 0]
        yi = coords[i, 1]
        for j in range(i + 2, n):
            if seq[j] != 1:
                continue
            dx = coords[j, 0] - xi
            dy = coords[j, 1] - yi
            if dx * dx + dy * dy == 1:
                e -= 1
    return e


@njit(cache=False)
def delta_e_single(coords, seq, n, idx, nx, ny):
    """Energy change from moving residue ``idx`` to ``(nx, ny)``.

    Only valid for single-residue moves, where the change is local: contacts
    are lost at the old site and gained at the new one.
    """
    if seq[idx] != 1:
        return 0
    ox = coords[idx, 0]
    oy = coords[idx, 1]
    de = 0
    for d in range(4):
        ax = ox + DX[d]
        ay = oy + DY[d]
        bx = nx + DX[d]
        by = ny + DY[d]
        for k in range(n):
            if k == idx or abs(k - idx) == 1 or seq[k] != 1:
                continue
            if coords[k, 0] == ax and coords[k, 1] == ay:
                de += 1
            if coords[k, 0] == bx and coords[k, 1] == by:
                de -= 1
    return de


@njit(cache=False)
def clear_grid(grid, gs):
    for a in range(gs):
        for b in range(gs):
            grid[a, b] = 0


@njit(cache=False)
def rebuild_grid_inplace(coords, grid, n, off, gs):
    """Rewrite ``grid`` from ``coords``. False if out of bounds or overlapping."""
    clear_grid(grid, gs)
    for i in range(n):
        gx = coords[i, 0] + off
        gy = coords[i, 1] + off
        if gx < 0 or gy < 0 or gx >= gs or gy >= gs:
            return False
        if grid[gx, gy] != 0:
            return False
        grid[gx, gy] = 1
    return True


@njit(cache=False)
def copy_coords(dst, src, n):
    for i in range(n):
        dst[i, 0] = src[i, 0]
        dst[i, 1] = src[i, 1]


@njit(cache=False)
def valid_walk(coords, n):
    """True when ``coords`` is a self-avoiding walk with unit-length bonds."""
    for i in range(n):
        for j in range(i + 1, n):
            if coords[i, 0] == coords[j, 0] and coords[i, 1] == coords[j, 1]:
                return False
    for i in range(n - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        if dx * dx + dy * dy != 1:
            return False
    return True


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #

@njit(cache=False)
def init_snake(n):
    """Compact boustrophedon fold. Always succeeds, so it is the fallback."""
    gs = 2 * n + 5
    off = n + 2
    coords = np.zeros((n, 2), dtype=np.int32)
    grid = np.zeros((gs, gs), dtype=np.int8)
    base = int(math.sqrt(n))
    if base < 2:
        base = 2
    width = base + np.random.randint(0, max(2, base))
    if width < 2:
        width = 2
    for i in range(n):
        row = i // width
        col = i - row * width
        if row % 2 == 0:
            x = col
        else:
            x = width - 1 - col
        y = row
        coords[i, 0] = x
        coords[i, 1] = y
        grid[x + off, y + off] = 1
    return coords, grid, off, gs, True


@njit(cache=False)
def init_saw(n):
    """Random self-avoiding walk, falling back to a snake when growth traps."""
    gs = 2 * n + 5
    off = n + 2
    tries = 600
    for _ in range(tries):
        coords = np.zeros((n, 2), dtype=np.int32)
        grid = np.zeros((gs, gs), dtype=np.int8)
        coords[0, 0] = 0
        coords[0, 1] = 0
        grid[off, off] = 1
        ok = True
        for i in range(1, n):
            x = coords[i - 1, 0]
            y = coords[i - 1, 1]
            start = np.random.randint(4)
            stride = 1
            if np.random.random() < 0.5:
                stride = 3
            placed = False
            for dd in range(4):
                d = (start + stride * dd) % 4
                nx = x + DX[d]
                ny = y + DY[d]
                gx = nx + off
                gy = ny + off
                if gx >= 0 and gy >= 0 and gx < gs and gy < gs and grid[gx, gy] == 0:
                    coords[i, 0] = nx
                    coords[i, 1] = ny
                    grid[gx, gy] = 1
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return coords, grid, off, gs, True
    return init_snake(n)


@njit(cache=False)
def apply_single(coords, grid, idx, nx, ny, off):
    ox = coords[idx, 0]
    oy = coords[idx, 1]
    grid[ox + off, oy + off] = 0
    grid[nx + off, ny + off] = 1
    coords[idx, 0] = nx
    coords[idx, 1] = ny


@njit(cache=False)
def accept_move(de, t):
    """Metropolis criterion: P = min(1, exp(-dE / T))."""
    if de <= 0:
        return True
    if t <= 1e-12:
        return False
    return np.random.random() < math.exp(-de / t)


# --------------------------------------------------------------------------- #
# Move operators
# --------------------------------------------------------------------------- #

@njit(cache=False)
def propose_corner(coords, grid, n, off, gs):
    """Corner flip: reflect an L-shaped residue across the diagonal."""
    i = np.random.randint(1, n - 1)
    xp = coords[i - 1, 0]
    yp = coords[i - 1, 1]
    xc = coords[i, 0]
    yc = coords[i, 1]
    xn = coords[i + 1, 0]
    yn = coords[i + 1, 1]
    if xp == xn or yp == yn:
        return -1, 0, 0
    nx = xp + xn - xc
    ny = yp + yn - yc
    gx = nx + off
    gy = ny + off
    if gx < 0 or gy < 0 or gx >= gs or gy >= gs:
        return -1, 0, 0
    if grid[gx, gy] != 0:
        return -1, 0, 0
    return i, nx, ny


@njit(cache=False)
def propose_end(coords, grid, n, off, gs):
    """End flip: swing a terminal residue to a free neighbour of its anchor."""
    end = np.random.randint(2)
    idx = 0
    anchor = 1
    if end == 1:
        idx = n - 1
        anchor = n - 2
    ax = coords[anchor, 0]
    ay = coords[anchor, 1]
    ox = coords[idx, 0]
    oy = coords[idx, 1]
    start = np.random.randint(4)
    stride = 1
    if np.random.random() < 0.5:
        stride = 3
    for dd in range(4):
        d = (start + stride * dd) % 4
        nx = ax + DX[d]
        ny = ay + DY[d]
        if nx == ox and ny == oy:
            continue
        gx = nx + off
        gy = ny + off
        if gx < 0 or gy < 0 or gx >= gs or gy >= gs:
            continue
        if grid[gx, gy] != 0:
            continue
        return idx, nx, ny
    return -1, 0, 0


@njit(cache=False)
def propose_pull(coords, grid, n, off, gs):
    """Simple pull: move residue i to a free site adjacent to both neighbours.

    Searched from both chain sides. Always validity-preserving, which is the
    point: an earlier "cascade pull" variant demanded a site adjacent to two
    *bonded* residues at once. Two adjacent lattice sites have disjoint
    4-neighbourhoods, so that precondition can never hold on the square lattice
    and the move silently never fired. See Section 5 of the paper.
    """
    i = np.random.randint(1, n - 1)
    start = np.random.randint(4)
    stride = 1
    if np.random.random() < 0.5:
        stride = 3
    for side in range(2):
        anchor = i + 1
        other = i - 1
        if side == 1:
            anchor = i - 1
            other = i + 1
        for dd in range(4):
            d = (start + stride * dd) % 4
            cx = coords[anchor, 0] + DX[d]
            cy = coords[anchor, 1] + DY[d]
            if cx == coords[i, 0] and cy == coords[i, 1]:
                continue
            ddx = cx - coords[other, 0]
            ddy = cy - coords[other, 1]
            if ddx * ddx + ddy * ddy != 1:
                continue
            gx = cx + off
            gy = cy + off
            if gx < 0 or gy < 0 or gx >= gs or gy >= gs:
                continue
            if grid[gx, gy] != 0:
                continue
            return i, cx, cy
    return -1, 0, 0


@njit(cache=False)
def do_crankshaft(coords, grid, seq, n, off, gs):
    """Crankshaft: swap a bonded pair to the complementary diagonal corners.

    The occupied set is unchanged, so the grid needs no update.
    """
    if n < 4:
        return False, 0
    i = np.random.randint(1, n - 2)
    px = coords[i - 1, 0]
    py = coords[i - 1, 1]
    qx = coords[i + 2, 0]
    qy = coords[i + 2, 1]
    if abs(px - qx) != 1 or abs(py - qy) != 1:
        return False, 0
    ax = px
    ay = qy
    bx = qx
    by = py
    cix = coords[i, 0]
    ciy = coords[i, 1]
    cjx = coords[i + 1, 0]
    cjy = coords[i + 1, 1]
    nx1 = bx
    ny1 = by
    nx2 = ax
    ny2 = ay
    if cix == bx and ciy == by and cjx == ax and cjy == ay:
        nx1 = ax
        ny1 = ay
        nx2 = bx
        ny2 = by
    elif not (cix == ax and ciy == ay and cjx == bx and cjy == by):
        return False, 0
    e_before = energy_full(coords, seq, n)
    coords[i, 0] = nx1
    coords[i, 1] = ny1
    coords[i + 1, 0] = nx2
    coords[i + 1, 1] = ny2
    return True, energy_full(coords, seq, n) - e_before


@njit(cache=False)
def do_pivot(coords, grid, seq, n, off, gs):
    """Rotate a chain segment by 90/180/270 degrees about a pivot residue.

    Collision detection reads the occupancy grid, which is O(s) in the segment
    length, rather than comparing every rotated position against every
    non-rotated residue at O(n*s). A rotation is a rigid isometry, so it
    preserves all pairwise distances: a self-avoiding segment stays
    self-avoiding, and only collisions against the *rest* of the chain need
    checking. Total pivot cost is then dominated by the O(n^2) energy
    recompute, which is why the measured end-to-end gain is 1.4-1.5x rather
    than the ~n-fold speedup of the collision check alone.
    """
    pivot = np.random.randint(0, n)
    rot = np.random.randint(3)        # 0 = 90 CW, 1 = 180, 2 = 270 CW
    direction = np.random.randint(2)  # 0 = rotate the tail, 1 = rotate the head
    px = coords[pivot, 0]
    py = coords[pivot, 1]

    seg_s = pivot + 1 if direction == 0 else 0
    seg_e = n if direction == 0 else pivot
    seg_len = seg_e - seg_s
    if seg_len <= 0:
        return False, 0

    new_pos = np.empty((seg_len, 2), dtype=np.int32)
    for j in range(seg_len):
        idx = seg_s + j
        dx = coords[idx, 0] - px
        dy = coords[idx, 1] - py
        if rot == 0:
            new_pos[j, 0] = px + dy
            new_pos[j, 1] = py - dx
        elif rot == 1:
            new_pos[j, 0] = px - dx
            new_pos[j, 1] = py - dy
        else:
            new_pos[j, 0] = px - dy
            new_pos[j, 1] = py + dx

    for j in range(seg_len):
        gx = new_pos[j, 0] + off
        gy = new_pos[j, 1] + off
        if gx < 0 or gy < 0 or gx >= gs or gy >= gs:
            return False, 0

    # Lift the old segment out of the grid so it cannot collide with itself.
    for j in range(seg_s, seg_e):
        grid[coords[j, 0] + off, coords[j, 1] + off] = 0

    placed = 0
    valid = True
    for j in range(seg_len):
        gx = new_pos[j, 0] + off
        gy = new_pos[j, 1] + off
        if grid[gx, gy] != 0:
            valid = False
            break
        grid[gx, gy] = 1
        placed += 1

    if not valid:
        for j in range(placed):
            grid[new_pos[j, 0] + off, new_pos[j, 1] + off] = 0
        for j in range(seg_s, seg_e):
            grid[coords[j, 0] + off, coords[j, 1] + off] = 1
        return False, 0

    e_before = energy_full(coords, seq, n)
    for j in range(seg_len):
        idx = seg_s + j
        coords[idx, 0] = new_pos[j, 0]
        coords[idx, 1] = new_pos[j, 1]
    e_after = energy_full(coords, seq, n)
    return True, e_after - e_before


@njit(cache=False)
def do_rebridging(coords, grid, seq, n, off, gs):
    """Bond rebridging after Wuest and Landau (2012).

    Find a lattice-adjacent, non-bonded pair (i, j) with |i-j| > 2; if
    reversing the internal segment [i+1, j-1] leaves both splice bonds intact,
    apply the reversal. The occupied set is unchanged.
    """
    i0 = np.random.randint(0, n)
    start = np.random.randint(4)
    stride = 1
    if np.random.random() < 0.5:
        stride = 3
    for dd in range(4):
        d = (start + stride * dd) % 4
        jx = coords[i0, 0] + DX[d]
        jy = coords[i0, 1] + DY[d]
        j0 = -1
        for k in range(n):
            if coords[k, 0] == jx and coords[k, 1] == jy:
                j0 = k
                break
        if j0 < 0 or abs(j0 - i0) <= 2:
            continue
        i = i0
        j = j0
        if i > j:
            i = j0
            j = i0
        if i + 1 >= n or j - 1 < 0:
            continue
        dx1 = coords[j - 1, 0] - coords[i, 0]
        dy1 = coords[j - 1, 1] - coords[i, 1]
        if dx1 * dx1 + dy1 * dy1 != 1:
            continue
        dx2 = coords[j, 0] - coords[i + 1, 0]
        dy2 = coords[j, 1] - coords[i + 1, 1]
        if dx2 * dx2 + dy2 * dy2 != 1:
            continue
        if j - i - 1 < 1:
            continue
        e_before = energy_full(coords, seq, n)
        lo = i + 1
        hi = j - 1
        while lo < hi:
            tx = coords[lo, 0]
            ty = coords[lo, 1]
            coords[lo, 0] = coords[hi, 0]
            coords[lo, 1] = coords[hi, 1]
            coords[hi, 0] = tx
            coords[hi, 1] = ty
            lo += 1
            hi -= 1
        return True, energy_full(coords, seq, n) - e_before
    return False, 0


@njit(cache=False)
def sa_step(coords, grid, seq, n, off, gs, t):
    """One Metropolis move attempt, drawn from the tuned operator mix.

    Cumulative thresholds: corner 0.12, end 0.20, pull 0.28, crankshaft 0.43,
    pivot 0.78, rebridging 1.00.
    """
    r = np.random.random()

    if r < 0.12:
        idx, nx, ny = propose_corner(coords, grid, n, off, gs)
        if idx < 0:
            return 0
        de = delta_e_single(coords, seq, n, idx, nx, ny)
        if accept_move(de, t):
            apply_single(coords, grid, idx, nx, ny, off)
            return de
        return 0

    if r < 0.20:
        idx, nx, ny = propose_end(coords, grid, n, off, gs)
        if idx < 0:
            return 0
        de = delta_e_single(coords, seq, n, idx, nx, ny)
        if accept_move(de, t):
            apply_single(coords, grid, idx, nx, ny, off)
            return de
        return 0

    if r < 0.28:
        idx, nx, ny = propose_pull(coords, grid, n, off, gs)
        if idx < 0:
            return 0
        de = delta_e_single(coords, seq, n, idx, nx, ny)
        if accept_move(de, t):
            apply_single(coords, grid, idx, nx, ny, off)
            return de
        return 0

    # Multi-residue moves: snapshot first, since rejection means a full restore.
    save = coords.copy()

    if r < 0.43:
        ok, de = do_crankshaft(coords, grid, seq, n, off, gs)
    elif r < 0.78:
        ok, de = do_pivot(coords, grid, seq, n, off, gs)
    else:
        ok, de = do_rebridging(coords, grid, seq, n, off, gs)

    if not ok:
        return 0
    if accept_move(de, t):
        return de
    copy_coords(coords, save, n)
    rebuild_grid_inplace(coords, grid, n, off, gs)
    return 0


# --------------------------------------------------------------------------- #
# Search drivers
# --------------------------------------------------------------------------- #

@njit(cache=False)
def run_sa(seq, n, nsteps, t_start, t_end, seed, start_coords, has_start):
    """Simulated annealing, cooling geometrically from ``t_start`` to ``t_end``.

    Energy is tracked incrementally but recomputed from scratch every
    ``nsteps/40`` steps, because accumulated delta-E drifts. The same checkpoint
    revalidates the walk and rewinds to the best fold if it ever breaks.
    """
    np.random.seed(seed)
    gs = 2 * n + 5
    off = n + 2
    if has_start:
        coords = np.zeros((n, 2), dtype=np.int32)
        grid = np.zeros((gs, gs), dtype=np.int8)
        for i in range(n):
            coords[i, 0] = start_coords[i, 0]
            coords[i, 1] = start_coords[i, 1]
        ok = rebuild_grid_inplace(coords, grid, n, off, gs)
        if not ok:
            coords, grid, off, gs, ok = init_saw(n)
    else:
        coords, grid, off, gs, ok = init_saw(n)
    if not ok:
        return 0, np.zeros((n, 2), dtype=np.int32)

    e = energy_full(coords, seq, n)
    best_e = e
    best_c = coords.copy()
    recompute_every = max(128, nsteps // 40)

    for step in range(nsteps):
        frac = step / max(nsteps - 1, 1)
        t = t_start * ((t_end / t_start) ** frac)
        de = sa_step(coords, grid, seq, n, off, gs, t)
        e += de
        if step % recompute_every == 0:
            e = energy_full(coords, seq, n)
            if not valid_walk(coords, n):
                copy_coords(coords, best_c, n)
                rebuild_grid_inplace(coords, grid, n, off, gs)
                e = energy_full(coords, seq, n)
        if e < best_e:
            best_e = e
            copy_coords(best_c, coords, n)

    best_e = energy_full(best_c, seq, n)
    return best_e, best_c


@njit(cache=False)
def run_remc(seq, n, nrep, steps_per_exchange, nexchanges, seed,
             t_min, t_max, start_coords, has_start):
    """Replica-exchange Monte Carlo over a geometric temperature ladder.

    Replica 0 (coldest) is seeded with the global best fold when one is
    supplied. After each block of ``steps_per_exchange`` moves per replica,
    neighbours attempt a swap with P = min(1, exp[(bi-bj)(Ej-Ei)]), alternating
    even and odd pairings between blocks.
    """
    np.random.seed(seed)
    gs = 2 * n + 5
    off = n + 2
    temps = np.zeros(nrep, dtype=np.float64)
    for r in range(nrep):
        temps[r] = t_min * ((t_max / t_min) ** (r / max(nrep - 1, 1)))

    all_c = np.zeros((nrep, n, 2), dtype=np.int32)
    all_g = np.zeros((nrep, gs, gs), dtype=np.int8)
    all_e = np.zeros(nrep, dtype=np.int32)

    for r in range(nrep):
        if r == 0 and has_start:
            for i in range(n):
                all_c[r, i, 0] = start_coords[i, 0]
                all_c[r, i, 1] = start_coords[i, 1]
            ok0 = rebuild_grid_inplace(all_c[r], all_g[r], n, off, gs)
            if not ok0:
                c, g, _, _, _ = init_saw(n)
                for i in range(n):
                    all_c[r, i, 0] = c[i, 0]
                    all_c[r, i, 1] = c[i, 1]
                for a in range(gs):
                    for b in range(gs):
                        all_g[r, a, b] = g[a, b]
        else:
            c, g, _, _, _ = init_saw(n)
            for i in range(n):
                all_c[r, i, 0] = c[i, 0]
                all_c[r, i, 1] = c[i, 1]
            for a in range(gs):
                for b in range(gs):
                    all_g[r, a, b] = g[a, b]
        all_e[r] = energy_full(all_c[r], seq, n)

    best_e = 999999
    best_c = np.zeros((n, 2), dtype=np.int32)
    for r in range(nrep):
        if all_e[r] < best_e:
            best_e = all_e[r]
            copy_coords(best_c, all_c[r], n)

    for ex in range(nexchanges):
        for r in range(nrep):
            t = temps[r]
            cr = all_c[r]
            gr = all_g[r]
            e = all_e[r]
            for _ in range(steps_per_exchange):
                e += sa_step(cr, gr, seq, n, off, gs, t)
            e = energy_full(cr, seq, n)
            if not valid_walk(cr, n):
                copy_coords(cr, best_c, n)
                rebuild_grid_inplace(cr, gr, n, off, gs)
                e = energy_full(cr, seq, n)
            all_e[r] = e
            if e < best_e:
                best_e = e
                copy_coords(best_c, cr, n)

        start = ex % 2
        r = start
        while r + 1 < nrep:
            ei = all_e[r]
            ej = all_e[r + 1]
            bi = 1.0 / max(temps[r], 1e-12)
            bj = 1.0 / max(temps[r + 1], 1e-12)
            log_accept = (bi - bj) * (ei - ej)
            if log_accept >= 0.0 or np.random.random() < math.exp(log_accept):
                for i in range(n):
                    tx = all_c[r, i, 0]
                    ty = all_c[r, i, 1]
                    all_c[r, i, 0] = all_c[r + 1, i, 0]
                    all_c[r, i, 1] = all_c[r + 1, i, 1]
                    all_c[r + 1, i, 0] = tx
                    all_c[r + 1, i, 1] = ty
                for a in range(gs):
                    for b in range(gs):
                        tg = all_g[r, a, b]
                        all_g[r, a, b] = all_g[r + 1, a, b]
                        all_g[r + 1, a, b] = tg
                te = all_e[r]
                all_e[r] = all_e[r + 1]
                all_e[r + 1] = te
            r += 2

    best_e = energy_full(best_c, seq, n)
    return best_e, best_c


# --------------------------------------------------------------------------- #
# Pure-Python scoring, used for independent validation of solver output
# --------------------------------------------------------------------------- #

def contact_count(seq_str: str, coords) -> int:
    """Count non-sequential H-H contacts. Deliberately not JIT-compiled: this
    is the independent scorer used to check what the compiled kernels claim."""
    coords = [tuple(map(int, xy)) for xy in coords]
    n = len(seq_str)
    c = 0
    for i in range(n):
        if seq_str[i] != "H":
            continue
        xi, yi = coords[i]
        for j in range(i + 2, n):
            if seq_str[j] != "H":
                continue
            xj, yj = coords[j]
            if abs(xj - xi) + abs(yj - yi) == 1:
                c += 1
    return c


def radius_gyration2(coords) -> float:
    """Squared radius of gyration, used as a compactness tiebreaker."""
    arr = np.asarray(coords, dtype=np.float64)
    cen = arr.mean(axis=0)
    return float(((arr - cen) ** 2).sum(axis=1).mean())


def validate_coords(seq_str: str, coords, expected_contacts=None, verbose=True):
    """Check length, self-avoidance and bond lengths; return ``(ok, info)``."""
    if coords is None:
        return False, {"error": "coords is None"}
    n = len(seq_str)
    coords = [tuple(map(int, xy)) for xy in coords]
    if len(coords) != n:
        return False, {"error": f"length mismatch: coords={len(coords)} sequence={n}"}
    seen = set()
    for i, xy in enumerate(coords):
        if xy in seen:
            return False, {"error": f"self-overlap at residue {i}: {xy}"}
        seen.add(xy)
        if i > 0:
            px, py = coords[i - 1]
            x, y = xy
            if abs(x - px) + abs(y - py) != 1:
                return False, {"error": f"broken chain bond at residues {i-1}-{i}"}
    contacts = contact_count(seq_str, coords)
    rg2 = radius_gyration2(coords)
    info = {"contacts": contacts, "radius_gyration2": rg2}
    if expected_contacts is not None and contacts != int(expected_contacts):
        info["warning"] = (
            f"expected_contacts={expected_contacts}, actual_contacts={contacts}"
        )
    if verbose:
        print(f"[validate] valid fold: contacts={contacts}, rg2={rg2:.4f}")
    return True, info
