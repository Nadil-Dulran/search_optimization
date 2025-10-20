# student_sa.py
from __future__ import annotations
from typing import List, Tuple, Set, Optional, Callable
import math, random, collections

"""
===========================================================
Simulated Annealing — Overall Pseudocode (Path Improvement)
===========================================================
Goal: improve a feasible S→G path on a grid by local mutations.

1) Build an initial feasible path P0 (e.g., BFS on the grid).
2) Set current := P0, best := P0, T := T0.
3) Repeat for k = 1..iters:
     a) Candidate := mutate(current)      # either "exploit" (shortcut) or "explore" (detour)
     b) Δ := cost(Candidate) - cost(Current)
     c) If Δ < 0, accept Candidate.
        Else accept with probability exp( -Δ / T ).      ← KEY CONCEPT
     d) If accepted, current := Candidate.
     e) If current better than best, best := current.
     f) Record best cost in history.
     g) Cool the temperature: T := alpha * T.            ← KEY CONCEPT
     h) (Optional) If stuck for long, perform a small restart around best.
4) Return (best, history of best_costs).
Notes:
- Mutations should always yield valid S→G paths (no obstacles).
- Objective is provided by the runner (default: length + 0.2*turns).
"""

# ----------------------------
# Types
# ----------------------------
Coord = Tuple[int, int]

# ----------------------------
# Small Utilities (implemented)
# ----------------------------
def _bfs_path(start: Coord, goal: Coord, neighbors_fn: Callable[[Coord], List[Coord]]) -> List[Coord]:
    """Feasible S→G path on unweighted grid."""
    if start == goal:
        return [start]
    q = collections.deque([start])
    parent = {start: None}
    while q:
        u = q.popleft()
        for v in neighbors_fn(u):
            if v not in parent:
                parent[v] = u
                if v == goal:
                    path = [v]
                    while path[-1] is not None:
                        p = parent[path[-1]]
                        if p is None: break
                        path.append(p)
                    path.reverse()
                    return path
                q.append(v)
    return []  # no path found

def _turns_in_path(path: List[Coord]) -> int:
    if len(path) < 3:
        return 0
    def step(a: Coord, b: Coord) -> Coord:
        return (b[0]-a[0], b[1]-a[1])
    t = 0
    for i in range(2, len(path)):
        if step(path[i-2], path[i-1]) != step(path[i-1], path[i]):
            t += 1
    return t

def _cost_default(path: List[Coord]) -> float:
    if not path:
        return float("inf")
    return float(len(path) + 0.2 * _turns_in_path(path))

def _splice(base: List[Coord], i: int, j: int, mid: List[Coord]) -> List[Coord]:
    """Return base[:i+1] + mid[1:-1] + base[j:] (keeps endpoints base[i], base[j])."""
    if not base or i < 0 or j >= len(base) or i >= j:
        return base[:]
    out = base[:i+1]
    core = mid[:]
    if core and core[0] == base[i]:
        core = core[1:]
    if core and core[-1] == base[j]:
        core = core[:-1]
    out.extend(core)
    out.extend(base[j:])
    return out

def _random_walk_connect(a: Coord, b: Coord, neighbors_fn: Callable[[Coord], List[Coord]],
                         rng: random.Random, budget: int = 24) -> List[Coord]:
    """Biased random walk that tends to move closer to b; returns a..b path or [] if failed."""
    def manhattan(u: Coord, v: Coord) -> int:
        return abs(u[0]-v[0]) + abs(u[1]-v[1])
    cur = a
    path = [cur]
    seen = {cur}
    for _ in range(budget):
        nbrs = neighbors_fn(cur)
        if not nbrs:
            break
        nbrs.sort(key=lambda x: (manhattan(x, b), rng.random()))
        chosen = None
        for cand in nbrs[:3]:
            if cand not in seen:
                chosen = cand
                break
        if chosen is None:
            chosen = rng.choice(nbrs)
        cur = chosen
        path.append(cur)
        seen.add(cur)
        if cur == b:
            return path
    return []

# ----------------------------
# Mutation Operators (implemented baseline)
# ----------------------------
def _mutate_shortcut(path: List[Coord],
                     neighbors_fn: Callable[[Coord], List[Coord]],
                     rng: random.Random) -> List[Coord]:
    """Try to replace a short segment i..j by a shorter connector (exploit)."""
    n = len(path)
    if n < 6:
        return path[:]
    i = rng.randrange(1, n-3)
    j = rng.randrange(i+2, min(i+6, n-1))
    a, b = path[i], path[j]
    mid = _random_walk_connect(a, b, neighbors_fn, rng, budget=18)
    if mid and len(mid) < (j - i + 1):
        return _splice(path, i, j, mid)
    return path[:]

def _mutate_detour(path: List[Coord],
                   neighbors_fn: Callable[[Coord], List[Coord]],
                   rng: random.Random) -> List[Coord]:
    """Try a small detour i..j via an alternative connector (explore)."""
    n = len(path)
    if n < 6:
        return path[:]
    i = rng.randrange(1, n-3)
    j = rng.randrange(i+2, min(i+6, n-1))
    a, b = path[i], path[j]
    mid = _random_walk_connect(a, b, neighbors_fn, rng, budget=30)
    if mid:
        return _splice(path, i, j, mid)
    return path[:]

# ----------------------------
# Simulated Annealing (only a few KEY lines to fill)
# ----------------------------
def simulated_annealing(
    neighbors_fn: Callable[[Coord], List[Coord]],
    objective_fn: Callable[[List[Coord]], float],
    obstacles: Set[Coord],
    seed: str,
    iters: int = 1200,
    T0: float = 1.3,
    alpha: float = 0.995
):
    """
    Return (best_path, history). History logs best-so-far cost after each iteration.

    KEY LINES FOR STUDENTS:
      • Mutation policy: choose between shortcut (exploit) and detour (explore).
      • Acceptance probability: exp(-Δ / T) for Δ>0.
      • Cooling schedule: update T each iteration.
      • (Optional) Restart trigger thresholds (when 'no_improve' is large).
    """
    rng = random.Random(str(seed))

    # 1) Initial feasible path: prefer BFS from START->GOAL to show improvement over baseline
    path0: List[Coord] = _bfs_path((0,0), (5,5), neighbors_fn)
    if not path0:
        # fallback: try a few nearby corners/goals to obtain any feasible path
        common_starts = [(0,0), (0,1), (1,0)]
        common_goals  = [(5,5), (5,4), (4,5)]
        for s in common_starts:
            for g in common_goals:
                p = _bfs_path(s, g, neighbors_fn)
                if p:
                    path0 = p
                    break
            if path0:
                break
    if not path0:
        return []  # no feasible start

    # Objective wrapper
    def safe_cost(pth: List[Coord]) -> float:
        try:
            val = objective_fn(pth)
            if val is None or not math.isfinite(val):
                return _cost_default(pth)
            return float(val)
        except Exception:
            return _cost_default(pth)

    current = path0[:]
    cur_cost  = safe_cost(current)

    # Make the starting solution intentionally a bit worse to create room for staged improvements.
    target_worse = cur_cost + 1.0
    worsen_attempts = 0
    while cur_cost < target_worse and worsen_attempts < 40:
        trial = _mutate_detour(current, neighbors_fn, rng)
        tcost = safe_cost(trial)
        # accept only if truly worse to climb up the landscape
        if tcost > cur_cost + 0.05:
            current = trial
            cur_cost = tcost
        worsen_attempts += 1

    best    = current[:]
    best_cost = cur_cost
    history: List[float] = [best_cost]
    T = float(T0)

    no_improve = 0
    for k in range(1, int(iters)+1):

        # --- (1) Mutation policy: choose exploit vs explore -----------------
        # TODO (fill this line): pick one of the two operators; e.g., mostly shortcut, sometimes detour.
        # Example idea (don’t copy literally): if k % 3 == 0 -> detour else shortcut
        # Bias toward shortcutting (exploitation) while occasionally taking detours (exploration).
        # Use temperature-scaled probability to explore more when T is high.
        explore_prob = min(0.5, max(0.15, 0.6 * (T / max(T0, 1e-9))))
        if (k % 7 == 0) or (rng.random() < explore_prob):
            cand = _mutate_detour(current, neighbors_fn, rng)
        else:
            cand = _mutate_shortcut(current, neighbors_fn, rng)

        cand_cost = safe_cost(cand)
        delta = cand_cost - cur_cost

        # --- (2) Acceptance probability for uphill moves --------------------
        accept = False
        if delta < 0:
            accept = True
        else:
            # TODO (fill this line): classic Metropolis acceptance using T
            # prob = exp( ? )  and compare with rng.random()
            prob = math.exp(-delta / max(T, 1e-12))
            if rng.random() < prob:
                accept = True

        if accept:
            current = cand
            cur_cost = cand_cost

        # Track global best & stagnation
        if cur_cost < best_cost:
            best = current[:]
            best_cost = cur_cost
            no_improve = 0
        else:
            no_improve += 1

        history.append(best_cost)

        # --- (3) Cooling schedule ------------------------------------------
        # TODO (fill this line): update temperature each iteration with alpha
        # e.g., T = alpha * T
        T = max(1e-9, alpha * T)

        # --- (4) Optional: simple restart if stuck --------------------------
        # You may tune these thresholds (bonus, not required).
        # TODO (optional): adjust these numbers or leave as-is.
        if no_improve > 250 and k < int(iters * 0.9):
            current = best[:]
            cur_cost = best_cost
            # small shake using an exploratory mutation
            current = _mutate_detour(current, neighbors_fn, rng)
            cur_cost = safe_cost(current)
            no_improve = 0

    return best, history
