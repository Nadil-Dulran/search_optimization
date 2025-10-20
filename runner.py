# runner.py
from __future__ import annotations
import json, math, random, argparse, importlib, sys
from typing import List, Tuple, Set, Callable, Dict, Any, Optional
from collections import deque

# --------------------------
# Types & global config
# --------------------------
Coord = Tuple[int, int]
START: Coord = (0, 0)

# --------------------------
# Utilities
# --------------------------
def set_seed_from_any(seed: str | int) -> random.Random:
    rs = str(seed)
    h = 1469598103934665603  # FNV-1a 64-bit basis
    for ch in rs:
        h ^= ord(ch); h = (h * 1099511628211) & ((1<<64)-1)
    r = random.Random(h)
    return r

def neighbors_4(rows: int, cols: int, obstacles: Set[Coord]) -> Callable[[Coord], List[Coord]]:
    def fn(u: Coord) -> List[Coord]:
        (r, c) = u
        out = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles:
                out.append((nr, nc))
        return out
    return fn

def turns_in_path(path: List[Coord]) -> int:
    if len(path) < 3:
        return 0
    def step(a: Coord, b: Coord) -> Coord:
        return (b[0]-a[0], b[1]-a[1])
    t = 0
    for i in range(2, len(path)):
        if step(path[i-2], path[i-1]) != step(path[i-1], path[i]):
            t += 1
    return t

def objective_path(path: List[Coord]) -> float:
    if not path: return float('inf')
    return float(len(path) + 0.2 * turns_in_path(path))

def _bfs_path_local(start: Coord, goal: Coord, neighbors_fn: Callable[[Coord], List[Coord]]) -> List[Coord]:
    if start == goal: return [start]
    q = deque([start]); parent = {start: None}
    while q:
        u = q.popleft()
        for v in neighbors_fn(u):
            if v not in parent:
                parent[v] = u
                if v == goal:
                    p = [v]
                    while parent[p[-1]] is not None:
                        p.append(parent[p[-1]])
                    p.append(start)
                    p.reverse()
                    return p
                q.append(v)
    return []

def _grid_bfs_dist(start: Coord, goal: Coord, neighbors_fn) -> float:
    if start == goal: return 0
    q = deque([start]); dist = {start: 0}
    while q:
        u = q.popleft()
        for v in neighbors_fn(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                if v == goal: return dist[v]
                q.append(v)
    return float("inf")

def build_grid(rows: int, cols: int, density: float, rng: random.Random) -> Set[Coord]:
    # density is probability of an obstacle in a non-start/non-goal cell
    attempts = 0
    while attempts < 200:
        obstacles: Set[Coord] = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) == START: continue
                if (r, c) == (rows-1, cols-1): continue
                if rng.random() < density:
                    obstacles.add((r, c))
        n4 = neighbors_4(rows, cols, obstacles)
        if _bfs_path_local(START, (rows-1, cols-1), n4):
            return obstacles
        density *= 0.95
        attempts += 1
    return set()

class Trace:
    def __init__(self): self.expanded = []
    def expand(self, node: Coord): self.expanded.append(node)

# --------------------------
# Integrity helpers
# --------------------------
def _add_check(checks, name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

def _count_plateau_changes(arr, eps=1e-9):
    changes, last = 0, None
    for v in (arr or []):
        if last is None or abs(v - last) > eps:
            changes += 1
            last = v
    return changes

# --------------------------
# Grading helpers
# --------------------------
def grade_bfs(student, rows, cols, obstacles) -> Dict[str, Any]:
    n4 = neighbors_4(rows, cols, obstacles)
    goal = (rows-1, cols-1)
    trace = Trace()
    try:
        path = student.bfs(START, goal, n4, trace)  # required signature
    except Exception:
        path = []
    ok = bool(path and path[-1] == goal)
    best = _bfs_path_local(START, goal, n4)
    best_len = len(best) if best else None
    path_len = len(path) if path else None
    score = 0
    if ok and best and path_len == best_len:
        score = 10
    elif ok:
        score = 5
    return {
        "ok": ok,
        "path": path,
        "path_len": path_len or 0,
        "best_len": best_len or 0,
        "expansions": len(trace.expanded),
        "score": score,
    }

def grade_heuristics(heur, rows, cols, obstacles) -> Dict[str, Any]:
    # 20% total: Manhattan 5, Straight-line 5, Custom 10
    n4 = neighbors_4(rows, cols, obstacles)
    goal = (rows-1, cols-1)
    rng = random.Random(12345)
    samples = []
    for _ in range(30):
        r = rng.randrange(rows); c = rng.randrange(cols)
        u = (r,c)
        if u in obstacles: continue
        td = _grid_bfs_dist(u, goal, n4)
        if not math.isfinite(td) or td == float("inf"): continue
        samples.append((u, td))
    def test_h(name, hf, weight):
        try:
            ok_cnt = 0; neg = 0; nan = 0; above = 0
            for (u, td) in samples:
                h = hf(u, goal)
                if not (isinstance(h,(int,float)) and math.isfinite(h)):
                    nan += 1; continue
                if h < -1e-9: neg += 1
                if h > td + 1e-9: above += 1
                if (h >= -1e-9) and (h <= td + 1e-9): ok_cnt += 1
            ok = (ok_cnt >= max(5, int(0.6*len(samples)))) and (neg == 0)
            score = weight if ok else 0
            detail = f"ok={ok_cnt}/{len(samples)}, neg={neg}, above={above}"
            return score, ok, detail
        except Exception as e:
            return 0, False, f"error: {e}"
    m_score, m_ok, m_det = test_h("manhattan", heur.heuristic_manhattan, 5)
    e_score, e_ok, e_det = test_h("straight", heur.heuristic_straight_line, 5)
    c_score, c_ok, c_det = test_h("custom", heur.heuristic_custom, 10)
    total = m_score + e_score + c_score
    return {
        "score": total,
        "manhattan": {"ok": m_ok, "detail": m_det, "score": m_score},
        "straight": {"ok": e_ok, "detail": e_det, "score": e_score},
        "custom": {"ok": c_ok, "detail": c_det, "score": c_score},
    }

def grade_astar(student, heur, rows, cols, obstacles) -> Dict[str, Any]:
    n4 = neighbors_4(rows, cols, obstacles)
    goal = (rows-1, cols-1)
    trace = Trace()
    # choose manhattan by default for grading path optimality
    try:
        path = student.astar(
            START, goal, n4, heur.heuristic_manhattan, trace  # required signature
        )
    except Exception:
        path = []
    ok = bool(path and path[-1] == goal)
    best = _bfs_path_local(START, goal, n4)
    best_len = len(best) if best else None
    path_len = len(path) if path else None
    score = 0
    if ok and best and path_len == best_len:
        score = 15
    elif ok:
        score = 8
    return {
        "ok": ok,
        "path": path,
        "path_len": path_len or 0,
        "best_len": best_len or 0,
        "final_cost": objective_path(path) if ok else None,
        "expansions": len(trace.expanded),
        "score": score,
    }

def grade_ids(student, rows, cols, obstacles) -> Dict[str, Any]:
    n4 = neighbors_4(rows, cols, obstacles)
    goal = (rows-1, cols-1)
    trace = Trace()
    try:
        path = student.ids(START, goal, n4, trace)  # required signature
    except Exception:
        path = []
    ok = bool(path and path[-1] == goal)
    best = _bfs_path_local(START, goal, n4)
    best_len = len(best) if best else None
    path_len = len(path) if path else None
    score = 0
    if ok and best and path_len == best_len:
        score = 15
    elif ok:
        score = 8
    return {
        "ok": ok,
        "path": path,
        "path_len": path_len or 0,
        "best_len": best_len or 0,
        "expansions": len(trace.expanded),
        "score": score,
    }

def grade_sa(student_sa, rows, cols, obstacles, seed) -> Dict[str, Any]:
    n4 = neighbors_4(rows, cols, obstacles)
    goal = (rows-1, cols-1)
    bfs0 = _bfs_path_local(START, goal, n4)
    bfs0_cost = objective_path(bfs0) if bfs0 else float("inf")
    try:
        res = student_sa.simulated_annealing(
            neighbors_fn=n4,
            objective_fn=objective_path,
            obstacles=obstacles,
            seed=str(seed),
            iters=900,
            T0=1.3,
            alpha=0.995
        )
        if isinstance(res, tuple) and len(res) == 2:
            best_path, history = res
        else:
            best_path, history = res, None
        ok = bool(best_path and best_path[-1] == goal)
        final_cost = objective_path(best_path) if ok else None
    except Exception:
        best_path, history, ok, final_cost = [], None, False, None

    # strict scoring on 3 conditions
    IMPROVE_THR = 1.0
    improvement = -1e9
    if ok and math.isfinite(bfs0_cost) and math.isfinite(final_cost or float('inf')):
        improvement = bfs0_cost - final_cost
    history_ok = isinstance(history, list) and len(history) >= 2
    nonconstant = history_ok and (max(history) - min(history) > 1e-6)
    changes_ok = history_ok and (_count_plateau_changes(history) >= 3)
    satisfied = sum([improvement > IMPROVE_THR, history_ok and nonconstant, changes_ok])

    if ok:
        if satisfied == 3: score = 15
        elif satisfied == 2: score = 12
        elif satisfied == 1: score = 9
        else: score = 6
    else:
        score = 0

    return {
        "ok": ok,
        "path": best_path if ok else [],
        "path_len": len(best_path) if ok else 0,
        "final_cost": final_cost,
        "history": history,
        "bfs0_cost": bfs0_cost,
        "improvement": improvement,
        "score": score,
    }

def grade_lp_dp(student_lpdp, rng: random.Random) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    # --- LP instance (max c^T x, Ax <= b, x>=0) ---
    # small, nice polygon; fixed but fine for all seeds
    constraints = [
        (1.0, 1.0, 6.0),
        (1.0, 0.0, 4.0),
        (0.0, 1.0, 5.0),
        (2.0, 1.0, 8.0),
    ]
    c1, c2 = 3.0, 5.0

    try:
        verts = student_lpdp.feasible_vertices(constraints)
        best_pt, best_val = student_lpdp.maximize_objective(verts, c1, c2)
        lp_ok = (isinstance(best_pt, tuple) and len(best_pt)==2 and isinstance(best_val,(int,float)))
    except Exception:
        verts, best_pt, best_val, lp_ok = [], (0.0,0.0), None, False

    lp_score = 12.5 if lp_ok else 0.0

    lp_out = {
        "ok": lp_ok,
        "score": lp_score,
        "constraints": constraints,
        "c": [c1, c2],
        "vertices": verts,
        "best_point": best_pt,
        "optimum": best_val
    }

    # --- DP instance (0/1 knapsack) ---
    values = [6, 5, 18, 15, 10]
    weights = [2, 2,  6,  5,  4]
    capacity = 10

    # bottom-up & top-down
    td_val = bu_val = None
    try:
        bu_val = student_lpdp.knapsack_bottom_up(values, weights, capacity)
    except Exception:
        bu_val = None
    try:
        td_val = student_lpdp.knapsack_top_down(values, weights, capacity)
    except Exception:
        td_val = None

    dp_ok = (bu_val is not None) and (td_val is not None) and (bu_val == td_val)
    dp_score = 12.5 if dp_ok else 0.0

    # for visualization, build a small bottom-up table (runner-side) without using student code
    n = len(values)
    dp_table = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        for cap in range(0, capacity+1):
            best = dp_table[i+1][cap]
            if weights[i] <= cap:
                best = max(best, values[i] + dp_table[i+1][cap - weights[i]])
            dp_table[i][cap] = best

    dp_out = {
        "ok": dp_ok,
        "score": dp_score,
        "values": values,
        "weights": weights,
        "capacity": capacity,
        "table": dp_table,
        "bottom_up_value": bu_val,
        "top_down_value": td_val,
        "optimum": bu_val if bu_val is not None else td_val
    }

    return lp_out, dp_out

# --------------------------
# Hidden checks
# --------------------------
def build_hidden_checks(seed, rows, cols, obstacles, out_all, heur_mod, sa_out, lp_out, dp_out):
    checks = []
    _add_check(checks, "Seed match", True, f"seed={seed}")

    # Trace usage (if present)
    try:
        bfs_exp = out_all["bfs"]["expansions"]
        astar_exp = out_all["astar"]["expansions"]
        ids_exp = out_all["ids"]["expansions"]
        ok = (bfs_exp > 0 and astar_exp > 0 and ids_exp > 0)
        _add_check(checks, "Trace usage", ok, f"BFS={bfs_exp}, A*={astar_exp}, IDS={ids_exp}")
    except Exception as e:
        _add_check(checks, "Trace usage", False, f"error: {e}")

    # A* admissibility (spot check using Manhattan)
    try:
        n4 = neighbors_4(rows, cols, obstacles)
        goal = (rows-1, cols-1)
        rng = random.Random(4242)
        samples = []
        for _ in range(20):
            r = rng.randrange(rows); c = rng.randrange(cols)
            u = (r,c)
            if u in obstacles: continue
            td = _grid_bfs_dist(u, goal, n4)
            if math.isfinite(td) and td < float('inf'):
                samples.append((u, td))
        bad = 0; neg = 0
        for (u, td) in samples:
            h = heur_mod.heuristic_manhattan(u, goal)
            if h < -1e-9: neg += 1
            if h > td + 1e-9: bad += 1
        ok = (neg == 0 and bad == 0 and len(samples) >= 5)
        _add_check(checks, "A* heuristic admissibility", ok, f"samples={len(samples)}, neg={neg}, above={bad}")
    except Exception as e:
        _add_check(checks, "A* heuristic admissibility", False, f"error: {e}")

    # SA annealing signals
    try:
        n4 = neighbors_4(rows, cols, obstacles)
        goal = (rows-1, cols-1)
        bfs0 = _bfs_path_local(START, goal, n4)
        bfs0_cost = objective_path(bfs0) if bfs0 else float("inf")
        sa_hist = sa_out.get("history")
        sa_cost = sa_out.get("final_cost")
        improvement = (bfs0_cost - sa_cost) if (math.isfinite(bfs0_cost) and math.isfinite(sa_cost or float('inf'))) else -1e9
        hist_ok = isinstance(sa_hist, list) and len(sa_hist) >= 2 and (max(sa_hist) - min(sa_hist) > 1e-6)
        changes_ok = _count_plateau_changes(sa_hist) >= 3 if sa_hist else False
        all_ok = (improvement > 1.0) and hist_ok and changes_ok
        _add_check(checks, "SA annealing", all_ok, f"improve={improvement:.3f}, hist_len={len(sa_hist or [])}, changes={_count_plateau_changes(sa_hist)}")
    except Exception as e:
        _add_check(checks, "SA annealing", False, f"error: {e}")

    # LP best is vertex
    try:
        best_pt = lp_out.get("best_point")
        verts = lp_out.get("vertices") or []
        in_verts = any(abs(best_pt[0]-vx)<1e-6 and abs(best_pt[1]-vy)<1e-6 for (vx,vy) in verts) if best_pt else False
        _add_check(checks, "LP best is vertex", in_verts, f"best={best_pt}, |V|={len(verts)}")
    except Exception as e:
        _add_check(checks, "LP best is vertex", False, f"error: {e}")

    # DP cross-check
    try:
        bu = dp_out.get("bottom_up_value"); td = dp_out.get("top_down_value")
        ok = (bu is not None) and (td is not None) and (bu == td)
        _add_check(checks, "DP cross-check", ok, f"bottom_up={bu}, top_down={td}")
    except Exception as e:
        _add_check(checks, "DP cross-check", False, f"error: {e}")

    # Hidden multi-seed quick SA
    try:
        SA = importlib.import_module("student_sa")
        res = SA.simulated_annealing(
            neighbors_fn=neighbors_4(rows, cols, obstacles),
            objective_fn=objective_path,
            obstacles=obstacles,
            seed=str(seed) + "_h1",
            iters=200, T0=1.3, alpha=0.995
        )
        best_path = res[0] if isinstance(res, tuple) else res
        ok = bool(best_path and best_path[-1] == (rows-1, cols-1))
        _add_check(checks, "Hidden multi-seed run", ok, "SA sanity on hidden seed")
    except Exception as e:
        _add_check(checks, "Hidden multi-seed run", False, f"error: {e}")

    return checks

# --------------------------
# Main run
# --------------------------
def run_suite(student_id: str, seed: str | None = None,
              rows: int = 6, cols: int = 6, density: float = 0.22) -> None:
    if seed is None: seed = student_id
    rng = set_seed_from_any(seed)
    obstacles = build_grid(rows, cols, density, rng)
    goal = (rows-1, cols-1)

    # Save the generated problem for HTML viz
    problem = {
        "rows": rows, "cols": cols,
        "start": list(START), "goal": [rows-1, cols-1],
        "obstacles": sorted(list(obstacles)),
        "seed": str(seed)
    }
    with open("problem.json","w",encoding="utf-8") as f:
        json.dump(problem, f, indent=2)

    # Import student modules
    BFS = importlib.import_module("student_bfs")
    IDS = importlib.import_module("student_ids")
    ASTAR = importlib.import_module("student_astar")
    SA = importlib.import_module("student_sa")
    LPDP = importlib.import_module("student_lp_dp")
    HEUR = importlib.import_module("heuristics")

    bfs_out = grade_bfs(BFS, rows, cols, obstacles)
    heur_out = grade_heuristics(HEUR, rows, cols, obstacles)
    astar_out = grade_astar(ASTAR, HEUR, rows, cols, obstacles)
    ids_out = grade_ids(IDS, rows, cols, obstacles)
    sa_out = grade_sa(SA, rows, cols, obstacles, seed)
    lp_out, dp_out = grade_lp_dp(LPDP, rng)

    hidden_checks = build_hidden_checks(seed, rows, cols, obstacles,
                                        {"bfs":bfs_out, "astar":astar_out, "ids":ids_out},
                                        HEUR, sa_out, lp_out, dp_out)

    out = {
        "rows": rows, "cols": cols, "obstacles": sorted(list(obstacles)),
        "seed": str(seed), "student_id": student_id,
        "start": list(START), "goal": [rows-1, cols-1],
        "bfs": bfs_out,
        "heuristics": heur_out,     # 20%
        "astar": astar_out,         # 15%
        "ids": ids_out,             # 15%
        "sa": sa_out,               # 15%
        "lp": lp_out,               # 12.5%
        "dp": dp_out,               # 12.5%
        "hidden_checks": hidden_checks
    }
    with open("results.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote results.json and problem.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_id", default="TEST", help="Unique student id string")
    ap.add_argument("--seed", default="None", help="Optional override seed")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--density", type=float, default=0.22)
    args = ap.parse_args()
    run_suite(args.student_id, seed=args.seed, rows=args.rows, cols=args.cols, density=args.density)
