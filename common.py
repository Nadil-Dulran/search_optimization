# common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional, Set
import os, random, hashlib, math, json

Coord = Tuple[int, int]
Edge = Tuple[Coord, int]
Graph = Dict[Coord, List[Edge]]

ROWS = COLS = 6  # a bit larger than 5x5 for variability
START: Coord = (0, 0)
GOAL:  Coord = (ROWS-1, COLS-1)

# ---------------- Seeding & obstacle/weight generation ----------------
CANDIDATE_SLOTS = [(1,2),(2,2),(3,1),(1,3),(2,1),(3,3),(4,2)]
DEFAULT_OBS = 4

def get_seed() -> str:
    return os.getenv("STUDENT_ID", "DEMO").strip() or "DEMO"

def get_secret_seed() -> Optional[str]:
    s = os.getenv("GRADER_SECRET", "").strip()
    return s or None

def rng_from(s: str) -> random.Random:
    return random.Random(int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10**9+7))

def make_obstacles(seed: str, k: int = DEFAULT_OBS) -> Set[Coord]:
    r = rng_from("OBS"+seed)
    pool = CANDIDATE_SLOTS[:]
    r.shuffle(pool)
    return set(pool[:k])

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS

def base_neighbors(rc: Coord) -> List[Coord]:
    r,c = rc
    return [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]

def weight_pattern(rc: Coord) -> int:
    # mild, deterministic terrain pattern (1 or 2)
    r,c = rc
    return 1 if (r+c) % 3 != 0 else 2

def build_graph(seed: str, weighted: bool) -> Tuple[Graph, Set[Coord]]:
    obs = make_obstacles(seed)
    g: Graph = {}
    for r in range(ROWS):
        for c in range(COLS):
            u = (r,c)
            if u in obs:
                continue
            edges: List[Edge] = []
            for v in base_neighbors(u):
                if in_bounds(*v) and v not in obs:
                    w = weight_pattern(v) if weighted else 1
                    edges.append((v, w))
            g[u] = edges
    return g, obs

# ---------------- Tracing hook (must be used) ----------------
@dataclass
class Trace:
    expanded: List[Coord]
    cap: int
    def expand(self, node: Coord):
        self.expanded.append(node)
        if len(self.expanded) > self.cap:
            raise RuntimeError("Exceeded expansion cap; check for loops or poor pruning.")

# ---------------- ASCII rendering ----------------
def draw_ascii(path: List[Coord], obstacles: Set[Coord]) -> str:
    grid = [["·" for _ in range(COLS)] for _ in range(ROWS)]
    for (r,c) in obstacles:
        grid[r][c] = "#"
    if path:
        for (r,c) in path:
            if (r,c) not in (START, GOAL):
                grid[r][c] = "*"
        sr,sc = START; gr,gc = GOAL
        grid[sr][sc] = "S"; grid[gr][gc] = "G"
    return "\n".join(" ".join(row) for row in grid)

# ---------------- LP helpers ----------------
Constraint = Tuple[float, float, float]  # a1, a2, b  => a1*x + a2*y <= b

def intersect(l1: Constraint, l2: Constraint) -> Optional[Tuple[float, float]]:
    a1,a2,b = l1; c1,c2,d = l2
    det = a1*c2 - a2*c1
    if abs(det) < 1e-12: return None
    x = (b*c2 - a2*d)/det
    y = (a1*d - b*c1)/det
    return (x,y)

def round_pair(p: Tuple[float,float]) -> Tuple[float,float]:
    return (round(p[0], 10), round(p[1], 10))

def gen_lp(seed: str) -> Tuple[List[Constraint], Tuple[float,float]]:
    # vary constraints/profits slightly per student
    r = rng_from("LP"+seed)
    # Base: 2x + 1y <= B1, 1x + 2y <= B2, x<=U1, y<=U2
    B1 = r.randint(14, 20)
    B2 = r.randint(14, 20)
    U1 = r.randint(7, 10)
    U2 = r.randint(7, 10)
    c1 = r.randint(6, 9)  # profits
    c2 = r.randint(4, 7)
    constraints = [(2,1,B1), (1,2,B2), (1,0,U1), (0,1,U2)]
    return constraints, (float(c1), float(c2))

# ---------------- DP (0/1 knapsack) data ----------------
def gen_knapsack(seed: str):
    r = rng_from("DP"+seed)
    # 6–7 items with weights 1..6 and values 3..14
    n = 6
    items = []
    for i in range(n):
        w = r.randint(1,6)
        v = r.randint(3,14)
        items.append((f"I{i+1}", w, v))
    capacity = r.randint(9, 12)
    return items, capacity

# ---------------- scoring utils ----------------
def pct(points, total) -> float:
    return 100.0 * (points / total)

def sha_code_fingerprint(files: List[str]) -> str:
    h = hashlib.sha256()
    for fp in files:
        try:
            with open(fp, "rb") as f: h.update(f.read())
        except Exception: pass
    return h.hexdigest()

def save_results(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
