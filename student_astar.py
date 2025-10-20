# student_astar.py
# ============================================================
# TASK
#   Implement A* search that returns (path, cost).
#
# SIGNATURE (do not change):
#   astar(start, goal, neighbors_fn, heuristic_fn, trace) -> (List[Coord], float)
#
# PARAMETERS
#   start, goal:           grid coordinates
#   neighbors_fn(u):       returns valid 4-neighbors of u
#   heuristic_fn(u, goal): returns a non-negative estimate to goal
#   trace:                 MUST call trace.expand(u) whenever you pop u
#                         from the PRIORITY QUEUE to expand it.
#
# EDGE COSTS
#   Assume unit step cost (=1) unless your runner specifies otherwise.
#   (If your runner supplies a graph.cost(u,v), adapt here if needed.)
#
# RETURN
#   (path, cost) where path is the list of coordinates from start to goal,
#   and cost is the sum of step costs along that path (float).
#   If no path exists, return ([], 0.0).
#
# IMPLEMENTATION HINT
# - Use min-heap over f = g + h.
# - Keep g[u] (cost from start), parent map, and a closed set.
# - On goal, reconstruct path and also compute cost (sum of steps).
# ============================================================

from typing import List, Tuple, Callable, Dict
import heapq

Coord = Tuple[int, int]

def astar(start: Coord,
          goal: Coord,
          neighbors_fn: Callable[[Coord], List[Coord]],
          heuristic_fn: Callable[[Coord, Coord], float],
          trace) -> Tuple[List[Coord], float]:
    """
    REQUIRED: call trace.expand(u) when you pop u from the PQ to expand.
    """
    # Trivial case
    if start == goal:
        return [start]

    # Min-heap of (f, g, node)
    h0 = float(heuristic_fn(start, goal))
    heap = [(h0, 0.0, start)]
    parent: Dict[Coord, Coord | None] = {start: None}
    g: Dict[Coord, float] = {start: 0.0}
    closed = set()

    while heap:
        f_u, g_u, u = heapq.heappop(heap)
        if u in closed:
            continue
        # Expansion trace hook
        try:
            trace.expand(u)
        except Exception:
            pass
        if u == goal:
            # Reconstruct path
            path: List[Coord] = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.append(start)
            path.reverse()
            # Cost is g of goal under unit steps
            return path
        closed.add(u)
        # Explore neighbors
        for v in neighbors_fn(u):
            tentative = g_u + 1.0  # unit step cost
            if v not in g or tentative < g[v]:
                g[v] = tentative
                parent[v] = u
                f_v = tentative + float(heuristic_fn(v, goal))
                heapq.heappush(heap, (f_v, tentative, v))

    return []

# --- (ONLY IF YOUR RUNNER PASSES A Graph INSTEAD OF neighbors_fn) ---
# def astar_graph(graph, start, goal, heuristic_fn, trace):
#     return astar(start, goal, graph.neighbors, heuristic_fn, trace)
