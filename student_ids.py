# student_ids.py
# ============================================================
# TASK
#   Implement Iterative Deepening Search (IDS).
#
# SIGNATURE (do not change):
#   ids(start, goal, neighbors_fn, trace, max_depth=64) -> (List[Coord], int)
#
# PARAMETERS
#   start, goal:       coordinates
#   neighbors_fn(u):   returns valid 4-neighbors of u
#   trace:             MUST call trace.expand(u) when you EXPAND u
#                      in the depth-limited search (DLS).
#   max_depth:         upper cap for the iterative deepening
#
# RETURN
#   (path, depth_limit_used)
#   - If found at depth L, return the path and L.
#   - If not found up to max_depth, return ([], max_depth).
#
# IMPLEMENTATION HINT
# - Outer loop: for limit in [0..max_depth]:
#       run DLS(start, limit) with its own parent dict and visited set
#       DLS(u, remaining):
#           trace.expand(u)
#           if u == goal: return True
#           if remaining == 0: return False
#           for v in neighbors_fn(u):
#               if v not seen in THIS DLS: mark parent[v]=u and recurse
# - Reconstruct the path when DLS reports success.
# ============================================================

from typing import List, Tuple, Callable, Dict, Optional, Set

Coord = Tuple[int, int]

def ids(start: Coord,
        goal: Coord,
        neighbors_fn: Callable[[Coord], List[Coord]],
        trace,
        max_depth: int = 64) -> Tuple[List[Coord], int]:
    """
    REQUIRED: call trace.expand(u) in the DLS when you expand u.
    """
    # Return early if start is the goal
    if start == goal:
        return [start]

    def reconstruct_path(parent: Dict[Coord, Optional[Coord]], v: Coord) -> List[Coord]:
        # Match the runner's reconstruction style (duplicates start before reverse)
        p: List[Coord] = [v]
        while parent[p[-1]] is not None:
            p.append(parent[p[-1]])
        p.append(start)
        p.reverse()
        return p

    for limit in range(0, max_depth + 1):
        parent: Dict[Coord, Optional[Coord]] = {start: None}
        path_stack: Set[Coord] = {start}

        def dls(u: Coord, remaining: int) -> Optional[List[Coord]]:
            # Expansion hook must be called upon expanding u
            try:
                trace.expand(u)
            except Exception:
                pass
            if u == goal:
                return reconstruct_path(parent, u)
            if remaining == 0:
                return None
            for v in neighbors_fn(u):
                if v in path_stack:
                    continue  # avoid cycles on current path
                parent[v] = u
                path_stack.add(v)
                res = dls(v, remaining - 1)
                if res is not None:
                    return res
                path_stack.remove(v)
            return None

        result = dls(start, limit)
        if result is not None:
            return result

    # Not found up to max_depth
    return []
