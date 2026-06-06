from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from geometry import FACES, OPPOSITE, P_REST, cube_orientations, grasp_template
from grasping import Grasp
from robot import RobotModel


@dataclass
class Reposition:
    pick:      Grasp
    place:     Grasp
    target:    int
    from_hold: bool = False   # pick face already grasped — no re-grasp needed


class CubePlanner:
    def __init__(self, robot: RobotModel, p_rest: np.ndarray = P_REST):
        self.robot  = robot
        self.p_rest = p_rest
        self.R      = cube_orientations()
        self.Tg     = {f: grasp_template(f) for f in FACES}
        self._grasp_cache: Dict[int, List[Grasp]] = {}
        self.edges = self._build_edges()

    def feasible_grasps(self, ori: int) -> List[Grasp]:
        if ori not in self._grasp_cache:
            self._grasp_cache[ori] = [
                g for f in FACES
                for g in [self.robot.solve_grasp(f, self.R[ori], self.p_rest, self.Tg[f])]
                if g is not None
            ]
        return self._grasp_cache[ori]

    def _build_edges(self) -> Dict[int, List[Tuple[int, Reposition]]]:
        edges = {a: [] for a in range(24)}
        for A in range(24):
            for g in self.feasible_grasps(A):
                seen: set = set()
                for B in range(24):
                    if B == A or B in seen:
                        continue
                    place = self.robot.solve_grasp(g.face, self.R[B], self.p_rest, self.Tg[g.face])
                    if place is None:
                        continue
                    edges[A].append((B, Reposition(pick=g, place=place, target=B)))
                    seen.add(B)
        return edges

    def bfs(self, start: int, goals: set) -> Optional[List[Reposition]]:
        if start in goals:
            return []
        parent: Dict[int, Optional[Tuple[int, Reposition]]] = {start: None}
        q = deque([start])
        while q:
            a = q.popleft()
            for b, action in self.edges[a]:
                if b in parent:
                    continue
                parent[b] = (a, action)
                if b in goals:
                    return self._reconstruct(parent, b)
                q.append(b)
        return None

    @staticmethod
    def _reconstruct(parent, node) -> List[Reposition]:
        path, cur = [], node
        while parent[cur] is not None:
            prev, action = parent[cur]
            path.append(action)
            cur = prev
        return list(reversed(path))

    def move_ready(self, ori: int, move: str) -> bool:
        return any(g.face == OPPOSITE[move] for g in self.feasible_grasps(ori))

    def _move_goals(self, move: str) -> set:
        """Orientations from which `move` executes with no repositioning."""
        return {o for o in range(24) if self.move_ready(o, move)}

    def plan_move(self, ori: int, move: str) -> List[Reposition]:
        goals = self._move_goals(move)
        if not goals:
            raise RuntimeError(f"move {move} never executable")
        path = self.bfs(ori, goals)
        if path is None:
            raise RuntimeError(f"cannot reposition from {ori} to execute {move}")
        return path

    def plan_move_reusing_hold(self, ori: int, move: str,
                               held_face: str) -> Optional[List[Reposition]]:
        """Shortest path for `move` whose first pick reuses `held_face` (no re-grasp)."""
        goals = self._move_goals(move)
        if ori in goals:
            return None
        best: Optional[List[Reposition]] = None
        for b, action in self.edges[ori]:
            if action.pick.face != held_face:
                continue
            tail = [] if b in goals else self.bfs(b, goals)
            if tail is None:
                continue
            cand = [action, *tail]
            if best is None or len(cand) < len(best):
                best = cand
        return best

    def plan_sequence(self, moves: List[str], ori0: int = 0):
        """Returns (actions, final_ori).

        Action stream: ('reposition', Reposition) | ('hold', Grasp)
                     | ('turn', (move, Grasp))    | ('release', Grasp)

        Optimizations:
          * Consecutive moves on same face stay grasped (no round-trip).
          * When reposition's first pick reuses the held face (equal-or-shorter
            path), the grasp is kept closed — no release + re-grasp.
        """
        actions, ori = [], ori0
        current_hold: Optional[Grasp] = None

        for m in moves:
            desired_face = OPPOSITE[m]

            if current_hold is not None and current_hold.face == desired_face:
                actions.append(("turn", (m, current_hold)))
            else:
                normal_path = self.plan_move(ori, m)
                reuse_path = (
                    self.plan_move_reusing_hold(ori, m, current_hold.face)
                    if current_hold is not None else None
                )

                if reuse_path and len(reuse_path) <= len(normal_path):
                    path = list(reuse_path)
                    path[0] = replace(path[0], from_hold=True)
                else:
                    if current_hold is not None:
                        actions.append(("release", current_hold))
                    path = normal_path

                current_hold = None
                for rep in path:
                    actions.append(("reposition", rep))
                    ori = rep.target

                hold = next(g for g in self.feasible_grasps(ori) if g.face == desired_face)
                actions.append(("hold", hold))
                actions.append(("turn", (m, hold)))
                current_hold = hold

        if current_hold is not None:
            actions.append(("release", current_hold))

        return actions, ori
