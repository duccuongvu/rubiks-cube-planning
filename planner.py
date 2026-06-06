from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from geometry import FACES, OPPOSITE, P_REST, cube_orientations, grasp_template
from grasping import Grasp
from robot import RobotModel

N_ORI = 24  # discrete cube orientations (proper rotation subgroup)


def _face(move: str) -> str:
    """Face letter of a move token, dropping any ``'`` / ``2`` suffix."""
    return move[0]


@dataclass
class Reposition:
    """A single pick-and-place that re-seats the cube in a new orientation.

    `from_hold` marks that the pick reuses the face already grasped, so the
    gripper need not release and re-grasp (see ``CubePlanner.plan_sequence``).
    """
    pick:      Grasp
    place:     Grasp
    target:    int
    from_hold: bool = False


class CubePlanner:
    """Plans gripper repositions + face turns for a Rubik's move sequence.

    The cube's pose is one of ``N_ORI`` discrete orientations. At construction
    the planner builds the full reposition graph: an edge ``A -> B`` exists for
    every face graspable at both A and B (one pick-and-place). Per move it BFS's
    from the current orientation to any "turn-ready" orientation — one where the
    face opposite the move is graspable.
    """

    def __init__(self, robot: RobotModel, p_rest: np.ndarray = P_REST):
        self.robot  = robot
        self.p_rest = p_rest
        self.R      = cube_orientations()
        self.Tg     = {f: grasp_template(f) for f in FACES}
        self._grasp_cache: Dict[int, List[Grasp]] = {}
        # Per-orientation {face: Grasp}; the single source of truth for edges.
        self._by_face: Dict[int, Dict[str, Grasp]] = {
            o: {g.face: g for g in self.feasible_grasps(o)} for o in range(N_ORI)
        }
        self._goal_cache: Dict[str, Set[int]] = {}
        self.edges = self._build_edges()

    # ── feasibility ───────────────────────────────────────────────────────── #

    def feasible_grasps(self, ori: int) -> List[Grasp]:
        """Grasps the robot can achieve at orientation `ori` (memoized)."""
        if ori not in self._grasp_cache:
            self._grasp_cache[ori] = [
                g for f in FACES
                if (g := self.robot.solve_grasp(
                    f, self.R[ori], self.p_rest, self.Tg[f])) is not None
            ]
        return self._grasp_cache[ori]

    def _build_edges(self) -> Dict[int, List[Tuple[int, Reposition]]]:
        """For each orientation A, an edge to every B reachable by re-seating
        the cube on a face graspable at both A and B."""
        edges: Dict[int, List[Tuple[int, Reposition]]] = {a: [] for a in range(N_ORI)}
        for A in range(N_ORI):
            for face, pick in self._by_face[A].items():
                for B in range(N_ORI):
                    if B == A:
                        continue
                    place = self._by_face[B].get(face)
                    if place is not None:
                        edges[A].append((B, Reposition(pick=pick, place=place, target=B)))
        return edges

    # ── orientation graph search ──────────────────────────────────────────── #

    def bfs(self, start: int, goals: Set[int]) -> Optional[List[Reposition]]:
        """Shortest reposition path (by count) from `start` into `goals`."""
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

    # ── turn readiness ────────────────────────────────────────────────────── #

    def move_ready(self, ori: int, move: str) -> bool:
        """True if `move` turns without repositioning (opposite face graspable)."""
        return OPPOSITE[_face(move)] in self._by_face[ori]

    def _move_goals(self, move: str) -> Set[int]:
        """Orientations from which `move` executes with no repositioning."""
        face = _face(move)
        if face not in self._goal_cache:
            self._goal_cache[face] = {
                o for o in range(N_ORI) if OPPOSITE[face] in self._by_face[o]
            }
        return self._goal_cache[face]

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
        """Shortest path for `move` whose first pick reuses `held_face`
        (so the grasp is never released), or ``None`` if none exists."""
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

    # ── sequence planning ─────────────────────────────────────────────────── #

    def plan_sequence(self, moves: List[str], ori0: int = 0):
        """Plan a full move sequence.

        Returns ``(actions, final_ori)`` where the action stream is::

            ('reposition', Reposition) | ('hold', Grasp)
            ('turn', (move, Grasp))    | ('release', Grasp)

        Optimizations:
          * Consecutive moves on the same face stay grasped (no round-trip).
          * When a reposition's first pick reuses the held face and is no longer
            than a fresh plan, the grasp is kept closed (``from_hold=True``) —
            skipping a release + re-grasp.
        """
        actions, ori = [], ori0
        current_hold: Optional[Grasp] = None

        for m in moves:
            desired_face = OPPOSITE[_face(m)]

            if current_hold is not None and current_hold.face == desired_face:
                actions.append(("turn", (m, current_hold)))
                continue

            normal_path = self.plan_move(ori, m)
            reuse_path = (
                self.plan_move_reusing_hold(ori, m, current_hold.face)
                if current_hold is not None else None
            )

            if reuse_path is not None and len(reuse_path) <= len(normal_path):
                path = list(reuse_path)
                path[0] = replace(path[0], from_hold=True)
            else:
                if current_hold is not None:
                    actions.append(("release", current_hold))
                path = normal_path

            for rep in path:
                actions.append(("reposition", rep))
                ori = rep.target

            hold = self._by_face[ori][desired_face]
            actions.append(("hold", hold))
            actions.append(("turn", (m, hold)))
            current_hold = hold

        if current_hold is not None:
            actions.append(("release", current_hold))

        return actions, ori
