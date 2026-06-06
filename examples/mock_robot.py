"""
MockRobot + visualization for the reposition planner.

Run: python3 examples/mock_robot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry import CUBE_HALF, FACES, N, OPPOSITE, T_OP, _T, pre_grasp_pose
from grasping import GRIPPER, Grasp
from motion import to_motion
from planner import CubePlanner
from robot import RobotModel

# --------------------------------------------------------------------------- #
#  Visualization helpers                                                       #
# --------------------------------------------------------------------------- #

COLOR_EMOJI = {
    "U": "⬜", "D": "🟨",
    "F": "🟩", "B": "🟦",
    "R": "🟥", "L": "🟧",
}

WORLD_AXES = {
    "right": np.array([0,  1, 0]),
    "left":  np.array([0, -1, 0]),
    "up":    np.array([0,  0, 1]),
    "down":  np.array([0,  0, -1]),
    "front": np.array([1,  0, 0]),
    "back":  np.array([-1, 0, 0]),
}


def describe_orientation(R: np.ndarray) -> dict:
    result = {}
    for face, n in N.items():
        w = R @ n
        for name, axis in WORLD_AXES.items():
            if np.allclose(w, axis):
                result[name] = face
    return result


def pretty_orientation(desc: dict) -> None:
    u = COLOR_EMOJI[desc["up"]]
    d = COLOR_EMOJI[desc["down"]]
    f = COLOR_EMOJI[desc["front"]]
    b = COLOR_EMOJI[desc["back"]]
    r = COLOR_EMOJI[desc["right"]]
    l = COLOR_EMOJI[desc["left"]]
    print(f"\n   {u}\n\n{l} {f} {r} {b}\n\n   {d}\n")


def finger_faces(T_ee: np.ndarray, R_cube: np.ndarray) -> tuple:
    x_world = T_ee[:3, :3] @ np.array([1.0, 0.0, 0.0])
    best_pos = best_neg = None
    best_pos_dot = best_neg_dot = -999.0
    for face, n in N.items():
        wn = R_cube @ n.astype(float)
        d1 = float(x_world @ wn)
        d2 = float((-x_world) @ wn)
        if d1 > best_pos_dot: best_pos_dot = d1; best_pos = face
        if d2 > best_neg_dot: best_neg_dot = d2; best_neg = face
    return best_pos, best_neg

# --------------------------------------------------------------------------- #
#  MockRobot                                                                   #
# --------------------------------------------------------------------------- #

class MockRobot(RobotModel):
    """Geometric stand-in — no IK. Reaches from +Y, -X, +Z directions."""
    REACH = [
        np.array([0.0,  1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0,  0.0, 1.0]),
    ]

    def _reachable(self, world_normal: np.ndarray) -> bool:
        return max(float(world_normal @ r) for r in self.REACH) > 0.9

    def solve_grasp(self, face, R, p_rest, T_g_body):
        n_world = R @ N[face].astype(float)
        if not self._reachable(n_world):
            return None
        table_z = p_rest[2] - CUBE_HALF
        T_ee    = _T(R, p_rest) @ T_g_body
        if GRIPPER.hits_table(T_ee, table_z):
            return None
        T_pre = pre_grasp_pose(T_ee)
        return Grasp(face=face, T_ee=T_ee, q=np.zeros(6),
                     T_pre=T_pre, q_pre=np.zeros(6), R_cube=R, T_op=T_OP)

# --------------------------------------------------------------------------- #
#  Demo                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    planner = CubePlanner(MockRobot())

    n_edges = sum(len(v) for v in planner.edges.values())
    print(f"orientations : 24")
    print(f"edges        : {n_edges}  (avg {n_edges/24:.1f} per node)")
    print()

    for mv in ["R", "U", "F", "L", "D", "B"]:
        path  = planner.plan_move(0, mv)
        steps = " -> ".join(str(r.target) for r in path) or "(already ready)"
        print(f"move {mv}: {len(path)} reposition(s)   path: {steps}")

    moves = ["F", "R", "L"]
    actions, final_ori = planner.plan_sequence(moves)

    print(f"\n============================== ACTION SEQUENCE  {moves}")
    ori = 0
    for kind, payload in actions:
        desc = describe_orientation(planner.R[ori])
        print(f"\n--- ORI={ori}")
        pretty_orientation(desc)

        if kind == "reposition":
            f1, f2 = finger_faces(payload.pick.T_ee, planner.R[ori])
            tag = "REPOSITION (reuse hold)" if payload.from_hold else "REPOSITION"
            print(f"[{tag}] pick={COLOR_EMOJI[payload.pick.face]}  "
                  f"fingers={COLOR_EMOJI[f1]}<->{COLOR_EMOJI[f2]}  "
                  f"target={payload.target}")
            ori = payload.target

        elif kind == "hold":
            f1, f2 = finger_faces(payload.T_ee, planner.R[ori])
            print(f"[HOLD]  face={COLOR_EMOJI[payload.face]}  "
                  f"fingers={COLOR_EMOJI[f1]}<->{COLOR_EMOJI[f2]}")

        elif kind == "turn":
            move_str, _ = payload
            print(f"[TURN]  move={move_str}")

        elif kind == "release":
            print(f"[RELEASE]  face={COLOR_EMOJI[payload.face]}")

    print(f"\n============================== FINAL ORI={final_ori}")

    motions = to_motion(actions)
    print(f"\nlowered motion: {len(motions)} waypoints")
    for m in motions:
        if m.kind == "turn":
            tag = m.move
        else:
            parts = [m.note]
            if m.face:    parts.append(f"face {COLOR_EMOJI[m.face]}")
            if m.fingers: parts.append(f"fingers {COLOR_EMOJI[m.fingers[0]]} <-> {COLOR_EMOJI[m.fingers[1]]}")
            tag = "  ".join(parts)
        print(f"  {m.kind:5}  {tag}")
