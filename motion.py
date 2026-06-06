from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from geometry import N
from grasping import Grasp


@dataclass
class Motion:
    """One executable waypoint.

    ``kind``:
      * ``move`` / ``lin`` — go to ``T_ee`` (free-space vs. slow linear). ``q``
        is the optional pre-solved joint target.
      * ``close`` / ``open`` — actuate the gripper.
      * ``turn`` — spin the rotate table for ``move``.
      * ``wait`` — dwell.
    """
    kind:    str
    T_ee:    Optional[np.ndarray] = None
    q:       Optional[np.ndarray] = None
    move:    Optional[str]        = None
    note:    str                  = ""
    face:    Optional[str]        = None
    fingers: Optional[tuple]      = None


def _to_pre(kind: str, g: Grasp, note: str, face: Optional[str] = None) -> Motion:
    """Waypoint at a grasp's pre-grasp (idle) pose."""
    return Motion(kind, g.T_pre, g.q_pre, note=note, face=face)


def _to_grasp(kind: str, g: Grasp, note: str, face: Optional[str] = None) -> Motion:
    """Waypoint at a grasp's contact pose."""
    return Motion(kind, g.T_ee, g.q, note=note, face=face)


def _finger_faces(grasp: Grasp) -> Optional[tuple]:
    """Cube faces the two fingers press against (along ±gripper-X), for display."""
    if grasp.R_cube is None:
        return None
    x_world = grasp.T_ee[:3, :3] @ np.array([1.0, 0.0, 0.0])
    best_pos = best_neg = None
    best_pos_dot = best_neg_dot = -999.0
    for face, n in N.items():
        wn = grasp.R_cube @ n.astype(float)
        d1 = float(x_world @ wn)
        d2 = float((-x_world) @ wn)
        if d1 > best_pos_dot: best_pos_dot = d1; best_pos = face
        if d2 > best_neg_dot: best_neg_dot = d2; best_neg = face
    return best_pos, best_neg


def to_motion(actions) -> List[Motion]:
    """Lower the planner's abstract action stream into ordered waypoints."""
    out: List[Motion] = []
    action_list = list(actions)
    n = len(action_list)

    for i, (kind, payload) in enumerate(action_list):
        prev_kind    = action_list[i - 1][0] if i > 0     else None
        next_kind    = action_list[i + 1][0] if i + 1 < n else None
        next_payload = action_list[i + 1][1] if i + 1 < n else None

        if kind == "reposition":
            rep = payload
            if rep.from_hold:
                # Cube still grasped at the turn pose — carry straight to placement.
                out += [_to_pre("move", rep.place, "idle@place"),
                        _to_grasp("lin", rep.place, "place"),
                        Motion("open")]
            else:
                out += [_to_pre("move", rep.pick, "idle@pick"),   # approach
                        _to_grasp("lin", rep.pick, "grasp"),
                        Motion("close"),
                        _to_pre("lin", rep.pick, "idle"),         # lift
                        _to_pre("move", rep.place, "idle@place"), # carry
                        _to_grasp("lin", rep.place, "place"),     # lower
                        Motion("open")]
            if next_kind != "hold":
                out.append(_to_pre("lin", rep.place, "idle"))

        elif kind == "hold":
            out += [_to_pre("move", payload, "idle", payload.face),
                    _to_grasp("lin", payload, "grasp", payload.face),
                    Motion("close", note="close", fingers=_finger_faces(payload)),
                    _to_pre("move", payload, "idle")]

        elif kind == "turn":
            move_str, hold_grasp = payload
            T_op = hold_grasp.T_op if hold_grasp.T_op is not None else hold_grasp.T_ee
            out += [Motion("move", T_op, note="operation", face=hold_grasp.face),
                    Motion("turn", move=move_str, note="turn", face=hold_grasp.face),
                    Motion("wait", note="dwell 1s")]
            next_from_hold = next_kind == "reposition" and next_payload.from_hold
            if next_kind not in ("turn", "release") and not next_from_hold:
                out.append(_to_pre("move", hold_grasp, "idle"))

        elif kind == "release":
            if prev_kind == "turn":
                out.append(_to_pre("move", payload, "idle@release", payload.face))
            out += [_to_grasp("lin", payload, "lower", payload.face),
                    Motion("open", note="open")]
            if next_kind != "reposition":
                out.append(_to_pre("move", payload, "idle", payload.face))

    return out
