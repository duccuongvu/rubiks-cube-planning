from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from geometry import LIFT_POSE, N
from grasping import Grasp
from planner import Reposition


@dataclass
class Motion:
    kind:    str
    T_ee:    Optional[np.ndarray] = None
    q:       Optional[np.ndarray] = None
    move:    Optional[str]        = None
    note:    str                  = ""
    face:    Optional[str]        = None
    fingers: Optional[tuple]      = None


def _finger_faces(grasp: Grasp) -> Optional[tuple]:
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
                # Cube still grasped at turn pose — carry straight to new placement.
                out.append(Motion("move", rep.place.T_pre, rep.place.q_pre, note="idle@place"))
                out.append(Motion("lin",  rep.place.T_ee,  rep.place.q,     note="place"))
                out.append(Motion("open"))
            else:
                # Pick: approach → grasp → lift
                out.append(Motion("move", rep.pick.T_pre,  rep.pick.q_pre,  note="idle@pick"))
                out.append(Motion("lin",  rep.pick.T_ee,   rep.pick.q,      note="grasp"))
                out.append(Motion("close"))
                out.append(Motion("lin",  rep.pick.T_pre,  rep.pick.q_pre,  note="idle"))
                out.append(Motion("move", LIFT_POSE, None,                   note="lift"))
                # Place: lower → open
                out.append(Motion("move", rep.place.T_pre, rep.place.q_pre, note="idle@place"))
                out.append(Motion("lin",  rep.place.T_ee,  rep.place.q,     note="place"))
                out.append(Motion("open"))
            if next_kind != "hold":
                out.append(Motion("lin", rep.place.T_pre, rep.place.q_pre, note="idle"))

        elif kind == "hold":
            fp = _finger_faces(payload)
            out.append(Motion("move", payload.T_pre, payload.q_pre, note="idle",  face=payload.face))
            out.append(Motion("lin",  payload.T_ee,  payload.q,     note="grasp", face=payload.face))
            out.append(Motion("close",                               note="close", fingers=fp))
            out.append(Motion("move", payload.T_pre, payload.q_pre, note="idle"))

        elif kind == "turn":
            move_str, hold_grasp = payload
            T_op = hold_grasp.T_op if hold_grasp.T_op is not None else hold_grasp.T_ee
            out.append(Motion("move", T_op,          note="operation", face=hold_grasp.face))
            out.append(Motion("turn", move=move_str, note="turn", face=hold_grasp.face))
            out.append(Motion("wait",                note="dwell 1s"))
            next_from_hold = next_kind == "reposition" and next_payload.from_hold
            if next_kind not in ("turn", "release") and not next_from_hold:
                out.append(Motion("move", hold_grasp.T_pre, hold_grasp.q_pre, note="idle"))

        elif kind == "release":
            if prev_kind == "turn":
                out.append(Motion("move", payload.T_pre, payload.q_pre, note="idle@release",
                                  face=payload.face))
            out.append(Motion("lin",  payload.T_ee, payload.q,     note="lower", face=payload.face))
            out.append(Motion("open",                               note="open"))
            if next_kind != "reposition":
                out.append(Motion("move", payload.T_pre, payload.q_pre, note="idle",
                                  face=payload.face))

    return out
