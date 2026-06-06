from __future__ import annotations

from typing import Dict, List

import numpy as np

FACES    = ["R", "L", "U", "D", "F", "B"]
OPPOSITE = {"R": "L", "L": "R", "U": "D", "D": "U", "F": "B", "B": "F"}

N = {
    "R": np.array([0,  1, 0]), "L": np.array([0, -1, 0]),
    "U": np.array([0,  0, 1]), "D": np.array([0,  0, -1]),
    "F": np.array([1,  0, 0]), "B": np.array([-1, 0, 0]),
}

GRIP_OFFSET = 0.04   # gripper palm distance from face center [m]
STANDOFF    = 0.05   # pre-grasp back-off along approach axis [m]
P_REST      = np.array([0.0, 0.0, 0.0])
CUBE_SIDE   = 0.057
CUBE_HALF   = CUBE_SIDE / 2

T_OP: np.ndarray = np.array([
    [1., 0., 0., 0.20],
    [0., 1., 0., 0.00],
    [0., 0., 1., 0.30],
    [0., 0., 0., 1.  ],
], dtype=float)

LIFT_POSE = np.eye(4)
LIFT_POSE[:3, 3] = [0.0, 0.0, 0.20]


def Tz(d: float) -> np.ndarray:
    T = np.eye(4); T[2, 3] = d; return T


def _rot(axis: str, k: int) -> np.ndarray:
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s,  c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1,  0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T


def pre_grasp_pose(T_ee: np.ndarray) -> np.ndarray:
    return T_ee @ Tz(-STANDOFF)


def grasp_template(face: str) -> np.ndarray:
    """Gripper pose in cube body frame for `face`. Z = approach inward, X = finger axis."""
    n        = N[face].astype(float)
    approach = -n
    gravity  = np.array([0.0, 0.0, 1.0])
    if abs(approach @ gravity) > 0.9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = np.cross(approach, gravity)
        x /= np.linalg.norm(x)
    y = np.cross(approach, x)
    y /= np.linalg.norm(y)
    return _T(np.column_stack([x, y, approach]), n * GRIP_OFFSET)


def cube_orientations() -> List[np.ndarray]:
    """Closure of {Rx90, Ry90, Rz90} — 24 rotation matrices, index 0 = identity."""
    gens = [_rot("x", 1), _rot("y", 1), _rot("z", 1)]
    seen: Dict[tuple, np.ndarray] = {}
    I = np.eye(3, dtype=int)
    seen[tuple(I.flatten())] = I
    frontier = [I]
    while frontier:
        R = frontier.pop()
        for g in gens:
            R2  = g @ R
            key = tuple(R2.flatten())
            if key not in seen:
                seen[key] = R2
                frontier.append(R2)
    assert len(seen) == 24
    keys   = sorted(seen)
    id_key = tuple(I.flatten())
    keys.remove(id_key)
    return [seen[id_key]] + [seen[k] for k in keys]
