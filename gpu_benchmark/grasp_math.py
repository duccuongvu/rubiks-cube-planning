"""Shared constants + reference math for the grasp-feasibility check.

Mirrors ``MockRobot.solve_grasp`` (examples/mock_robot.py) and the geometry
helpers it depends on (geometry.py, grasping.py), using the real config.yaml
values. Kept dependency-free from the rest of the repo so this folder can be
run standalone in the `ai` conda env (which has torch/cuda but not this
project's own env).
"""
from __future__ import annotations

import numpy as np

# ── config.yaml values (hardcoded copy, see ../config.yaml) ─────────────────
CUBE_SIDE     = 0.057
CUBE_HALF     = CUBE_SIDE / 2.0
P_REST        = np.array([0.0, 0.0, 0.0])
GRIP_OFFSET   = 0.040
STANDOFF      = 0.050
OPEN_WIDTH    = 0.065
FINGER_LENGTH = 0.045
N_SAMPLES     = 3
MARGIN        = 0.0

FACES = ["R", "L", "U", "D", "F", "B"]
N = {
    "R": np.array([0.0,  1.0, 0.0]), "L": np.array([0.0, -1.0, 0.0]),
    "U": np.array([0.0,  0.0, 1.0]), "D": np.array([0.0,  0.0, -1.0]),
    "F": np.array([1.0,  0.0, 0.0]), "B": np.array([-1.0, 0.0, 0.0]),
}
REACH = [
    np.array([0.0,  1.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0,  0.0, 1.0]),
]


def grasp_template(face: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (R_g_body, t_g_body) — rotation + translation of the gripper
    pose in the cube body frame, i.e. geometry.grasp_template() split apart."""
    n = N[face]
    approach = -n
    gravity = np.array([0.0, 0.0, 1.0])
    if abs(approach @ gravity) > 0.9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = np.cross(approach, gravity)
        x /= np.linalg.norm(x)
    y = np.cross(approach, x)
    y /= np.linalg.norm(y)
    R_g = np.column_stack([x, y, approach])
    t_g = n * GRIP_OFFSET
    return R_g, t_g


def finger_points() -> np.ndarray:
    """Gripper-local finger sample points, shape (2*N_SAMPLES, 3)."""
    hw = OPEN_WIDTH / 2.0
    zs = np.linspace(0.0, FINGER_LENGTH, N_SAMPLES)
    return np.array([[s * hw, 0.0, z] for s in (-1.0, 1.0) for z in zs])


def random_rotation_matrices(n: int, rng: np.random.Generator) -> np.ndarray:
    """n proper (det=+1) rotation matrices via QR of random Gaussians."""
    A = rng.standard_normal((n, 3, 3))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diagonal(R, axis1=1, axis2=2))
    Q = Q * d[:, None, :]
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1.0
    return Q


def reference_loop_numpy(R_batch: np.ndarray) -> np.ndarray:
    """Ground-truth per-item Python loop, identical math to
    ``MockRobot.solve_grasp`` in examples/mock_robot.py. Shape (B,6) bool."""
    P = finger_points()
    table_z = P_REST[2] - CUBE_HALF
    templates = {f: grasp_template(f) for f in FACES}
    B = R_batch.shape[0]
    out = np.zeros((B, len(FACES)), dtype=bool)
    for b in range(B):
        R = R_batch[b]
        for fi, face in enumerate(FACES):
            n_world = R @ N[face]
            reachable = max(float(n_world @ r) for r in REACH) > 0.9
            if not reachable:
                continue
            R_g, t_g = templates[face]
            R_ee = R @ R_g
            p_ee = R @ t_g + P_REST
            world_z = (R_ee @ P.T)[2, :] + p_ee[2]
            hits_table = float(world_z.min()) < table_z + MARGIN
            out[b, fi] = not hits_table
    return out
