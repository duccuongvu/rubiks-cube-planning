"""Batched (CPU or CUDA) grasp-feasibility check via torch tensor ops.

Vectorizes the same math as grasp_math.reference_loop_numpy across the batch
dimension AND the 6 faces at once, so it can run on GPU. This is the
"embarrassingly parallel" piece identified in CubePlanner: solve_grasp(face,
orientation) is independent per (face, orientation) pair.
"""
from __future__ import annotations

import numpy as np
import torch

from grasp_math import (CUBE_HALF, FACES, FINGER_LENGTH, GRIP_OFFSET, MARGIN,
                         N, N_SAMPLES, OPEN_WIDTH, P_REST, REACH, STANDOFF,
                         grasp_template)


def _face_constants(device, dtype):
    """Precompute the fixed (face-indexed) tensors once per device."""
    N_mat  = torch.tensor(np.array([N[f] for f in FACES]), device=device, dtype=dtype)  # (6,3)
    REACH_mat = torch.tensor(np.array(REACH), device=device, dtype=dtype)          # (3,3)

    Rg_list, tg_list = [], []
    for f in FACES:
        Rg, tg = grasp_template(f)
        Rg_list.append(Rg)
        tg_list.append(tg)
    Rg_body = torch.tensor(np.array(Rg_list), device=device, dtype=dtype)   # (6,3,3)
    tg_body = torch.tensor(np.array(tg_list), device=device, dtype=dtype)   # (6,3)

    hw = OPEN_WIDTH / 2.0
    zs = torch.linspace(0.0, FINGER_LENGTH, N_SAMPLES, device=device, dtype=dtype)
    xs = torch.tensor([-hw, hw], device=device, dtype=dtype)
    P = torch.stack(torch.meshgrid(xs, zs, indexing="ij"), dim=-1).reshape(-1, 2)  # (S,2) -> (x,z)
    P = torch.stack([P[:, 0], torch.zeros_like(P[:, 0]), P[:, 1]], dim=-1)         # (S,3)

    p_rest = torch.tensor(P_REST, device=device, dtype=dtype)
    table_z = p_rest[2] - CUBE_HALF
    return N_mat, REACH_mat, Rg_body, tg_body, P, p_rest, table_z


def batched_solve_grasp(R_batch: torch.Tensor, device: str, dtype=torch.float32) -> torch.Tensor:
    """R_batch: (B,3,3) rotation matrices (any device/dtype, will be moved).
    Returns bool feasibility mask (B,6) on `device`."""
    R_batch = R_batch.to(device=device, dtype=dtype)
    N_mat, REACH_mat, Rg_body, tg_body, P, p_rest, table_z = _face_constants(device, dtype)

    n_world   = torch.einsum("bij,fj->bfi", R_batch, N_mat)              # (B,6,3)
    reach_dot = torch.einsum("bfi,ri->bfr", n_world, REACH_mat)          # (B,6,3)
    reachable = reach_dot.max(dim=-1).values > 0.9                       # (B,6)

    R_ee = torch.einsum("bij,fjk->bfik", R_batch, Rg_body)               # (B,6,3,3)
    p_ee = torch.einsum("bij,fj->bfi", R_batch, tg_body) + p_rest        # (B,6,3)

    world_pts = torch.einsum("bfij,sj->bfsi", R_ee, P)                   # (B,6,S,3)
    world_z   = world_pts[..., 2] + p_ee[..., 2:3]                       # (B,6,S)
    min_z     = world_z.min(dim=-1).values                               # (B,6)
    hits_table = min_z < (table_z + MARGIN)

    return reachable & (~hits_table)
