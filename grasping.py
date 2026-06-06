from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Gripper:
    open_width:    float = 0.065
    finger_length: float = 0.045
    n_samples:     int   = 3

    def finger_points(self) -> np.ndarray:
        hw = self.open_width / 2.0
        zs = np.linspace(0.0, self.finger_length, self.n_samples)
        return np.array([[s * hw, 0.0, z] for s in (-1.0, 1.0) for z in zs])

    def world_points(self, T_ee: np.ndarray) -> np.ndarray:
        P = self.finger_points()
        return (T_ee[:3, :3] @ P.T).T + T_ee[:3, 3]

    def hits_table(self, T_ee: np.ndarray, table_z: float, margin: float = 0.0) -> bool:
        return float(self.world_points(T_ee)[:, 2].min()) < table_z + margin


GRIPPER = Gripper()


@dataclass
class Grasp:
    face:   str
    T_ee:   np.ndarray
    q:      Optional[np.ndarray]
    T_pre:  Optional[np.ndarray] = None
    q_pre:  Optional[np.ndarray] = None
    R_cube: Optional[np.ndarray] = None
    T_op:   Optional[np.ndarray] = None   # world-frame pose for face-turn operation
