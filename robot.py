from __future__ import annotations

from typing import Optional

import numpy as np

from grasping import Grasp


class RobotModel:
    def solve_grasp(self, face: str, R: np.ndarray, p_rest: np.ndarray,
                    T_g_body: np.ndarray) -> Optional[Grasp]:
        raise NotImplementedError

    def can_turn(self, move: str, R: np.ndarray) -> bool:
        raise NotImplementedError
