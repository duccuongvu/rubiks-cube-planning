from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from grasping import Grasp


class RobotModel(ABC):
    """Executor-agnostic feasibility oracle used by :class:`CubePlanner`.

    A concrete model decides whether the gripper can grasp a given cube face at
    a given orientation. The planner treats this as a pure, side-effect-free
    function and caches its results, so implementations must be deterministic
    and cheap.
    """

    @abstractmethod
    def solve_grasp(self, face: str, R: np.ndarray, p_rest: np.ndarray,
                    T_g_body: np.ndarray) -> Optional[Grasp]:
        """Return a :class:`Grasp` for `face` when the cube sits at rotation `R`
        and position `p_rest`, or ``None`` if that grasp is infeasible
        (unreachable, or in collision — e.g. ``Gripper.hits_table``).

        `T_g_body` is the grasp template in the cube body frame
        (``geometry.grasp_template(face)``).
        """
        raise NotImplementedError
