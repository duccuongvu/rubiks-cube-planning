"""
MuJoCo sim: kinematic 2F-85 gripper (freejoint) + real 3x3x3 Rubik cube.

Gripper control: freejoint on base_mount, qpos written directly — no constraint lag.
Grasp pinning:   T_tcp_to_cube latched on close, cube pose follows TCP each step.

Run:
    conda run -n manipsim python3 examples/mujoco_sim.py F R
    conda run -n manipsim python3 examples/mujoco_sim.py --no-render F R U
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from manipsim.core.paths import DEFAULT_SCENE_XML
from manipsim.core.scene import _merge_scene_settings

from geometry import CUBE_HALF, N, P_REST, T_OP, _T, pre_grasp_pose
from grasping import GRIPPER, Grasp
from motion import Motion, to_motion
from planner import CubePlanner
from robot import RobotModel
from utils import ctrl_close, ctrl_open, pinch_z

# --------------------------------------------------------------------------- #
#  Paths / constants                                                           #
# --------------------------------------------------------------------------- #

GRIPPER_XML      = ROOT / "robotiq_2f85" / "2f85.xml"
CUBE_XML         = ROOT / "cube" / "cube_3x3x3.xml"
ROTATE_TABLE_XML = ROOT / "rotate_table" / "rotate_table.xml"

PINCH_Z     = pinch_z()
TABLE_POS   = [0.0, 0.0, 0.5] 
TABLE_EULER = [180.0, 0.0, 0.0]
HOME_T_EE   = np.array([
    [1.,  0.,  0.,  0.00],
    [0., -1.,  0.,  0.00],
    [0.,  0., -1.,  0.30],
    [0.,  0.,  0.,  1.  ],
])

# --------------------------------------------------------------------------- #
#  Scene builder                                                               #
# --------------------------------------------------------------------------- #

def _build_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec()
    spec.modelname = "gripper_cube"
    spec.option.timestep = 0.002
    spec.option.gravity = [0, 0, -9.81]
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    ground = mujoco.MjSpec.from_file(str(DEFAULT_SCENE_XML))
    _merge_scene_settings(spec, ground)
    scene = spec.worldbody.add_body()
    scene.name = "scene"
    scene.add_frame().attach_body(ground.worldbody, "scene", "")

    g_spec = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    g_root = spec.worldbody.add_body()
    g_root.name = "gripper_root"
    g_root.add_freejoint().name = "gripper_free"
    g_root.add_frame().attach_body(g_spec.worldbody, "gripper", "")

    c_spec = mujoco.MjSpec.from_file(str(CUBE_XML))
    c_root = spec.worldbody.add_body()
    c_root.name = "cube_root"
    c_root.pos = [0.0, 0.0, CUBE_HALF]
    c_root.add_freejoint().name = "cube_free"
    c_root.add_frame().attach_body(c_spec.worldbody, "cube", "")

    # Rotate table: fixed at TABLE_POS, Z axis pointing down (180° around X)
    t_spec   = mujoco.MjSpec.from_file(str(ROTATE_TABLE_XML))
    t_anchor = spec.worldbody.add_body()
    t_anchor.name  = "table_anchor"
    t_anchor.pos   = TABLE_POS
    t_anchor.quat  = [0.0, 1.0, 0.0, 0.0]   # wxyz: 180° around X → Z points down
    t_anchor.add_frame().attach_body(t_spec.worldbody, "rt", "")

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data

# --------------------------------------------------------------------------- #
#  Pose helpers                                                                #
# --------------------------------------------------------------------------- #

def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _slerp_T(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3]  = (1 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    rots      = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    T[:3, :3] = Slerp([0.0, 1.0], rots)([alpha]).as_matrix()[0]
    return T


def _ee_to_base_mount(T_ee: np.ndarray) -> np.ndarray:
    T = T_ee.copy()
    T[:3, 3] = T_ee[:3, 3] - T_ee[:3, 2] * PINCH_Z
    return T



def _fmt_pose(T: np.ndarray) -> str:
    pos = T[:3, 3]
    return (f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})  "
            f"approach={T[:3,2].round(2)}  fingers={T[:3,0].round(2)}")

# --------------------------------------------------------------------------- #
#  Simulator                                                                   #
# --------------------------------------------------------------------------- #

class GripperCubeSim:
    DT         = 0.002
    CTRL_OPEN  = ctrl_open()
    CTRL_CLOSE = ctrl_close()
    MOVE_SPEED = 0.25
    LIN_SPEED  = 0.10
    MIN_STEPS  = 60

    def __init__(self, render: bool = True):
        self._render         = render
        self._grasped        = False
        self._T_tcp_to_cube: np.ndarray | None = None
        self._current_T_ee:  np.ndarray | None = None
        self._pending_turn:  tuple | None       = None

        self.model, self.data = _build_scene()

        def bid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,     name)
        def sid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,     name)
        def aid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        def jid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    name)

        self._bid_gripper      = bid("gripper_root")
        self._qpos_adr_gripper = self.model.jnt_qposadr[jid("gripper_free")]
        self._dof_adr_gripper  = self.model.body_dofadr[self._bid_gripper]
        self._sid_pinch        = sid("gripperpinch")
        self._aid_fingers      = aid("gripperfingers_actuator")

        self._bid_cube      = bid("cube_root")
        self._qpos_adr_cube = self.model.jnt_qposadr[jid("cube_free")]
        self._dof_adr_cube  = self.model.body_dofadr[self._bid_cube]

        self._aid_table_motor = aid("rttable_motor")
        self._table_angle     = 0.0   # cumulative hinge angle [rad]

        self.set_pose(HOME_T_EE)
        mujoco.mj_forward(self.model, self.data)

        self._viewer = None
        if render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _write_gripper_qpos(self, T_base: np.ndarray) -> None:
        q    = self._qpos_adr_gripper
        xyzw = Rotation.from_matrix(T_base[:3, :3]).as_quat()
        self.data.qpos[q:q+3] = T_base[:3, 3]
        self.data.qpos[q+3]   = xyzw[3]
        self.data.qpos[q+4]   = xyzw[0]
        self.data.qpos[q+5]   = xyzw[1]
        self.data.qpos[q+6]   = xyzw[2]
        self.data.qvel[self._dof_adr_gripper:self._dof_adr_gripper + 6] = 0.0

    def set_pose(self, T_ee: np.ndarray) -> None:
        self._write_gripper_qpos(_ee_to_base_mount(T_ee))
        self._current_T_ee = T_ee.copy()

    def get_pinch_pose(self) -> np.ndarray:
        sid = self._sid_pinch
        T = np.eye(4)
        T[:3, 3]  = self.data.site_xpos[sid].copy()
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
        return T

    def get_cube_pose(self) -> np.ndarray:
        bid = self._bid_cube
        T = np.eye(4)
        T[:3, 3]  = self.data.xpos[bid].copy()
        T[:3, :3] = _quat_wxyz_to_mat(self.data.xquat[bid].copy())
        return T

    def set_cube_pose(self, T: np.ndarray) -> None:
        q    = self._qpos_adr_cube
        xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        self.data.qpos[q:q+3] = T[:3, 3]
        self.data.qpos[q+3]   = xyzw[3]
        self.data.qpos[q+4]   = xyzw[0]
        self.data.qpos[q+5]   = xyzw[1]
        self.data.qpos[q+6]   = xyzw[2]
        self.data.qvel[self._dof_adr_cube:self._dof_adr_cube + 6] = 0.0

    def open_fingers(self) -> None:
        self.data.ctrl[self._aid_fingers] = self.CTRL_OPEN

    def close_fingers(self) -> None:
        self.data.ctrl[self._aid_fingers] = self.CTRL_CLOSE

    def latch_grasp(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self._T_tcp_to_cube = np.linalg.inv(self.get_pinch_pose()) @ self.get_cube_pose()
        self._grasped = True

    def release_grasp(self) -> None:
        self._grasped       = False
        self._T_tcp_to_cube = None

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self._current_T_ee is not None:
                self.set_pose(self._current_T_ee)
            if self._grasped and self._T_tcp_to_cube is not None:
                mujoco.mj_forward(self.model, self.data)
                self.set_cube_pose(self.get_pinch_pose() @ self._T_tcp_to_cube)

            t0 = time.time()
            mujoco.mj_step(self.model, self.data)

            if self._current_T_ee is not None:
                self.set_pose(self._current_T_ee)
            mujoco.mj_forward(self.model, self.data)

            if self._viewer is not None:
                self._viewer.sync()
                remaining = self.DT - (time.time() - t0)
                if remaining > 0.0:
                    time.sleep(remaining)

    def wait(self, duration: float) -> None:
        self.step(max(1, round(duration / self.DT)))

    def move_to(self, T_ee: np.ndarray, speed: float | None = None) -> None:
        T0 = self._current_T_ee
        if T0 is None:
            self.set_pose(T_ee); self.step(5); return
        spd  = speed if speed is not None else self.MOVE_SPEED
        dist = float(np.linalg.norm(T_ee[:3, 3] - T0[:3, 3]))
        n    = max(self.MIN_STEPS, round(max(0.3, dist / spd) / self.DT))
        for i in range(n + 1):
            self.set_pose(_slerp_T(T0, T_ee, i / n))
            self.step(1)

    def lin_to(self, T_ee: np.ndarray) -> None:
        self.move_to(T_ee, speed=self.LIN_SPEED)

    def execute(self, m: Motion) -> None:
        if m.kind == "turn":
            print(f"  [turn ] {m.move}  (face={m.face})")
            # Store for the following "wait" motion to animate
            self._pending_turn = (m.face, m.move)
            return

        parts = [m.note or m.kind]
        if m.face: parts.append(f"face={m.face}")
        label = " | ".join(parts)

        if m.kind in ("move", "lin") and m.T_ee is not None:
            print(f"  [{m.kind:5}] {label}")
            print(f"           {_fmt_pose(m.T_ee)}")
        else:
            print(f"  [{m.kind:5}] {label}")

        if m.kind == "move":
            self.move_to(m.T_ee)
        elif m.kind == "lin":
            self.lin_to(m.T_ee)
        elif m.kind == "close":
            self.close_fingers()
            self.wait(0.5)
            self.latch_grasp()
        elif m.kind == "open":
            self.release_grasp()
            self.open_fingers()
            self.wait(0.3)
        elif m.kind == "wait":
            pending = getattr(self, "_pending_turn", None)
            if pending is not None:
                hold_face, move_str = pending
                self._pending_turn = None
                print(f"  [table] spin {move_str}")
                self.animate_table_turn(hold_face, move_str)
            else:
                self.wait(1.0)

    # ------------------------------------------------------------------
    #  Rotate-table helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_move(move_str: str):
        suffix = move_str[1:] if len(move_str) > 1 else ""
        angle  = np.pi if suffix == "2" else (-np.pi / 2 if "'" in suffix else np.pi / 2)
        return move_str[0], angle

    def _spin_table(self, angle: float, duration: float) -> None:
        """Spin table hinge by `angle` [rad] over `duration` [s]. Cube untouched."""
        n  = max(1, round(duration / self.DT))
        a0 = self._table_angle
        for i in range(1, n + 1):
            self.data.ctrl[self._aid_table_motor] = a0 + angle * i / n
            self.step(1)
        self._table_angle += angle

    def animate_table_turn(self, hold_face: str, move_str: str) -> None:
        """Spin table for the turn move. Table fixed at TABLE_POS, cube untouched."""
        _, angle = self._parse_move(move_str)
        self._spin_table(angle, 1.0)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

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

    def _reachable(self, n_world: np.ndarray) -> bool:
        return max(float(n_world @ r) for r in self.REACH) > 0.9

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
#  Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("moves", nargs="*", default=["F"], help="Rubik moves e.g. F R U")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    moves  = [m.upper() for m in args.moves]
    render = not args.no_render

    p_rest = np.array([0.0, 0.0, CUBE_HALF])
    planner = CubePlanner(MockRobot(), p_rest=p_rest)
    print(f"Planning: {moves}")
    actions, final_ori = planner.plan_sequence(moves)
    motions = to_motion(actions)
    print(f"{len(motions)} waypoints planned\n")

    sim = GripperCubeSim(render=render)
    sim.wait(0.5)

    print("Executing motions...")
    for m in motions:
        sim.execute(m)

    print("\nReturn to home...")
    sim.move_to(HOME_T_EE)
    print(f"\nDone. Final orientation: {final_ori}")

    if render and sim._viewer is not None:
        print("Viewer open — close window to exit.")
        while sim._viewer.is_running():
            sim.step(1)

    sim.close()
