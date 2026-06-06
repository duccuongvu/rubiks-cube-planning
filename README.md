# rubiks-cube-planning

Motion planning and MuJoCo simulation for solving a Rubik's cube with a single robotic gripper and a motorized rotate table.

## Overview

The system plans the minimal sequence of gripper repositions and face turns needed to execute any Rubik's cube move sequence. A MuJoCo simulation visualizes the full motion with a Robotiq 2F-85 gripper and a motorized rotate table.

```
Rubik move sequence  →  CubePlanner  →  motion waypoints  →  MuJoCo sim
                         (BFS over                            (gripper + table)
                          24 orientations)
```

## Repository structure

```
config.yaml              — tunable physical parameters (cube, gripper)
utils.py                 — config loader; typed accessors for all parameters
geometry.py              — cube geometry, face normals, grasp templates, SE(3) helpers
grasping.py              — Gripper collision model, Grasp dataclass
robot.py                 — RobotModel abstract interface
planner.py               — CubePlanner: BFS reposition planner over 24 cube orientations
motion.py                — to_motion(): converts planner actions → Motion waypoint list
examples/
  mujoco_sim.py          — GripperCubeSim: MuJoCo scene + kinematic gripper execution
  mock_robot.py          — MockRobot (no IK) + planner demo / orientation visualizer
cube/
  cube_3x3x3.xml         — MuJoCo Rubik's cube model
robotiq_2f85/
  2f85.xml               — Robotiq 2F-85 gripper model
rotate_table/
  rotate_table.xml       — Motorized rotate table (hinge around Z, position-controlled)
```

## Setup

Requires the `manipsim` conda environment (MuJoCo 3.x, numpy, scipy):

```bash
conda activate manipsim
```

## Running

### MuJoCo simulation

```bash
# from repo root
conda run -n manipsim python3 examples/mujoco_sim.py F R U
conda run -n manipsim python3 examples/mujoco_sim.py --no-render F R U
```

Executes the given Rubik moves in simulation. The gripper repositions the cube as needed; the rotate table spins independently on each face turn.

### Planner demo (no simulation)

```bash
python3 examples/mock_robot.py
```

Prints the full action sequence and motion waypoints for a hardcoded move set, with emoji cube-orientation visualization.

## Core concepts

### Cube orientations

The cube has 24 distinct orientations (rotation group SO(3) restricted to 90° steps). `CubePlanner` builds a graph over all 24 and uses BFS to find the shortest reposition path before each move.

### Action stream

`CubePlanner.plan_sequence()` returns a flat action list:

| Action | Meaning |
|--------|---------|
| `reposition` | Pick cube from one face, place it at another to reach a turn-ready orientation |
| `hold` | Close gripper on the opposite face of the upcoming turn |
| `turn` | Execute the Rubik face turn (rotate table spins) |
| `release` | Open gripper |

### Motion waypoints

`to_motion()` expands actions into `Motion` objects with `kind ∈ {move, lin, close, open, turn, wait}`. The sim executes each in order.

### Rotate table

Fixed at `[0, 0, 0.5]` in world frame, Z axis pointing down. Position-controlled hinge actuator. Activates only on `turn` commands — spins independently of the cube.

### Grasp model

Gripper approach direction = inward face normal. Pre-grasp pose backs off `standoff` distance along approach axis. Collision with the support table is checked via finger-point sampling.

## Configuration

Edit `config.yaml` to change physical parameters:

```yaml
cube:
  side: 0.057          # cube side length [m]
  p_rest: [0,0,0]      # cube rest position
  t_op: ...            # gripper pose during face turn

gripper:
  pinch_z: 0.125       # base → pinch site offset [m]
  grip_offset: 0.040   # palm distance from face center [m]
  standoff: 0.050      # pre-grasp back-off [m]
  ctrl_open/close: ... # finger actuator values
```

Sim timing, speeds, and scene poses are hardcoded in `mujoco_sim.py`.

## Extending

**New robot**: subclass `RobotModel`, implement `solve_grasp()` (returns `Grasp` or `None`) and `can_turn()`. Pass instance to `CubePlanner`.

**New move sequences**: pass any standard Rubik notation to `plan_sequence()` — moves from `{R L U D F B}` with optional `'` (prime) or `2` suffix.
