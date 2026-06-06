from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_cache: dict[str, Any] | None = None


def load_config(path: Path | str = _CONFIG_PATH) -> dict[str, Any]:
    global _cache
    if _cache is None:
        with open(path) as f:
            _cache = yaml.safe_load(f)
    return _cache


def _cfg() -> dict[str, Any]:
    return load_config()


def _make_T(block: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.array(block["rot"], dtype=float)
    T[:3,  3] = np.array(block["pos"], dtype=float)
    return T


# ── cube ──────────────────────────────────────────────────────────────────── #

def cube_side() -> float:
    return float(_cfg()["cube"]["side"])

def cube_half() -> float:
    return cube_side() / 2.0

def p_rest() -> np.ndarray:
    return np.array(_cfg()["cube"]["p_rest"], dtype=float)

def t_op() -> np.ndarray:
    return _make_T(_cfg()["cube"]["t_op"])

def lift_pose() -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array(_cfg()["cube"]["lift_pos"], dtype=float)
    return T


# ── gripper ───────────────────────────────────────────────────────────────── #

def pinch_z() -> float:
    return float(_cfg()["gripper"]["pinch_z"])

def grip_offset() -> float:
    return float(_cfg()["gripper"]["grip_offset"])

def standoff() -> float:
    return float(_cfg()["gripper"]["standoff"])

def gripper_open_width() -> float:
    return float(_cfg()["gripper"]["open_width"])

def gripper_finger_length() -> float:
    return float(_cfg()["gripper"]["finger_length"])

def gripper_n_samples() -> int:
    return int(_cfg()["gripper"]["n_samples"])

def ctrl_open() -> float:
    return float(_cfg()["gripper"]["ctrl_open"])

def ctrl_close() -> float:
    return float(_cfg()["gripper"]["ctrl_close"])


