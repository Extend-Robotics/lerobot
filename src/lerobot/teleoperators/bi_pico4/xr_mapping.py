# Copyright 2025 HangZhou YuShu TECHNOLOGY CO.,LTD. (Unitree Robotics)
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modified from xr_teleoperate 817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b
# and televuer 766de45e74373ae0ea66321d942ce538385655a5 (MIT; see
# examples/unitree_g1d/UNITREE_XR_NOTICES.md for the complete upstream notices).
"""Pure Unitree G1-29 XR coordinate mapping for PICO tracking samples."""

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _scipy_available, require_package

SIDES = ("left", "right")

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

# televuer's OpenXR -> robot change of basis, applied on both sides of SE(3).
ROBOT_FROM_XR = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)


def sdk_pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert native PICO xyzw quaternion poses to OpenXR SE(3) matrices."""
    require_package("scipy", extra="scipy-dep")
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-6:
        raise ValueError("Expected a finite XR pose with a nonzero xyzw quaternion")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def head_yaw_rotation(head_rotation: np.ndarray) -> np.ndarray:
    """televuer get_Brobot_world_head_yaw_rot, including its degenerate fallback."""
    x_axis = head_rotation[:, 0].copy()
    x_axis[2] = 0.0
    x_norm = np.linalg.norm(x_axis)
    if not np.isfinite(x_norm) or np.isclose(x_norm, 0.0, atol=1e-6):
        return np.eye(3)
    x_axis /= x_norm
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def map_xr_wrist_targets(poses: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Map controller poses into Unitree arm tool frames."""
    head = ROBOT_FROM_XR @ sdk_pose_to_matrix(poses["head"]) @ ROBOT_FROM_XR.T
    yaw = head_yaw_rotation(head[:3, :3])
    targets = []
    for side in SIDES:
        wrist = ROBOT_FROM_XR @ sdk_pose_to_matrix(poses[side]) @ ROBOT_FROM_XR.T
        wrist[:3, :3] = yaw.T @ wrist[:3, :3]
        wrist[:3, 3] = yaw.T @ (wrist[:3, 3] - head[:3, 3])
        # Upstream fixed head -> waist origin translation, in metres.
        wrist[0, 3] += 0.15
        wrist[2, 3] += 0.45
        targets.append(wrist)
    return targets[0], targets[1]
