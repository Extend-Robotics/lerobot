"""PICO OpenXR skeletons to normalized Revo2 actions, independent of transport.

Mapping and kinematic assets follow unitreerobotics/xr_teleoperate at
817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b. See assets/brainco/LICENSE.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.utils.import_utils import _dex_retargeting_available, require_package

if TYPE_CHECKING or _dex_retargeting_available:
    from dex_retargeting.retargeting_config import RetargetingConfig

MOTOR_NAMES = ("motor_0", "motor_1", "index", "middle", "ring", "pinky")
JOINT_SUFFIXES = (
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
ANGLE_RANGES = np.array([1.52, 1.05, 1.47, 1.47, 1.47, 1.47])
HUMAN_INDICES = np.array(
    [
        [9, 14, 19, 24, 14, 19, 24, 19, 24, 24, 0, 0, 0, 0, 0],
        [4, 4, 4, 4, 9, 9, 9, 14, 14, 19, 4, 9, 14, 19, 24],
    ]
)


def hand_points(joints: np.ndarray) -> np.ndarray:
    """Drop OpenXR palm and express 25 points in the upstream hand frame."""
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (26, 7) or not np.isfinite(joints).all():
        raise ValueError("Expected a finite (26, 7) OpenXR hand skeleton")
    wrist = joints[1]
    if np.linalg.norm(wrist[3:]) < 1e-6:
        raise ValueError("Invalid wrist quaternion")
    local = (joints[1:, :3] - wrist[:3]) @ Rotation.from_quat(wrist[3:]).as_matrix()
    # Equivalent to T_TO_UNITREE_HAND @ R_ROBOT_OPENXR in TeleVuer.
    points = local[:, [1, 2, 0]]
    if np.any(np.linalg.norm(points[[4, 9, 14, 19, 24]], axis=1) < 1e-6):
        raise ValueError("Hand skeleton contains missing or collapsed fingertips")
    return points


class PicoRevo2Retargeter:
    """One stateful DexPilot solver per side; returns G1-compatible feature keys.

    A whole-system caller can pass skeletons from the same PicoHandInput snapshot
    used for arm IK. This class never reads the SDK or sends robot commands.
    """

    def __init__(
        self, sides: tuple[str, ...] = ("left", "right"), *, invert_thumb_base: tuple[str, ...] = ()
    ):
        if not sides or len(set(sides)) != len(sides) or any(s not in ("left", "right") for s in sides):
            raise ValueError("Select left, right, or both hands")
        if not set(invert_thumb_base).issubset(sides):
            raise ValueError("Thumb-base inversion must select a configured hand")
        require_package("dex-retargeting", extra="pico_revo2", import_name="dex_retargeting")
        self.sides = sides
        self.invert_thumb_base = frozenset(invert_thumb_base)
        self.solvers = {}
        self.motor_indices = {}
        for side in sides:
            names = [f"{side}_{suffix}" for suffix in JOINT_SUFFIXES]
            config = RetargetingConfig.from_dict(
                {
                    "type": "DexPilot",
                    "urdf_path": str(Path(__file__).parent / "assets" / "brainco" / f"brainco_{side}.urdf"),
                    "target_joint_names": names,
                    "wrist_link_name": "base_link",
                    "finger_tip_link_names": [
                        f"{side}_{finger}_tip" for finger in ("thumb", "index", "middle", "ring", "pinky")
                    ],
                    "target_link_human_indices": HUMAN_INDICES.tolist(),
                    "scaling_factor": 1.0,
                    "low_pass_alpha": 0.2,
                }
            )
            solver = config.build()
            self.solvers[side] = solver
            self.motor_indices[side] = [solver.joint_names.index(name) for name in names]

    def retarget(self, skeletons: dict[str, np.ndarray]) -> dict[str, float]:
        # Validate all input before advancing either stateful solver.
        points = {side: hand_points(skeletons[side]) for side in self.sides}
        action = {}
        for side in self.sides:
            reference = points[side][HUMAN_INDICES[1]] - points[side][HUMAN_INDICES[0]]
            q = np.asarray(self.solvers[side].retarget(reference), dtype=float)[self.motor_indices[side]]
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError(f"Invalid {side} Revo2 retargeting output")
            normalized = np.clip(q / ANGLE_RANGES, 0, 1)
            # Driver calibration belongs after anatomical retargeting: motor 0
            # is the thumb metacarpal/base, independent of the thumb bend motor.
            if side in self.invert_thumb_base:
                normalized[0] = 1.0 - normalized[0]
            action.update(
                {
                    f"{side}_hand.{name}.pos": float(value)
                    for name, value in zip(MOTOR_NAMES, normalized, strict=True)
                }
            )
        return action
