"""Measured-pose PICO clutching and checked G1 arm IK, without owning transport."""

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _scipy_available, require_package

from .g1_utils import G1_29_JointArmIndex, action_to_pose

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

ARM_KEYS = tuple(f"{joint.name}.q" for joint in G1_29_JointArmIndex)
SIDES = ("left", "right")


def pose_to_quaternion(pose: np.ndarray) -> np.ndarray:
    """Convert a model-frame transform to PICO's [x,y,z,qw,qx,qy,qz,gripper]."""
    require_package("scipy", extra="scipy-dep")
    quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate((pose[:3, 3], np.roll(quaternion, 1), [0.0]))


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return relative rotation angle in radians."""
    return float(np.arccos(np.clip((np.trace(first @ second.T) - 1) / 2, -1, 1)))


# MVP tuning constants: metres, radians, and seconds.
TIMEOUT_S = 0.25
JOINT_VELOCITY = 0.3
MAX_TRACKING_ERROR = 0.25


class PicoArmSession:
    """One Pico-to-IK loop. Faults propagate to the caller for shutdown."""

    def __init__(self, robot: Any, teleop: Any, clock: Callable[[], float] = time.monotonic):
        if (
            robot.control_space != "cartesian_abs"
            or not robot.config.arm_commands_only
            or robot.controller is not None
            or robot.config.gravity_compensation
        ):
            raise ValueError("PICO requires absolute Cartesian arm-only control without gravity compensation")
        self.robot, self.teleop, self.clock = robot, teleop, clock
        self.solver = robot.arm_ik
        self._stamp: int | None = None
        self._stream_at = clock()
        self._last_command_at = clock()
        self._ready = self._started = False
        self._held_q: np.ndarray | None = None

    def initialize(self) -> None:
        """Connect Pico using measured wrist poses; wait for grip release before arming."""
        q = self._measured()
        poses = self.solver.forward_kinematics(q)
        self.teleop.connect(
            **{
                f"{side}_tcp_pose_quat": pose_to_quaternion(pose)
                for side, pose in zip(SIDES, poses, strict=True)
            }
        )
        self._stream_at = self._last_command_at = self.clock()

    def _measured(self) -> np.ndarray:
        observation = self.robot.get_observation()
        q = np.array([observation[key] for key in ARM_KEYS], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite arm feedback")
        return q

    def _stream(self, stamp: int) -> bool:
        now = self.clock()
        if stamp <= 0 or (self._stamp is not None and stamp < self._stamp):
            raise TimeoutError("Invalid or restarted PICO source timestamp")
        advanced = self._stamp is not None and stamp > self._stamp
        if advanced:
            self._stream_at = now
        self._stamp = stamp
        if now - self._stream_at > TIMEOUT_S:
            raise TimeoutError("PICO stream is stale")
        return advanced

    def tick(self) -> dict[str, Any]:
        """Read inputs, solve once, bound the joints, and optionally publish."""
        now = self.clock()
        q = self._measured()
        stamp, _ = self.teleop.get_control_status()
        fresh = self._stream(stamp)
        action = self.teleop.get_action()
        active = self.teleop.get_arm_enabled()
        if not self._ready:
            if not fresh or any(active.values()):
                return {"status": "waiting", "reason": "Release both grips with PICO streaming"}
            self._held_q = q.copy()
            poses = self.solver.forward_kinematics(q)
            for side, pose in zip(SIDES, poses, strict=True):
                self.teleop.reset_arm_to_pose(side, pose_to_quaternion(pose)[:7])
            self._last_command_at = self.clock()
            self._ready = True
            return {"status": "ready", "reason": "Squeeze a grip to control its arm"}
        assert self._held_q is not None
        if (
            self._started
            and not self.robot.config.read_only
            and np.max(np.abs(q - self._held_q)) > MAX_TRACKING_ERROR
        ):
            raise ValueError("Measured joints deviate from held commands")
        held_poses = self.solver.forward_kinematics(self._held_q)
        targets = []
        for side, held in zip(SIDES, held_poses, strict=True):
            if not active[side]:
                # Pico owns grip hysteresis. Rebase released arms to the actual
                # command, so the next engagement cannot resume a lagging target.
                self.teleop.reset_arm_to_pose(side, pose_to_quaternion(held)[:7])
            target = action_to_pose(action, side) if active[side] else held
            targets.append(target)
        candidate = self._held_q.copy()
        residuals = {}
        if any(active.values()):
            seed = self._held_q if self.robot.config.read_only else q
            candidate, _ = self.solver.solve_ik(*targets, seed, smooth=False)
            candidate = np.asarray(candidate, dtype=float).copy()
            if (
                not self.solver.last_solve_success
                or candidate.shape != (14,)
                or not np.isfinite(candidate).all()
            ):
                raise ValueError("IK failed or returned invalid joints")
            for index, side in enumerate(SIDES):
                if not active[side]:
                    candidate[index * 7 : (index + 1) * 7] = self._held_q[index * 7 : (index + 1) * 7]
            model = self.solver.reduced_robot.model
            order = self.solver._arm_reorder_pin_to_g1
            if np.any(candidate < model.lowerPositionLimit[order]) or np.any(
                candidate > model.upperPositionLimit[order]
            ):
                raise ValueError("IK violates model joint limits")
            residuals = {
                side: {
                    "position_m": float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
                    "rotation_rad": rotation_distance(target[:3, :3], actual[:3, :3]),
                }
                for side, target, actual in zip(
                    SIDES, targets, self.solver.forward_kinematics(candidate), strict=True
                )
            }
        # Recheck inputs after the native solve; never publish an expired candidate.
        stamp, grips = self.teleop.get_control_status()
        self._stream(stamp)
        if self.clock() - now > TIMEOUT_S:
            raise TimeoutError("IK exceeded control deadline")
        self._measured()
        for index, side in enumerate(SIDES):
            if grips[side] <= self.teleop.config.grip_disable_threshold:
                candidate[index * 7 : (index + 1) * 7] = self._held_q[index * 7 : (index + 1) * 7]
        command_dt = self.clock() - self._last_command_at
        if not 0 < command_dt <= TIMEOUT_S:
            raise TimeoutError("Control loop missed its deadline")
        step = float(np.max(np.abs(candidate - self._held_q)))
        if step > JOINT_VELOCITY * command_dt:
            candidate = self._held_q + (candidate - self._held_q) * (JOINT_VELOCITY * command_dt / step)
        joint_action = dict(zip(ARM_KEYS, candidate.tolist(), strict=True))
        self._started = self._started or any(active.values())
        if self._started and not self.robot.config.read_only:
            self.robot.send_arm_joint_action(joint_action, TIMEOUT_S, check_state_age=False)
        self._last_command_at = self.clock()
        self._held_q = candidate
        return {
            "status": ("executing" if self._started else "ready")
            if not self.robot.config.read_only
            else "preview",
            "active": active,
            "joint_action": joint_action,
            "ik_residuals": residuals,
        }
