# Copyright 2025 HangZhou YuShu TECHNOLOGY CO.,LTD. (Unitree Robotics)
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modified from xr_teleoperate 817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b
# and televuer 766de45e74373ae0ea66321d942ce538385655a5 (MIT; see
# examples/unitree_g1d/UNITREE_XR_NOTICES.md for the complete upstream notices).
"""Unitree G1-29 XR mapping and arm control, adapted to PICO SDK and our bridge."""

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _scipy_available, require_package

from .pico_teleop import ARM_KEYS, SIDES, TIMEOUT_S, rotation_distance

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation

# televuer's OpenXR -> robot change of basis, applied on both sides of SE(3).
ROBOT_FROM_XR = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
CONTROL_DT = 1 / 250
ARM_VELOCITY_LIMIT = 30.0
ARM_KP = ([80.0] * 4 + [40.0] * 3) * 2
ARM_KD = ([3.0] * 4 + [1.5] * 3) * 2


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
    """Port televuer's controller/head_yaw path, without clutch offsets or scaling."""
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


def clip_arm_target(target: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """G1_29_ArmController.clip_arm_q_target, at its original 250 Hz / 30 rad/s."""
    delta = target - measured
    motion_scale = np.max(np.abs(delta)) / (ARM_VELOCITY_LIMIT * CONTROL_DT)
    return measured + delta / max(motion_scale, 1.0)


class UnitreeXrArmSession:
    """30 Hz upstream IK targets with an independent 250 Hz command publisher.

    The hardware adapters retain read-only preview, passive legs, input expiry,
    explicit start, and fault cleanup. Preview uses ideal virtual joint feedback.
    """

    def __init__(self, robot: Any, teleop: Any, clock: Callable[[], float] = time.monotonic):
        if (
            robot.config.arm_ik_profile != "unitree_xr"
            or robot.control_space != "cartesian_abs"
            or not robot.config.arm_commands_only
            or robot.controller is not None
        ):
            raise ValueError("Unitree XR requires its IK profile and Cartesian arm-only mode")
        self.robot, self.teleop, self.clock = robot, teleop, clock
        self.solver = robot.arm_ik
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._following = False
        self._error: Exception | None = None
        self._stamp: int | None = None
        self._source_at = self._target_at = clock()
        self._target_q: np.ndarray | None = None
        self._target_tau = np.zeros(14)
        self._preview_q: np.ndarray | None = None

    def _measured(self) -> tuple[np.ndarray, np.ndarray]:
        observation = self.robot.get_observation()
        q = np.array([observation[key] for key in ARM_KEYS], dtype=float)
        dq = np.array([observation.get(key.replace(".q", ".dq"), 0.0) for key in ARM_KEYS], dtype=float)
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError("Nonfinite arm feedback")
        return q, dq

    def initialize(self) -> None:
        """Connect raw PICO tracking and initialize commands from measured joints."""
        q, _ = self._measured()
        self.teleop.connect()
        self._target_q = self._preview_q = q.copy()
        self._source_at = self._target_at = self.clock()
        self.tick()  # Require valid headset and controller poses before starting.

    def start(self) -> None:
        """Equivalent to upstream's explicit 'r' start; both arms follow together."""
        with self._lock:
            self._following = True
            self._target_at = self.clock()

    def start_publisher(self) -> None:
        """Run publication separately from native IK so solve jitter cannot set its rate."""
        if self._thread is not None:
            raise RuntimeError("Publisher already started")
        self._thread = threading.Thread(target=self._publish_loop, name="g1-xr-publisher", daemon=True)
        self._thread.start()

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"XR publisher stopped: {self._error}") from self._error

    def tick(self) -> dict[str, Any]:
        """Read absolute controller poses and solve using upstream parameters/filtering."""
        self._check()
        began = self.clock()
        stamp, poses = self.teleop.get_xr_sample()
        with self._lock:
            if stamp <= 0 or (self._stamp is not None and stamp < self._stamp):
                raise TimeoutError("Invalid or restarted PICO source timestamp")
            if self._stamp is None or stamp > self._stamp:
                self._source_at = began
            self._stamp = stamp
            if began - self._source_at > TIMEOUT_S:
                raise TimeoutError("PICO stream is stale")
            following = self._following
            preview_q = None if self._preview_q is None else self._preview_q.copy()
        targets = map_xr_wrist_targets(poses)
        if not following:
            return {"status": "ready", "reason": "Press r then Enter to follow both controllers"}
        measured, dq = self._measured()
        seed = preview_q if self.robot.config.read_only else measured
        candidate, tau = self.solver.solve_ik(*targets, seed, dq)
        candidate, tau = np.asarray(candidate, dtype=float), np.asarray(tau, dtype=float)
        if (
            not self.solver.last_solve_success
            or candidate.shape != (14,)
            or tau.shape != (14,)
            or not np.isfinite(candidate).all()
            or not np.isfinite(tau).all()
        ):
            raise ValueError("IK failed or returned invalid arm commands")
        self._check()
        if self.clock() - began > TIMEOUT_S:
            raise TimeoutError("IK exceeded control deadline")
        actual = self.solver.forward_kinematics(candidate)
        with self._lock:
            self._target_q, self._target_tau = candidate.copy(), tau.copy()
            self._target_at = self.clock()
        return {
            "status": "preview" if self.robot.config.read_only else "executing",
            "joint_action": dict(zip(ARM_KEYS, candidate.tolist(), strict=True)),
            "ik_residuals": {
                side: {
                    "position_m": float(np.linalg.norm(target[:3, 3] - pose[:3, 3])),
                    "rotation_rad": rotation_distance(target[:3, :3], pose[:3, :3]),
                }
                for side, target, pose in zip(SIDES, targets, actual, strict=True)
            },
        }

    def publish_once(self) -> np.ndarray | None:
        """One upstream motor-control tick, using the latest measured joint snapshot."""
        with self._lock:
            if not self._following or self._stop.is_set():
                return None
            if self.clock() - min(self._source_at, self._target_at) > TIMEOUT_S:
                raise TimeoutError("XR input or IK target is stale")
            assert self._target_q is not None and self._preview_q is not None
            target, tau = self._target_q.copy(), self._target_tau.copy()
        if self.robot.config.read_only:
            # Upstream simulation bypasses clipping; use the same behavior in preview.
            with self._lock:
                self._preview_q = target.copy()
            return target
        measured, _ = self._measured()
        command = clip_arm_target(target, measured)
        self.robot.send_arm_joint_action(
            dict(zip(ARM_KEYS, command.tolist(), strict=True)), TIMEOUT_S, arm_tau=tau
        )
        return command

    def _publish_loop(self) -> None:
        try:
            while not self._stop.is_set():
                began = self.clock()
                self.publish_once()
                self._stop.wait(max(0.0, CONTROL_DT - (self.clock() - began)))
        except Exception as exc:
            self._error = exc
            self._stop.set()
            if not self.robot.config.read_only:
                self.robot._send_zero_torque()

    def close(self) -> None:
        """Join the sole command writer before robot disconnect makes the arms passive."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("XR publisher did not stop")
