"""Clutch-relative Tianji wrist tracking using the Wuji PICO parameters."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep, TransitionKey
from lerobot.robots.tianji.kinematics import TianjiKinematics
from lerobot.robots.tianji.tianji import JOINT_NAMES

from .wrist_calibration import WristCalibration
from .xr_mapping import ROBOT_FROM_XR, map_xr_wrist_targets, sdk_pose_to_matrix

SIDES = ("left", "right")
POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
CHEST_FROM_XR = tuple(
    Rotation.from_euler("x", angle, degrees=True).as_matrix() @ ROBOT_FROM_XR[:3, :3] for angle in (90, -90)
)
logger = logging.getLogger(__name__)


@dataclass
class WristTarget:
    controller: np.ndarray
    flange: np.ndarray
    seed: np.ndarray
    stamp: float
    basis: np.ndarray = field(default_factory=lambda: np.eye(3))
    pose: np.ndarray = field(init=False)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    angular_speed: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.pose = self.flange.copy()

    def update(self, controller: np.ndarray, basis: np.ndarray, stamp: float) -> np.ndarray:
        dt = stamp - self.stamp
        if dt <= 0:
            return self.pose
        self.stamp = stamp
        pos = self.flange[:3, 3] + basis @ (controller[:3, 3] - self.controller[:3, 3])
        delta = basis @ controller[:3, :3] @ self.controller[:3, :3].T @ basis.T
        target_rot = delta @ self.flange[:3, :3]
        position_error = pos - self.pose[:3, 3]
        rotation_error = Rotation.from_matrix(self.pose[:3, :3].T @ target_rot).as_rotvec()

        # Wuji One-Euro filters, with the cadence of the samples actually consumed.
        derivative_alpha = self._alpha(1.0, dt)
        self.velocity += derivative_alpha * (position_error / dt - self.velocity)
        self.angular_speed += derivative_alpha * (np.linalg.norm(rotation_error) / dt - self.angular_speed)
        self.pose[:3, 3] += self._alpha(1 + 14 * np.linalg.norm(self.velocity), dt) * position_error
        rot_alpha = self._alpha(1 + 2.8 * self.angular_speed, dt)
        self.pose[:3, :3] = self.pose[:3, :3] @ Rotation.from_rotvec(rot_alpha * rotation_error).as_matrix()
        return self.pose

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return dt / (dt + 1 / (2 * np.pi * cutoff))


@ProcessorStepRegistry.register("pico_to_tianji_joints")
@dataclass
class PicoToTianjiJoints(RobotActionProcessorStep):
    solver: TianjiKinematics = field(repr=False)
    last_sent_action: Callable[[], RobotAction | None] | None = field(default=None, repr=False)
    calibration: WristCalibration | None = None
    timeout_s: float = 0.5
    max_target_delta_deg: float = 2.0
    max_seed_delta_deg: float = 15.0
    grip_enable_threshold: float = 0.5
    grip_disable_threshold: float = 0.3
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _wrists: list[WristTarget | None] = field(default_factory=lambda: [None, None], init=False)
    _await_release: list[bool] = field(default_factory=lambda: [False, False], init=False)
    _ready: bool = field(default=False, init=False)
    _command: np.ndarray | None = field(default=None, init=False)
    _grips: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    diagnostics: dict[str, dict] = field(default_factory=dict, init=False)

    def reset(self):
        self._wrists = [None, None]
        self._await_release = [False, False]
        self._ready = False
        self._command = None
        self._grips = np.zeros(2)
        self.diagnostics = {}

    @property
    def recording_features(self) -> dict[str, dict[str, Any]]:
        return {
            key: {"dtype": "float32", "shape": (2,), "names": list(SIDES)}
            for key in ("teleop.grip", "teleop.clutch")
        }

    @property
    def recording_frame(self) -> dict[str, np.ndarray]:
        return {
            "teleop.grip": self._grips.astype(np.float32),
            "teleop.clutch": np.array([wrist is not None for wrist in self._wrists], dtype=np.float32),
        }

    def released(self, action: RobotAction) -> bool:
        return all(0 <= action[f"{side}.grip"] <= self.grip_disable_threshold for side in SIDES)

    def action(self, action: RobotAction) -> RobotAction:
        stamp = action["sample_time"]
        self._check_age(stamp)
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Tianji Pico teleoperation requires measured joints")
        measured = np.array([observation[key] for key in JOINT_NAMES], dtype=float).reshape(2, 7)
        if not np.isfinite(measured).all():
            raise ValueError("Nonfinite Tianji measured joints")
        self._grips = np.array([action[f"{side}.grip"] for side in SIDES], dtype=float)
        if not np.isfinite(self._grips).all() or np.any((self._grips < 0) | (self._grips > 1)):
            raise ValueError("Pico grips must be in [0, 1]")
        poses = [
            sdk_pose_to_matrix(np.array([action[f"{side}.{field}"] for field in POSE_FIELDS]))
            for side in SIDES
        ]
        if self.calibration is not None:
            for pose, offset in zip(poses, self.calibration.offsets, strict=True):
                pose[:3, 3] += pose[:3, :3] @ offset
        if self.calibration is None:
            raw = {side: np.array([action[f"{side}.{f}"] for f in POSE_FIELDS]) for side in ("head", *SIDES)}
            poses = map_xr_wrist_targets(raw)
        self._ready = self._ready or self.released(action)
        previous = self._command
        if self.last_sent_action is not None:
            sent = self.last_sent_action()
            if sent is not None:
                previous = np.array([sent[key] for key in JOINT_NAMES], dtype=float).reshape(2, 7)
        command = measured.copy() if previous is None else previous.copy()
        self.diagnostics = {}
        for index, side in enumerate(SIDES):
            wrist = self._wrists[index]
            grip = self._grips[index]
            current = measured[index]
            reference = current if self.last_sent_action is None else command[index].copy()
            if self._await_release[index]:
                if grip <= self.grip_disable_threshold:
                    self._await_release[index] = False
                continue
            if wrist is None:
                if self._ready and grip > self.grip_enable_threshold:
                    xr = ROBOT_FROM_XR[:3, :3]
                    # G1D mapping already returns poses in the robot world basis.
                    basis = CHEST_FROM_XR[index] @ xr.T
                    if self.calibration is not None:
                        basis = basis @ self.calibration.world_from_xr
                    self._wrists[index] = WristTarget(
                        poses[index], self.solver.forward(index, current), current.copy(), stamp, basis
                    )
                    command[index] = current
            elif grip <= self.grip_disable_threshold:
                self._wrists[index] = None
                command[index] = current
            else:
                if np.array_equal(wrist.controller, poses[index]) and np.array_equal(
                    wrist.pose, wrist.flange
                ):
                    wrist.stamp = stamp
                    continue
                if self.calibration is None:
                    # G1D's independent FK translation and orientation clutch offsets.
                    target = wrist.flange.copy()
                    target[:3, 3] += wrist.basis @ (poses[index][:3, 3] - wrist.controller[:3, 3])
                    target[:3, :3] = (
                        wrist.basis
                        @ poses[index][:3, :3]
                        @ wrist.controller[:3, :3].T
                        @ wrist.basis.T
                        @ wrist.flange[:3, :3]
                    )
                    wrist.pose = target
                    candidate = self.solver.solve_local(
                        index, target, reference, self.max_target_delta_deg * 0.9, measured=current
                    )
                    self.diagnostics[side] = {"target": target.tolist(), "ik_mode": "g1_local"}
                    if candidate is not None:
                        self.diagnostics[side]["ik_deg"] = candidate.tolist()
                    else:
                        self.diagnostics[side]["status"] = "release_to_reengage"
                else:
                    target = wrist.update(poses[index], wrist.basis, stamp)
                    seed = wrist.seed if self.last_sent_action is None else reference
                    candidate = self._solve(index, side, target, seed)
                if candidate is None:
                    self._wrists[index] = None
                    self._await_release[index] = True
                    command[index] = current
                    logger.warning(
                        "Tianji %s wrist target rejected. Release and re-grip to resume this arm.", side
                    )
                else:
                    wrist.seed = candidate
                    delta = candidate - current
                    # Leave headroom for feedback to advance before the driver's check.
                    cap = 0.9 * self.max_target_delta_deg
                    command[index] = current + delta / max(1.0, np.max(np.abs(delta)) / cap)
        self._check_age(stamp)
        self._command = command
        return dict(zip(JOINT_NAMES, command.ravel().tolist(), strict=True))

    def _check_age(self, stamp: float):
        if not 0 <= self.clock() - stamp <= self.timeout_s:
            raise TimeoutError("Pico sample expired in Tianji teleoperation")

    def _solve(self, index, side, target, seed):
        # Compare consecutive IK solutions. During preview, feedback stays at the
        # initial pose, so the feedback-limited command is not a useful IK seed.
        attempts = {}
        diagnostic = {"target": target.tolist(), "ik_attempts": attempts}
        self.diagnostics[side] = diagnostic
        for mode in (0, 1):
            candidate = self.solver.inverse(index, target, seed, mode)
            if candidate is None:
                attempts[str(mode)] = {"result": "invalid_solution"}
                continue
            step = float(np.max(np.abs(candidate - seed)))
            attempts[str(mode)] = {"result": "valid_solution", "max_step_deg": step}
            if step <= self.max_seed_delta_deg:
                diagnostic.update(ik_deg=candidate.tolist(), ik_mode=mode)
                return candidate
        # Near workspace limits, try smaller wrist increments from the last
        # solved flange. Keep the original absolute target for the next tick.
        origin = self.solver.forward(index, seed)
        rotation = Rotation.from_matrix(origin[:3, :3].T @ target[:3, :3]).as_rotvec()
        for fraction in (0.5, 0.25, 0.125, 0.0625):
            partial = origin.copy()
            partial[:3, 3] += fraction * (target[:3, 3] - origin[:3, 3])
            partial[:3, :3] = origin[:3, :3] @ Rotation.from_rotvec(fraction * rotation).as_matrix()
            candidate = self.solver.inverse(index, partial, seed, 0)
            if candidate is not None and np.max(np.abs(candidate - seed)) <= self.max_seed_delta_deg:
                diagnostic.update(ik_deg=candidate.tolist(), ik_mode=0, target_fraction=fraction)
                return candidate
        diagnostic["status"] = "release_to_reengage"
        return None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION] = {
            key: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for key in JOINT_NAMES
        }
        return features


def connect_tianji_pico(robot, teleop, processor):
    """Arm the standard CLI paths after observing both released grips."""
    step = next(step for step in processor.steps if isinstance(step, PicoToTianjiJoints))
    # The headset may take a few frames to publish all tracking poses.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            action = teleop.get_action()
        except (TimeoutError, ValueError):
            time.sleep(0.02)
            continue
        if step.released(action):
            robot.connect()
            if not robot.config.move_to_start_on_connect:
                robot.enable_position_control()
            return
        time.sleep(0.02)
    raise TimeoutError("Tianji startup requires fresh Pico tracking with both grips released")
