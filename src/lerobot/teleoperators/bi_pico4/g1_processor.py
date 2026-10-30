"""PICO tracking to joint targets. This processor never connects or commands hardware."""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep, TransitionKey
from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ArmIK
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex

from .xr_mapping import map_xr_wrist_targets

ARM_KEYS = tuple(f"{joint.name}.q" for joint in G1_29_JointArmIndex)
POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")


@ProcessorStepRegistry.register("pico_to_g1_joints")
@dataclass
class PicoToG1Joints(RobotActionProcessorStep):
    """Apply Unitree XR mapping and IK; retain only arm/hand joint actions."""

    hand_features: tuple[str, ...] = ()
    timeout_s: float = 0.5
    solver: Any = field(default=None, repr=False)
    grip_enable_threshold: float = 0.5
    grip_disable_threshold: float = 0.3
    _enabled: tuple[bool, bool] = field(default=(False, False), init=False, repr=False)
    _held_q: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.grip_disable_threshold < self.grip_enable_threshold <= 1:
            raise ValueError("PICO grip thresholds must satisfy 0 <= disable < enable <= 1")
        if self.solver is None:
            self.solver = G1_29_ArmIK(unitree_xr=True)

    def reset(self) -> None:
        self._enabled = (False, False)
        self._held_q = None
        self.solver.smooth_filter.reset()

    def action(self, action: RobotAction) -> RobotAction:
        began = time.monotonic()
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("PICO IK requires measured arm joints")
        q = np.array([observation[key] for key in ARM_KEYS], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite measured joints")
        if not 0 <= began - action["sample_time"] <= self.timeout_s:
            raise TimeoutError("PICO sample expired before IK")
        poses = {
            side: np.array([action[f"{side}.{key}"] for key in POSE_FIELDS])
            for side in ("head", "left", "right")
        }
        # Validate tracking even while released; clutching does not bypass watchdogs.
        targets = list(map_xr_wrist_targets(poses))
        grips = np.array([action[f"{side}.grip"] for side in ("left", "right")], dtype=float)
        if not np.isfinite(grips).all() or np.any((grips < 0) | (grips > 1)):
            raise ValueError("PICO grips must be in [0, 1]")
        enabled_values = [
            bool(grip > (self.grip_disable_threshold if previous else self.grip_enable_threshold))
            for grip, previous in zip(grips, self._enabled, strict=True)
        ]
        enabled = (enabled_values[0], enabled_values[1])
        held_q = q.copy() if self._held_q is None else self._held_q.copy()
        for index, (previous, active) in enumerate(zip(self._enabled, enabled, strict=True)):
            if previous and not active:
                # Stop at the measured position, not an unfinished IK target.
                held_q[index * 7 : (index + 1) * 7] = q[index * 7 : (index + 1) * 7]
        candidate = held_q.copy()
        if any(enabled):
            if not all(enabled):
                held_targets = self.solver.forward_kinematics(held_q)
                for index, active in enumerate(enabled):
                    if not active:
                        targets[index] = held_targets[index]
            solved, _ = self.solver.solve_ik(*targets, q, np.zeros(14))
            solved = np.asarray(solved, dtype=float)
            if not self.solver.last_solve_success or solved.shape != (14,) or not np.isfinite(solved).all():
                raise ValueError("IK failed or returned invalid arm joints")
            order = self.solver._arm_reorder_pin_to_g1
            model = self.solver.reduced_robot.model
            if np.any(solved < model.lowerPositionLimit[order]) or np.any(
                solved > model.upperPositionLimit[order]
            ):
                raise ValueError("IK violates model joint limits")
            for index, active in enumerate(enabled):
                if active:
                    candidate[index * 7 : (index + 1) * 7] = solved[index * 7 : (index + 1) * 7]
        if time.monotonic() - action["sample_time"] > self.timeout_s:
            raise TimeoutError("PICO sample expired during IK")
        result = dict(zip(ARM_KEYS, candidate.tolist(), strict=True))
        for key in self.hand_features:
            side = key.split("_hand.", 1)[0]
            trigger = float(action[f"{side}.trigger"])
            if not np.isfinite(trigger) or not 0 <= trigger <= 1:
                raise ValueError("PICO triggers must be in [0, 1]")
            result[key] = trigger
        self._enabled = enabled
        self._held_q = held_q
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        joint_keys = (*ARM_KEYS, *self.hand_features)
        existing = features.get(PipelineFeatureType.ACTION, {})
        # Rollout may already have aligned joint features to checkpoint order.
        # Only replace the schema when converting raw tracking features.
        if set(existing) == set(joint_keys):
            return features
        features[PipelineFeatureType.ACTION] = {
            key: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for key in joint_keys
        }
        return features
