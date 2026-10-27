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

ARM_KEYS = tuple(f"{joint.name}.pos" for joint in G1_29_JointArmIndex)
POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")


@ProcessorStepRegistry.register("pico_to_g1_joints")
@dataclass
class PicoToG1Joints(RobotActionProcessorStep):
    """Apply Unitree XR mapping and IK; retain only arm/hand joint actions."""

    hand_features: tuple[str, ...] = ()
    timeout_s: float = 0.25
    solver: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.solver is None:
            self.solver = G1_29_ArmIK(unitree_xr=True)

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
        targets = map_xr_wrist_targets(poses)
        candidate, _ = self.solver.solve_ik(*targets, q, np.zeros(14))
        candidate = np.asarray(candidate, dtype=float)
        if not self.solver.last_solve_success or candidate.shape != (14,) or not np.isfinite(candidate).all():
            raise ValueError("IK failed or returned invalid arm joints")
        order = self.solver._arm_reorder_pin_to_g1
        model = self.solver.reduced_robot.model
        if np.any(candidate < model.lowerPositionLimit[order]) or np.any(
            candidate > model.upperPositionLimit[order]
        ):
            raise ValueError("IK violates model joint limits")
        if time.monotonic() - action["sample_time"] > self.timeout_s:
            raise TimeoutError("PICO sample expired during IK")
        result = dict(zip(ARM_KEYS, candidate.tolist(), strict=True))
        for key in self.hand_features:
            side = key.split("_hand.", 1)[0]
            trigger = float(action[f"{side}.trigger"])
            if not np.isfinite(trigger) or not 0 <= trigger <= 1:
                raise ValueError("PICO triggers must be in [0, 1]")
            result[key] = trigger
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
