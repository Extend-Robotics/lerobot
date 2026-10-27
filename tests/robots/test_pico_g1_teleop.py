"""PICO-to-joints processing, without robot writes or native IK dependencies."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.configs import PipelineFeatureType
from lerobot.processor import make_default_processors
from lerobot.teleoperators.bi_pico4.g1_processor import ARM_KEYS, POSE_FIELDS, PicoToG1Joints


@pytest.fixture
def pipeline_state(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    solver = Mock()
    solver.last_solve_success = True
    solver.solve_ik.return_value = (np.full(14, 0.2), np.zeros(14))
    solver._arm_reorder_pin_to_g1 = np.arange(14)
    solver.reduced_robot.model = SimpleNamespace(
        lowerPositionLimit=np.full(14, -1.0), upperPositionLimit=np.full(14, 1.0)
    )
    processor = make_default_processors()[0]
    step = PicoToG1Joints(solver=solver)
    processor.steps = [step]
    poses = {
        "head": [0, 1.5, 0, 0, 0, 0, 1],
        "left": [-0.2, 1.2, -0.4, 0, 0, 0, 1],
        "right": [0.2, 1.2, -0.4, 0, 0, 0, 1],
    }
    action = {
        f"{side}.{key}": value
        for side, pose in poses.items()
        for key, value in zip(POSE_FIELDS, pose, strict=True)
    }
    action.update({"sample_time": now[0], "left.trigger": 0.3, "right.trigger": 0.8})
    return SimpleNamespace(
        processor=processor,
        step=step,
        solver=solver,
        now=now,
        observation=dict.fromkeys(ARM_KEYS, 0.0),
        action=action,
    )


def test_raw_tracking_becomes_only_joint_targets(pipeline_state):
    s = pipeline_state
    result = s.processor((s.action, s.observation))
    assert result == dict.fromkeys(ARM_KEYS, 0.2)
    left, right, measured, _ = s.solver.solve_ik.call_args.args
    np.testing.assert_allclose(left[:3, 3], [0.55, 0.2, 0.15])
    np.testing.assert_allclose(right[:3, 3], [0.55, -0.2, 0.15])
    np.testing.assert_array_equal(measured, np.zeros(14))


@pytest.mark.parametrize("failure", ["failed", "nan", "shape", "limits"])
def test_invalid_ik_never_becomes_an_action(pipeline_state, failure):
    s = pipeline_state
    if failure == "failed":
        s.solver.last_solve_success = False
    else:
        value = {"nan": np.full(14, np.nan), "shape": np.zeros(13), "limits": np.full(14, 2.0)}[failure]
        s.solver.solve_ik.return_value = (value, np.zeros(14))
    with pytest.raises(ValueError):
        s.processor((s.action, s.observation))


def test_sample_expiry_before_and_during_native_solve(pipeline_state):
    s = pipeline_state
    s.action["sample_time"] -= 1
    with pytest.raises(TimeoutError, match="before IK"):
        s.processor((s.action, s.observation))
    s.solver.solve_ik.assert_not_called()
    s.action["sample_time"] = s.now[0]

    def slow_solve(*args):
        s.now[0] += 0.3
        return np.zeros(14), np.zeros(14)

    s.solver.solve_ik.side_effect = slow_solve
    with pytest.raises(TimeoutError, match="during IK"):
        s.processor((s.action, s.observation))


def test_dataset_features_exclude_tracking_and_tcp(pipeline_state):
    s = pipeline_state
    features = s.step.transform_features({PipelineFeatureType.ACTION: {"head.x": object()}})
    assert set(features[PipelineFeatureType.ACTION]) == set(ARM_KEYS)


def test_joint_features_retain_checkpoint_order(pipeline_state):
    keys = list(reversed(ARM_KEYS))
    features = {PipelineFeatureType.ACTION: dict.fromkeys(keys, object())}
    result = pipeline_state.step.transform_features(features)
    assert list(result[PipelineFeatureType.ACTION]) == keys


def test_default_factory_selects_joint_processor(monkeypatch, pipeline_state):
    s = pipeline_state
    monkeypatch.setattr("lerobot.teleoperators.bi_pico4.g1_processor.G1_29_ArmIK", lambda **kwargs: s.solver)
    robot = SimpleNamespace(
        name="unitree_g1",
        config=SimpleNamespace(arm_commands_only=True, command_timeout_s=0.25),
        action_features=dict.fromkeys(ARM_KEYS, float),
    )
    teleop = SimpleNamespace(name="bi_pico4", config=SimpleNamespace(output_mode="xr", input_timeout_s=0.25))
    pipeline, _, _ = make_default_processors(robot, teleop)
    assert pipeline((s.action, s.observation)) == dict.fromkeys(ARM_KEYS, 0.2)
    teleop.config.output_mode = "cartesian"
    with pytest.raises(ValueError, match="output_mode"):
        make_default_processors(robot, teleop)
