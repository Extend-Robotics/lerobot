"""PICO-to-joints processing, without robot writes or native IK dependencies."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.configs import PipelineFeatureType
from lerobot.processor import make_default_processors
from lerobot.teleoperators.bi_pico4.g1_processor import ARM_KEYS, POSE_FIELDS, PicoToG1Joints
from lerobot.teleoperators.bi_pico4.xr_mapping import map_xr_wrist_targets


@pytest.fixture
def pipeline_state(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    solver = Mock()
    solver.forward_kinematics.return_value = (np.eye(4), np.eye(4))
    solver.last_solve_success = True
    solver.solve_ik.return_value = (np.full(14, 0.2), np.zeros(14))
    solver._arm_reorder_pin_to_g1 = np.arange(14)
    solver.reduced_robot.model = SimpleNamespace(
        lowerPositionLimit=np.full(14, -1.0), upperPositionLimit=np.full(14, 1.0)
    )
    processor = make_default_processors()[0]
    step = PicoToG1Joints(solver=solver, timeout_s=0.25)
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
    action.update(
        {
            "sample_time": now[0],
            "left.trigger": 0.3,
            "right.trigger": 0.8,
            "left.grip": 1.0,
            "right.grip": 1.0,
        }
    )
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
    teleop = SimpleNamespace(
        name="bi_pico4",
        config=SimpleNamespace(
            output_mode="xr", input_timeout_s=0.25, grip_enable_threshold=0.7, grip_disable_threshold=0.2
        ),
    )
    pipeline, _, _ = make_default_processors(robot, teleop)
    assert pipeline((s.action, s.observation)) == dict.fromkeys(ARM_KEYS, 0.2)
    assert pipeline.steps[0].grip_enable_threshold == 0.7
    assert pipeline.steps[0].grip_disable_threshold == 0.2
    teleop.config.output_mode = "cartesian"
    with pytest.raises(ValueError, match="output_mode"):
        make_default_processors(robot, teleop)


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 1)])
def test_released_arm_holds_measured_position_independently(pipeline_state, side, index):
    s = pipeline_state
    s.processor((s.action, s.observation))
    keys = ARM_KEYS[index * 7 : (index + 1) * 7]
    s.observation.update(dict.fromkeys(keys, 0.1))
    s.action[f"{side}.grip"] = 0.0
    result = s.processor((s.action, s.observation))
    assert all(result[key] == (0.1 if key in keys else 0.2) for key in ARM_KEYS)
    # Neither tracking changes nor subsequent feedback drift updates the latch.
    s.action[f"{side}.x"] = 100
    s.action["head.x"] = 1.0
    s.observation.update(dict.fromkeys(keys, 0.05))
    result = s.processor((s.action, s.observation))
    assert all(result[key] == 0.1 for key in keys)
    np.testing.assert_array_equal(s.solver.solve_ik.call_args.args[index], np.eye(4))
    s.action[f"{side}.grip"] = 1.0
    result = s.processor((s.action, s.observation))
    poses = {
        name: np.array([s.action[f"{name}.{field}"] for field in POSE_FIELDS])
        for name in ("head", "left", "right")
    }
    expected = map_xr_wrist_targets(poses)
    np.testing.assert_allclose(s.solver.solve_ik.call_args.args[index], expected[index])
    assert all(result[key] == 0.2 for key in keys)


def test_both_released_skip_ik_keep_hands_and_validate_freshness(pipeline_state):
    s = pipeline_state
    s.action.update({"left.grip": 0.0, "right.grip": 0.0})
    s.step.hand_features = ("left_hand.index.pos",)
    s.observation = dict.fromkeys(ARM_KEYS, 0.15)
    result = s.processor((s.action, s.observation))
    assert all(result[key] == 0.15 for key in ARM_KEYS)
    assert result["left_hand.index.pos"] == 0.3
    s.solver.solve_ik.assert_not_called()
    s.observation = dict.fromkeys(ARM_KEYS, 0.1)
    assert s.processor((s.action, s.observation)) == result
    s.action["sample_time"] -= 1
    with pytest.raises(TimeoutError):
        s.processor((s.action, s.observation))


def test_clutch_hysteresis_and_reset(pipeline_state):
    s = pipeline_state
    s.action["right.grip"] = 0.0
    for grip, expected in [(0.5, False), (0.51, True), (0.4, True), (0.3, False), (0.4, False)]:
        s.action["left.grip"] = grip
        result = s.processor((s.action, s.observation))
        assert result[ARM_KEYS[0]] == (0.2 if expected else 0.0)
    s.processor.reset()
    s.solver.smooth_filter.reset.assert_called_once()
    s.observation = dict.fromkeys(ARM_KEYS, 0.1)
    assert s.processor((s.action, s.observation)) == dict.fromkeys(ARM_KEYS, 0.1)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_grip_never_enables_arm(pipeline_state, value):
    s = pipeline_state
    s.action["left.grip"] = value
    with pytest.raises(ValueError, match="grips"):
        s.processor((s.action, s.observation))
    s.solver.solve_ik.assert_not_called()


def test_invalid_tracking_is_rejected_while_released(pipeline_state):
    s = pipeline_state
    s.action.update({"left.grip": 0.0, "right.grip": 0.0, "head.qw": 0.0})
    with pytest.raises(ValueError, match="XR pose"):
        s.processor((s.action, s.observation))
