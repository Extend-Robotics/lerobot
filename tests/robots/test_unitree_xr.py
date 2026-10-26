"""Upstream XR mapping and motor-control behavior, without headset or robot writes."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ArmIK, WeightedMovingFilter
from lerobot.robots.unitree_g1.pico_teleop import ARM_KEYS
from lerobot.robots.unitree_g1.unitree_xr import (
    ROBOT_FROM_XR,
    UnitreeXrArmSession,
    clip_arm_target,
    map_xr_wrist_targets,
)


def sample_poses():
    return {
        "head": np.array([0, 1.5, 0, 0, 0, 0, 1.0]),
        "left": np.array([-0.2, 1.2, -0.4, 0, 0, 0, 1.0]),
        "right": np.array([0.2, 1.2, -0.4, 0, 0, 0, 1.0]),
    }


def test_absolute_targets_have_upstream_head_to_waist_offsets():
    left, right = map_xr_wrist_targets(sample_poses())
    np.testing.assert_allclose(left[:3, 3], [0.55, 0.2, 0.15])
    np.testing.assert_allclose(right[:3, 3], [0.55, -0.2, 0.15])
    np.testing.assert_allclose(left[:3, :3], np.eye(3))


def test_common_head_yaw_and_translation_do_not_change_targets():
    poses = sample_poses()
    expected = map_xr_wrist_targets(poses)
    # Rotate the entire XR scene about vertical and translate it.
    yaw = Rotation.from_euler("y", 70, degrees=True)
    for pose in poses.values():
        pose[:3] = yaw.apply(pose[:3]) + [1, 0.3, -2]
        pose[3:] = yaw.as_quat()
    for actual, target in zip(map_xr_wrist_targets(poses), expected, strict=True):
        np.testing.assert_allclose(actual, target, atol=1e-12)


def test_full_absolute_rotation_does_not_depend_on_clutch_or_initial_tool_pose():
    poses = sample_poses()
    rotation = Rotation.from_euler("xyz", [25, -35, 70], degrees=True)
    poses["left"][3:] = rotation.as_quat()
    left, _ = map_xr_wrist_targets(poses)
    change_basis = ROBOT_FROM_XR[:3, :3]
    np.testing.assert_allclose(left[:3, :3], change_basis @ rotation.as_matrix() @ change_basis.T)


def test_head_pitch_and_roll_do_not_rotate_wrist_targets():
    poses = sample_poses()
    expected = map_xr_wrist_targets(poses)
    # Robot-frame pitch/roll with zero yaw, converted back to the XR convention.
    basis = ROBOT_FROM_XR[:3, :3]
    rotation = Rotation.from_euler("xyz", [20, 30, 0], degrees=True).as_matrix()
    poses["head"][3:] = Rotation.from_matrix(basis.T @ rotation @ basis).as_quat()
    for actual, target in zip(map_xr_wrist_targets(poses), expected, strict=True):
        np.testing.assert_allclose(actual, target, atol=1e-12)


def test_upstream_joint_filter_puts_largest_weight_on_newest_sample():
    filtered = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], 14, newest_first=True)
    for value in [0, 1, 2, 3]:
        filtered.add_data(np.full(14, value))
    np.testing.assert_allclose(filtered.filtered_data, np.full(14, 2.0))
    filtered.add_data(np.full(14, 3))  # Upstream ignores duplicates.
    np.testing.assert_allclose(filtered.filtered_data, np.full(14, 2.0))


def test_native_ik_filter_and_gravity_match_pinned_upstream_sequence():
    pytest.importorskip("pinocchio.casadi")
    pytest.importorskip("casadi")
    reference = json.loads(
        (Path(__file__).parents[1] / "fixtures/unitree_g1/xr_ik_reference.json").read_text()
    )
    ik = G1_29_ArmIK(unitree_xr=True)
    measured = np.array(reference["measured_q"])
    for frame in reference["sequence"]:
        q, tau = ik.solve_ik(np.array(frame["left"]), np.array(frame["right"]), measured, np.zeros(14))
        assert ik.last_solve_success
        np.testing.assert_allclose(q, frame["q"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(tau, frame["tau"], atol=5e-5, rtol=0)


def test_upstream_clipping_is_relative_to_measured_joints_and_scales_both_arms():
    measured = np.full(14, 0.2)
    target = measured.copy()
    target[0] += 1.0
    target[7] += 0.5
    command = clip_arm_target(target, measured)
    assert command[0] == pytest.approx(0.32)
    assert command[7] == pytest.approx(0.26)


@pytest.fixture
def session_state():
    now = [1.0]
    robot = Mock()
    robot.config = SimpleNamespace(read_only=False, arm_commands_only=True, arm_ik_profile="unitree_xr")
    robot.control_space = "cartesian_abs"
    robot.controller = None
    robot.get_observation.return_value = dict.fromkeys(ARM_KEYS, 0.0)
    solver = robot.arm_ik
    solver.last_solve_success = True
    solver.solve_ik.return_value = (np.full(14, 0.5), np.arange(14, dtype=float))
    solver.forward_kinematics.return_value = (np.eye(4), np.eye(4))
    teleop = Mock()
    teleop.get_xr_sample.return_value = (10, sample_poses())
    session = UnitreeXrArmSession(robot, teleop, clock=lambda: now[0])
    session.initialize()
    return SimpleNamespace(session=session, robot=robot, teleop=teleop, solver=solver, now=now)


def test_explicit_start_full_ik_and_feedforward_publication(session_state):
    state = session_state
    assert state.session.publish_once() is None
    state.robot.send_arm_joint_action.assert_not_called()
    state.session.start()
    state.session.tick()
    state.solver.solve_ik.assert_called_once()
    assert state.solver.solve_ik.call_args.kwargs == {}  # Upstream filter is enabled.
    first = state.session.publish_once()
    second = state.session.publish_once()
    np.testing.assert_allclose(first, np.full(14, 0.12))
    np.testing.assert_array_equal(first, second)  # Unchanged measured state; no command accumulation.
    np.testing.assert_array_equal(
        state.robot.send_arm_joint_action.call_args.kwargs["arm_tau"], np.arange(14)
    )


def test_preview_uses_virtual_feedback_and_never_sends_commands(session_state):
    state = session_state
    state.robot.config.read_only = True
    state.session.start()
    assert state.session.tick()["status"] == "preview"
    state.session.publish_once()
    state.session.tick()
    np.testing.assert_allclose(state.solver.solve_ik.call_args.args[2], np.full(14, 0.5))
    state.robot.send_arm_joint_action.assert_not_called()


def test_expired_targets_cannot_be_published(session_state):
    state = session_state
    state.session.start()
    state.session.tick()
    state.now[0] += 0.3
    with pytest.raises(TimeoutError):
        state.session.publish_once()
    state.robot.send_arm_joint_action.assert_not_called()


def test_bad_ik_cannot_replace_motor_targets(session_state):
    state = session_state
    state.session.start()
    state.solver.solve_ik.return_value = (np.full(14, np.nan), np.zeros(14))
    with pytest.raises(ValueError, match="IK failed"):
        state.session.tick()
    state.robot.send_arm_joint_action.assert_not_called()


def test_publisher_runs_independently_and_stops_before_disconnect(session_state):
    state = session_state
    state.session.start()
    state.session.tick()
    published = threading.Event()
    state.robot.send_arm_joint_action.side_effect = lambda *args, **kwargs: published.set()
    try:
        state.session.start_publisher()
        assert published.wait(1.0)
    finally:
        state.session.close()
    assert not state.session._thread.is_alive()
    count = state.robot.send_arm_joint_action.call_count
    assert state.session.publish_once() is None
    assert state.robot.send_arm_joint_action.call_count == count
