"""Hands-only arm control, including pause races and bounded supported hold."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.robots.unitree_g1.pico_hand_teleop import HOLD_TIMEOUT_S, PicoHandArmSession
from lerobot.robots.unitree_g1.pico_teleop import ARM_KEYS
from lerobot.robots.unitree_g1.unitree_xr import UnitreeXrArmSession, map_xr_wrist_targets
from lerobot.teleoperators.pico4.hand_input import HandTrackingUnavailableError
from tests.robots.test_unitree_xr import sample_poses


def test_hand_frame_corrections_are_local_and_side_specific():
    poses = sample_poses()
    poses["left"][3:] = Rotation.from_euler("xyz", [15, 25, 40], degrees=True).as_quat()
    controllers = map_xr_wrist_targets(poses)
    hands = map_xr_wrist_targets(poses, hand_tracking=True)
    corrections = (
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    )
    for controller, hand, correction in zip(controllers, hands, corrections, strict=True):
        np.testing.assert_allclose(hand[:3, 3], controller[:3, 3])
        np.testing.assert_allclose(hand[:3, :3], controller[:3, :3] @ correction, atol=1e-12)


@pytest.fixture
def state():
    now = [1.0]
    robot = Mock()
    robot.config = SimpleNamespace(read_only=False, arm_commands_only=True, arm_ik_profile="unitree_xr")
    robot.control_space = "cartesian_abs"
    robot.controller = None
    robot.get_observation.return_value = dict.fromkeys(ARM_KEYS, 0.0)
    solver = robot.arm_ik
    solver.last_solve_success = True
    solver.solve_ik.return_value = (np.full(14, 0.5), np.full(14, 99.0))
    solver.solve_tau.return_value = np.full(14, 2.0)
    solver.forward_kinematics.return_value = (np.eye(4), np.eye(4))
    teleop = Mock()
    teleop.get_xr_sample.return_value = (10, sample_poses())
    session = PicoHandArmSession(robot, teleop, clock=lambda: now[0])
    session.initialize()
    return SimpleNamespace(session=session, robot=robot, solver=solver, teleop=teleop, now=now)


def advance_and_publish(state):
    state.now[0] += 0.01
    return state.session.publish_once()


def test_explicit_start_uses_controller_speed(state):
    assert state.session.publish_once() is None
    state.session.start()
    state.session.tick()
    command = advance_and_publish(state)
    np.testing.assert_allclose(command, np.full(14, 0.12))
    # Gravity follows measured joints, not the distant IK target.
    np.testing.assert_array_equal(state.robot.send_arm_joint_action.call_args.kwargs["arm_tau"], 2.0)
    assert set(state.robot.send_arm_joint_action.call_args.args[0]) == set(ARM_KEYS)


def test_tracking_loss_holds_last_command_and_recovery_requires_resume(state):
    state.session.start()
    state.session.tick()
    held = advance_and_publish(state)
    state.teleop.get_xr_sample.side_effect = HandTrackingUnavailableError("left hand lost")
    assert state.session.tick()["status"] == "paused"
    np.testing.assert_array_equal(advance_and_publish(state), held)
    state.teleop.get_xr_sample.side_effect = None
    state.teleop.get_xr_sample.return_value = (11, sample_poses())
    assert state.session.tick()["status"] == "paused"
    np.testing.assert_array_equal(advance_and_publish(state), held)
    state.session.start()
    state.session.tick()
    np.testing.assert_allclose(advance_and_publish(state), np.full(14, 0.12))


def test_cannot_start_with_invalid_tracking(state):
    state.teleop.get_xr_sample.side_effect = HandTrackingUnavailableError("missing")
    state.session.start()
    assert state.session.publish_once() is None
    state.robot.send_arm_joint_action.assert_not_called()


def test_hold_expires_even_if_pause_is_repeated(state):
    state.session.start()
    state.session.tick()
    advance_and_publish(state)
    state.session.pause()
    state.now[0] += HOLD_TIMEOUT_S - 0.01
    state.session.pause()
    state.session.publish_once()
    state.now[0] += 0.02
    with pytest.raises(TimeoutError, match="5s"):
        state.session.publish_once()
    with pytest.raises(TimeoutError, match="5s"):
        state.session.tick()


def test_publisher_expiry_pauses_without_main_loop_progress(state):
    state.session.start()
    state.session.tick()
    held = advance_and_publish(state)
    state.now[0] += 0.3
    np.testing.assert_array_equal(state.session.publish_once(), held)
    assert not state.session._following
    state.now[0] += HOLD_TIMEOUT_S
    with pytest.raises(TimeoutError):
        state.session.publish_once()


def test_tracking_loss_during_ik_cannot_publish_new_candidate(state):
    state.session.start()
    state.teleop.get_xr_sample.side_effect = [
        (10, sample_poses()),
        HandTrackingUnavailableError("lost during solve"),
    ]
    assert state.session.tick()["status"] == "paused"
    np.testing.assert_array_equal(advance_and_publish(state), np.zeros(14))


def test_publisher_pause_during_ik_cannot_be_overwritten(state):
    state.session.start()

    def solve(*args):
        state.session.pause("concurrent pause")
        return np.full(14, 0.8), np.zeros(14)

    state.solver.solve_ik.side_effect = solve
    assert state.session.tick()["status"] == "paused"
    np.testing.assert_array_equal(advance_and_publish(state), np.zeros(14))


def test_preview_never_publishes_even_during_hold(state):
    state.robot.config.read_only = True
    state.session.start()
    assert state.session.tick()["status"] == "preview"
    np.testing.assert_allclose(advance_and_publish(state), np.full(14, 0.5))
    state.session.pause()
    advance_and_publish(state)
    state.robot.send_arm_joint_action.assert_not_called()


def test_bad_ik_remains_a_fault(state):
    state.session.start()
    state.solver.last_solve_success = False
    with pytest.raises(ValueError, match="IK failed"):
        state.session.tick()


def test_publisher_stops_before_disconnect(state):
    state.session.start()
    state.session.tick()
    state.session.start_publisher()
    state.session.close()
    assert not state.session._thread.is_alive()
    assert state.session.publish_once() is None


@pytest.mark.parametrize("read_only", [False, True])
def test_hand_and_controller_publication_match_for_identical_ik_targets(state, read_only):
    state.robot.config.read_only = read_only
    measured = np.linspace(-0.2, 0.2, 14)
    state.robot.get_observation.return_value = dict(zip(ARM_KEYS, measured, strict=True))
    target = measured + np.linspace(-0.5, 1.0, 14)
    state.solver.solve_ik.return_value = (target, np.zeros(14))
    controller = UnitreeXrArmSession(state.robot, state.teleop, clock=lambda: state.now[0])
    controller.initialize()
    for session in (state.session, controller):
        session.start()
        session.tick()
    state.now[0] += 0.01
    np.testing.assert_allclose(state.session.publish_once(), controller.publish_once())
