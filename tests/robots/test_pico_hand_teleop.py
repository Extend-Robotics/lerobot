"""Controllers and trigger-controlled Revo2 fingers, including pause races and hold."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.brainco_revo2 import BraincoRevo2Hand, BraincoRevo2HandConfig
from lerobot.robots.unitree_g1.pico_hand_teleop import HOLD_TIMEOUT_S, PicoHandArmSession
from lerobot.robots.unitree_g1.pico_teleop import ARM_KEYS
from lerobot.robots.unitree_g1.unitree_xr import UnitreeXrArmSession, map_xr_wrist_targets
from tests.robots.test_unitree_xr import sample_poses


@pytest.fixture
def state():
    now = [1.0]
    robot = Mock()
    robot.config = SimpleNamespace(read_only=False, arm_commands_only=True, arm_ik_profile="unitree_xr")
    robot.control_space = "cartesian_abs"
    robot.hands = {}
    robot.controller = None
    robot.get_observation.return_value = dict.fromkeys(ARM_KEYS, 0.0)
    solver = robot.arm_ik
    solver.last_solve_success = True
    solver.solve_ik.return_value = (np.full(14, 0.5), np.full(14, 99.0))
    solver.solve_tau.return_value = np.full(14, 2.0)
    solver.forward_kinematics.return_value = (np.eye(4), np.eye(4))
    teleop = Mock()
    teleop.get_controller_sample.return_value = (10, sample_poses(), {"left": 1.0, "right": 1.0})
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
    state.teleop.get_controller_sample.side_effect = RuntimeError("left hand lost")
    assert state.session.tick()["status"] == "paused"
    np.testing.assert_array_equal(advance_and_publish(state), held)
    state.teleop.get_controller_sample.side_effect = None
    state.teleop.get_controller_sample.return_value = (11, sample_poses(), {"left": 1.0, "right": 1.0})
    assert state.session.tick()["status"] == "paused"
    np.testing.assert_array_equal(advance_and_publish(state), held)
    state.session.start()
    state.session.tick()
    np.testing.assert_allclose(advance_and_publish(state), np.full(14, 0.12))


def test_cannot_start_with_invalid_tracking(state):
    state.teleop.get_controller_sample.side_effect = RuntimeError("missing")
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
    state.teleop.get_controller_sample.side_effect = [
        (10, sample_poses(), {"left": 1.0, "right": 1.0}),
        RuntimeError("lost during solve"),
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
    state.teleop.get_xr_sample.return_value = (10, sample_poses())
    controller = UnitreeXrArmSession(state.robot, state.teleop, clock=lambda: state.now[0])
    controller.initialize()
    for session in (state.session, controller):
        session.start()
        session.tick()
    state.now[0] += 0.01
    np.testing.assert_allclose(state.session.publish_once(), controller.publish_once())


@pytest.fixture
def combined(state, tmp_path):
    state.robot.hands = {}
    for side in ("left", "right"):
        hand = BraincoRevo2Hand(BraincoRevo2HandConfig(side=side, calibration_dir=tmp_path))
        hand._transport = Mock()
        hand._transport.read.side_effect = lambda: ([0.2] * 6, time.monotonic())
        hand._connected = True
        state.robot.hands[side] = hand
    state.teleop.reset_mock()
    state.sample = (10, sample_poses(), {"left": 1.0, "right": 1.0})
    state.teleop.get_controller_sample.return_value = state.sample
    state.session = PicoHandArmSession(state.robot, state.teleop, clock=lambda: state.now[0])
    state.mapper = Mock(wraps=state.session.trigger_mapper)
    state.mapper.map.return_value = {
        f"{side}_hand.{key}": 1.0 for side, hand in state.robot.hands.items() for key in hand.action_features
    }
    state.session.trigger_mapper = state.mapper
    state.session.initialize()
    return state


def assert_no_finger_commands(combined):
    for hand in combined.robot.hands.values():
        hand._transport.write.assert_not_called()


def test_combined_waits_for_start_and_uses_shared_snapshot(combined):
    combined.session.tick()
    assert_no_finger_commands(combined)
    assert combined.session.publish_once() is None
    combined.session.start()
    combined.now[0] += 0.02
    report = combined.session.tick()
    assert report["status"] == "executing"
    assert list(report["hand_sent_action"].values()) == pytest.approx([0.22] * 12)
    np.testing.assert_allclose(advance_and_publish(combined), np.full(14, 0.12))
    sample = combined.mapper.map.call_args.args[0]
    assert sample == combined.sample[2]
    expected = map_xr_wrist_targets(combined.sample[1])
    for target, actual in zip(expected, combined.solver.solve_ik.call_args.args[:2], strict=True):
        np.testing.assert_allclose(target, actual)


def test_combined_loss_during_solve_blocks_both_targets(combined):
    combined.session.start()
    combined.teleop.get_controller_sample.side_effect = [
        combined.sample,
        RuntimeError("lost"),
    ]
    assert combined.session.tick()["status"] == "paused"
    assert_no_finger_commands(combined)
    np.testing.assert_allclose(advance_and_publish(combined), np.zeros(14))
    combined.teleop.get_controller_sample.side_effect = None
    assert combined.session.tick()["status"] == "paused"
    assert_no_finger_commands(combined)
    combined.session.start()
    combined.now[0] += 0.02
    assert combined.session.tick()["status"] == "executing"


def test_combined_concurrent_pause_during_trigger_mapping_blocks_dispatch(combined):
    combined.session.start()

    def retarget(_):
        combined.session.pause("publisher expired while solving")
        return combined.mapper.map.return_value

    combined.mapper.map.side_effect = retarget
    assert combined.session.tick()["status"] == "paused"
    assert_no_finger_commands(combined)
    np.testing.assert_allclose(advance_and_publish(combined), np.zeros(14))


def test_combined_finger_solver_fault_pauses_arms(combined):
    combined.session.start()
    combined.mapper.map.side_effect = ValueError("bad trigger")
    with pytest.raises(ValueError, match="bad trigger"):
        combined.session.tick()
    assert not combined.session._following
    assert_no_finger_commands(combined)
    np.testing.assert_allclose(advance_and_publish(combined), np.zeros(14))


def test_combined_stale_hand_feedback_blocks_both_fingers_and_arm_target(combined):
    combined.session.start()
    combined.robot.hands["right"]._transport.read.return_value = None
    combined.robot.hands["right"]._transport.read.side_effect = None
    with pytest.raises(TimeoutError, match="stale"):
        combined.session.tick()
    assert_no_finger_commands(combined)
    assert not combined.session._following
    np.testing.assert_allclose(combined.session._target_q, np.zeros(14))


def test_combined_preview_never_writes_any_commands(combined):
    combined.robot.config.read_only = True
    for hand in combined.robot.hands.values():
        hand.config.read_only = True
    combined.session.start()
    report = combined.session.tick()
    assert report["status"] == "preview"
    assert len(report["hand_target_action"]) == 12
    assert report["hand_sent_action"] == {}
    advance_and_publish(combined)
    assert_no_finger_commands(combined)
    combined.robot.send_arm_joint_action.assert_not_called()


def test_combined_transport_fault_stops_accepting_arm_targets(combined):
    combined.session.start()
    combined.robot.hands["left"]._transport.write.side_effect = ConnectionError("hand bridge disconnected")
    with pytest.raises(ConnectionError):
        combined.session.tick()
    combined.robot.hands["right"]._transport.write.assert_not_called()
    assert not combined.session._following
    np.testing.assert_allclose(combined.session._target_q, np.zeros(14))


def test_combined_publisher_failure_blocks_fingers(combined):
    combined.session.start()

    def failed_publisher(_):
        combined.session._error = TimeoutError("arm bridge lost")
        return combined.mapper.map.return_value

    combined.mapper.map.side_effect = failed_publisher
    with pytest.raises(RuntimeError, match="publisher stopped"):
        combined.session.tick()
    assert_no_finger_commands(combined)


@pytest.fixture
def controller_combined(combined):
    combined.teleop.reset_mock()
    combined.teleop.get_controller_sample.return_value = (10, sample_poses(), {"left": 1.0, "right": 0.0})
    combined.session = PicoHandArmSession(
        combined.robot,
        combined.teleop,
        clock=lambda: combined.now[0],
    )
    combined.session.initialize()
    return combined


def test_controller_triggers_and_arm_poses_share_start_and_rate_limits(controller_combined):
    state = controller_combined
    assert state.session.tick()["status"] == "ready"
    assert_no_finger_commands(state)
    state.session.start()
    state.now[0] += 0.02
    report = state.session.tick()
    assert list(report["hand_target_action"].values()) == [1.0] * 6 + [0.0] * 6
    assert list(report["hand_sent_action"].values()) == pytest.approx([0.22] * 6 + [0.18] * 6)
    expected = map_xr_wrist_targets(sample_poses())
    for target, actual in zip(expected, state.solver.solve_ik.call_args.args[:2], strict=True):
        np.testing.assert_allclose(target, actual)
    state.teleop.get_hand_sample.assert_not_called()
    np.testing.assert_allclose(advance_and_publish(state), np.full(14, 0.12))
    state.session.pause()
    for hand in state.robot.hands.values():
        hand._transport.write.reset_mock()
    assert state.session.tick()["status"] == "paused"
    assert_no_finger_commands(state)


def test_controller_trigger_preview_never_writes(controller_combined):
    state = controller_combined
    state.robot.config.read_only = True
    state.session.start()
    report = state.session.tick()
    assert report["status"] == "preview" and report["hand_sent_action"] == {}
    advance_and_publish(state)
    assert_no_finger_commands(state)
    state.robot.send_arm_joint_action.assert_not_called()


def test_controller_source_restart_during_solve_blocks_commands(controller_combined):
    state = controller_combined
    state.session.start()
    state.teleop.get_controller_sample.side_effect = [
        (10, sample_poses(), {"left": 1, "right": 0}),
        (1, sample_poses(), {"left": 1, "right": 0}),
    ]
    assert state.session.tick()["status"] == "paused"
    assert_no_finger_commands(state)
    np.testing.assert_allclose(advance_and_publish(state), np.zeros(14))


def test_controller_frozen_stream_pauses_fingers_and_arms(controller_combined):
    state = controller_combined
    state.session.start()
    state.now[0] += 0.3
    assert state.session.tick()["status"] == "paused"
    assert_no_finger_commands(state)
