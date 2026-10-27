"""PICO mapping and lifecycle tests; no headset or native SDK required."""

from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.unitree_g1.pico_teleop import ARM_KEYS, PicoArmSession
from lerobot.teleoperators.bi_pico4 import BiPico4, BiPico4Config, bi_pico4 as bi_module
from lerobot.teleoperators.pico4 import Pico4, Pico4Config, teleop_pico4 as single_module
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError


@pytest.fixture
def sdk(monkeypatch):
    sdk = MagicMock()
    for side in ("left", "right"):
        getattr(sdk, f"get_{side}_controller_pose").return_value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        getattr(sdk, f"get_{side}_grip").return_value = 0.0
        getattr(sdk, f"get_{side}_trigger").return_value = 0.0
    for button in "ABXY":
        getattr(sdk, f"get_{button}_button").return_value = False
    for module in (single_module, bi_module):
        monkeypatch.setattr(module, "xrt", sdk, raising=False)
        monkeypatch.setattr(module, "require_package", lambda pkg_name, extra, import_name=None: None)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return sdk


@pytest.mark.parametrize("config", [Pico4Config(), BiPico4Config()])
def test_factory_and_features(config, sdk):
    teleop = make_teleoperator_from_config(config)
    try:
        teleop.connect()
        assert set(teleop.action_features) == set(teleop.get_action())
        assert all(t is float for t in teleop.action_features.values())
    finally:
        teleop.disconnect()
    sdk.init.assert_called_once()
    sdk.close.assert_called_once()


@pytest.mark.parametrize("kind", ["single", "bi", "pre_init"])
def test_timeout_closes_sdk(kind, sdk):
    sdk.get_left_controller_pose.return_value = [0.0] * 7
    sdk.get_right_controller_pose.return_value = [0.0] * 7
    teleop = Pico4(Pico4Config()) if kind == "single" else BiPico4(BiPico4Config())
    with pytest.raises(DeviceNotConnectedError):
        (teleop.pre_init if kind == "pre_init" else teleop.connect)()
    assert not teleop.is_connected
    assert teleop._xrt is None
    sdk.close.assert_called_once()


def test_pre_init_shares_one_connection(sdk):
    teleop = BiPico4(BiPico4Config())
    teleop.pre_init()
    teleop.connect()
    teleop.disconnect()
    sdk.init.assert_called_once()
    sdk.close.assert_called_once()
    assert teleop._left_pico4._xrt is None
    assert teleop._right_pico4._xrt is None


def test_clutch_mapping_and_independent_arms(sdk):
    teleop = BiPico4(BiPico4Config(max_pos_velocity=0.0, max_rot_velocity=0.0))
    try:
        teleop.connect()
        sdk.get_left_grip.return_value = 1.0
        teleop.get_action()  # Latch controller origin.
        sdk.get_left_controller_pose.return_value = [0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0]
        action = teleop.get_action()
        assert action["left_tcp.x"] == pytest.approx(0.02)
        assert action["right_tcp.x"] == 0.0
        sdk.get_left_grip.return_value = 0.0
        sdk.get_left_controller_pose.return_value = [0.0, 0.0, -0.04, 0.0, 0.0, 0.0, 1.0]
        assert teleop.get_action()["left_tcp.x"] == pytest.approx(0.02)
        # 6D identity columns are finite and orthonormal.
        cols = np.array([action[f"left_tcp.r{i}"] for i in range(1, 7)]).reshape(2, 3)
        np.testing.assert_allclose(cols @ cols.T, np.eye(2), atol=1e-6)
    finally:
        teleop.disconnect()


def test_control_status_uses_stream_timestamp_and_validates_raw_pose(sdk):
    sdk.get_time_stamp_ns.return_value = 1234
    teleop = BiPico4(BiPico4Config())
    try:
        teleop.connect()
        stamp, grips = teleop.get_control_status()
        assert stamp == 1234 and grips == {"left": 0.0, "right": 0.0}
        sdk.get_left_controller_pose.return_value = [0.0] * 7
        with pytest.raises(ValueError):
            teleop.get_control_status()
    finally:
        teleop.disconnect()


def test_raw_xr_sample_includes_head_and_bypasses_clutch_mapping(sdk):
    sdk.get_headset_pose.return_value = [0, 1.5, 0, 0, 0, 0, 1]
    sdk.get_time_stamp_ns.return_value = 1234
    teleop = BiPico4(BiPico4Config())
    try:
        teleop.connect()
        stamp, poses = teleop.get_xr_sample()
        assert stamp == 1234
        np.testing.assert_array_equal(poses["head"], sdk.get_headset_pose.return_value)
        assert set(poses) == {"head", "left", "right"}
        sdk.get_headset_pose.return_value = [0] * 7
        with pytest.raises(ValueError, match="head XR pose"):
            teleop.get_xr_sample()
    finally:
        teleop.disconnect()


def test_controller_sample_retries_stream_changes_and_reads_raw_triggers(sdk):
    sdk.get_headset_pose.return_value = [0, 1.5, 0, 0, 0, 0, 1]
    teleop = BiPico4(BiPico4Config())
    try:
        teleop.connect()
        sdk.get_time_stamp_ns.side_effect = [1, 1, 2, 2, 2, 2]
        sdk.get_left_trigger.return_value = 0.25
        sdk.get_right_trigger.return_value = 0.75
        stamp, poses, triggers = teleop.get_controller_sample()
        assert stamp == 2
        assert set(poses) == {"head", "left", "right"}
        assert triggers == {"left": 0.25, "right": 0.75}
        sdk.get_hand_input_sample.assert_not_called()
        sdk.get_left_grip.assert_not_called()
    finally:
        teleop.disconnect()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_controller_sample_rejects_invalid_trigger(sdk, value):
    sdk.get_headset_pose.return_value = [0, 1.5, 0, 0, 0, 0, 1]
    sdk.get_time_stamp_ns.return_value = 123
    teleop = BiPico4(BiPico4Config())
    try:
        teleop.connect()
        sdk.get_left_trigger.return_value = value
        with pytest.raises(ValueError, match="trigger"):
            teleop.get_controller_sample()
    finally:
        teleop.disconnect()


def test_controller_sample_rejects_unstable_stream(sdk):
    sdk.get_headset_pose.return_value = [0, 1.5, 0, 0, 0, 0, 1]
    teleop = BiPico4(BiPico4Config())
    try:
        teleop.connect()
        sdk.get_time_stamp_ns.side_effect = range(1, 10)
        with pytest.raises(RuntimeError):
            teleop.get_controller_sample()
    finally:
        teleop.disconnect()


def test_rebase_one_arm_preserves_other_target(sdk):
    teleop = BiPico4(BiPico4Config())
    try:
        teleop.connect()
        teleop.reset_arm_to_pose("left", np.array([0.3, 0.2, 0.5, 1, 0, 0, 0]))
        action = teleop.get_action()
        assert action["left_tcp.x"] == pytest.approx(0.3)
        assert action["right_tcp.x"] == 0.0
    finally:
        teleop.disconnect()


def test_cached_grip_states_use_pico_hysteresis(sdk):
    teleop = BiPico4(BiPico4Config())
    try:
        teleop.connect()
        sdk.get_left_grip.return_value = 1.0
        teleop.get_action()
        assert teleop.get_arm_enabled() == {"left": True, "right": False}
        middle = (teleop.config.grip_enable_threshold + teleop.config.grip_disable_threshold) / 2
        sdk.get_left_grip.return_value = middle
        teleop.get_action()
        assert teleop.get_arm_enabled()["left"]
        sdk.get_left_grip.return_value = 0.0
        teleop.get_action()
        assert not teleop.get_arm_enabled()["left"]
        sdk.get_left_grip.return_value = middle
        teleop.get_action()
        assert not teleop.get_arm_enabled()["left"]
    finally:
        teleop.disconnect()


def test_g1_session_uses_real_pico_clutch_and_rebases_limited_commands(sdk):
    now = [1.0]
    robot = MagicMock()
    robot.config = SimpleNamespace(read_only=False, arm_commands_only=True, gravity_compensation=False)
    robot.control_space, robot.controller = "cartesian_abs", None
    robot.state_age_s.return_value = 0.0
    robot.get_observation.return_value = dict.fromkeys(ARM_KEYS, 0.0)
    solver = robot.arm_ik

    def fk(q):
        poses = [np.eye(4), np.eye(4)]
        poses[0][0, 3], poses[1][0, 3] = q[0], q[7]
        return tuple(poses)

    def solve(left, right, q, **kwargs):
        q = q.copy()
        q[0], q[7] = left[0, 3], right[0, 3]
        return q, np.zeros(14)

    solver.forward_kinematics.side_effect = fk
    solver.solve_ik.side_effect = solve
    solver.last_solve_success = True
    solver._arm_reorder_pin_to_g1 = np.arange(14)
    solver.reduced_robot.model = SimpleNamespace(
        lowerPositionLimit=np.full(14, -1.0), upperPositionLimit=np.full(14, 1.0)
    )
    teleop = BiPico4(
        BiPico4Config(pos_sensitivity=1.0, filter_window_size=1, max_pos_velocity=0.0, max_rot_velocity=0.0)
    )
    session = PicoArmSession(robot, teleop, clock=lambda: now[0])

    def tick():
        now[0] += 1 / 30
        sdk.get_time_stamp_ns.return_value = int(now[0] * 1e9)
        return session.tick()

    try:
        session.initialize()
        assert tick()["status"] == "waiting"
        assert tick()["status"] == "ready"
        sdk.get_left_grip.return_value = 1.0
        tick()
        sdk.get_left_controller_pose.return_value = [0, 0, -0.02, 0, 0, 0, 1]
        moving = tick()
        assert moving["joint_action"][ARM_KEYS[0]] == pytest.approx(0.01)
        assert moving["joint_action"][ARM_KEYS[7]] == 0.0
        sdk.get_left_grip.return_value = 0.0
        assert tick()["joint_action"] == pytest.approx(moving["joint_action"])
        sdk.get_left_grip.return_value = 1.0
        assert tick()["joint_action"] == pytest.approx(moving["joint_action"])
        sdk.get_left_controller_pose.return_value = [0, 0, -0.025, 0, 0, 0, 1]
        assert tick()["joint_action"][ARM_KEYS[0]] == pytest.approx(0.015)
    finally:
        teleop.disconnect()


@pytest.mark.parametrize("pre_init", [False, True])
def test_delayed_controller_stream_can_start_after_eight_seconds(sdk, pre_init):
    valid = [0, 0, 0, 0, 0, 0, 1]
    sdk.get_left_controller_pose.side_effect = chain(repeat([0.0] * 7, 80), repeat(valid))
    sdk.get_right_controller_pose.side_effect = chain(repeat([0.0] * 7, 80), repeat(valid))
    teleop = BiPico4(BiPico4Config())
    try:
        if pre_init:
            teleop.pre_init()
        teleop.connect()
        assert teleop.is_connected
        assert sdk.get_left_controller_pose.call_count >= 81
    finally:
        if teleop.is_connected:
            teleop.disconnect()
    sdk.init.assert_called_once()
    sdk.close.assert_called_once()


def test_startup_timeout_identifies_missing_controller(sdk):
    sdk.get_right_controller_pose.return_value = [0.0] * 7
    teleop = BiPico4(BiPico4Config())
    with pytest.raises(DeviceNotConnectedError, match="left=ready, right=missing"):
        teleop.connect()
    assert not teleop.is_connected
    sdk.close.assert_called_once()
