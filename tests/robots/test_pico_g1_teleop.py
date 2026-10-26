"""PICO-to-IK session tests using synthetic frames, clocks, and robot feedback."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.unitree_g1.pico_teleop import (
    ARM_KEYS,
    SIDES,
    PicoArmSession,
    action_to_pose,
    pose_to_quaternion,
)
from lerobot.teleoperators.bi_pico4 import BiPico4Config


def fk(q):
    poses = [np.eye(4), np.eye(4)]
    for index, pose in enumerate(poses):
        pose[:3, 3] = [0.3 + q[index * 7], 0.2 if index == 0 else -0.2, 0.5]
        angle = q[index * 7 + 1]
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    return tuple(poses)


def pose_action(poses):
    result = {}
    for side, pose in zip(SIDES, poses, strict=True):
        values = np.concatenate((pose[:3, 3], pose[:3, 0], pose[:3, 1]))
        result.update(
            zip(
                (f"{side}_tcp.{field}" for field in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")),
                values,
                strict=True,
            )
        )
    return result


@pytest.fixture
def setup_session():
    now = [1.0]
    robot = Mock()
    robot.config = SimpleNamespace(read_only=False, arm_commands_only=True, gravity_compensation=False)
    robot.controller = None
    robot.control_space = "cartesian_abs"
    robot.state_age_s.return_value = 0.0
    robot.get_observation.return_value = dict.fromkeys(ARM_KEYS, 0.0)
    solver = robot.arm_ik
    solver.forward_kinematics.side_effect = fk
    solver.last_solve_success = True
    solver._arm_reorder_pin_to_g1 = np.arange(14)
    solver.reduced_robot.model = SimpleNamespace(
        lowerPositionLimit=np.full(14, -1.0), upperPositionLimit=np.full(14, 1.0)
    )

    def solve(left, right, q, *, smooth=True):
        candidate = q.copy()
        for index, target in enumerate((left, right)):
            candidate[index * 7] = target[0, 3] - 0.3
            candidate[index * 7 + 1] = np.arctan2(target[1, 0], target[0, 0])
        return candidate, np.zeros(14)

    solver.solve_ik.side_effect = solve
    teleop = Mock(config=BiPico4Config())
    grips = dict.fromkeys(SIDES, 0.0)
    stamp = [1]
    teleop.get_control_status.side_effect = lambda: (stamp[0], grips.copy())
    targets = list(fk(np.zeros(14)))
    teleop.get_action.side_effect = lambda: pose_action(targets)
    teleop.get_arm_enabled.side_effect = lambda: {side: grip > 0.8 for side, grip in grips.items()}

    def reset_arm(side, pose):
        index = SIDES.index(side)
        targets[index] = np.eye(4)
        targets[index][:3, 3] = pose[:3]
        w, x, y, z = pose[3:7]
        targets[index][:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]

    teleop.reset_arm_to_pose.side_effect = reset_arm
    session = PicoArmSession(robot, teleop, clock=lambda: now[0])
    session.initialize()

    def tick(dt=1 / 30, advance_stamp=True):
        now[0] += dt
        if advance_stamp:
            stamp[0] += 1
        return session.tick()

    tick()  # Observe source timestamp baseline; progression is required.
    assert tick()["status"] == "ready"
    return SimpleNamespace(
        session=session,
        robot=robot,
        teleop=teleop,
        solver=solver,
        now=now,
        grips=grips,
        stamp=stamp,
        targets=targets,
        tick=tick,
    )


def test_measured_pose_initialization(setup_session):
    state = setup_session
    kwargs = state.teleop.connect.call_args.kwargs
    np.testing.assert_allclose(kwargs["left_tcp_pose_quat"], [0.3, 0.2, 0.5, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(kwargs["right_tcp_pose_quat"], [0.3, -0.2, 0.5, 1, 0, 0, 0, 0])
    state.robot.send_arm_joint_action.assert_not_called()


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 1)])
def test_independent_arm_and_clutch_reengagement(setup_session, side, index):
    state = setup_session
    state.grips[side] = 1.0
    state.tick()  # Grip latches a new reference.
    state.targets[index][0, 3] += 0.005
    result = state.tick()
    assert result["status"] == "executing"
    values = np.array(list(result["joint_action"].values()))
    assert values[index * 7] == pytest.approx(0.005)
    np.testing.assert_array_equal(values[(1 - index) * 7 : (2 - index) * 7], np.zeros(7))
    state.grips[side] = 0.0
    state.targets[index][0, 3] += 0.05
    hold = state.tick()
    assert hold["joint_action"] == result["joint_action"]
    state.grips[side] = 1.0
    resumed = state.tick()
    assert resumed["joint_action"] == result["joint_action"]


def test_readonly_never_publishes_or_relaxes(setup_session):
    state = setup_session
    state.robot.config.read_only = True
    state.grips["left"] = 1.0
    assert state.tick()["status"] == "preview"
    state.targets[0][0, 3] += 0.005
    assert state.tick()["joint_action"][ARM_KEYS[0]] > 0
    state.robot.state_age_s.return_value = 1.0
    assert state.tick()["status"] == "preview"
    state.robot.send_arm_joint_action.assert_not_called()


def test_stationary_input_is_fresh_if_timestamp_advances(setup_session):
    state = setup_session
    state.grips["left"] = 1.0
    for _ in range(20):
        assert state.tick()["status"] == "executing"


def test_stalled_stream_raises_for_caller_cleanup(setup_session):
    state = setup_session
    state.grips["left"] = 1.0
    state.tick()
    state.robot.send_arm_joint_action.reset_mock()
    with pytest.raises(TimeoutError, match="PICO stream is stale"):
        for _ in range(9):
            state.tick(advance_stamp=False)
    state.robot.send_arm_joint_action.reset_mock()
    with pytest.raises(TimeoutError):
        state.tick(advance_stamp=False)
    state.robot.send_arm_joint_action.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    [
        "invalid",
        "ik_failure",
        "limits",
        "tracking",
        "source_restart",
        "disconnect",
    ],
)
def test_faults_raise_without_publishing(setup_session, kind):
    state = setup_session
    state.grips["left"] = 1.0
    state.tick()
    state.robot.send_arm_joint_action.reset_mock()
    if kind == "invalid":
        state.targets[0][0, 0] = np.nan
    elif kind == "ik_failure":
        state.solver.last_solve_success = False
    elif kind == "limits":
        state.solver.solve_ik.side_effect = lambda *args, **kwargs: (np.full(14, 2.0), np.zeros(14))
    elif kind == "tracking":
        state.robot.get_observation.return_value = dict.fromkeys(ARM_KEYS, 1.0)
    elif kind == "source_restart":
        state.stamp[0] = 0
    else:
        state.teleop.get_control_status.side_effect = ConnectionError("PICO disconnected")
    with pytest.raises((ValueError, TimeoutError, ConnectionError)):
        state.tick()
    state.robot.send_arm_joint_action.assert_not_called()


def test_joint_velocity_uses_elapsed_time(setup_session):
    state = setup_session
    state.grips["left"] = 1.0
    state.tick()
    state.targets[0][0, 3] += 0.05
    report = state.tick(dt=0.02)
    assert report["status"] == "executing"
    assert report["joint_action"][ARM_KEYS[0]] == pytest.approx(0.3 * 0.02)


def test_velocity_interval_starts_at_last_command_not_last_loop(setup_session):
    state = setup_session
    state.grips["left"] = 1.0
    state.tick()
    state.targets[0][0, 3] += 0.05
    state.now[0] += 0.1  # Emulate a previous tick publishing late.
    state.session._last_command_at = state.now[0]
    report = state.tick(dt=0.02)
    assert report["joint_action"][ARM_KEYS[0]] == pytest.approx(0.3 * 0.02)


def test_slow_ik_cannot_publish_an_expired_candidate(setup_session):
    state = setup_session
    state.grips["left"] = 1.0
    state.tick()
    state.robot.send_arm_joint_action.reset_mock()
    original = state.solver.solve_ik.side_effect

    def slow(*args, **kwargs):
        state.now[0] += 0.3
        return original(*args, **kwargs)

    state.solver.solve_ik.side_effect = slow
    with pytest.raises(TimeoutError):
        state.tick()
    state.robot.send_arm_joint_action.assert_not_called()


def test_quaternion_and_6d_rotation_agree():
    q = np.zeros(14)
    q[1] = 0.4
    pose = fk(q)[0]
    output = pose_to_quaternion(pose)
    np.testing.assert_allclose(output[3:7], [np.cos(0.2), 0, 0, np.sin(0.2)])
    np.testing.assert_allclose(action_to_pose(pose_action((pose, pose)), "left"), pose)


@pytest.mark.parametrize(
    "execute,shutdown_error", [(False, None), (True, None), (False, "robot"), (False, "teleop")]
)
def test_example_defaults_and_cleanup_order(monkeypatch, execute, shutdown_error):
    path = Path(__file__).resolve().parents[2] / "examples/unitree_g1d/teleop_pico4.py"
    spec = importlib.util.spec_from_file_location("pico_g1_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    robot, teleop = Mock(), Mock(is_connected=True)
    make_robot = Mock(return_value=robot)
    monkeypatch.setattr(example, "UnitreeG1", make_robot)
    monkeypatch.setattr(example, "BiPico4", Mock(return_value=teleop))
    monkeypatch.setattr(example, "PicoArmSession", Mock())
    arguments = ["teleop_pico4.py", "--robot-ip", "mock", "--duration", "0.000000001"]
    if execute:
        arguments.append("--execute")
    monkeypatch.setattr(sys, "argv", arguments)
    closed = []

    def close(name):
        closed.append(name)
        if name == shutdown_error:
            raise RuntimeError("Cleanup failure")

    robot.disconnect.side_effect = lambda: close("robot")
    teleop.disconnect.side_effect = lambda: close("teleop")
    if shutdown_error:
        with pytest.raises(RuntimeError, match="Cleanup failure"):
            example.main()
    else:
        example.main()
    config = make_robot.call_args.args[0]
    assert config.read_only is not execute
    assert config.arm_commands_only
    assert config.kp[:15] == [0.0] * 12 + [60.0, 0.0, 40.0]
    assert closed == ["robot", "teleop"]


def test_preview_can_move_without_physical_tracking(setup_session):
    state = setup_session
    state.robot.config.read_only = True
    state.grips["left"] = 1.0
    state.tick()
    for i in range(60):
        angle = (i + 1) * 0.005
        state.targets[0][:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        report = state.tick()
        assert report["status"] == "preview"
    assert report["joint_action"][ARM_KEYS[1]] == pytest.approx(0.3)
    state.robot.send_arm_joint_action.assert_not_called()
    assert state.solver.solve_ik.call_args.kwargs == {"smooth": False}


def test_ik_residual_is_diagnostic(setup_session):
    state = setup_session
    state.grips["left"] = 1.0
    state.tick()
    state.solver.solve_ik.side_effect = lambda *args, **kwargs: (np.full(14, 0.1), np.zeros(14))
    report = state.tick()
    assert report["status"] == "executing"
    assert report["ik_residuals"]["left"]["position_m"] == pytest.approx(0.1)


def test_example_fault_exits_and_cleans_up(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "examples/unitree_g1d/teleop_pico4.py"
    spec = importlib.util.spec_from_file_location("pico_g1_fault", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    robot, teleop, session = Mock(), Mock(is_connected=True), Mock()
    session.tick.side_effect = TimeoutError("PICO stream is stale")
    monkeypatch.setattr(example, "UnitreeG1", Mock(return_value=robot))
    monkeypatch.setattr(example, "BiPico4", Mock(return_value=teleop))
    monkeypatch.setattr(example, "PicoArmSession", Mock(return_value=session))
    monkeypatch.setattr(sys, "argv", ["teleop_pico4.py", "--robot-ip", "mock"])
    with pytest.raises(SystemExit) as exc:
        example.main()
    assert exc.value.code == 1
    robot.disconnect.assert_called_once()
    teleop.disconnect.assert_called_once()


def test_startup_can_wait_for_grip_release(setup_session):
    state = setup_session
    state.session._ready = False
    state.grips["left"] = 1.0
    for _ in range(20):
        assert state.tick()["status"] == "waiting"
    state.robot.send_arm_joint_action.assert_not_called()
    state.grips["left"] = 0.0
    assert state.tick()["status"] == "ready"
    state.grips["left"] = 1.0
    assert state.tick()["status"] == "executing"


def test_execute_idle_status_is_ready(setup_session):
    state = setup_session
    assert state.tick()["status"] == "ready"
    state.robot.send_arm_joint_action.assert_not_called()


def test_large_ik_step_is_rate_limited_instead_of_rejected(setup_session):
    state = setup_session
    state.grips["left"] = 1.0
    state.tick()
    state.targets[0][0, 3] += 0.4
    result = state.tick(dt=0.02)
    assert result["status"] == "executing"
    assert result["joint_action"][ARM_KEYS[0]] == pytest.approx(0.3 * 0.02)
    state.robot.send_arm_joint_action.assert_called()


def test_pico_mvp_explicitly_opts_out_of_robot_feedback_age_check(setup_session):
    state = setup_session
    state.robot.state_age_s.return_value = 10.0
    state.grips["left"] = 1.0
    report = state.tick()
    assert report["status"] == "executing"
    state.robot.send_arm_joint_action.assert_called_once_with(
        report["joint_action"], 0.25, check_state_age=False
    )
