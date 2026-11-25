from unittest.mock import Mock

import pytest

from lerobot.robots.tianji import Tianji, TianjiConfig, tianji as driver


@pytest.mark.parametrize("read_only, persistent", [(False, False), (True, False), (False, True)])
def test_connect_recovers_before_motion(monkeypatch, read_only, persistent):
    robot = Tianji(
        TianjiConfig(robot_ip="127.0.0.1", clear_faults_on_connect=True, move_to_start_on_connect=True)
    )
    sdk = Mock()
    monkeypatch.setattr(driver, "Marvin_Robot", lambda: sdk)
    sdk.connect.return_value = sdk.release_robot.return_value = 1
    feedback = {
        "outputs": [{"frame_serial": 1, "fb_joint_pos": [0.0] * 7}] * 2,
        "states": [{"err_code": 5, "cur_state": driver.FAULT}, {"err_code": 0, "cur_state": driver.IDLE}],
    }
    sdk.subscribe.return_value = feedback
    if not persistent:
        sdk.clear_faults.side_effect = lambda arms: feedback["states"][0].update(
            err_code=0, cur_state=driver.IDLE
        )
    clock = iter(range(100))
    monkeypatch.setattr(driver, "monotonic", lambda: next(clock))
    monkeypatch.setattr(driver, "sleep", lambda _: None)
    robot.enable_position_control = Mock()
    robot._move_to_start = Mock()
    if persistent:
        with pytest.raises(OSError, match="fault code"):
            robot.connect()
        assert not robot.is_connected
    else:
        robot.connect(read_only=read_only)
        robot.disconnect()
    if read_only:
        sdk.clear_faults.assert_not_called()
    else:
        sdk.clear_faults.assert_called_once_with(("A",))
    assert robot._move_to_start.called == (not persistent and not read_only)
    assert robot.enable_position_control.called == robot._move_to_start.called


def test_start_trajectory_follows_feedback(monkeypatch):
    robot = Tianji(TianjiConfig(robot_ip="127.0.0.1", max_target_delta_deg=2))
    positions = dict.fromkeys(driver.JOINT_NAMES, 0.0)
    clock = [0.0]
    monkeypatch.setattr(driver, "monotonic", lambda: clock[0])
    monkeypatch.setattr(driver, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    robot._read_feedback = lambda: {
        "outputs": [{"frame_serial": 1, "fb_joint_pos": list(positions.values())[i : i + 7]} for i in (0, 7)],
        "states": [{"cur_state": driver.POSITION_FOLLOW, "err_code": 0}] * 2,
    }

    def follow(action):
        for key, target in action.items():
            assert abs(target - positions[key]) <= 2
            positions[key] += 0.6 * (target - positions[key])

    robot.send_action = follow
    robot._move_to_start(positions.copy())
    assert list(positions.values()) == pytest.approx(driver.WUJI_START_JOINTS, abs=1)


@pytest.mark.parametrize("control_mode, state", [("position", 1), ("joint_impedance", 3)])
def test_activation_configures_control_mode_before_startup(monkeypatch, control_mode, state):
    from unittest.mock import call

    robot = Tianji(
        TianjiConfig(robot_ip="127.0.0.1", control_mode=control_mode, move_to_start_on_connect=True)
    )
    sdk = Mock()
    sdk.connect.return_value = sdk.release_robot.return_value = 1
    cabinet_state = driver.IDLE
    serial = 0

    def feedback(_):
        nonlocal serial
        serial += 1
        return {
            "outputs": [{"frame_serial": serial, "fb_joint_pos": [0.0] * 7}] * 2,
            "states": [{"cur_state": cabinet_state, "err_code": 0}] * 2,
        }

    def set_state(next_state):
        nonlocal cabinet_state
        cabinet_state = next_state

    sdk.subscribe.side_effect = feedback
    sdk.set_arm_state.side_effect = set_state
    monkeypatch.setattr(driver, "Marvin_Robot", lambda: sdk)
    monkeypatch.setattr(driver, "sleep", lambda _: None)
    robot._move_to_start = lambda initial: robot.send_action(initial)
    robot.connect()
    assert robot.last_sent_action == dict.fromkeys(driver.JOINT_NAMES, 0.0)
    robot.disconnect()
    assert robot.last_sent_action is None
    command_calls = [entry for entry in sdk.method_calls if entry[0] != "subscribe"]
    expected = [
        call.connect("127.0.0.1"),
        call.send_joint_positions([0.0] * 7, [0.0] * 7, speed_ratio=10),
        call.set_arm_state(state),
    ]
    if control_mode == "joint_impedance":
        expected.extend(
            [
                call.set_joint_impedance_gains((2, 2, 2, 1.6, 1, 1, 1), (0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2)),
                call.select_joint_impedance(),
            ]
        )
    expected += [
        call.send_joint_positions([0.0] * 7, [0.0] * 7),
        call.set_arm_state(driver.IDLE),
        call.release_robot(),
    ]
    assert command_calls == expected
