"""Hand contracts, DDS driver messages, stale-state behavior and transport lifecycle."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import draccus
import pytest

from lerobot.robots import make_robot_from_config
from lerobot.robots.brainco_revo2 import BraincoRevo2Hand, BraincoRevo2HandConfig
from lerobot.robots.brainco_revo2.transport import DdsHandTransport, ZmqHandTransport


@pytest.fixture
def hand(tmp_path):
    robot = BraincoRevo2Hand(BraincoRevo2HandConfig(calibration_dir=tmp_path, connect_timeout_s=0.02))
    robot._transport = MagicMock()
    robot._transport.read.side_effect = lambda: ([0.1] * 6, time.monotonic())
    yield robot
    robot.disconnect()


def test_factory_and_disconnected_schema(tmp_path):
    robot = make_robot_from_config(BraincoRevo2HandConfig(calibration_dir=tmp_path))
    assert isinstance(robot, BraincoRevo2Hand)
    assert len(robot.action_features) == 6
    assert robot.observation_features == robot.action_features
    assert not robot.is_connected
    with pytest.raises(ConnectionError):
        robot.get_observation()
    with pytest.raises(ConnectionError):
        robot.send_action(dict.fromkeys(robot.action_features, 0.5))


def test_config_cli():
    config = draccus.parse(BraincoRevo2HandConfig, args=["--side=right", "--read_only=true"])
    assert config.ports == (6004, 6005)
    assert config.read_only


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": "invalid"},
        {"transport": "serial"},
        {"state_timeout_s": float("nan")},
        {"connect_timeout_s": 0},
        {"speed": 2},
        {"motor_names": ("a",) * 6},
        {"state_port": 0},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        BraincoRevo2HandConfig(**kwargs)


def test_feedback_and_clipped_sent_action(hand):
    hand.connect()
    assert hand.get_observation() == dict.fromkeys(hand.action_features, 0.1)
    targets = dict(zip(hand.action_features, [-1, 0, 0.2, 0.5, 1, 2], strict=True))
    assert list(hand.send_action(targets).values()) == [0, 0, 0.2, 0.5, 1, 1]
    hand._transport.write.assert_called_once_with([0, 0, 0.2, 0.5, 1, 1])


def test_read_only_never_writes(hand):
    hand.config.read_only = True
    hand.connect()
    assert hand.get_observation()
    with pytest.raises(PermissionError):
        hand.send_action(dict.fromkeys(hand.action_features, 0.5))
    hand.disconnect()
    hand._transport.write.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_target_rejected(hand, value):
    hand.connect()
    with pytest.raises(ValueError):
        hand.send_action(dict.fromkeys(hand.action_features, value))
    hand._transport.write.assert_not_called()


def test_partial_target_rejected(hand):
    hand.connect()
    with pytest.raises(ValueError):
        hand.send_action({"index.pos": 0.4})
    hand._transport.write.assert_not_called()


def test_stale_state_blocks_observations_and_commands(hand):
    hand.connect()
    hand._transport.read.side_effect = lambda: ([0.1] * 6, time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        hand.get_observation()
    with pytest.raises(TimeoutError):
        hand.send_action(dict.fromkeys(hand.action_features, 0.5))
    hand._transport.write.assert_not_called()


def test_connect_timeout_cleans_up_and_allows_reconnect(hand):
    hand._transport.read.side_effect = None
    hand._transport.read.return_value = None
    with pytest.raises(TimeoutError):
        hand.connect()
    assert not hand.is_connected
    hand._transport.close.assert_called_once()
    hand._transport.read.side_effect = lambda: ([0.1] * 6, time.monotonic())
    hand.connect()
    assert hand.is_connected
    with pytest.raises(ConnectionError):
        hand.connect()


@pytest.mark.parametrize("read_only", [True, False])
def test_dds_topics_motor_order_and_read_only(read_only):
    module = "lerobot.robots.brainco_revo2.transport"
    transport = DdsHandTransport(BraincoRevo2HandConfig(side="right", transport="dds", read_only=read_only))
    with (
        patch(f"{module}.initialize_dds"),
        patch(f"{module}.ChannelSubscriber", create=True) as sub,
        patch(f"{module}.ChannelPublisher", create=True) as pub,
        patch(f"{module}.MotorStates_", create=True),
        patch(f"{module}.MotorCmds_", create=True) as msg,
        patch(f"{module}.unitree_go_msg_dds__MotorCmd_", side_effect=SimpleNamespace, create=True),
    ):
        transport.connect()
        assert sub.call_args.args[0] == "rt/brainco/right/state"
        callback = sub.return_value.Init.call_args.args[0]
        callback(SimpleNamespace(states=[SimpleNamespace(q=i / 10) for i in range(6)]))
        assert transport.read()[0] == [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        callback(SimpleNamespace(states=[SimpleNamespace(q=float("nan"))] * 6))
        assert transport.read()[0][0] == 0  # Malformed feedback doesn't refresh state.
        if read_only:
            pub.assert_not_called()
            with pytest.raises(PermissionError):
                transport.write([0.5] * 6)
        else:
            assert pub.call_args.args[0] == "rt/brainco/right/cmd"
            transport.write([i / 10 for i in range(6)])
            assert [c.q for c in msg.return_value.cmds] == [i / 10 for i in range(6)]
            assert all(c.dq == 1 for c in msg.return_value.cmds)
        transport.close()
        sub.return_value.Close.assert_called_once()
        assert transport.read() is None


def test_zmq_read_only_lifecycle_and_age():
    zmq = pytest.importorskip("zmq")
    transport = ZmqHandTransport(BraincoRevo2HandConfig(read_only=True))
    socket = MagicMock()
    with patch.object(zmq.Context, "instance") as context:
        context.return_value.socket.return_value = socket
        transport.connect()
        context.return_value.socket.assert_called_once_with(zmq.SUB)
        socket.recv_json.return_value = {
            "side": "left",
            "positions": [0.3] * 6,
            "age_s": 0.4,
            "observed_at": time.time() - 0.4,
        }
        before = time.monotonic()
        q, received = transport.read()
        assert q == [0.3] * 6
        assert received <= before - 0.39
        socket.recv_json.side_effect = zmq.Again
        assert transport.read() == (q, received)
        transport.close()
        socket.close.assert_called_once_with(linger=0)


def test_g1_nested_config_cli_and_simulation_rejection():
    from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

    config = draccus.parse(
        UnitreeG1Config,
        args=[
            "--is_simulation=false",
            "--read_only=true",
            "--left_hand.transport=dds",
            "--right_hand.transport=zmq",
        ],
    )
    assert config.left_hand.transport == "dds"
    assert config.right_hand.transport == "zmq"
    with pytest.raises(ValueError, match="no Revo2 simulation"):
        UnitreeG1Config(left_hand=BraincoRevo2HandConfig())


def test_zmq_roundtrip_with_real_sockets():
    zmq = pytest.importorskip("zmq")
    context = zmq.Context.instance()
    state = context.socket(zmq.PUB)
    commands = context.socket(zmq.PULL)
    state_port = state.bind_to_random_port("tcp://127.0.0.1")
    command_port = commands.bind_to_random_port("tcp://127.0.0.1")
    transport = ZmqHandTransport(
        BraincoRevo2HandConfig(
            side="right",
            robot_ip="127.0.0.1",
            state_port=state_port,
            command_port=command_port,
        )
    )
    try:
        transport.connect()
        deadline = time.monotonic() + 2
        sample = None
        while sample is None and time.monotonic() < deadline:
            state.send_json(
                {"side": "right", "positions": [0.2] * 6, "age_s": 0.1, "observed_at": time.time() - 0.1}
            )
            time.sleep(0.01)
            sample = transport.read()
        assert sample is not None and sample[0] == [0.2] * 6
        assert time.monotonic() - sample[1] >= 0.1
        transport.write([0.3] * 6)
        assert commands.poll(1000)
        command = commands.recv_json()
        assert command["positions"] == [0.3] * 6
        assert command["expires_at"] > time.time()
    finally:
        transport.close()
        state.close(linger=0)
        commands.close(linger=0)


@pytest.mark.parametrize("mode", ["readonly", "valid", "expired", "invalid_speed"])
def test_hand_bridge_read_only_and_command_validation(mode, tmp_path):
    zmq = pytest.importorskip("zmq")
    from lerobot.robots.brainco_revo2.bridge import serve_hand

    config = BraincoRevo2HandConfig(transport="dds", read_only=mode == "readonly", calibration_dir=tmp_path)
    hand = MagicMock()
    hand.config = config
    hand.action_features = dict.fromkeys([f"motor_{i}.pos" for i in range(6)], float)
    hand._transport.read.return_value = ([0.25] * 6, time.monotonic())
    stop = MagicMock()
    stop.is_set.side_effect = [False, True]
    state, commands = MagicMock(), MagicMock()
    commands.poll.return_value = True
    commands.recv_json.return_value = {
        "positions": [0.4] * 6,
        "speed": float("nan") if mode == "invalid_speed" else 0.3,
        "expires_at": time.time() + (-1 if mode == "expired" else 0.4),
    }
    with (
        patch("lerobot.robots.brainco_revo2.bridge.BraincoRevo2Hand", return_value=hand),
        patch.object(zmq.Context, "instance") as context,
    ):
        context.return_value.socket.side_effect = [state, commands]
        serve_hand(config, stop, "127.0.0.1")
        assert state.send_json.call_args.args[0]["positions"] == [0.25] * 6
        assert state.send_json.call_args.args[0]["observed_at"] <= time.time()
        if mode == "readonly":
            context.return_value.socket.assert_called_once_with(zmq.PUB)
        if mode == "valid":
            hand.send_action.assert_called_once_with(dict.fromkeys(hand.action_features, 0.4))
            assert config.speed == 0.3
        else:
            hand.send_action.assert_not_called()
        hand.disconnect.assert_called_once()
        state.close.assert_called_once_with(linger=0)


def test_zmq_old_buffered_feedback_preserves_source_age():
    pytest.importorskip("zmq")
    transport = ZmqHandTransport(BraincoRevo2HandConfig(read_only=True))
    transport.subscriber = MagicMock()
    transport.subscriber.recv_json.return_value = {
        "side": "left",
        "positions": [0.2] * 6,
        "age_s": 0.001,
        "observed_at": time.time() - 10,
    }
    assert time.monotonic() - transport.read()[1] >= 9.9
    transport.close()
