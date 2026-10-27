"""The launcher delays r without blocking tracking checks or pause/quit input."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.unitree_g1d import teleop_pico4


@pytest.mark.parametrize(
    "input_mode, keys, expected_starts",
    [
        ("controllers", [(0.0, "r")], [2.0]),
        ("controller_hands", [(0.0, "r")], [2.0]),
        ("controller_hands", [(0.0, "r"), (0.5, "p")], []),
        ("controller_hands", [(0.0, "r"), (0.5, "q")], []),
        ("controller_hands", [(0.0, "r"), (2.2, "p"), (2.3, "r")], [2.0, 4.3]),
    ],
)
def test_start_delay_keeps_loop_running(monkeypatch, input_mode, keys, expected_starts):
    combined = input_mode == "controller_hands"
    input_mode = "controllers"
    now = [0.0]
    pending = list(keys)
    started = []
    ticks = []
    session = Mock()
    session.start.side_effect = lambda: started.append(now[0])
    session.tick.side_effect = lambda: ticks.append(now[0]) or {"status": "ready"}
    robot = Mock()
    teleop = Mock(is_connected=True)
    for name, value in (
        ("UnitreeG1", robot),
        ("BiPico4", teleop),
        ("UnitreeG1Config", Mock()),
        ("PicoHandArmSession", session),
        ("UnitreeXrArmSession", session),
    ):
        monkeypatch.setattr(teleop_pico4, name, Mock(return_value=value))
    monkeypatch.setattr(teleop_pico4.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(teleop_pico4.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    stdin = SimpleNamespace(isatty=lambda: True, readline=lambda: pending.pop(0)[1])
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        teleop_pico4.select,
        "select",
        lambda *args: ([stdin], [], []) if pending and now[0] + 1e-9 >= pending[0][0] else ([], [], []),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teleop_pico4.py",
            "--robot-ip",
            "test",
            "--input",
            input_mode,
            "--execute",
            "--fps",
            "10",
            "--duration",
            "5",
            *(["--hands", "both"] if combined else []),
        ],
    )
    teleop_pico4.main()
    assert len(started) == len(expected_starts)
    for actual, expected in zip(started, expected_starts, strict=True):
        assert expected - 1e-9 <= actual <= expected + 0.1 + 1e-9
    assert sum(t < 2.0 for t in ticks) >= (5 if any(key == "q" for _, key in keys) else 19)
    if any(key == "p" for _, key in keys):
        session.pause.assert_called_once()
    session.close.assert_called_once()
    robot.disconnect.assert_called_once()
    if combined:
        config = teleop_pico4.UnitreeG1Config.call_args.kwargs
        assert config["left_hand"].transport == config["right_hand"].transport == "zmq"
