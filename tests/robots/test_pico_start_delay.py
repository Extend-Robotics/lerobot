"""The optional preview launcher uses the standard joint pipeline and explicit start."""

import sys
from unittest.mock import Mock

import pytest

from examples.unitree_g1d import teleop_pico4


@pytest.mark.parametrize(
    "execute, response, expected_writes",
    [
        (False, "r", 0),
        (True, "r", 1),
        (True, "q", 0),
    ],
)
def test_launcher_preview_and_explicit_start(monkeypatch, execute, response, expected_writes):
    now = [0.0]
    robot = Mock()
    teleop = Mock(is_connected=True)
    processor = Mock(return_value={"kLeftElbow.pos": 0.2})
    robot.send_action.side_effect = lambda action: action
    monkeypatch.setattr(teleop_pico4, "UnitreeG1", Mock(return_value=robot))
    monkeypatch.setattr(teleop_pico4, "BiPico4", Mock(return_value=teleop))
    monkeypatch.setattr(teleop_pico4, "make_default_processors", Mock(return_value=(processor, None, None)))
    monkeypatch.setattr("builtins.input", lambda _: response)
    monkeypatch.setattr(teleop_pico4.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(teleop_pico4.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teleop_pico4.py",
            "--robot-ip",
            "test",
            "--duration",
            "0.01",
            *(["--execute"] if execute else []),
        ],
    )
    teleop_pico4.main()
    assert robot.send_action.call_count == expected_writes
    if expected_writes:
        assert now[0] >= 2
        robot.send_action.assert_called_once_with(processor.return_value)
    robot.disconnect.assert_called_once()
    teleop.disconnect.assert_called_once()
