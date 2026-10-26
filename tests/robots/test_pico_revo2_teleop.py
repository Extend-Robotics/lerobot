"""Hands-only command gating without SDK or robot access."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.robots.brainco_revo2.pico_teleop import PicoRevo2Session
from lerobot.teleoperators.pico4.hand_input import HandTrackingUnavailableError
from lerobot.teleoperators.pico4.revo2_retargeting import MOTOR_NAMES


@pytest.fixture
def state():
    now = [1.0]
    hands = {}
    for side in ("left", "right"):
        hand = Mock()
        hand.config = SimpleNamespace(motor_names=MOTOR_NAMES, read_only=False)
        hand.action_features = dict.fromkeys((f"{name}.pos" for name in MOTOR_NAMES), float)
        hand.get_observation.return_value = dict.fromkeys(hand.action_features, 0.2)
        hand.prepare_action.side_effect = lambda action: action
        hand.send_action.side_effect = lambda action: action
        hands[side] = hand
    source = Mock()
    source.get_hand_sample.return_value = (10, {"left": None, "right": None})
    mapper = Mock(sides=("left", "right"))
    mapper.retarget.return_value = {
        f"{side}_hand.{key}": 1.0 for side, hand in hands.items() for key in hand.action_features
    }
    session = PicoRevo2Session(hands, source, mapper, execute=True, clock=lambda: now[0])
    return SimpleNamespace(session=session, hands=hands, source=source, mapper=mapper, now=now)


def test_no_commands_before_start_and_preview_never_writes(state):
    state.session.tick()
    state.session.execute = False
    state.session.start()
    assert state.session.tick()["status"] == "preview"
    for hand in state.hands.values():
        hand.send_action.assert_not_called()


def test_start_bounds_commands_from_measured_state_and_reuses_same_frame(state):
    state.session.start()
    state.now[0] += 0.02
    report = state.session.tick()
    assert report["status"] == "executing"
    assert list(report["sent_action"].values()) == pytest.approx([0.22] * 12)
    state.now[0] += 0.02
    state.session.tick()
    state.mapper.retarget.assert_called_once()


def test_loss_stops_updates_and_recovery_requires_resume(state):
    state.session.start()
    state.source.get_hand_sample.side_effect = HandTrackingUnavailableError("lost")
    assert state.session.tick()["status"] == "paused"
    state.source.get_hand_sample.side_effect = None
    assert state.session.tick()["status"] == "paused"
    for hand in state.hands.values():
        hand.send_action.assert_not_called()
    state.session.start()
    state.now[0] += 0.01
    assert state.session.tick()["status"] == "executing"


def test_loss_during_solve_sends_nothing(state):
    state.session.start()
    state.source.get_hand_sample.side_effect = [(10, {}), HandTrackingUnavailableError("lost during solve")]
    assert state.session.tick()["status"] == "paused"
    for hand in state.hands.values():
        hand.send_action.assert_not_called()


def test_stale_feedback_on_second_hand_blocks_both(state):
    state.session.start()
    state.hands["right"].get_observation.side_effect = TimeoutError("stale")
    with pytest.raises(TimeoutError):
        state.session.tick()
    for hand in state.hands.values():
        hand.send_action.assert_not_called()


def test_loop_deadline_requires_resume(state):
    state.session.start()
    state.now[0] += 0.3
    assert state.session.tick()["status"] == "paused"
    for hand in state.hands.values():
        hand.send_action.assert_not_called()


def test_slow_solver_cannot_publish(state):
    def slow(_):
        state.now[0] += 0.3
        return state.mapper.retarget.return_value

    state.mapper.retarget.side_effect = slow
    state.session.start()
    assert state.session.tick()["status"] == "paused"
    for hand in state.hands.values():
        hand.send_action.assert_not_called()
