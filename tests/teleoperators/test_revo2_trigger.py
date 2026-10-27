"""Controller trigger closure in Revo2 driver order."""

import pytest

from lerobot.teleoperators.pico4.revo2_trigger import PicoRevo2TriggerMapper


@pytest.mark.parametrize("left,right", [(0, 1), (1, 0), (0.25, 0.75)])
def test_trigger_closure_is_analog_and_independent(left, right):
    action = PicoRevo2TriggerMapper().map({"left": left, "right": right})
    assert list(action.values()) == [left] * 6 + [right] * 6
    assert list(action)[:2] == ["left_hand.motor_0.pos", "left_hand.motor_1.pos"]


def test_single_hand_only_requires_its_trigger():
    action = PicoRevo2TriggerMapper(("left",)).map({"left": 0.5})
    assert len(action) == 6 and all(key.startswith("left_hand.") for key in action)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_trigger_cannot_generate_commands(value):
    with pytest.raises(ValueError, match="triggers"):
        PicoRevo2TriggerMapper().map({"left": 0.5, "right": value})
