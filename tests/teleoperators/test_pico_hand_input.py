"""Hand input freshness and wrist selection without headset or robot access."""

import copy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.teleoperators.pico4.hand_input import HandTrackingUnavailableError, PicoHandInput


def packet():
    joints = np.tile([0, 0, 0, 0, 0, 0, 1.0], (26, 1))
    joints[0, :3] = [9, 9, 9]  # The palm must never become the wrist target.
    joints[1, :3] = [-0.2, 1.2, -0.4]
    return {
        "stamp": 10,
        "age_s": 0.01,
        "head_valid": True,
        "head": [0, 1.5, 0, 0, 0, 0, 1],
        "left": {"active": True, "joints": joints.tolist()},
        "right": {"active": True, "joints": joints.tolist()},
    }


@pytest.fixture
def source(monkeypatch):
    now = [1.0]
    monkeypatch.setattr("lerobot.teleoperators.pico4.hand_input.time.monotonic", lambda: now[0])
    data = packet()
    sdk = Mock()
    sdk.get_hand_input_sample.side_effect = lambda: copy.deepcopy(data)
    teleop = PicoHandInput()
    teleop._sdk = sdk
    return SimpleNamespace(teleop=teleop, sdk=sdk, data=data, now=now)


def test_selects_openxr_wrist_and_never_reads_controllers(source):
    stamp, poses = source.teleop.get_xr_sample()
    assert stamp == 10
    assert set(poses) == {"head", "left", "right"}
    np.testing.assert_allclose(poses["left"][:3], [-0.2, 1.2, -0.4])
    assert source.sdk.method_calls == [("get_hand_input_sample", (), {})]


@pytest.mark.parametrize("side", ["left", "right"])
def test_inactive_hand_invalidates_entire_sample(source, side):
    source.data[side]["active"] = False
    with pytest.raises(HandTrackingUnavailableError, match=side):
        source.teleop.get_xr_sample()


@pytest.mark.parametrize("age", [0.251, -1, float("nan")])
def test_old_or_invalid_receive_age_is_rejected(source, age):
    source.data["age_s"] = age
    with pytest.raises(HandTrackingUnavailableError, match="stale"):
        source.teleop.get_xr_sample()


def test_frozen_source_stamp_is_rejected_even_with_fresh_receive_age(source):
    source.teleop.get_xr_sample()
    source.now[0] += 0.3
    with pytest.raises(HandTrackingUnavailableError, match="stale"):
        source.teleop.get_xr_sample()
    source.data["stamp"] += 1
    source.teleop.get_xr_sample()  # A stationary hand with new frames is valid.


def test_restarted_source_requires_another_valid_read(source):
    source.teleop.get_xr_sample()
    source.data["stamp"] = 1
    with pytest.raises(HandTrackingUnavailableError, match="restarted"):
        source.teleop.get_xr_sample()
    source.data["stamp"] = 2
    source.teleop.get_xr_sample()


@pytest.mark.parametrize("fault", ["head_missing", "head_nan", "wrist_zero_quaternion", "bad_shape"])
def test_invalid_head_or_wrist_is_rejected(source, fault):
    if fault == "head_missing":
        source.data["head_valid"] = False
    elif fault == "head_nan":
        source.data["head"][0] = float("nan")
    elif fault == "wrist_zero_quaternion":
        source.data["left"]["joints"][1][3:] = [0, 0, 0, 0]
    else:
        source.data["right"]["joints"] = []
    with pytest.raises(HandTrackingUnavailableError):
        source.teleop.get_xr_sample()


def test_disconnect_closes_sdk_once(source):
    source.teleop.disconnect()
    source.teleop.disconnect()
    source.sdk.close.assert_called_once()
    assert not source.teleop.is_connected


def test_connect_requires_new_binding_and_does_not_initialize_old_sdk(monkeypatch):
    import lerobot.teleoperators.pico4.hand_input as module

    monkeypatch.setattr(module, "require_package", lambda *args, **kwargs: None)
    sdk = SimpleNamespace(init=Mock())
    monkeypatch.setitem(sys.modules, "xensevr_pc_service_sdk", sdk)
    with pytest.raises(RuntimeError, match="rebuilt bindings"):
        PicoHandInput().connect()
    sdk.init.assert_not_called()


def test_connect_does_not_wait_for_or_read_controllers(monkeypatch):
    import lerobot.teleoperators.pico4.hand_input as module

    monkeypatch.setattr(module, "require_package", lambda *args, **kwargs: None)
    sdk = SimpleNamespace(init=Mock(), close=Mock(), get_hand_input_sample=Mock(return_value=packet()))
    monkeypatch.setitem(sys.modules, "xensevr_pc_service_sdk", sdk)
    teleop = PicoHandInput()
    try:
        teleop.connect()
        assert teleop.is_connected
        teleop.get_xr_sample()
    finally:
        teleop.disconnect()
    sdk.init.assert_called_once()
    sdk.close.assert_called_once()
