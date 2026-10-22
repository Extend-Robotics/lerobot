"""PICO mapping and lifecycle tests; no headset or native SDK required."""

from unittest.mock import MagicMock

import numpy as np
import pytest

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
