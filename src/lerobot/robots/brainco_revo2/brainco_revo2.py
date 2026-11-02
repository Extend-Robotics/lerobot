"""One BrainCo Revo2 hand, exposed as a standard LeRobot Robot."""

import math
import time

from lerobot.lerobot_types import RobotAction, RobotObservation

from ..robot import Robot
from .config_brainco_revo2 import BraincoRevo2HandConfig
from .transport import DdsHandTransport, ZmqHandTransport


class BraincoRevo2Hand(Robot):
    config_class = BraincoRevo2HandConfig
    name = "brainco_revo2"

    def __init__(self, config: BraincoRevo2HandConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._transport = DdsHandTransport(config) if config.transport == "dds" else ZmqHandTransport(config)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.motor_names}

    @property
    def observation_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        # Calibration/encoder normalization is provided by the hand's driver.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise ConnectionError("Revo2 hand is already connected")
        try:
            self._transport.connect()
            deadline = time.monotonic() + self.config.connect_timeout_s
            while time.monotonic() < deadline:
                sample = self._transport.read()
                if sample is not None and time.monotonic() - sample[1] <= self.config.state_timeout_s:
                    self._connected = True
                    return
                time.sleep(0.01)
            raise TimeoutError(f"No fresh {self.config.side} Revo2 state before connection timeout")
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise ConnectionError("Revo2 hand is disconnected")
        sample = self._transport.read()
        if sample is None or time.monotonic() - sample[1] > self.config.state_timeout_s:
            raise TimeoutError(f"{self.config.side} Revo2 feedback is stale")
        return dict(zip(self.observation_features, sample[0], strict=True))

    def prepare_action(self, action: RobotAction) -> RobotAction:
        """Validate before a composite robot starts dispatching to any component."""
        if self.config.read_only:
            raise PermissionError("Revo2 command publishing is disabled in read-only mode")
        if set(action) != set(self.action_features):
            raise ValueError("Expected exactly six Revo2 position targets")
        values = {key: float(action[key]) for key in self.action_features}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Revo2 targets must be finite")
        self.get_observation()
        return {key: min(1.0, max(0.0, value)) for key, value in values.items()}

    def send_action(self, action: RobotAction) -> RobotAction:
        sent = self.prepare_action(action)
        self._transport.write(list(sent.values()))
        return sent

    def disconnect(self) -> None:
        # No open/close gesture or command on disconnect.
        self._connected = False
        self._transport.close()
