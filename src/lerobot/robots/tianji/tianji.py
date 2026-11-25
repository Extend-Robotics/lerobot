# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from math import isfinite
from time import monotonic, sleep

from lerobot.lerobot_types import RobotAction, RobotObservation

from ..config import RobotConfig
from ..robot import Robot
from ._sdk.fx_robot import Marvin_Robot
from ._sdk.structure_data import DCSS

# Marvin feedback is ordered arm A (left), then arm B (right).
JOINT_NAMES = tuple(f"{side}_joint_{i}.pos" for side in ("left", "right") for i in range(1, 8))
IDLE = 0
POSITION_FOLLOW = 1


@RobotConfig.register_subclass("tianji")
@dataclass(kw_only=True)
class TianjiConfig(RobotConfig):
    robot_ip: str
    max_target_delta_deg: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isfinite(self.max_target_delta_deg) or self.max_target_delta_deg <= 0:
            raise ValueError("max_target_delta_deg must be positive and finite")


class Tianji(Robot):
    """Dual-arm Ethernet driver. Joint positions and targets are in degrees.

    The Marvin SDK binary lives in _sdk/lib/ next to the vendored wrapper.
    """

    config_class = TianjiConfig
    name = "tianji"

    def __init__(self, config: TianjiConfig):
        super().__init__(config)
        self.config = config
        self._sdk: Marvin_Robot | None = None
        self._position_control_requested = False
        self._last_action_frames: tuple[int, int] | None = None

    @property
    def observation_features(self) -> dict[str, type]:
        return dict.fromkeys(JOINT_NAMES, float)

    @property
    def action_features(self) -> dict[str, type]:
        return self.observation_features

    @property
    def is_connected(self) -> bool:
        return self._sdk is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise ConnectionError("Tianji is already connected")
        sdk = Marvin_Robot()
        if sdk.connect(self.config.robot_ip) != 1:
            raise ConnectionError(f"Marvin SDK could not connect to {self.config.robot_ip}")
        self._sdk = sdk
        try:
            deadline = monotonic() + 2
            while not all(output["frame_serial"] for output in sdk.subscribe(DCSS())["outputs"]):
                if monotonic() >= deadline:
                    raise TimeoutError("Tianji did not return joint feedback")
                sdk.request_feedback()
                sleep(0.02)
        except Exception:
            self.disconnect()
            raise

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        return self._joint_positions(self._read_feedback())

    def _read_feedback(self) -> dict:
        if self._sdk is None:
            raise ConnectionError("Tianji is disconnected")
        return self._sdk.subscribe(DCSS())

    @staticmethod
    def _joint_positions(feedback: dict) -> dict[str, float]:
        left, right = feedback["outputs"]
        if not (left["frame_serial"] or right["frame_serial"]):
            raise OSError("Tianji joint feedback unavailable")
        positions = left["fb_joint_pos"] + right["fb_joint_pos"]
        if not all(isfinite(value) for value in positions):
            raise OSError("Tianji returned non-finite joint feedback")
        return dict(zip(JOINT_NAMES, positions, strict=True))

    def enable_position_control(self) -> RobotObservation:
        """Seed both targets from the current pose and enter position follow from idle."""
        if self._sdk is None:
            raise ConnectionError("Tianji is disconnected")
        self._last_action_frames = None
        feedback = self._read_feedback()
        self._check_arm_state(feedback, IDLE)
        current = self._joint_positions(feedback)
        joints = list(current.values())
        self._sdk.send_joint_positions(joints[:7], joints[7:], speed_ratio=10)
        sleep(0.02)

        self._position_control_requested = True
        self._sdk.set_arm_state(POSITION_FOLLOW)
        self._wait_for_arm_state(POSITION_FOLLOW)
        return current

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._sdk is None:
            raise ConnectionError("Tianji is disconnected")
        if not self._position_control_requested:
            raise RuntimeError("Call enable_position_control() before sending Tianji actions")
        if action.keys() != self.action_features.keys():
            raise ValueError("Tianji requires targets for all 14 joints")
        targets = {name: float(action[name]) for name in JOINT_NAMES}
        if not all(isfinite(value) for value in targets.values()):
            raise ValueError("Tianji joint targets must be finite")

        feedback = self._read_feedback()
        self._check_arm_state(feedback, POSITION_FOLLOW)
        left, right = feedback["outputs"]
        frames = (left["frame_serial"], right["frame_serial"])
        if self._last_action_frames == frames:
            raise OSError("Tianji feedback did not advance since the previous command")
        current = self._joint_positions(feedback)
        for name, target in targets.items():
            if abs(target - current[name]) > self.config.max_target_delta_deg:
                raise ValueError(f"Tianji target for {name} is too far from measured position")

        joints = list(targets.values())
        self._sdk.send_joint_positions(joints[:7], joints[7:])
        self._last_action_frames = frames
        return targets

    @staticmethod
    def _check_arm_state(feedback: dict, expected: int) -> None:
        for arm, state in zip(("left", "right"), feedback["states"], strict=True):
            if state["err_code"] or state["cur_state"] != expected:
                raise OSError(
                    f"Tianji {arm} arm state {state['cur_state']} (expected {expected}), "
                    f"fault code {state['err_code']}"
                )

    def _wait_for_arm_state(self, expected: int) -> None:
        deadline = monotonic() + 2
        while True:
            feedback = self._read_feedback()
            try:
                self._check_arm_state(feedback, expected)
            except OSError:
                if monotonic() >= deadline or any(state["err_code"] for state in feedback["states"]):
                    raise
                sleep(0.02)
            else:
                return

    def disconnect(self) -> None:
        if self._sdk is None:
            return
        try:
            if self._position_control_requested:
                sleep(0.02)  # Let the SDK transmit any queued position packet first.
                self._sdk.set_arm_state(IDLE)
                self._wait_for_arm_state(IDLE)
        finally:
            try:
                if self._sdk.release_robot() != 1:
                    raise OSError("Marvin SDK could not release the Tianji connection")
            finally:
                self._sdk = None
                self._position_control_requested = False
                self._last_action_frames = None
