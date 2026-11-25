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
from pathlib import Path
from time import monotonic, sleep
from typing import Literal

from lerobot.lerobot_types import RobotAction, RobotObservation

from ..config import RobotConfig
from ..robot import Robot
from ._sdk.fx_robot import Marvin_Robot
from ._sdk.structure_data import DCSS

# Marvin feedback is ordered arm A (left), then arm B (right).
JOINT_NAMES = tuple(f"{side}_joint_{i}.pos" for side in ("left", "right") for i in range(1, 8))
WUJI_START_JOINTS = (55, -65, -70, -60, -30, 0, 0, -55, -65, 70, -60, 30, 0, 0)
IDLE = 0
POSITION_FOLLOW = 1
IMPEDANCE = 3
FAULT = 100
WUJI_JOINT_STIFFNESS = (2.0, 2.0, 2.0, 1.6, 1.0, 1.0, 1.0)
WUJI_JOINT_DAMPING = (0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4)


@RobotConfig.register_subclass("tianji")
@dataclass(kw_only=True)
class TianjiConfig(RobotConfig):
    robot_ip: str
    control_mode: Literal["position", "joint_impedance"] = "position"
    clear_faults_on_connect: bool = False
    move_to_start_on_connect: bool = False
    max_target_delta_deg: float = 180.0
    kinematics_library: Path | None = None
    kinematics_model: Path | None = None
    pico_wrist_calibration: Path | None = None
    ik_posture_weight: float = 0.0
    ik_max_seed_delta_deg: float = 15.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_mode not in ("position", "joint_impedance"):
            raise ValueError(f"Unsupported Tianji control mode: {self.control_mode}")
        if not isfinite(self.max_target_delta_deg) or self.max_target_delta_deg <= 0:
            raise ValueError("max_target_delta_deg must be positive and finite")
        if not isfinite(self.ik_posture_weight) or self.ik_posture_weight < 0:
            raise ValueError("ik_posture_weight must be finite and nonnegative")
        if not isfinite(self.ik_max_seed_delta_deg) or self.ik_max_seed_delta_deg <= 0:
            raise ValueError("ik_max_seed_delta_deg must be positive and finite")


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
        self._last_sent_action: RobotAction | None = None
        self._active_state = IMPEDANCE if config.control_mode == "joint_impedance" else POSITION_FOLLOW

    @property
    def last_sent_action(self) -> RobotAction | None:
        return None if self._last_sent_action is None else self._last_sent_action.copy()

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

    def connect(self, calibrate: bool = True, *, read_only: bool = False) -> None:
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
            if self.config.clear_faults_on_connect and not read_only:
                self.clear_faults()
            if self.config.move_to_start_on_connect and not read_only:
                self._move_to_start(self.enable_position_control())
        except BaseException:
            self.disconnect()
            raise

    def _move_to_start(self, initial: RobotObservation) -> None:
        """Follow Wuji's quintic joint trajectory and verify both arms arrive."""
        goal = dict(zip(JOINT_NAMES, WUJI_START_JOINTS, strict=True))
        began = monotonic()
        while True:
            feedback = self._read_feedback()
            self._check_arm_state(feedback, self._active_state)
            measured = self._joint_positions(feedback)
            elapsed = monotonic() - began
            error = max(abs(goal[key] - measured[key]) for key in JOINT_NAMES)
            if elapsed > 30:
                raise TimeoutError(f"Tianji startup pose timed out with {error:.2f} degrees remaining")
            t = min(elapsed / 3.0, 1.0)
            blend = t**3 * (10 - 15 * t + 6 * t**2)
            delta = {
                key: initial[key] + blend * (goal[key] - initial[key]) - measured[key] for key in JOINT_NAMES
            }
            # Allow feedback to lag the nominal trajectory while preserving the driver limit.
            scale = max(1.0, max(map(abs, delta.values())) / (0.8 * self.config.max_target_delta_deg))
            self.send_action({key: measured[key] + delta[key] / scale for key in JOINT_NAMES})
            if t == 1 and scale == 1 and error <= 1.0:
                return
            sleep(0.03)

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
        if not (left["frame_serial"] and right["frame_serial"]):
            raise OSError("Tianji joint feedback unavailable")
        positions = left["fb_joint_pos"] + right["fb_joint_pos"]
        if not all(isfinite(value) for value in positions):
            raise OSError("Tianji returned non-finite joint feedback")
        return dict(zip(JOINT_NAMES, positions, strict=True))

    def clear_faults(self) -> None:
        """Clear latched faults and wait for idle after the stop condition is resolved."""
        if self._sdk is None:
            raise ConnectionError("Tianji is disconnected")
        if self._position_control_requested:
            raise OSError("Clear Tianji faults before enabling position control")

        feedback = self._read_feedback()
        self._joint_positions(feedback)
        states = feedback["states"]
        if any(state["cur_state"] not in (IDLE, FAULT) for state in states):
            raise OSError("Both Tianji arms must be idle or faulted to clear faults")

        faulted_arms = tuple(
            arm
            for arm, state in zip(("A", "B"), states, strict=True)
            if state["err_code"] or state["cur_state"] == FAULT
        )
        if faulted_arms:
            self._sdk.clear_faults(faulted_arms)
            self._wait_for_arm_state(IDLE, wait_for_fault_clear=True)

    def enable_position_control(self) -> RobotObservation:
        """Seed both joint position targets and enter the configured control mode from idle."""
        if self._sdk is None:
            raise ConnectionError("Tianji is disconnected")
        self._last_action_frames = None
        feedback = self._read_feedback()
        self._check_arm_state(feedback, IDLE)
        previous_frames = [output["frame_serial"] for output in feedback["outputs"]]
        deadline = monotonic() + 2
        while True:
            feedback = self._read_feedback()
            self._check_arm_state(feedback, IDLE)
            if all(
                output["frame_serial"] and output["frame_serial"] != previous
                for previous, output in zip(previous_frames, feedback["outputs"], strict=True)
            ):
                break
            if monotonic() >= deadline:
                raise TimeoutError("Both Tianji feedback streams must advance before position control")
            sleep(0.01)
        current = self._joint_positions(feedback)
        joints = list(current.values())
        self._sdk.send_joint_positions(joints[:7], joints[7:], speed_ratio=10)
        self._last_sent_action = current.copy()
        sleep(0.02)

        self._position_control_requested = True
        self._sdk.set_arm_state(self._active_state)
        self._wait_for_arm_state(self._active_state)
        if self._active_state == IMPEDANCE:
            # The state readback confirms mode entry. Gain and type writes apply separately.
            self._sdk.set_joint_impedance_gains(WUJI_JOINT_STIFFNESS, WUJI_JOINT_DAMPING)
            sleep(0.5)
            self._check_arm_state(self._read_feedback(), IMPEDANCE)
            self._sdk.select_joint_impedance()
            sleep(0.5)
            self._check_arm_state(self._read_feedback(), IMPEDANCE)
            current = self.get_observation()
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
        self._check_arm_state(feedback, self._active_state)
        left, right = feedback["outputs"]
        frames = (left["frame_serial"], right["frame_serial"])
        if self._last_action_frames is not None:
            for side, previous, current_frame in zip(
                ("left", "right"), self._last_action_frames, frames, strict=True
            ):
                if current_frame == previous:
                    raise OSError(f"Tianji {side} feedback did not advance since the previous command")
        current = self._joint_positions(feedback)
        for name, target in targets.items():
            if abs(target - current[name]) > self.config.max_target_delta_deg:
                raise ValueError(f"Tianji target for {name} is too far from measured position")

        joints = list(targets.values())
        self._sdk.send_joint_positions(joints[:7], joints[7:])
        self._last_sent_action = targets.copy()
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

    def _wait_for_arm_state(
        self, expected: int, *, wait_for_fault_clear: bool = False, timeout_s: float = 2
    ) -> None:
        deadline = monotonic() + timeout_s
        while True:
            feedback = self._read_feedback()
            try:
                self._check_arm_state(feedback, expected)
            except OSError:
                faulted = any(state["err_code"] for state in feedback["states"])
                if monotonic() >= deadline or (faulted and not wait_for_fault_clear):
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
                self._wait_for_arm_state(IDLE, timeout_s=5 if self._active_state == IMPEDANCE else 2)
        finally:
            try:
                if self._sdk.release_robot() != 1:
                    raise OSError("Marvin SDK could not release the Tianji connection")
            finally:
                self._sdk = None
                self._position_control_requested = False
                self._last_action_frames = None
                self._last_sent_action = None
