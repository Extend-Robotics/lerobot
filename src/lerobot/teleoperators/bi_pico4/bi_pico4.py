#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
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

"""BiPico4: Bimanual Pico4 VR teleoperator for any bimanual robot using the shared
world-frame TCP schema (BiFlexivRizon4RT, BiEliteCS66RT, ...).

Uses both Pico4 VR controllers simultaneously via a single xrt SDK connection.
Delegates all per-arm tracking to two Pico4 instances (left and right), sharing
one xrt handle initialised in connect() and closed in disconnect(). Output is in
the world frame (x=forward, y=left, z=up), so it drives any robot that speaks it.

Output action keys (shared world-frame TCP schema; Flexiv & Elite):
    left_tcp.{x, y, z, r1-r6}   left_gripper.pos
    right_tcp.{x, y, z, r1-r6}  right_gripper.pos

Control scheme (same for each controller):
    Grip held  -> enable arm control; controller pose maps to TCP target pose
    Grip release -> freeze arm at current target
    Trigger    -> gripper position (0 = open, 1 = closed mapped to gripper_width)
    A button (right controller) -> reset both arms to initial pose (rising edge)
"""

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.teleoperators.bi_pico4.config_bi_pico4 import BiPico4Config
from lerobot.teleoperators.pico4 import Pico4
from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.import_utils import _pico4_sdk_available, require_package

if TYPE_CHECKING or _pico4_sdk_available:
    import xensevr_pc_service_sdk as xrt
from lerobot.teleoperators.pico4.math_utils import normalize_quaternion


class BiPico4(Teleoperator):
    """Bimanual Pico4 VR teleoperator for any bimanual robot on the shared
    world-frame TCP schema (BiFlexivRizon4RT, BiEliteCS66RT, ...).

    Wraps two Pico4 instances (left and right) sharing a single xrt SDK
    connection.  BiPico4.connect() initialises the SDK once, injects the
    shared handle into both Pico4 instances, and seeds their initial poses.
    BiPico4.get_action() delegates to each Pico4 and prefixes the keys with
    "left_" / "right_".

    Output action keys (world-frame TCP; Flexiv & Elite use the same):
        left_tcp.{x,y,z,r1-r6}, left_gripper.pos
        right_tcp.{x,y,z,r1-r6}, right_gripper.pos

    Example:
        >>> config = BiPico4Config()
        >>> teleop = BiPico4(config)
        >>> robot = BiFlexivRizon4RT(robot_config)
        >>> robot.connect()
        >>> left_pose, right_pose = robot.get_current_tcp_pose_quat()
        >>> teleop.connect(left_tcp_pose_quat=left_pose, right_tcp_pose_quat=right_pose)
        >>> while True:
        ...     action = teleop.get_action()
        ...     robot.send_action(action)
    """

    config_class = BiPico4Config
    name = "bi_pico4"

    def __init__(self, config: BiPico4Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._xrt = None
        self.logger = logging.getLogger("BiPico4")

        # Edge detection for the controller button currently used by teleop / record flows
        self._was_reset_button_pressed: bool = False

        # Build per-arm Pico4 configs from shared BiPico4Config fields
        left_config = Pico4Config(
            id="bi_pico4_left",
            use_left_controller=True,
            use_right_controller=False,
            pos_sensitivity=config.pos_sensitivity,
            ori_sensitivity=config.ori_sensitivity,
            filter_window_size=config.filter_window_size,
            max_pos_velocity=config.max_pos_velocity,
            max_rot_velocity=config.max_rot_velocity,
            gripper_width=config.left_gripper_width,
            grip_enable_threshold=config.grip_enable_threshold,
            grip_disable_threshold=config.grip_disable_threshold,
            orientation_offset_warning_deg=config.orientation_offset_warning_deg,
            position_jump_threshold=config.position_jump_threshold,
        )
        right_config = Pico4Config(
            id="bi_pico4_right",
            use_left_controller=False,
            use_right_controller=True,
            pos_sensitivity=config.pos_sensitivity,
            ori_sensitivity=config.ori_sensitivity,
            filter_window_size=config.filter_window_size,
            max_pos_velocity=config.max_pos_velocity,
            max_rot_velocity=config.max_rot_velocity,
            gripper_width=config.right_gripper_width,
            grip_enable_threshold=config.grip_enable_threshold,
            grip_disable_threshold=config.grip_disable_threshold,
            orientation_offset_warning_deg=config.orientation_offset_warning_deg,
            position_jump_threshold=config.position_jump_threshold,
        )
        self._left_pico4 = Pico4(left_config)
        self._right_pico4 = Pico4(right_config)

    def pre_init(self) -> None:
        """Initialize XenseVR SDK and wait for controllers early (without TCP poses).

        Call this in a background thread while the robot is connecting/moving to
        start position, then call connect() with the TCP poses once the robot is ready.
        The SDK init and controller wait (up to 15s) will overlap with robot startup.
        """
        if self._xrt is not None:
            return  # Already pre-initialized
        try:
            require_package("xensevr-pc-service-sdk", extra="pico4", import_name="xensevr_pc_service_sdk")
        except ImportError as e:
            raise ImportError("Install the PICO native SDK with: bash scripts/pico4/install_sdk.sh") from e

        try:
            xrt.init()
            self._xrt = xrt
            self.logger.info("XenseVR SDK pre-initialized.")

            self._wait_for_controllers()
        except Exception:
            self._close_sdk()
            raise

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_connected

    @property
    def action_features(self) -> dict[str, type]:
        """Cartesian target fields emitted by get_action()."""
        return dict.fromkeys(
            [
                "left_tcp.x",
                "left_tcp.y",
                "left_tcp.z",
                "left_tcp.r1",
                "left_tcp.r2",
                "left_tcp.r3",
                "left_tcp.r4",
                "left_tcp.r5",
                "left_tcp.r6",
                "left_gripper.pos",
                "right_tcp.x",
                "right_tcp.y",
                "right_tcp.z",
                "right_tcp.r1",
                "right_tcp.r2",
                "right_tcp.r3",
                "right_tcp.r4",
                "right_tcp.r5",
                "right_tcp.r6",
                "right_gripper.pos",
            ],
            float,
        )

    @property
    def feedback_features(self) -> dict[str, Any]:
        return {}

    def connect(
        self,
        calibrate: bool = True,
        left_tcp_pose_quat: np.ndarray | None = None,
        right_tcp_pose_quat: np.ndarray | None = None,
    ) -> None:
        """Connect to Pico4 VR via xrt SDK (single shared connection for both arms).

        Args:
            calibrate: Unused, kept for interface compatibility.
            left_tcp_pose_quat: Left arm current TCP pose [x,y,z,qw,qx,qy,qz,gripper_pos].
            right_tcp_pose_quat: Right arm current TCP pose [x,y,z,qw,qx,qy,qz,gripper_pos].
        """
        if left_tcp_pose_quat is None:
            left_tcp_pose_quat = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
        if right_tcp_pose_quat is None:
            right_tcp_pose_quat = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        self.logger.info("Connecting to Pico4 VR (bimanual)...")
        try:
            require_package("xensevr-pc-service-sdk", extra="pico4", import_name="xensevr_pc_service_sdk")
        except ImportError as e:
            raise ImportError("Install the PICO native SDK with: bash scripts/pico4/install_sdk.sh") from e

        try:
            if self._xrt is None:
                # Not pre-initialized — do full init now (slower path)
                xrt.init()
                self._xrt = xrt
                self.logger.info("XenseVR SDK initialized.")

                self._wait_for_controllers()
            else:
                # pre_init() was already called — reuse the existing xrt handle
                self.logger.info("XenseVR SDK already pre-initialized, skipping init.")

            # Inject shared xrt handle and seed initial state into both Pico4 instances,
            # bypassing their connect() to avoid calling xrt.init() a second time.
            self._init_pico4_instance(self._left_pico4, self._xrt, left_tcp_pose_quat)
            self._init_pico4_instance(self._right_pico4, self._xrt, right_tcp_pose_quat)

            self._is_connected = True

            # Seed edge-detection state from the actual current A-button state so
            # that a held button at connect time does not fire as a false
            # "just pressed" event on the first poll_buttons() call.
            self.poll_buttons()
            self._was_reset_button_pressed = self._right_pico4._last_a_button

            self.logger.info("BiPico4 connected (left + right controllers).")

        except DeviceNotConnectedError:
            self._close_sdk()
            raise
        except Exception as e:
            self._close_sdk()
            raise DeviceNotConnectedError(f"Failed to connect BiPico4: {e}") from e

    def _wait_for_controllers(self) -> None:
        """Allow delayed tracking startup; a service connection alone is insufficient."""
        ready = {}
        for attempt in range(150):  # Up to 15 seconds at 10 Hz.
            for side in ("left", "right"):
                pose = np.asarray(getattr(self._xrt, f"get_{side}_controller_pose")(), dtype=float)
                ready[side] = (
                    pose.shape == (7,) and np.isfinite(pose).all() and np.linalg.norm(pose[3:]) > 1e-6
                )
            if all(ready.values()):
                self.logger.info("Both PICO controller poses are available")
                return
            if attempt % 10 == 0:
                self.logger.warning(
                    "Waiting for PICO tracking: left=%s, right=%s",
                    *["ready" if ready[side] else "missing" for side in ("left", "right")],
                )
            time.sleep(0.1)
        raise DeviceNotConnectedError(
            "No PICO controller poses after 15s "
            f"(left={'ready' if ready['left'] else 'missing'}, right={'ready' if ready['right'] else 'missing'}). "
            "Enable Tracking > Controller and Data & Control > Send in the headset app, "
            "and wake both controllers. A headset connection alone does not provide tracking."
        )

    def _init_pico4_instance(self, pico4: Pico4, xrt, tcp_pose_quat: np.ndarray) -> None:
        """Inject shared xrt handle and seed initial pose into a Pico4 instance.

        This mirrors what Pico4.connect() does after xrt.init(), so we avoid
        initialising the SDK a second time for the second arm.

        Args:
            pico4: The Pico4 instance to initialise.
            xrt: The already-initialised xrt SDK module.
            tcp_pose_quat: Initial TCP pose [x,y,z,qw,qx,qy,qz,gripper_pos].
        """
        pico4._xrt = xrt
        pico4._target_pos = tcp_pose_quat[:3].copy().astype(np.float32)
        pico4._target_quat = normalize_quaternion(tcp_pose_quat[3:7], input_format="wxyz")
        pico4._target_gripper_pos = float(tcp_pose_quat[7])
        pico4._start_pos = tcp_pose_quat[:3].copy().astype(np.float32)
        pico4._ref_pos = None
        pico4._quat_offset = None
        pico4._enabled = False
        pico4._was_enabled = False
        pico4._orientation_control_active = True
        pico4._last_raw_pose = None
        pico4._last_a_button = False
        pico4._last_b_button = False
        pico4._last_x_button = False
        pico4._last_y_button = False
        pico4._is_connected = True

    def _close_sdk(self) -> None:
        sdk, self._xrt = self._xrt, None
        self._is_connected = False
        for child in (self._left_pico4, self._right_pico4):
            child._xrt = None
            child._is_connected = False
        if sdk is not None:
            sdk.close()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        """Get action from both controllers.

        Delegates to each Pico4 instance and prefixes the returned keys with
        "left_" / "right_" to match the bimanual world-frame TCP action space.

        Returns:
            Flat dict with keys: left_tcp.{x,y,z,r1-r6}, left_gripper.pos,
            right_tcp.{x,y,z,r1-r6}, right_gripper.pos.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_action = self._left_pico4.get_action()
        right_action = self._right_pico4.get_action()

        return {f"left_{k}": v for k, v in left_action.items()} | {
            f"right_{k}": v for k, v in right_action.items()
        }

    def get_arm_enabled(self) -> dict[str, bool]:
        """Return grip enable states from the last get_action(), with Pico hysteresis."""
        return {"left": self._left_pico4._enabled, "right": self._right_pico4._enabled}

    def get_control_status(self) -> tuple[int, dict[str, float]]:
        """Return source-stream timestamp and grips, rejecting invalid raw poses.

        The timestamp covers the stream, not per-controller tracking confidence.
        Its progression distinguishes a stationary controller from a stalled stream.
        """
        if not self.is_connected or self._xrt is None:
            raise DeviceNotConnectedError("PICO is disconnected")
        grips = {}
        for side in ("left", "right"):
            pose = np.asarray(getattr(self._xrt, f"get_{side}_controller_pose")(), dtype=float)
            grip = float(getattr(self._xrt, f"get_{side}_grip")())
            if (
                pose.shape != (7,)
                or not np.isfinite(pose).all()
                or np.linalg.norm(pose[3:]) < 1e-6
                or not np.isfinite(grip)
                or not 0 <= grip <= 1
            ):
                raise ValueError(f"Invalid {side} controller sample")
            grips[side] = grip
        return int(self._xrt.get_time_stamp_ns()), grips

    def reset_arm_to_pose(self, side: str, pose: np.ndarray) -> None:
        """Rebase one arm's clutch to [x,y,z,qw,qx,qy,qz] without affecting the other."""
        if side not in ("left", "right"):
            raise ValueError("Side must be left or right")
        child = self._left_pico4 if side == "left" else self._right_pico4
        child.reset_to_pose(pose)

    def reset_to_pose(
        self,
        left_pose_7d: np.ndarray,
        right_pose_7d: np.ndarray,
        left_gripper_pos: float = 0.0,
        right_gripper_pos: float = 0.0,
    ) -> None:
        """Reset both arm target poses (e.g. after robot moves to start position).

        Args:
            left_pose_7d: [x, y, z, qw, qx, qy, qz] in world frame.
            right_pose_7d: [x, y, z, qw, qx, qy, qz] in world frame.
            left_gripper_pos: Left gripper position.
            right_gripper_pos: Right gripper position.
        """
        self._left_pico4.reset_to_pose(left_pose_7d, left_gripper_pos)
        self._right_pico4.reset_to_pose(right_pose_7d, right_gripper_pos)

    def poll_buttons(self) -> None:
        """Refresh cached controller button states without a full pose/action computation.

        Call this at the top of a control loop when button state is needed before get_action().
        get_action() will re-read the same buttons, which is safe (idempotent reads).
        Only the right-controller A button is currently consumed by teleop / record flows.
        """
        if not self._is_connected or self._xrt is None:
            return
        self._right_pico4._last_a_button = bool(self._xrt.get_A_button())

    def get_reset_button(self) -> bool:
        """Get rising-edge state of the A button (right controller).

        Returns True only on the first frame the button is pressed.
        Both arms share the same reset button — pressing A resets both simultaneously.

        Note: Call get_action() before this method each frame so that the right
        Pico4 instance has polled the SDK and cached the A button state.
        """
        current = self._right_pico4._last_a_button
        just_pressed = current and not self._was_reset_button_pressed
        self._was_reset_button_pressed = current
        return just_pressed

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("Feedback is not supported by BiPico4.")

    def disconnect(self) -> None:
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Closing XenseVR SDK...")

        # Mark both Pico4 instances as disconnected before closing the shared handle
        self._left_pico4._is_connected = False
        self._left_pico4._xrt = None
        self._right_pico4._is_connected = False
        self._right_pico4._xrt = None

        try:
            self._xrt.close()
        except Exception as e:
            self.logger.warning(f"Error closing XenseVR SDK: {e}")
        finally:
            self._is_connected = False
            self._xrt = None
            self.logger.info("BiPico4 disconnected.")

    def __del__(self):
        if getattr(self, "_xrt", None) is not None:
            try:
                self._close_sdk()
            # No logging: __del__ can run during interpreter shutdown, where the
            # logger's sinks may already be gone.
            except Exception:  # nosec B110
                pass
            finally:
                self._is_connected = False
                self._xrt = None
