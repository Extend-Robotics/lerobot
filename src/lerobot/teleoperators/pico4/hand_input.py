"""Raw PICO hand wrists for arm teleoperation; no finger or gripper commands."""

import time
from typing import Any

import numpy as np

from lerobot.utils.import_utils import require_package


class HandTrackingUnavailableError(RuntimeError):
    """A hand packet is missing, invalid, or no longer advancing."""


class PicoHandInput:
    """Own the native SDK and expose the same raw-pose interface as BiPico4.

    OpenXR joint 0 is the palm and joint 1 is the wrist. Poses are metres and
    xyzw quaternions in the SDK's OpenXR world frame. Finger joints are ignored.
    """

    def __init__(self, timeout_s: float = 0.25):
        self.timeout_s = timeout_s
        self._sdk: Any = None
        self._stamp: int | None = None
        self._advanced_at = 0.0
        self._await_advance = False

    @property
    def is_connected(self) -> bool:
        return self._sdk is not None

    def connect(self) -> None:
        if self.is_connected:
            raise RuntimeError("PICO hand input is already connected")
        require_package("xensevr-pc-service-sdk", extra="pico4", import_name="xensevr_pc_service_sdk")
        import xensevr_pc_service_sdk as sdk

        if not hasattr(sdk, "get_hand_input_sample"):
            raise RuntimeError(
                "PICO hands require rebuilt bindings: "
                "PICO_PYTHON=../envs/unitree-g1/bin/python bash scripts/pico4/install_sdk.sh"
            )
        sdk.init()
        self._sdk = sdk
        self._stamp = None
        self._await_advance = False
        self._advanced_at = time.monotonic()

    def disconnect(self) -> None:
        sdk, self._sdk = self._sdk, None
        if sdk is not None:
            sdk.close()

    def get_xr_sample(self) -> tuple[int, dict[str, np.ndarray]]:
        if self._sdk is None:
            raise RuntimeError("PICO hand input is disconnected")
        sample = self._sdk.get_hand_input_sample()
        now = time.monotonic()
        stamp = int(sample["stamp"])
        if stamp <= 0 or (self._stamp is not None and stamp < self._stamp):
            # A restarted source must advance again before it may be resumed.
            self._stamp, self._advanced_at = stamp, now
            self._await_advance = True
            raise HandTrackingUnavailableError("Invalid or restarted PICO hand timestamp")
        if self._await_advance and self._stamp is not None and stamp <= self._stamp:
            raise HandTrackingUnavailableError("Waiting for restarted PICO hand timestamp to advance")
        if self._stamp is None or stamp > self._stamp:
            self._await_advance = False
            self._advanced_at = now
        self._stamp = stamp
        age = float(sample["age_s"])
        if not np.isfinite(age) or not 0 <= age <= self.timeout_s or now - self._advanced_at > self.timeout_s:
            raise HandTrackingUnavailableError("PICO hand packet is stale")
        if not sample["head_valid"]:
            raise HandTrackingUnavailableError("Headset tracking is missing from the hand packet")
        poses = {"head": np.asarray(sample["head"], dtype=float)}
        for side in ("left", "right"):
            hand = sample[side]
            joints = np.asarray(hand["joints"], dtype=float)
            if not hand["active"] or joints.shape != (26, 7):
                raise HandTrackingUnavailableError(f"{side} hand tracking is unavailable")
            poses[side] = joints[1].copy()
        for name, pose in poses.items():
            if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-6:
                raise HandTrackingUnavailableError(f"Invalid {name} wrist/head pose")
        return stamp, poses
