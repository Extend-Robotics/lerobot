"""Synchronous hands-only session using the shared PICO skeleton interface."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot.teleoperators.pico4.hand_input import HandTrackingUnavailableError


class PicoRevo2Session:
    """Explicit start/resume, bounded commands, and no writes on tracking loss.

    The caller owns SDK and hand connections. A tick never publishes body
    commands. Retargeted actions use the same prefixes as UnitreeG1.send_action.
    """

    def __init__(
        self,
        hands: dict[str, Any],
        source: Any,
        retargeter: Any,
        *,
        execute: bool = False,
        max_speed: float = 1.0,
        timeout_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not hands or set(hands) != set(retargeter.sides):
            raise ValueError("Connected hands must match retargeter sides")
        if any(
            hand.config.motor_names != ("motor_0", "motor_1", "index", "middle", "ring", "pinky")
            for hand in hands.values()
        ):
            raise ValueError("Pico Revo2 teleop requires default driver motor names")
        if not np.isfinite(max_speed) or max_speed <= 0 or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Speed and timeout must be finite and positive")
        if execute and any(hand.config.read_only for hand in hands.values()):
            raise ValueError("Execution requires writable hand connections")
        self.hands, self.source, self.retargeter = hands, source, retargeter
        self.execute, self.max_speed, self.timeout_s, self.clock = execute, max_speed, timeout_s, clock
        self.following = False
        self.reason = "Press r then Enter to start"
        self._stamp: int | None = None
        self._targets: dict[str, float] = {}
        self._last: dict[str, dict[str, float]] = {}
        self._sent_at = clock()

    def pause(self, reason: str = "Paused by operator") -> None:
        self.following = False
        self.reason = reason

    def start(self) -> None:
        self.pause("Waiting for valid tracking and feedback")
        try:
            self.source.get_hand_sample(tuple(self.hands))
        except HandTrackingUnavailableError as exc:
            self.pause(str(exc))
            return
        self._last = {side: hand.get_observation() for side, hand in self.hands.items()}
        self._sent_at = self.clock()
        self.following = True

    def tick(self) -> dict[str, Any]:
        try:
            return self._tick()
        except Exception:
            self.pause("Hand teleop fault; check tracking, feedback and transport before resuming")
            raise

    def _tick(self) -> dict[str, Any]:
        began = self.clock()
        try:
            stamp, skeletons = self.source.get_hand_sample(tuple(self.hands))
            if self._stamp != stamp:
                targets = self.retargeter.retarget(skeletons)
                # Retargeting can release the GIL; check input again before sending.
                self.source.get_hand_sample(tuple(self.hands))
                if self.clock() - began > self.timeout_s:
                    raise HandTrackingUnavailableError("Retargeting exceeded input deadline")
                self._targets, self._stamp = targets, stamp
        except HandTrackingUnavailableError as exc:
            self.pause(str(exc))
            return {"status": "paused", "reason": self.reason}
        observations = {side: hand.get_observation() for side, hand in self.hands.items()}
        commands = {}
        if self.execute and self.following:
            now = self.clock()
            dt = now - self._sent_at
            if not 0 <= dt <= self.timeout_s:
                self.pause("Command loop missed its deadline; press r to resume")
            else:
                step = self.max_speed * dt
                for side, hand in self.hands.items():
                    requested = {key: self._targets[f"{side}_hand.{key}"] for key in hand.action_features}
                    bounded = {
                        key: float(np.clip(value, self._last[side][key] - step, self._last[side][key] + step))
                        for key, value in requested.items()
                    }
                    commands[side] = hand.prepare_action(bounded)
                # All selected feedback and targets validated before the first write.
                try:
                    self.source.get_hand_sample(tuple(self.hands))
                    if self.clock() - began > self.timeout_s:
                        raise HandTrackingUnavailableError("Command preparation exceeded input deadline")
                except HandTrackingUnavailableError as exc:
                    self.pause(str(exc))
                    return {"status": "paused", "reason": self.reason}
                for side, command in commands.items():
                    self._last[side] = self.hands[side].send_action(command)
                self._sent_at = now
        return {
            "status": "executing"
            if self.execute and self.following
            else "paused"
            if self.execute
            else "preview",
            "reason": self.reason if not self.following else None,
            "target_action": self._targets.copy(),
            "observation": {
                f"{side}_hand.{key}": value
                for side, obs in observations.items()
                for key, value in obs.items()
            },
            "sent_action": {
                f"{side}_hand.{key}": value for side, cmd in commands.items() for key, value in cmd.items()
            },
        }
