"""PICO controllers and Revo2 fingers, with coordinated pause and resume."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot.teleoperators.pico4.revo2_trigger import PicoRevo2TriggerMapper

from .pico_teleop import ARM_KEYS, SIDES, TIMEOUT_S, rotation_distance
from .unitree_xr import UnitreeXrArmSession, clip_arm_target, map_xr_wrist_targets

HOLD_TIMEOUT_S = 5.0


class PicoHandArmSession(UnitreeXrArmSession):
    """Follow controller poses and trigger closure with coordinated pause/resume.

    Input loss latches a pause. Healthy feedback and transport are required
    during supported hold, which expires after five seconds.
    """

    def __init__(
        self,
        robot: Any,
        teleop: Any,
        clock: Callable[[], float] = time.monotonic,
        *,
        hand_max_speed: float = 1.0,
    ):
        super().__init__(robot, teleop, clock)
        if not np.isfinite(hand_max_speed) or hand_max_speed <= 0:
            raise ValueError("Hand speed must be finite and positive")
        self.hand_max_speed = hand_max_speed
        self.trigger_mapper = PicoRevo2TriggerMapper(tuple(robot.hands)) if robot.hands else None
        if not robot.config.read_only and any(hand.config.read_only for hand in robot.hands.values()):
            raise ValueError("Combined execution requires writable hands")
        self._last_hand: dict[str, dict[str, float]] = {}
        self._hand_at = self.clock()
        self._finger_stamp: int | None = None
        self._finger_target: dict[str, float] = {}
        self._holding = False
        self._paused_at = 0.0
        self._reason = "Press r then Enter to follow controllers"
        self._generation = 0
        self._command_q: np.ndarray | None = None
        self._command_at = self.clock()

    def _read_controllers(self) -> tuple[int, dict[str, np.ndarray], dict[str, float]]:
        return self.teleop.get_controller_sample()

    def start(self) -> None:
        """Require a valid current packet before starting or resuming."""
        self._check()
        try:
            self._read_controllers()
        except RuntimeError as exc:
            self.pause(str(exc))
            return
        with self._lock:
            self._check_hold_locked()
            if self._following:
                return
            q, _ = self._measured()
            tau = np.asarray(self.solver.solve_tau(q), dtype=float)
            if tau.shape != (14,) or not np.isfinite(tau).all():
                raise ValueError("Invalid hand hold gravity torques")
            if self.trigger_mapper is not None:
                self._last_hand = {side: hand.get_observation() for side, hand in self.robot.hands.items()}
                self._hand_at = self.clock()
            self._target_tau = tau.copy()
            self._command_q = q.copy()
            self._command_at = self._source_at = self._target_at = self.clock()
            self._target_q = q.copy()
            self._holding = False
            self._following = True
            self._generation += 1

    def _pause_locked(self, reason: str) -> None:
        if self._following:
            self._following = False
            self._holding = True
            self._paused_at = self.clock()
            self._generation += 1
            if self._command_q is not None:
                self._target_q = self._command_q.copy()
        self._reason = reason

    def _input_loss(self, exc: RuntimeError) -> dict[str, Any]:
        with self._lock:
            self._pause_locked(str(exc))
            return {"status": "paused", "reason": str(exc)}

    def pause(self, reason: str = "Paused by operator") -> None:
        with self._lock:
            self._pause_locked(reason)

    def _check_hold_locked(self) -> None:
        if self._holding and self.clock() - self._paused_at >= HOLD_TIMEOUT_S:
            raise TimeoutError("Hand pause exceeded 5s; stopping supported hold")

    def tick(self) -> dict[str, Any]:
        try:
            return self._tick()
        except Exception:
            self.pause("Arm/hand teleop fault; restart the script")
            raise

    def _tick(self) -> dict[str, Any]:
        self._check()
        began = self.clock()
        with self._lock:
            self._check_hold_locked()
            supporting = self._following or self._holding
        if supporting:
            measured, _ = self._measured()
            tau = np.asarray(self.solver.solve_tau(measured), dtype=float)
            if tau.shape != (14,) or not np.isfinite(tau).all():
                raise ValueError("Invalid hand hold gravity torques")
            with self._lock:
                self._target_tau = tau.copy()
        try:
            stamp, poses, finger_input = self._read_controllers()
        except RuntimeError as exc:
            return self._input_loss(exc)
        with self._lock:
            if stamp <= 0 or (self._stamp is not None and stamp < self._stamp):
                self._stamp = stamp
                self._pause_locked("Invalid or restarted PICO source timestamp; press r to resume")
                return {"status": "paused", "reason": self._reason}
            if self._stamp is None or stamp > self._stamp:
                self._source_at = began
            self._stamp = stamp
            if began - self._source_at > TIMEOUT_S:
                self._pause_locked("PICO input stopped advancing")
            if not self._following:
                return {"status": "paused" if self._holding else "ready", "reason": self._reason}
            generation = self._generation
            preview_q = self._preview_q.copy()
            last_hand = {side: action.copy() for side, action in self._last_hand.items()}
            hand_at = self._hand_at
        targets = map_xr_wrist_targets(poses)
        measured, dq = self._measured()
        seed = preview_q if self.robot.config.read_only else measured
        candidate, tau = self.solver.solve_ik(*targets, seed, dq)
        candidate, tau = np.asarray(candidate, dtype=float), np.asarray(tau, dtype=float)
        if (
            not self.solver.last_solve_success
            or candidate.shape != (14,)
            or tau.shape != (14,)
            or not np.isfinite(candidate).all()
            or not np.isfinite(tau).all()
        ):
            raise ValueError("IK failed or returned invalid arm commands")
        hand_commands = {}
        hand_targets = {}
        if self.trigger_mapper is not None:
            if stamp != self._finger_stamp:
                self._finger_target = self.trigger_mapper.map(finger_input)
                self._finger_stamp = stamp
            hand_targets = self._finger_target.copy()
            dt = self.clock() - hand_at
            if not 0 <= dt <= TIMEOUT_S:
                self.pause("Finger command deadline expired; press r to resume")
                return {"status": "paused", "reason": self._reason}
            step = self.hand_max_speed * dt
            for side, hand in self.robot.hands.items():
                hand.get_observation()  # Validate every selected hand before any write.
                requested = {key: hand_targets[f"{side}_hand.{key}"] for key in hand.action_features}
                bounded = {
                    key: float(np.clip(value, last_hand[side][key] - step, last_hand[side][key] + step))
                    for key, value in requested.items()
                }
                hand_commands[side] = bounded if self.robot.config.read_only else hand.prepare_action(bounded)
        self._check()
        if self.clock() - began > TIMEOUT_S:
            raise TimeoutError("Arm/hand solve exceeded control deadline")
        # Native IK releases the GIL; revalidate controller input before accepting its result.
        try:
            checked_stamp, _, _ = self._read_controllers()
            if checked_stamp < stamp:
                raise RuntimeError("PICO source restarted during solve; press r to resume")
        except RuntimeError as exc:
            return self._input_loss(exc)
        actual = self.solver.forward_kinematics(candidate)
        with self._lock:
            self._check()
            if generation != self._generation or self._stop.is_set():
                return {"status": "paused", "reason": self._reason}
            if self.clock() - began > TIMEOUT_S:
                raise TimeoutError("Arm/hand target exceeded control deadline")
            # Serialize finger dispatch with pause and the arm publisher's expiry
            # check: an in-flight solve can never resume either component.
            for side, command in hand_commands.items():
                if self.clock() - began > TIMEOUT_S:
                    raise TimeoutError("Hand dispatch exceeded control deadline")
                self._last_hand[side] = (
                    command if self.robot.config.read_only else self.robot.hands[side].send_action(command)
                )
            if self.clock() - began > TIMEOUT_S:
                raise TimeoutError("Arm/hand dispatch exceeded control deadline")
            self._hand_at = self.clock()
            self._target_q = candidate.copy()
            # Feedforward follows measured joints, including during supported
            # pause, rather than the potentially distant IK solution.
            self._target_at = self.clock()
        return {
            "status": "preview" if self.robot.config.read_only else "executing",
            "wrist_targets": {side: target.tolist() for side, target in zip(SIDES, targets, strict=True)},
            "joint_action": dict(zip(ARM_KEYS, candidate.tolist(), strict=True)),
            "hand_target_action": hand_targets,
            "hand_sent_action": {}
            if self.robot.config.read_only
            else {
                f"{side}_hand.{key}": value
                for side, action in hand_commands.items()
                for key, value in action.items()
            },
            "ik_residuals": {
                side: {
                    "position_m": float(np.linalg.norm(target[:3, 3] - pose[:3, 3])),
                    "rotation_rad": rotation_distance(target[:3, :3], pose[:3, :3]),
                }
                for side, target, pose in zip(SIDES, targets, actual, strict=True)
            },
        }

    def publish_once(self) -> np.ndarray | None:
        # Serialize pause/start with sending so no in-flight solve can override a hold.
        with self._lock:
            if self._stop.is_set():
                return None
            now = self.clock()
            if self._following and now - min(self._source_at, self._target_at) > TIMEOUT_S:
                self._pause_locked("Controller input or IK target expired; press r to resume")
            self._check_hold_locked()
            if not self._following and not self._holding:
                return None
            assert self._target_q is not None and self._command_q is not None
            target = self._target_q.copy()
            if self._following:
                dt = now - self._command_at
                if not 0 <= dt <= TIMEOUT_S:
                    self._pause_locked("Hand publisher missed its deadline")
                    target = self._target_q.copy()
            if not self.robot.config.read_only:
                # Keep the existing measured-joint clip and feedback age check.
                measured, _ = self._measured()
                target = clip_arm_target(target, measured)
                self.robot.send_arm_joint_action(
                    dict(zip(ARM_KEYS, target.tolist(), strict=True)),
                    TIMEOUT_S,
                    arm_tau=self._target_tau.copy(),
                )
            self._command_q = self._preview_q = target.copy()
            self._command_at = now
            return target
