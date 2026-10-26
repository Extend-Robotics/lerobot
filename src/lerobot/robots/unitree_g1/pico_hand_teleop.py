"""PICO hand wrists to the existing G1 arm IK, with bounded pause and resume."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot.teleoperators.pico4.hand_input import HandTrackingUnavailableError

from .pico_teleop import ARM_KEYS, SIDES, TIMEOUT_S, rotation_distance
from .unitree_xr import UnitreeXrArmSession, clip_arm_target, map_xr_wrist_targets

HOLD_TIMEOUT_S = 5.0


class PicoHandArmSession(UnitreeXrArmSession):
    """Follow both hand wrists; loss of either hand pauses both arms.

    Healthy robot feedback/transport is still required while holding. After five
    seconds paused, fail through the existing passive cleanup. No auto-resume.
    """

    def __init__(self, robot: Any, teleop: Any, clock: Callable[[], float] = time.monotonic):
        super().__init__(robot, teleop, clock)
        self._holding = False
        self._paused_at = 0.0
        self._reason = "Press r then Enter with both hands tracked"
        self._generation = 0
        self._command_q: np.ndarray | None = None
        self._command_at = self.clock()

    def start(self) -> None:
        """Require a valid current packet before starting or resuming."""
        self._check()
        try:
            self.teleop.get_xr_sample()
        except HandTrackingUnavailableError as exc:
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

    def pause(self, reason: str = "Paused by operator") -> None:
        with self._lock:
            self._pause_locked(reason)

    def _check_hold_locked(self) -> None:
        if self._holding and self.clock() - self._paused_at >= HOLD_TIMEOUT_S:
            raise TimeoutError("Hand pause exceeded 5s; stopping supported hold")

    def tick(self) -> dict[str, Any]:
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
            stamp, poses = self.teleop.get_xr_sample()
        except HandTrackingUnavailableError as exc:
            self.pause(str(exc))
            return {"status": "paused", "reason": str(exc)}
        with self._lock:
            if self._stamp is None or stamp > self._stamp:
                self._source_at = began
            self._stamp = stamp
            if began - self._source_at > TIMEOUT_S:
                self._pause_locked("PICO hands stopped advancing")
            if not self._following:
                return {"status": "paused" if self._holding else "ready", "reason": self._reason}
            generation = self._generation
            preview_q = self._preview_q.copy()
        targets = map_xr_wrist_targets(poses, hand_tracking=True)
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
        self._check()
        if self.clock() - began > TIMEOUT_S:
            raise TimeoutError("IK exceeded control deadline")
        # Native IK releases the GIL; revalidate tracking before accepting its result.
        try:
            self.teleop.get_xr_sample()
        except HandTrackingUnavailableError as exc:
            self.pause(str(exc))
            return {"status": "paused", "reason": str(exc)}
        actual = self.solver.forward_kinematics(candidate)
        with self._lock:
            if generation != self._generation:
                return {"status": "paused", "reason": self._reason}
            self._target_q = candidate.copy()
            # Feedforward follows measured joints, including during supported
            # pause, rather than the potentially distant IK solution.
            self._target_at = self.clock()
        return {
            "status": "preview" if self.robot.config.read_only else "executing",
            "wrist_targets": {side: target.tolist() for side, target in zip(SIDES, targets, strict=True)},
            "joint_action": dict(zip(ARM_KEYS, candidate.tolist(), strict=True)),
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
                self._pause_locked("Hand input or IK target expired; press r to resume")
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
