#!/usr/bin/env python
"""Preview PICO-to-G1-D arm IK, or explicitly enable arm execution with --execute."""

import argparse
import json
import logging
import math
import time

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.robots.unitree_g1.pico_teleop import PicoArmSession
from lerobot.teleoperators.bi_pico4 import BiPico4, BiPico4Config


def main() -> None:
    """Connect measured robot wrists to clutched controllers, initially read-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument(
        "--execute", action="store_true", help="Publish arm commands; faults/shutdown make arms passive"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until Ctrl+C")
    parser.add_argument("--position-scale", type=float, default=0.5)
    parser.add_argument("--orientation-scale", type=float, default=0.5)
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 100:
        parser.error("--fps must be between 1 and 100")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    if any(
        not math.isfinite(value) or not 0 < value <= 1
        for value in (args.position_scale, args.orientation_scale)
    ):
        parser.error("Controller scales must be in (0, 1]")
    logging.basicConfig(level=logging.INFO)
    config = UnitreeG1Config(
        robot_ip=args.robot_ip,
        is_simulation=False,
        control_space="cartesian_abs",
        controller=None,
        gravity_compensation=False,
        cameras={},
        read_only=not args.execute,
        arm_commands_only=True,
        kp=[0.0] * 12 + [60.0, 0.0, 40.0] + [40.0] * 14,
        kd=[0.0] * 12 + [1.0, 0.0, 1.0] + [1.0] * 14,
    )
    try:
        robot = UnitreeG1(config)
    except ImportError as exc:
        parser.exit(
            2,
            f"Missing dependency: {exc}. Pinocchio must include pinocchio.casadi; see docs/source/unitree_g1.mdx.\n",
        )
    teleop = BiPico4(
        BiPico4Config(
            pos_sensitivity=args.position_scale,
            ori_sensitivity=args.orientation_scale,
            max_pos_velocity=0.0,
            max_rot_velocity=0.0,
        )
    )
    try:
        robot.connect()
        session = PicoArmSession(robot, teleop)
        session.initialize()
        print(
            "EXECUTION ENABLED: faults and shutdown make arms passive."
            if args.execute
            else "READ-ONLY PREVIEW: no robot commands will be published.",
            flush=True,
        )
        print(
            "Release both grips, then squeeze either grip. Triggers and A/B/X/Y are ignored. Ctrl+C exits.",
            flush=True,
        )
        start = time.monotonic()
        next_print = start
        previous_status = None
        while not args.duration or time.monotonic() - start < args.duration:
            tick = time.monotonic()
            report = session.tick()
            status = (report["status"], report.get("reason"))
            if tick >= next_print or status != previous_status:
                print(json.dumps(report), flush=True)
                next_print = tick + 0.2
            previous_status = status
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    except (ValueError, KeyError, ConnectionError, TimeoutError, RuntimeError) as exc:
        logging.error("Teleop stopped: %s. Restart the script to retry.", exc)
        raise SystemExit(1) from exc
    finally:
        # Remove torque before closing the SDK, which may wait for its native threads.
        try:
            robot.disconnect()
        finally:
            if teleop.is_connected:
                teleop.disconnect()


if __name__ == "__main__":
    main()
