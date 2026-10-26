#!/usr/bin/env python
"""Unitree XR mapping/IK for G1-D; preview by default, --execute enables commands."""

import argparse
import json
import logging
import math
import select
import sys
import time

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.robots.unitree_g1.pico_hand_teleop import PicoHandArmSession
from lerobot.robots.unitree_g1.unitree_xr import ARM_KD, ARM_KP, UnitreeXrArmSession
from lerobot.teleoperators.bi_pico4 import BiPico4, BiPico4Config
from lerobot.teleoperators.pico4.hand_input import PicoHandInput


def main() -> None:
    """Follow absolute, head-yaw-relative wrist targets using upstream G1-29 tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--input", choices=("controllers", "hands"), default="controllers")
    parser.add_argument(
        "--execute", action="store_true", help="Publish arm commands; faults/shutdown make arms passive"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until Ctrl+C")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 100:
        parser.error("--fps must be between 1 and 100")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    logging.basicConfig(level=logging.INFO)
    config = UnitreeG1Config(
        robot_ip=args.robot_ip,
        is_simulation=False,
        control_space="cartesian_abs",
        arm_ik_profile="unitree_xr",
        controller=None,
        gravity_compensation=True,
        cameras={},
        read_only=not args.execute,
        arm_commands_only=True,
        kp=[0.0] * 12 + [300.0, 0.0, 300.0] + ARM_KP,
        kd=[0.0] * 12 + [3.0, 0.0, 3.0] + ARM_KD,
    )
    try:
        robot = UnitreeG1(config)
    except ImportError as exc:
        parser.exit(
            2,
            f"Missing dependency: {exc}. Pinocchio must include pinocchio.casadi; see docs/source/unitree_g1.mdx.\n",
        )
    # Raw SDK poses bypass BiPico4's clutch, scaling, jump filter and Cartesian limits.
    teleop = PicoHandInput() if args.input == "hands" else BiPico4(BiPico4Config())
    session = None
    try:
        robot.connect()
        session_class = PicoHandArmSession if args.input == "hands" else UnitreeXrArmSession
        session = session_class(robot, teleop)
        session.initialize()
        print(
            "EXECUTION ENABLED: r + Enter starts both arms after 2 seconds; q + Enter exits."
            if args.execute
            else "READ-ONLY PREVIEW: no robot commands will be published.",
            flush=True,
        )
        if args.input == "hands":
            print(
                "Hand wrists control arms only. p + Enter pauses; r + Enter resumes. "
                "Tracking loss pauses both arms; a 5s pause ends in passive cleanup.",
                flush=True,
            )
        print(
            "Absolute Unitree XR mapping: headset yaw sets forward; grips do not clutch. Ctrl+C exits.",
            flush=True,
        )
        if not args.execute:
            session.start()
        session.start_publisher()
        start = time.monotonic()
        next_print = start
        previous_status = None
        start_at: float | None = None
        while not args.duration or time.monotonic() - start < args.duration:
            tick = time.monotonic()
            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip().lower()
                if key == "q":
                    break
                if key == "p" and args.input == "hands":
                    start_at = None
                    session.pause()
                if key == "r":
                    start_at = time.monotonic() + 2.0
                    print("Starting in 2 seconds. Return hands to the tracking area.", flush=True)
            if start_at is not None and time.monotonic() >= start_at:
                start_at = None
                session.start()
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
        # Stop the command writer before making arms passive, then close native tracking.
        try:
            if session is not None:
                session.close()
        finally:
            try:
                robot.disconnect()
            finally:
                if teleop.is_connected:
                    teleop.disconnect()


if __name__ == "__main__":
    main()
