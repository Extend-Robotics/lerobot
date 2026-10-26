#!/usr/bin/env python
"""PICO skeletons to Revo2 fingers only; preview by default, no body connection."""

import argparse
import json
import math
import select
import sys
import time
from contextlib import ExitStack

from lerobot.robots.brainco_revo2 import BraincoRevo2Hand, BraincoRevo2HandConfig
from lerobot.robots.brainco_revo2.pico_teleop import PicoRevo2Session
from lerobot.teleoperators.pico4.hand_input import PicoHandInput
from lerobot.teleoperators.pico4.revo2_retargeting import PicoRevo2Retargeter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.0.89")
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    parser.add_argument(
        "--invert-thumb-base",
        choices=("left", "right", "both"),
        help="Reverse the selected hand's thumb-base motor (motor_0): target becomes 1 - target",
    )
    parser.add_argument("--transport", choices=("zmq", "dds"), default="zmq")
    parser.add_argument("--network-interface")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until Ctrl+C")
    parser.add_argument(
        "--max-speed", type=float, default=1.0, help="Maximum normalized position change per second"
    )
    parser.add_argument("--motor-speed", type=float, default=0.3, help="Driver speed in (0, 1]")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 100:
        parser.error("--fps must be between 1 and 100")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    if not math.isfinite(args.max_speed) or args.max_speed <= 0:
        parser.error("--max-speed must be finite and positive")
    if not math.isfinite(args.motor_speed) or not 0 < args.motor_speed <= 1:
        parser.error("--motor-speed must be in (0, 1]")
    if args.execute and not sys.stdin.isatty():
        parser.error("--execute requires an interactive terminal for start/pause controls")
    sides = ("left", "right") if args.side == "both" else (args.side,)
    inverted = (
        sides
        if args.invert_thumb_base == "both"
        else (args.invert_thumb_base,)
        if args.invert_thumb_base
        else ()
    )
    if not set(inverted).issubset(sides):
        parser.error("--invert-thumb-base must select a hand enabled by --side")
    try:
        retargeter = PicoRevo2Retargeter(sides, invert_thumb_base=inverted)
        with ExitStack() as stack:
            source = PicoHandInput()
            source.connect()
            stack.callback(source.disconnect)
            hands = {}
            for side in sides:
                hand = BraincoRevo2Hand(
                    BraincoRevo2HandConfig(
                        side=side,
                        transport=args.transport,
                        robot_ip=args.robot_ip,
                        network_interface=args.network_interface,
                        read_only=not args.execute,
                        speed=args.motor_speed,
                    )
                )
                stack.callback(hand.disconnect)
                hand.connect()
                hands[side] = hand
            session = PicoRevo2Session(
                hands, source, retargeter, execute=args.execute, max_speed=args.max_speed
            )
            print(
                "EXECUTION ENABLED: r + Enter starts/resumes after 2s; p + Enter pauses; q + Enter exits."
                if args.execute
                else "READ-ONLY PREVIEW: no commands. q + Enter exits.",
                flush=True,
            )
            print(
                "Tracking loss stops finger updates; press r to resume. Disconnect sends no gesture.",
                flush=True,
            )
            began = next_print = time.monotonic()
            start_at = None
            previous_status = None
            while not args.duration or time.monotonic() - began < args.duration:
                tick = time.monotonic()
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip().lower()
                    if key == "q":
                        break
                    if key == "p":
                        start_at = None
                        session.pause()
                    if key == "r" and args.execute:
                        session.pause("Starting in 2 seconds")
                        start_at = time.monotonic() + 2
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
    except (ImportError, ValueError, KeyError, ConnectionError, TimeoutError, RuntimeError) as exc:
        parser.exit(1, f"Hand teleop stopped: {exc}\n")


if __name__ == "__main__":
    main()
