#!/usr/bin/env python
"""Preview joint targets or execute the same G1 pipeline used by standard LeRobot scripts."""

import argparse
import json
import math
import time

from lerobot.processor import make_default_processors
from lerobot.robots.brainco_revo2 import BraincoRevo2HandConfig
from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.teleoperators.bi_pico4 import BiPico4, BiPico4Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--hands", choices=("none", "left", "right", "both"), default="none")
    parser.add_argument("--hand-transport", choices=("zmq", "dds"), default="zmq")
    parser.add_argument("--hand-network-interface")
    parser.add_argument("--hand-max-speed", type=float, default=1.0)
    parser.add_argument("--hand-motor-speed", type=float, default=0.3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 100:
        parser.error("fps must be in [1, 100]")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and nonnegative")
    sides = ("left", "right") if args.hands == "both" else () if args.hands == "none" else (args.hands,)
    robot = UnitreeG1(
        UnitreeG1Config(
            robot_ip=args.robot_ip,
            is_simulation=False,
            read_only=not args.execute,
            arm_commands_only=True,
            gravity_compensation=True,
            hand_max_speed=args.hand_max_speed,
            kp=[0.0] * 12 + [300.0, 0.0, 300.0] + ([80.0] * 4 + [40.0] * 3) * 2,
            kd=[0.0] * 12 + [3.0, 0.0, 3.0] + ([3.0] * 4 + [1.5] * 3) * 2,
            **{
                f"{side}_hand": BraincoRevo2HandConfig(
                    speed=args.hand_motor_speed,
                    transport=args.hand_transport,
                    network_interface=args.hand_network_interface,
                )
                for side in sides
            },
        )
    )
    teleop = BiPico4(BiPico4Config(output_mode="xr"))
    processor, _, _ = make_default_processors(robot, teleop)
    try:
        teleop.connect()
        robot.connect()
        if args.execute:
            if input("Type r then Enter to start arms and configured hands: ").strip() != "r":
                return
            print("Starting in two seconds. Ctrl+C stops execution.", flush=True)
            time.sleep(2.0)
        start = time.monotonic()
        while not args.duration or time.monotonic() - start < args.duration:
            began = time.monotonic()
            observation = robot.get_observation()
            action = processor((teleop.get_action(), observation))
            sent = robot.send_action(action) if args.execute else action
            print(json.dumps({"status": "executing" if args.execute else "preview", "joint_action": sent}))
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - began)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            robot.disconnect()
        finally:
            if teleop.is_connected:
                teleop.disconnect()


if __name__ == "__main__":
    main()
