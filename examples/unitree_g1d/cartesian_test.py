#!/usr/bin/env python

"""Read G1-D TCP poses, optionally moving both arms along a small Cartesian sinusoid.

Examples:
    uv run --no-sync python examples/unitree_g1d/cartesian_test.py --robot-ip 192.168.123.164
    uv run --no-sync python examples/unitree_g1d/cartesian_test.py --robot-ip 192.168.123.164 \
        --control-space cartesian_abs --move-arms
    uv run --no-sync python examples/unitree_g1d/cartesian_test.py --robot-ip 192.168.123.164 \
        --control-space cartesian_rel --move-arms

TCP positions are metres in the IK model's base frame (legs/waist fixed at zero).
Orientation uses Pico4's r1-r6: the first two columns of the rotation matrix.
Grippers are not controlled. Requires the G1 IK dependencies (Pinocchio and CasADi).
"""

import argparse
import json
import math
import time

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument(
        "--control-space", choices=("cartesian_abs", "cartesian_rel"), default="cartesian_abs"
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--move-arms", action="store_true", help="Enable arm motion (default: read only)")
    parser.add_argument("--amplitude", type=float, default=0.09, help="Peak vertical offset in metres")
    parser.add_argument("--frequency", type=float, default=0.2, help="Sinusoid frequency in Hz")
    args = parser.parse_args()

    config = UnitreeG1Config(
        robot_ip=args.robot_ip,
        is_simulation=False,
        control_space=args.control_space,
        controller=None,
        gravity_compensation=False,
        cameras={},
        kp=[0.0] * 12 + [60.0, 0.0, 40.0] + [40.0] * 14,
        kd=[0.0] * 12 + [1.0, 0.0, 1.0] + [1.0] * 14,
    )
    robot = UnitreeG1(config)
    if not args.move_arms:
        robot._send_zero_torque = lambda: None
    try:
        robot.connect()
        initial = robot.get_observation()
        starting_pose = {key: initial[key] for key in robot.action_features}
        start = time.monotonic()
        next_print = start
        # IK runs at the caller's rate; it does not need the 250 Hz DDS read rate.
        period = 1.0 / 100.0
        while (elapsed := time.monotonic() - start) < args.duration:
            tick = time.monotonic()
            observation = robot.get_observation()
            if args.move_arms:
                envelope = math.sin(math.pi * elapsed / args.duration) ** 2
                offset = args.amplitude * envelope * math.sin(2 * math.pi * args.frequency * elapsed)
                action = starting_pose.copy()
                for side in ("left", "right"):
                    action[f"{side}_tcp.z"] += offset
                    if args.control_space == "cartesian_rel":
                        # Correct from measured position to the desired point on the path.
                        for axis in ("x", "y", "z"):
                            action[f"{side}_tcp.{axis}"] -= observation[f"{side}_tcp.{axis}"]
                        # Identity increment keeps the measured orientation.
                        for i, value in enumerate((1, 0, 0, 0, 1, 0), start=1):
                            action[f"{side}_tcp.r{i}"] = value
                robot.send_action(action)
            if tick >= next_print:
                print(json.dumps({key: observation[key] for key in robot.action_features}))
                next_print = tick + 0.1
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
