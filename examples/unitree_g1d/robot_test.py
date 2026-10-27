#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Read G1-D state, optionally moving both elbows with a small sinusoid."""

import argparse
import json
import math
import time

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--move-elbows", action="store_true", help="Enable elbow motion (default: read only)")
    parser.add_argument(
        "--amplitude", type=float, default=0.1, help="Peak elbow offset in radians (max 0.05)"
    )
    parser.add_argument("--frequency", type=float, default=0.2, help="Sinusoid frequency in Hz (max 0.5)")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not math.isfinite(args.amplitude) or not 0 < args.amplitude <= 0.1:
        parser.error("--amplitude must be in (0, 0.05] radians")
    if not math.isfinite(args.frequency) or not 0 < args.frequency <= 0.5:
        parser.error("--frequency must be in (0, 0.5] Hz")

    # Gains from Unitree's g1d_arm_example.cpp.
    config = UnitreeG1Config(
        robot_ip=args.robot_ip,
        is_simulation=False,
        controller=None,
        gravity_compensation=False,
        cameras={},
        kp=[0.0] * 12 + [60.0, 0.0, 40.0] + [40.0] * 14,
        kd=[0.0] * 12 + [1.0, 0.0, 1.0] + [1.0] * 14,
    )
    robot = UnitreeG1(config)
    # Preserve the read-only behavior; motion mode uses the stock passive shutdown.
    if not args.move_elbows:
        robot._send_zero_torque = lambda: None
    try:
        robot.connect()
        elbow_keys = ("kLeftElbow.pos", "kRightElbow.pos")
        initial = robot.get_observation()
        starting_elbows = {key: initial[key] for key in elbow_keys}
        if not all(math.isfinite(value) for value in starting_elbows.values()):
            raise ValueError("Nonfinite elbow feedback")
        start = time.monotonic()
        next_tick = start
        next_print = start
        period = config.control_dt if args.move_elbows else 0.1
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= args.duration:
                break
            if args.move_elbows:
                # Smoothly fade in/out so the target starts and finishes at the initial pose.
                envelope = math.sin(math.pi * elapsed / args.duration) ** 2
                offset = args.amplitude * envelope * math.sin(2 * math.pi * args.frequency * elapsed)
                robot.send_action({key: value + offset for key, value in starting_elbows.items()})
            if now >= next_print:
                observation = robot.get_observation()
                print(
                    json.dumps(
                        {
                            f"{joint.name}.{field}": observation[f"{joint.name}.{field}"]
                            for joint in G1_29_JointArmIndex
                            for field in ("q", "dq", "tau")
                        }
                    )
                )
                next_print = now + 0.1
            next_tick = max(next_tick + period, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))
        if args.move_elbows:
            # Briefly hold the exact initial elbow targets before stock disconnect.
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                robot.send_action(starting_elbows)
                time.sleep(config.control_dt)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
