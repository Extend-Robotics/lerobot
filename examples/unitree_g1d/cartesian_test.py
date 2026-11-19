#!/usr/bin/env python
"""Read-only wrist FK diagnostic. The G1 robot interface exposes joint positions."""

import argparse
import json
import math
import time

import numpy as np

from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ArmIK
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    model = G1_29_ArmIK()
    robot = UnitreeG1(
        UnitreeG1Config(
            robot_ip=args.robot_ip,
            is_simulation=False,
            read_only=True,
            arm_commands_only=True,
        )
    )
    try:
        robot.connect()
        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            observation = robot.get_observation()
            q = np.array([observation[f"{joint.name}.q"] for joint in G1_29_JointArmIndex])
            poses = model.forward_kinematics(q)
            print(
                json.dumps({side: pose.tolist() for side, pose in zip(("left", "right"), poses, strict=True)})
            )
            time.sleep(0.1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
