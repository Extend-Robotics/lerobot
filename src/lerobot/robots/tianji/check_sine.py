# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Check command tracking with one ±1° sine cycle on the right wrist joint."""

import argparse
from math import sin, sqrt, tau
from socket import gethostbyname
from time import sleep

from . import Tianji, TianjiConfig

JOINT = "right_joint_7.pos"
PERIOD_S = 6
SAMPLE_RATE_HZ = 25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="Arm controller hostname or IPv4 address")
    parser.add_argument(
        "--clear-fault",
        action="store_true",
        help="Clear latched faults before motion after resolving the cause and releasing the E-stop",
    )
    args = parser.parse_args()

    errors = []
    measured = []
    try:
        robot = Tianji(TianjiConfig(robot_ip=gethostbyname(args.host)))
        try:
            robot.connect()
            if args.clear_fault:
                robot.clear_faults()
            baseline = robot.enable_position_control()
            print(f"Moving {JOINT} by ±1° for one {PERIOD_S}s cycle", flush=True)
            center = baseline[JOINT]
            steps = PERIOD_S * SAMPLE_RATE_HZ
            for step in range(steps + 1):
                target = center + sin(tau * step / steps)
                robot.send_action({**baseline, JOINT: target})
                sleep(1 / SAMPLE_RATE_HZ)
                position = robot.get_observation()[JOINT]
                measured.append(position)
                errors.append(position - target)

            for _ in range(SAMPLE_RATE_HZ // 2):
                robot.send_action(baseline)
                sleep(1 / SAMPLE_RATE_HZ)
            final = robot.get_observation()[JOINT]
        finally:
            robot.disconnect()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Tianji sine check stopped: {exc}\n")

    excursion = max(measured) - min(measured)
    rmse = sqrt(sum(error**2 for error in errors) / len(errors))
    print(f"Measured travel: {excursion:.3f}° peak-to-peak (commanded 2.000°)")
    print(f"Tracking RMSE: {rmse:.3f}°")
    print(f"Start: {center:.3f}° | End: {final:.3f}° | Return error: {final - center:+.3f}°")
    if max(measured) - center < 0.5 or center - min(measured) < 0.5 or abs(final - center) > 0.3:
        parser.exit(1, "Feedback did not follow both halves of the sine and return within 0.3°.\n")


if __name__ == "__main__":
    main()
