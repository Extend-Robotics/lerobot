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

"""Read one joint state from the Tianji cabinet over Ethernet."""

import argparse
import socket

from . import Tianji, TianjiConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "host", nargs="?", default="tanji-marvin-pro", help="Cabinet hostname or IPv4 address"
    )
    args = parser.parse_args()

    try:
        robot_ip = socket.gethostbyname(args.host)
    except OSError as exc:
        parser.exit(1, f"Could not resolve {args.host!r} to an IPv4 address: {exc}\n")

    print(f"Reading Tianji joint state from {args.host} ({robot_ip})", flush=True)
    try:
        robot = Tianji(TianjiConfig(robot_ip=robot_ip))
        try:
            robot.connect()
            positions = robot.get_observation()
        finally:
            robot.disconnect()
    except OSError as exc:
        parser.exit(1, f"Tianji state check failed: {exc}\n")

    print("Joint positions (degrees)")
    for name, degrees in positions.items():
        print(f"  {name}: {degrees:.4f}")


if __name__ == "__main__":
    main()
