"""Read Revo2 feedback without creating command publishers or switching robot modes."""

import argparse
import time

from lerobot.robots.brainco_revo2 import BraincoRevo2Hand, BraincoRevo2HandConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--transport", choices=("dds", "zmq"), default="dds")
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--network-interface")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    config = BraincoRevo2HandConfig(
        side=args.side,
        transport=args.transport,
        robot_ip=args.robot_ip,
        network_interface=args.network_interface,
        read_only=True,
    )
    with BraincoRevo2Hand(config) as hand:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            print(hand.get_observation(), flush=True)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
