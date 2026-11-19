"""Read bimanual LeRobot Cartesian actions without connecting to a robot."""

import argparse
import json
import time

import numpy as np

from lerobot.teleoperators.bi_pico4 import BiPico4, BiPico4Config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=0, help="0 runs until Ctrl+C")
args = parser.parse_args()
teleop = BiPico4(BiPico4Config())
left = np.array([0, 0.2, 0, 1, 0, 0, 0, 0], dtype=np.float32)
right = np.array([0, -0.2, 0, 1, 0, 0, 0, 0], dtype=np.float32)
try:
    teleop.connect(left_tcp_pose_quat=left, right_tcp_pose_quat=right)
    print("Connected. Hold each side grip and move its controller. Ctrl+C exits.", flush=True)
    start = time.monotonic()
    next_print = start
    while not args.seconds or time.monotonic() - start < args.seconds:
        action = teleop.get_action()
        if time.monotonic() >= next_print:
            print(json.dumps({k: round(float(v), 4) for k, v in action.items()}), flush=True)
            next_print = time.monotonic() + 0.2
        time.sleep(1 / 30)
except KeyboardInterrupt:
    pass
finally:
    if teleop.is_connected:
        teleop.disconnect()
