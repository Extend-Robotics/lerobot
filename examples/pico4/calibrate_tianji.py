"""Measure handheld Pico wrist pivots and robot-forward direction, without connecting to Tianji."""

import argparse
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from lerobot.teleoperators.bi_pico4 import BiPico4, BiPico4Config
from lerobot.teleoperators.bi_pico4.tianji_processor import POSE_FIELDS
from lerobot.teleoperators.bi_pico4.wrist_calibration import WristCalibration, fit_wrist_offset
from lerobot.teleoperators.bi_pico4.xr_mapping import sdk_pose_to_matrix


def collect(teleop, side, instruction):
    print(
        f"{side.capitalize()} controller: {instruction}\nHold the grip during the gesture, release to finish.",
        flush=True,
    )
    poses = []
    previous = None
    released = False
    while True:
        sample = teleop.get_action()
        grip = sample[f"{side}.grip"]
        if grip <= 0.3:
            if poses:
                return np.stack(poses)
            released = True
        if released and (grip > 0.5 or poses) and previous != sample["sample_time"]:
            previous = sample["sample_time"]
            poses.append(sdk_pose_to_matrix(np.array([sample[f"{side}.{f}"] for f in POSE_FIELDS])))
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/tianji_wrist_calibration.json"))
    args = parser.parse_args()
    teleop = BiPico4(BiPico4Config(output_mode="xr", input_timeout_s=0.5))
    teleop.connect()
    try:
        offsets = []
        for side in ("left", "right"):
            while True:
                poses = collect(
                    teleop, side, "Support your wrist at one point. Slowly bend it up/down and sideways."
                )
                try:
                    offset, rms = fit_wrist_offset(poses)
                except ValueError as error:
                    print(error)
                    continue
                offsets.append(offset)
                print(f"Pivot residual: {rms * 1000:.1f} mm")
                break
        while True:
            poses = collect(
                teleop,
                "right",
                "Push straight in the direction you want the robot to move forward, then release at the endpoint.",
            )
            wrists = poses[:, :3, 3] + poses[:, :3, :3] @ offsets[1]
            n = min(5, max(1, len(wrists) // 4))
            stroke = wrists[-n:].mean(axis=0) - wrists[:n].mean(axis=0)
            try:
                calibration = WristCalibration(offsets, stroke)
            except ValueError as error:
                print(error)
                continue
            break
    finally:
        teleop.disconnect()
    with NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as output:
        json.dump(
            {
                "controller_to_wrist_m": calibration.offsets.tolist(),
                "xr_forward": calibration.forward.tolist(),
            },
            output,
            indent=2,
        )
        output.write("\n")
    Path(output.name).replace(args.output)
    print(
        f"Saved {args.output}. Repeat calibration after resetting the Pico tracking origin or changing grip."
    )


if __name__ == "__main__":
    main()
