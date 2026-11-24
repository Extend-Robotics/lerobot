"""Measure SDK stream freshness without connecting to a robot.

Poll at 500 Hz; observed update gaps include poll scheduling jitter. Source
clock deltas are meaningful, but absolute headset/host clock offsets are not.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path


def stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "median_ms": ordered[len(ordered) // 2] * 1000,
        "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1] * 1000,
        "p99_ms": ordered[math.ceil(len(ordered) * 0.99) - 1] * 1000,
        "max_ms": ordered[-1] * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, default=Path("/tmp/pico-stream.csv"))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")
    import xensevr_pc_service_sdk as sdk

    gaps, source_gaps, polls, reads = [], [], [], []
    previous_stamp = previous_update = previous_poll = previous_pose = None
    resets = updates = pose_changes = pose_changes_without_stamp = 0
    first_update = None
    max_age = 0.0
    last_report = -5.0
    sdk.init()
    start = time.monotonic()
    try:
        with args.output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("elapsed_s", "source_ns", "age_s", "poll_gap_s", "read_s", "pose_changed"))
            while (elapsed := time.monotonic() - start) < args.seconds:
                began = time.monotonic()
                stamp = int(sdk.get_time_stamp_ns())
                pose = tuple(sdk.get_left_controller_pose()) + tuple(sdk.get_right_controller_pose())
                now = time.monotonic()
                reads.append(now - began)
                poll_gap = now - previous_poll if previous_poll is not None else 0.0
                if previous_poll is not None:
                    polls.append(poll_gap)
                changed = previous_pose is not None and pose != previous_pose
                pose_changes += int(changed)
                if stamp > 0 and stamp != previous_stamp:
                    if previous_update is not None:
                        gaps.append(now - previous_update)
                        if stamp > previous_stamp:
                            source_gaps.append((stamp - previous_stamp) / 1e9)
                        else:
                            resets += 1
                    updates += 1
                    previous_update = now
                    if first_update is None:
                        first_update = now
                elif changed:
                    pose_changes_without_stamp += 1
                age = now - (previous_update if previous_update is not None else start)
                max_age = max(max_age, age)
                writer.writerow((elapsed, stamp, age, poll_gap, now - began, int(changed)))
                if elapsed - last_report >= 5:
                    print(
                        f"t={elapsed:.1f}s updates={updates} age={age:.3f}s max_age={max_age:.3f}s",
                        flush=True,
                    )
                    last_report = elapsed
                previous_stamp, previous_pose, previous_poll = stamp, pose, now
                time.sleep(0.002)
    finally:
        sdk.close()
    duration = time.monotonic() - start
    result = {
        "duration_s": duration,
        "updates": updates,
        "observed_updates_per_s": updates / duration,
        "first_update_s": None if first_update is None else first_update - start,
        "max_observed_age_s": max_age,
        "trailing_age_s": age,
        "timestamp_resets": resets,
        "pose_changes": pose_changes,
        "pose_changes_without_stamp": pose_changes_without_stamp,
        "observed_update_gaps": stats(gaps),
        "source_timestamp_steps": stats(source_gaps),
        "poll_intervals": stats(polls),
        "sdk_read_duration": stats(reads),
        "completed_gaps_over_250ms": sum(gap > 0.25 for gap in gaps),
        "completed_gaps_over_500ms": sum(gap > 0.5 for gap in gaps),
        "csv": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
