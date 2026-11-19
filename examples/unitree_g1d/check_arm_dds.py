"""Check direct DDS feedback for both G1 arms, without command publishers.

uv run --no-project --python ../envs/unitree-g1/bin/python python \
    examples/unitree_g1d/check_arm_dds.py --network-interface enx9c69d328e90f \
    --disable-config-trace

Subscribes to the native rt/lowstate topic (both arms share one stream).
Does not connect to ZMQ, switch robot modes, or test command delivery.
Timing describes local SDK callbacks, not one-way latency or actuator response.
"""

import argparse
import json
import math
import statistics
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core import channel
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


class ArmFeedbackProbe:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.arrivals: list[float] = []
        self.last_joints: dict[str, dict[str, float]] = {}
        self.invalid = 0
        self.first_tick: int | None = None
        self.last_tick: int | None = None
        self.tick_changes = 0

    def receive(self, message: Any) -> None:
        received = time.monotonic()
        try:
            joints = {
                joint.name: {
                    "q": float(message.motor_state[joint.value].q),
                    "dq": float(message.motor_state[joint.value].dq),
                    "tau_est": float(message.motor_state[joint.value].tau_est),
                }
                for joint in G1_29_JointArmIndex
            }
            if not all(math.isfinite(v) for state in joints.values() for v in state.values()):
                raise ValueError("Nonfinite arm feedback")
            tick = int(message.tick)
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            with self.lock:
                self.invalid += 1
            return
        with self.lock:
            self.arrivals.append(received)
            self.last_joints = joints
            if self.first_tick is None:
                self.first_tick = tick
            if self.last_tick is not None and tick != self.last_tick:
                self.tick_changes += 1
            self.last_tick = tick

    def report(self, start: float, end: float) -> dict[str, Any]:
        with self.lock:
            times = [t for t in self.arrivals if start <= t <= end]
            joints, invalid = self.last_joints.copy(), self.invalid
            first_tick, last_tick, changes = self.first_tick, self.last_tick, self.tick_changes
        intervals = [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]
        jitter = [abs(b - a) for a, b in zip(intervals, intervals[1:], strict=False)]
        silence = [(b - a) * 1000 for a, b in zip([start, *times], [*times, end], strict=True)]
        return {
            "topic": "rt/lowstate",
            "accessible": bool(times),
            "samples": len(times),
            "receive_hz": len(times) / (end - start),
            "first_sample_after_ms": (times[0] - start) * 1000 if times else None,
            "last_sample_age_ms": (end - times[-1]) * 1000 if times else None,
            "interval_mean_ms": statistics.mean(intervals) if intervals else None,
            "interval_std_ms": statistics.pstdev(intervals) if intervals else None,
            "interval_max_ms": max(intervals) if intervals else None,
            "jitter_mean_ms": statistics.mean(jitter) if jitter else None,
            "jitter_std_ms": statistics.pstdev(jitter) if jitter else None,
            "max_silence_ms": max(silence),
            "silences_over_100ms": sum(v > 100 for v in silence),
            "invalid_samples": invalid,
            "first_tick": first_tick,
            "last_tick": last_tick,
            "tick_changes": changes,
            "last_arm_joints": joints,
        }


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--seconds", type=positive, default=30.0)
    parser.add_argument("--disable-config-trace", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/arm_dds_check.json"))
    args = parser.parse_args()
    require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
    if args.disable_config_trace:
        channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
            "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
        )
    probe = ArmFeedbackProbe()
    subscriber = None
    errors = []
    start = time.monotonic()
    try:
        channel.ChannelFactoryInitialize(args.domain_id, args.network_interface)
        subscriber = channel.ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(probe.receive, 1)
        print(
            f"Listening to arm DDS feedback for {args.seconds:g}s on {args.network_interface} (read-only).",
            flush=True,
        )
        stop = threading.Event()
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            stop.wait(min(1.0, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("Interrupted; saving partial results.", flush=True)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        end = time.monotonic()
        if subscriber is not None:
            try:
                subscriber.Close()
            except Exception as exc:
                errors.append(f"Close: {type(exc).__name__}: {exc}")
    feedback = probe.report(start, end)
    report = {
        "network_interface": args.network_interface,
        "domain_id": args.domain_id,
        "read_only": True,
        "config_trace_disabled": args.disable_config_trace,
        "duration_s_including_setup": end - start,
        "timing": "Local DDS callback intervals; std is population std; jitter is absolute interval change.",
        "feedback": feedback,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")
    return 0 if feedback["accessible"] and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
