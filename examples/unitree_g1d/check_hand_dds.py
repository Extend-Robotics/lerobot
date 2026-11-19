"""Read-only direct DDS connectivity check for both Revo2 hands.

From the lerobot directory, using the existing Unitree environment:
    uv run --no-project --python ../envs/unitree-g1/bin/python python \
        examples/unitree_g1d/check_hand_dds.py --network-interface enx9c69d328e90f

Subscribes only to rt/brainco/{left,right}/state. No command publishers, ZMQ
bridge, body connections or mode changes. Success establishes feedback access,
not command delivery. Callback timing excludes driver and network sample age.
"""

import argparse
import json
import math
import statistics
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.robots.brainco_revo2.config_brainco_revo2 import BraincoRevo2HandConfig
from lerobot.robots.brainco_revo2.transport import DdsHandTransport
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core import channel


class FeedbackProbe(DdsHandTransport):
    def __init__(self, config: BraincoRevo2HandConfig):
        super().__init__(config)
        self.arrivals: list[float] = []
        self.last_positions: list[float] | None = None
        self.invalid = 0

    def _receive(self, msg: Any) -> None:
        received = time.monotonic()
        try:
            q = [float(state.q) for state in msg.states]
            if len(q) != 6 or not all(math.isfinite(value) for value in q):
                raise ValueError("Expected six finite positions")
        except (AttributeError, TypeError, ValueError, OverflowError):
            with self._lock:
                self.invalid += 1
            return
        with self._lock:
            self.arrivals.append(received)
            self.last_positions = q

    def report(self, start: float, end: float) -> dict[str, Any]:
        with self._lock:
            times = [t for t in self.arrivals if start <= t <= end]
            q, invalid = self.last_positions, self.invalid
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]
        jitter = [abs(b - a) for a, b in zip(gaps, gaps[1:], strict=False)]
        silence = [b - a for a, b in zip([start, *times], [*times, end], strict=True)]
        return {
            "topic": f"rt/brainco/{self.config.side}/state",
            "accessible": bool(times),
            "samples": len(times),
            "receive_hz": len(times) / (end - start),
            "first_sample_after_ms": (times[0] - start) * 1000 if times else None,
            "last_sample_age_ms": (end - times[-1]) * 1000 if times else None,
            "interval_mean_ms": statistics.mean(gaps) if gaps else None,
            "interval_std_ms": statistics.pstdev(gaps) if gaps else None,
            "interval_max_ms": max(gaps) if gaps else None,
            "jitter_mean_ms": statistics.mean(jitter) if jitter else None,
            "jitter_std_ms": statistics.pstdev(jitter) if jitter else None,
            "max_silence_ms": max(silence) * 1000,
            "last_positions": q,
            "invalid_samples": invalid,
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
    parser.add_argument("--network-interface", required=True, help="Host interface connected to the robot")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--seconds", type=positive, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/hand_dds_check.json"))
    parser.add_argument(
        "--disable-config-trace",
        action="store_true",
        help="Disable SDK config tracing in this process (workaround for native config-print crash)",
    )
    args = parser.parse_args()
    if args.disable_config_trace:
        require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
        channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
            "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
        )
    probes = {
        side: FeedbackProbe(
            BraincoRevo2HandConfig(
                side=side,
                transport="dds",
                read_only=True,
                network_interface=args.network_interface,
                domain_id=args.domain_id,
            )
        )
        for side in ("left", "right")
    }
    errors: dict[str, str] = {}
    started = time.monotonic()
    try:
        for side, probe in probes.items():
            try:
                probe.connect()
            except Exception as exc:
                errors[side] = f"{type(exc).__name__}: {exc}"
        print(
            f"Listening for {args.seconds:g}s on {args.network_interface}, DDS domain {args.domain_id} (read-only).",
            flush=True,
        )
        stop = threading.Event()
        deadline = time.monotonic() + args.seconds
        try:
            while time.monotonic() < deadline:
                stop.wait(min(1.0, max(0.0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            print("Interrupted; saving partial results.", flush=True)
        ended = time.monotonic()
        hands = {side: probe.report(started, ended) for side, probe in probes.items()}
    finally:
        for side, probe in probes.items():
            try:
                probe.close()
            except Exception as exc:
                errors[f"{side}_close"] = f"{type(exc).__name__}: {exc}"
    report = {
        "config_trace_disabled": args.disable_config_trace,
        "network_interface": args.network_interface,
        "domain_id": args.domain_id,
        "duration_s_including_setup": ended - started,
        "read_only": True,
        "timing": "Local DDS callback arrivals; jitter is absolute change of consecutive intervals.",
        "hands": hands,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")
    return 0 if all(hand["accessible"] for hand in hands.values()) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
