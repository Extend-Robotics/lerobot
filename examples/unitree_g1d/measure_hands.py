"""Measure the Revo2 ZMQ bridge for 30 seconds, without moving the hands.

uv run --no-sync python examples/unitree_g1d/measure_hands.py --robot-ip 192.168.123.164

Run read-only alongside teleoperation to observe feedback under load. Optional
--probe-commands sends expired targets: the repository's bridge rejects these
and logs warnings. Stop other command publishers before probing: bridge
conflation can replace their commands with these probes. Send timings measure
local enqueue/blocking, NOT delivery,
acknowledgement, round-trip time or actuator response. This mode requires the
bridge.py deadline validation and should not be used with a different server.

SUB conflation matches the production client. Rates/gaps describe messages
received by this diagnostic, not DDS samples or packet loss. The bridge can
republish cached samples. Clock delta includes server sample age and clock skew;
absolute sample age requires --clock-offset-ms (client clock minus robot clock).
Jitter is the absolute change between consecutive receive intervals. Delay
variation is also reported; constant clock offsets cancel, clock steps do not.
"""

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import zmq


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo = math.floor(index)
        hi = math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
    }


def summarize(rows: list[dict[str, Any]], seconds: float, offset_ms: float | None) -> dict[str, Any]:
    feedback = [r for r in rows if r["event"] == "feedback"]
    times = [r["elapsed_s"] for r in feedback]
    intervals = [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]
    silence = [(b - a) * 1000 for a, b in zip([0.0, *times], [*times, seconds], strict=True)]
    deltas = [r["clock_delta_ms"] for r in feedback]
    ages = (
        [max(r["bridge_age_ms"], r["clock_delta_ms"] - offset_ms) for r in feedback]
        if offset_ms is not None
        else []
    )
    sends = [r for r in rows if r["event"] in ("probe_sent", "probe_timeout")]
    return {
        "received": len(feedback),
        "receive_hz": len(feedback) / seconds,
        "receive_interval_ms": distribution(intervals),
        "receive_decode_ms": distribution([r["duration_ms"] for r in feedback]),
        "interval_jitter_ms": distribution(
            [abs(b - a) for a, b in zip(intervals, intervals[1:], strict=False)]
        ),
        "max_silence_ms_including_start_and_end": max(silence),
        "silences_over_100ms": sum(g > 100 for g in silence),
        "silences_over_250ms": sum(g > 250 for g in silence),
        "bridge_sample_age_ms": distribution([r["bridge_age_ms"] for r in feedback]),
        "client_minus_source_clock_ms": distribution(deltas),
        "sample_age_ms_clock_corrected": distribution(ages),
        "delay_change_ms": distribution([b - a for a, b in zip(deltas, deltas[1:], strict=False)]),
        "absolute_delay_change_ms": distribution(
            [abs(b - a) for a, b in zip(deltas, deltas[1:], strict=False)]
        ),
        "bridge_samples_over_500ms": sum(r["bridge_age_ms"] > 500 for r in feedback),
        "corrected_samples_over_500ms": sum(a > 500 for a in ages) if offset_ms is not None else None,
        "invalid_messages": sum(r["event"] == "invalid" for r in rows),
        "probe_send_ms": distribution([r["duration_ms"] for r in sends]),
        "probe_timeouts": sum(r["event"] == "probe_timeout" for r in sends),
    }


def receive(socket: zmq.Socket, side: str, start: float) -> dict[str, Any] | None:
    began = time.monotonic()
    try:
        data = socket.recv_json(zmq.NOBLOCK)
    except zmq.Again:
        return None
    except (ValueError, UnicodeError) as exc:
        return {"event": "invalid", "elapsed_s": time.monotonic() - start, "error": str(exc)}
    now, wall = time.monotonic(), time.time()
    row: dict[str, Any] = {"event": "feedback", "elapsed_s": now - start, "duration_ms": (now - began) * 1000}
    try:
        age, observed = float(data["age_s"]), float(data["observed_at"])
        q = [float(v) for v in data["positions"]]
        if data["side"] != side or not math.isfinite(age) or age < 0 or not math.isfinite(observed):
            raise ValueError("Invalid side, age or timestamp")
        if len(q) != 6 or not all(math.isfinite(v) for v in q):
            raise ValueError("Expected six finite positions")
        row.update(bridge_age_ms=age * 1000, clock_delta_ms=(wall - observed) * 1000)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        row.update(event="invalid", error=str(exc))
    return row


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--seconds", type=positive, default=30.0)
    parser.add_argument("--sides", nargs="+", choices=["left", "right"], default=["left", "right"])
    parser.add_argument("--left-state-port", type=int, default=6003)
    parser.add_argument("--right-state-port", type=int, default=6005)
    parser.add_argument("--left-command-port", type=int, default=6002)
    parser.add_argument("--right-command-port", type=int, default=6004)
    parser.add_argument(
        "--clock-offset-ms", type=float, help="Client wall clock minus robot wall clock; 0 if synced"
    )
    parser.add_argument(
        "--probe-commands", action="store_true", help="Send expired commands; bridge logs rejections"
    )
    parser.add_argument("--probe-hz", type=positive, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/hand_communication"))
    args = parser.parse_args()
    if args.clock_offset_ms is not None and not math.isfinite(args.clock_offset_ms):
        parser.error("--clock-offset-ms must be finite")
    for side in ("left", "right"):
        for kind in ("state", "command"):
            if not 0 < getattr(args, f"{side}_{kind}_port") < 65536:
                parser.error("Ports must be between 1 and 65535")
    args.output.mkdir(parents=True, exist_ok=True)
    sides = list(dict.fromkeys(args.sides))
    rows: dict[str, list[dict[str, Any]]] = {side: [] for side in sides}
    context = zmq.Context()
    sockets: list[zmq.Socket] = []
    subscribers, publishers = {}, {}
    poller = zmq.Poller()
    try:
        for side in sides:
            sub = context.socket(zmq.SUB)
            sockets.append(sub)
            sub.setsockopt(zmq.LINGER, 0)
            sub.setsockopt(zmq.CONFLATE, 1)
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            sub.connect(f"tcp://{args.robot_ip}:{getattr(args, side + '_state_port')}")
            subscribers[side] = sub
            poller.register(sub, zmq.POLLIN)
            if args.probe_commands:
                pub = context.socket(zmq.PUSH)
                sockets.append(pub)
                for option, value in (
                    (zmq.LINGER, 0),
                    (zmq.IMMEDIATE, 1),
                    (zmq.SNDTIMEO, 100),
                    (zmq.SNDHWM, 1),
                ):
                    pub.setsockopt(option, value)
                pub.connect(f"tcp://{args.robot_ip}:{getattr(args, side + '_command_port')}")
                publishers[side] = pub
        print(f"Measuring {', '.join(sides)} for {args.seconds:g}s; output: {args.output}", flush=True)
        if args.probe_commands:
            print(
                "Expired probes enabled: stop other command publishers; probes can replace queued targets. "
                "Send timings are not round-trip latency.",
                flush=True,
            )
        start = time.monotonic()
        end = start + args.seconds
        next_probe = start
        next_report = start + 5
        try:
            while time.monotonic() < end:
                now = time.monotonic()
                wake = min(end, next_report, next_probe if publishers else end)
                ready = dict(poller.poll(max(0, math.ceil((wake - now) * 1000))))
                for side, sub in subscribers.items():
                    if sub in ready:
                        row = receive(sub, side, start)
                        if row is not None:
                            rows[side].append(row)
                now = time.monotonic()
                if publishers and now >= next_probe:
                    for side, pub in publishers.items():
                        if time.monotonic() >= end:
                            break
                        began = time.monotonic()
                        event = "probe_sent"
                        try:
                            pub.send_json({"positions": [0.0] * 6, "speed": 1.0, "expires_at": 0.0})
                        except zmq.Again:
                            event = "probe_timeout"
                        finished = time.monotonic()
                        rows[side].append(
                            {
                                "event": event,
                                "elapsed_s": finished - start,
                                "duration_ms": (finished - began) * 1000,
                            }
                        )
                    next_probe = time.monotonic() + 1 / args.probe_hz
                if time.monotonic() >= next_report:
                    counts = {
                        side: sum(r["event"] == "feedback" for r in data) for side, data in rows.items()
                    }
                    print(f"{time.monotonic() - start:.1f}s: received {counts}", flush=True)
                    next_report = time.monotonic() + 5
        except KeyboardInterrupt:
            print("Interrupted; saving partial measurements.")
        elapsed = max(time.monotonic() - start, 1e-9)
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()
    summary = {
        "duration_s": elapsed,
        "robot_ip": args.robot_ip,
        "clock_offset_ms": args.clock_offset_ms,
        "command_probes": args.probe_commands,
        "notes": [
            "std is population standard deviation; all distributions are in milliseconds.",
            "Interval jitter = absolute difference between successive receive intervals.",
            "Rates are received bridge messages, not unique DDS samples; conflation can skip messages.",
            "Clock deltas are not latency without clock synchronization; no RTT or loss fraction is measured.",
            "Probe sends run sequentially and can delay feedback collection by 100 ms per hand.",
            "Silence includes initial connection/subscription setup and trailing outages.",
        ],
        "hands": {side: summarize(data, elapsed, args.clock_offset_ms) for side, data in rows.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "side",
                "event",
                "elapsed_s",
                "duration_ms",
                "bridge_age_ms",
                "clock_delta_ms",
                "error",
            ],
        )
        writer.writeheader()
        for side, data in rows.items():
            writer.writerows({"side": side, **row} for row in data)
    print(json.dumps(summary["hands"], indent=2))
    print(f"Saved {args.output / 'summary.json'} and {args.output / 'samples.csv'}")


if __name__ == "__main__":
    main()
