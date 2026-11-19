"""Measure the G1 arm ZMQ stream for 30 seconds (read-only by default).

Both arms share rt/lowstate on port 6001. No source timestamp or sequence is
available: this measures receive intervals, not sample age, RTT or packet loss.
Optional probes use an ignored topic, never rt/lowcmd. Only use them with the
repository's run_g1_server.py, which forwards commands only for rt/lowcmd.
Stop other command publishers before probing: conflation can replace commands.
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

PROBE_TOPIC = "diagnostic/arm_communication"


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
    }


def summarize(rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    feedback = [r for r in rows if r["event"] == "feedback"]
    times = [r["elapsed_s"] for r in feedback]
    intervals = [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]
    silence = [(b - a) * 1000 for a, b in zip([0, *times], [*times, seconds], strict=True)]
    probes = [r for r in rows if r["event"] in ("probe_sent", "probe_timeout")]
    return {
        "received": len(feedback),
        "receive_hz": len(feedback) / seconds,
        "receive_interval_ms": distribution(intervals),
        "interval_jitter_ms": distribution(
            [abs(b - a) for a, b in zip(intervals, intervals[1:], strict=False)]
        ),
        "receive_decode_validate_ms": distribution([r["duration_ms"] for r in feedback]),
        "max_silence_ms_including_start_and_end": max(silence),
        "silences_over_100ms": sum(g > 100 for g in silence),
        "silences_over_250ms": sum(g > 250 for g in silence),
        "invalid_messages": sum(r["event"] == "invalid" for r in rows),
        "unexpected_topics": sum(r["event"] == "unexpected_topic" for r in rows),
        "probe_send_ms": distribution([r["duration_ms"] for r in probes]),
        "probe_timeouts": sum(r["event"] == "probe_timeout" for r in probes),
    }


def receive(socket: zmq.Socket, start: float) -> dict[str, Any] | None:
    began = time.monotonic()
    try:
        payload = socket.recv(zmq.NOBLOCK)
    except zmq.Again:
        return None
    arrived = time.monotonic()
    row: dict[str, Any] = {"event": "feedback", "elapsed_s": arrived - start, "bytes": len(payload)}
    try:
        message = json.loads(payload)
        if message["topic"] != "rt/lowstate":
            row["event"] = "unexpected_topic"
        else:
            motors = message["data"]["motor_state"]
            if not isinstance(motors, list) or len(motors) < 29:
                raise ValueError("Expected at least 29 motor states")
            # G1 arm joints occupy indices 15..28 (seven per arm).
            for motor in motors[15:29]:
                if not all(math.isfinite(float(motor[key])) for key in ("q", "dq", "tau_est")):
                    raise ValueError("Nonfinite arm feedback")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        row.update(event="invalid", error=str(exc))
    row["duration_ms"] = (time.monotonic() - began) * 1000
    return row


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--seconds", type=positive, default=30.0)
    parser.add_argument("--state-port", type=int, default=6001)
    parser.add_argument("--command-port", type=int, default=6000)
    parser.add_argument(
        "--probe-commands", action="store_true", help="Send ignored-topic probes to the repository bridge"
    )
    parser.add_argument("--probe-hz", type=positive, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/arm_communication"))
    args = parser.parse_args()
    if not all(0 < p < 65536 for p in (args.state_port, args.command_port)):
        parser.error("Ports must be between 1 and 65535")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    context = zmq.Context()
    sockets: list[zmq.Socket] = []
    try:
        sub = context.socket(zmq.SUB)
        sockets.append(sub)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(f"tcp://{args.robot_ip}:{args.state_port}")
        pub = None
        if args.probe_commands:
            pub = context.socket(zmq.PUSH)
            sockets.append(pub)
            # Match unitree_sdk2_socket.py (including its lack of IMMEDIATE).
            for option, value in ((zmq.LINGER, 0), (zmq.CONFLATE, 1), (zmq.SNDTIMEO, 100)):
                pub.setsockopt(option, value)
            pub.connect(f"tcp://{args.robot_ip}:{args.command_port}")
            print(
                "Ignored-topic probes enabled. Stop other command publishers before running probes.",
                flush=True,
            )
        print(f"Measuring shared arm feedback for {args.seconds:g}s; output: {args.output}", flush=True)
        start = time.monotonic()
        end, next_report, next_probe = start + args.seconds, start + 5, start
        try:
            while time.monotonic() < end:
                wake = min(end, next_report, next_probe if pub is not None else end)
                if sub.poll(max(0, math.ceil((wake - time.monotonic()) * 1000))):
                    row = receive(sub, start)
                    if row is not None:
                        rows.append(row)
                now = time.monotonic()
                if pub is not None and next_probe <= now < end:
                    began = time.monotonic()
                    event = "probe_sent"
                    try:
                        pub.send_json({"topic": PROBE_TOPIC, "data": {"motor_cmd": []}})
                    except zmq.Again:
                        event = "probe_timeout"
                    finished = time.monotonic()
                    rows.append(
                        {
                            "event": event,
                            "elapsed_s": finished - start,
                            "duration_ms": (finished - began) * 1000,
                        }
                    )
                    next_probe = finished + 1 / args.probe_hz
                if time.monotonic() >= next_report:
                    count = sum(r["event"] == "feedback" for r in rows)
                    print(f"{time.monotonic() - start:.1f}s: received {count}", flush=True)
                    next_report = time.monotonic() + 5
        except KeyboardInterrupt:
            print("Interrupted; saving partial measurements.")
        elapsed = max(time.monotonic() - start, 1e-9)
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()
    report = {
        "duration_s": elapsed,
        "robot_ip": args.robot_ip,
        "state_port": args.state_port,
        "command_probes": args.probe_commands,
        "notes": [
            "Both arms share one stream; stats describe received bridge messages, not unique DDS samples.",
            "std is population standard deviation. Timing distributions are in milliseconds.",
            "Jitter is absolute change between successive receive intervals.",
            "No source timestamps/sequence numbers: sample age, one-way latency, RTT and loss fraction are unavailable.",
            "Silence includes startup and trailing outages. SUB conflation may skip messages.",
            "Probe send success is local enqueue only; CONFLATE and no IMMEDIATE can mask a disconnected receiver.",
            "Probe payloads are smaller than motor commands and do not exercise DDS writes or actuator response.",
            "Synchronous probes can stall feedback collection; compare with a read-only baseline.",
        ],
        "arms": summarize(rows, elapsed),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["event", "elapsed_s", "duration_ms", "bytes", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["arms"], indent=2))
    print(f"Saved {args.output / 'summary.json'} and {args.output / 'samples.csv'}")


if __name__ == "__main__":
    main()
