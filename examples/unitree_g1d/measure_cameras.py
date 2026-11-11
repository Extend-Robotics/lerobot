"""Measure G1 camera transport and host contention without opening motor connections.

Run with the camera publisher active and other host camera/teleop clients stopped:
    uv run --no-sync python examples/unitree_g1d/measure_cameras.py \
        --config configs/unitree_g1_pico.json --seconds 20 --ik --output outputs/camera_diagnostic

Requires the G1 environment's numpy, OpenCV, pyzmq and psutil. Output includes a
Markdown report, JSON summaries and a CSV of one-second host resource samples.
Source clock differences are used only for intervals, never absolute latency.
--ik uses a cached robot model for a synthetic IK workload; it sends no commands.
The one-camera phase still receives bundled images for all published cameras.
shared_decode is an experimental single receiver, not a production adapter change.
"""

import argparse
import base64
import csv
import json
import logging
import math
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import zmq

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return dict(
        zip(
            ("count", "mean", "p50", "p95", "p99", "max"),
            (len(a), float(a.mean()), *map(float, np.percentile(a, [50, 95, 99, 100]))),
            strict=True,
        )
    )


def timing(times: list[float], start: float, end: float) -> dict[str, Any]:
    # Include leading/trailing silence so dead streams cannot look healthy.
    gaps = np.diff(times).tolist() if times else []
    silence = np.diff([start, *times, end]).tolist()
    return {
        "frames": len(times),
        "fps": len(times) / (end - start),
        "gaps_ms": distribution([v * 1000 for v in gaps]),
        "max_silence_ms": max(silence) * 1000,
        "gaps_over_100ms": sum(v > 0.1 for v in silence),
        "gaps_over_250ms": sum(v > 0.25 for v in silence),
    }


def process_snapshot() -> dict[tuple[int, float], tuple[str, float]]:
    result = {}
    for p in psutil.process_iter(["pid", "name", "create_time", "cpu_times"]):
        try:
            info = p.info
            t = info["cpu_times"]
            result[(info["pid"], info["create_time"])] = (info["name"], t.user + t.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def net_snapshot() -> dict[str, dict[str, int]]:
    return {name: value._asdict() for name, value in psutil.net_io_counters(pernic=True).items()}


class Monitor:
    def __init__(self, phase: str, rows: list[dict[str, Any]]) -> None:
        self.phase, self.rows = phase, rows
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.first_net = net_snapshot()
        self.first_processes = process_snapshot()
        self.first_cpu = sum(self.process.cpu_times()[:2])
        self.start = time.perf_counter()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        psutil.cpu_percent(percpu=True)
        self.process.cpu_percent()
        while not self.stop.wait(1):
            cores = psutil.cpu_percent(percpu=True)
            row = {
                "phase": self.phase,
                "elapsed_s": time.perf_counter() - self.start,
                "system_cpu_percent": sum(cores) / len(cores),
                "busiest_core_percent": max(cores),
                "process_cpu_percent": self.process.cpu_percent(),
                "rss_mb": self.process.memory_info().rss / 1e6,
                "threads": self.process.num_threads(),
                "available_memory_mb": psutil.virtual_memory().available / 1e6,
            }
            self.samples.append(row)
            self.rows.append(row)

    def finish(self) -> dict[str, Any]:
        self.stop.set()
        self.thread.join()
        duration = time.perf_counter() - self.start
        net = net_snapshot()
        network = {}
        for name, before in self.first_net.items():
            if name not in net:
                continue
            delta = {k: net[name][k] - value for k, value in before.items()}
            if delta["bytes_recv"] or delta["bytes_sent"]:
                network[name] = {
                    **delta,
                    "rx_mbps": delta["bytes_recv"] * 8 / duration / 1e6,
                    "tx_mbps": delta["bytes_sent"] * 8 / duration / 1e6,
                }
        top = []
        for key, (name, cpu) in process_snapshot().items():
            if key in self.first_processes:
                used = (cpu - self.first_processes[key][1]) / duration * 100
                top.append({"pid": key[0], "name": name, "cpu_percent": used})
        return {
            "duration_s": duration,
            "process_cpu_percent": (sum(self.process.cpu_times()[:2]) - self.first_cpu) / duration * 100,
            "sample_stats": {
                k: distribution([r[k] for r in self.samples])
                for k in ("system_cpu_percent", "busiest_core_percent", "rss_mb", "threads")
            },
            "network": network,
            "top_processes": sorted(top, key=lambda p: p["cpu_percent"], reverse=True)[:10],
        }


class MeasuredCamera(ZMQCamera):
    """Instrument the real adapter; preserve its socket, conflation and decoding."""

    def __init__(self, config: ZMQCameraConfig) -> None:
        super().__init__(config)
        self.arrivals = []
        self.errors = []
        self.measure_start = float("inf")
        self.shape = None

    def _read_from_hardware(self) -> np.ndarray:
        try:
            frame = super()._read_from_hardware()
        except Exception as exc:
            if time.perf_counter() >= self.measure_start and not self.stop_event.is_set():
                self.errors.append(str(exc))
            raise
        now = time.perf_counter()
        if now >= self.measure_start:
            self.arrivals.append(now)
            self.shape = list(frame.shape)
        return frame


class WireReader:
    """One subscriber per endpoint; optionally decode all named views once."""

    def __init__(self, endpoint: str, decode_all: bool = False) -> None:
        self.endpoint = endpoint
        self.decode_all = decode_all
        self.decode_ms = []
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.messages = self.bytes = 0
        self.arrivals, self.stamps, self.sizes, self.samples = {}, {}, {}, {}
        self.parse_ms, self.errors = [], []
        self.measure_start = float("inf")
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            raise TimeoutError("Wire subscriber failed to start")

    def run(self) -> None:
        try:
            with zmq.Context() as context, context.socket(zmq.SUB) as sock:
                sock.setsockopt(zmq.SUBSCRIBE, b"")
                sock.setsockopt(zmq.CONFLATE, 1)
                sock.connect(self.endpoint)
                self.ready.set()
                while not self.stop.is_set():
                    if not sock.poll(100):
                        continue
                    data = sock.recv()
                    began = time.perf_counter()
                    packet = json.loads(data)
                    now = time.perf_counter()
                    if now < self.measure_start:
                        continue
                    self.messages += 1
                    self.bytes += len(data)
                    self.parse_ms.append((now - began) * 1000)
                    for name, encoded in packet.get("images", {}).items():
                        if self.decode_all:
                            decode_start = time.perf_counter()
                            image = cv2.imdecode(
                                np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR
                            )
                            if image is None:
                                raise ValueError(f"Invalid JPEG for {name}")
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            self.decode_ms.append((time.perf_counter() - decode_start) * 1000)
                        self.arrivals.setdefault(name, []).append(time.perf_counter())
                        self.sizes.setdefault(name, []).append(len(encoded))
                        stamp = packet.get("timestamps", {}).get(name)
                        if stamp is not None:
                            self.stamps.setdefault(name, []).append(stamp)
                        samples = self.samples.setdefault(name, [])
                        if len(samples) < 20:
                            samples.append(encoded)
        except Exception as exc:
            self.errors.append(repr(exc))
            self.ready.set()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("Wire reader did not stop")


def decode_benchmark(samples: dict[str, list[str]]) -> dict[str, Any]:
    """Offline microbenchmark of received images, outside the live phase."""
    result = {}
    for name, frames in samples.items():
        b64, jpeg, rgb, shapes, sizes = [], [], [], [], []
        for encoded in frames:
            t0 = time.perf_counter()
            data = base64.b64decode(encoded)
            t1 = time.perf_counter()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            t2 = time.perf_counter()
            if image is None:
                raise ValueError(f"Invalid JPEG for {name}")
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            t3 = time.perf_counter()
            b64.append((t1 - t0) * 1000)
            jpeg.append((t2 - t1) * 1000)
            rgb.append((t3 - t2) * 1000)
            sizes.append(len(data))
            shapes.append(list(image.shape))
        result[name] = {
            "base64_ms": distribution(b64),
            "jpeg_ms": distribution(jpeg),
            "bgr_to_rgb_ms": distribution(rgb),
            "jpeg_bytes": distribution(sizes),
            "shapes": sorted({tuple(s) for s in shapes}),
        }
    return result


def probe(
    seconds: float,
    fps: float,
    cameras: dict[str, MeasuredCamera],
    workload: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    deadline = start
    ticks, work, late, gaps, ik_ms = [], [], [], [], []
    ages = {name: [] for name in cameras}
    repeated = dict.fromkeys(cameras, 0)
    previous = {}
    errors = {name: [] for name in cameras}
    while time.perf_counter() - start < seconds:
        now = time.perf_counter()
        if now < deadline:
            time.sleep(deadline - now)
        began = time.perf_counter()
        if began - start >= seconds:
            break
        ticks.append(began)
        late.append(max(0, began - deadline) * 1000)
        for name, camera in cameras.items():
            try:
                camera.read_latest()
                with camera.frame_lock:
                    stamp = camera.latest_timestamp
                ages[name].append((time.perf_counter() - stamp) * 1000)
                repeated[name] += int(previous.get(name) == stamp)
                previous[name] = stamp
            except Exception as exc:
                errors[name].append(str(exc))
        read_end = time.perf_counter()
        work.append((read_end - began) * 1000)
        if workload is not None:
            workload(len(ticks) - 1)
            ik_ms.append((time.perf_counter() - read_end) * 1000)
        # Match teleop's pacing: an overrun starts a new period, no catch-up burst.
        deadline = began + 1 / fps
    end = time.perf_counter()
    gaps = np.diff(ticks).tolist()
    return {
        "duration_s": end - start,
        "ticks": len(ticks),
        "hz": len(gaps) / sum(gaps) if gaps else None,
        "tick_gaps_ms": distribution([v * 1000 for v in gaps]),
        "wake_lateness_ms": distribution(late),
        "read_work_ms": distribution(work),
        "ik_work_ms": distribution(ik_ms),
        "gaps_over_250ms": sum(v > 0.25 for v in gaps),
        "frames": {
            name: {
                "receive_age_ms": distribution(ages[name]),
                "repeated_reads": repeated[name],
                "read_errors": len(errors[name]),
                "error_examples": errors[name][:3],
            }
            for name in cameras
        },
    }


def report(result: dict[str, Any]) -> str:
    lines = [
        "# G1 camera diagnostic",
        "",
        f"Host: {result['host']}; target probe: {result['fps']} Hz.",
        "",
        "Camera-only run: no robot, hand or PICO control connections were created.",
        "",
        "| Phase | Probe Hz | Tick gap p99 / max (ms) | Read work p99 (ms) | IK p99 (ms) | Process CPU % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in result["phases"]:
        probe_data = p["probe"]
        g = probe_data["tick_gaps_ms"]
        ik = probe_data.get("ik_work_ms")
        ik_text = f"{ik['p99']:.2f}" if ik else "disabled"
        lines.append(
            f"| {p['name']} | {probe_data['hz']:.2f} | {g['p99']:.2f} / {g['max']:.2f} | "
            f"{probe_data['read_work_ms']['p99']:.3f} | {ik_text} | {p['resources']['process_cpu_percent']:.1f} |"
        )
    lines.extend(
        ["", "CPU 100% means one logical core; network counters cover all traffic on each interface.", ""]
    )
    for p in result["phases"]:
        lines += [f"## {p['name']}", "", f"OpenCV thread limit: {p['opencv_threads']}.", ""]
        for name, c in p.get("cameras", {}).items():
            g = c["gaps_ms"]
            lines.append(
                f"- {name}: {c['fps']:.2f} received/decoded FPS; "
                f"gap p99/max {g['p99']:.2f}/{g['max']:.2f} ms; "
                f"max silence {c['max_silence_ms']:.2f} ms; "
                f"gaps >250 ms: {c['gaps_over_250ms']}; errors: {c.get('errors', [])[:3]}"
                if g
                else f"- {name}: no completed frame gaps; max silence {c['max_silence_ms']:.2f} ms."
            )
        for name, net in p["resources"]["network"].items():
            lines.append(
                f"- {name}: RX {net['rx_mbps']:.2f}, TX {net['tx_mbps']:.2f} Mbit/s; "
                f"RX drops/errors {net['dropin']}/{net['errin']}."
            )
        for name, c in p["probe"]["frames"].items():
            a = c["receive_age_ms"]
            if a:
                lines.append(
                    f"- {name} buffered receive age p99/max: {a['p99']:.2f}/{a['max']:.2f} ms; "
                    f"repeated reads {c['repeated_reads']}; read errors {c['read_errors']}."
                )
        for address, output in p.get("ping", {}).items():
            lines.append(f"- Ping {address}: {'; '.join(output.strip().splitlines()[-2:])}")
        lines.append("")
    lines += [
        "## Interpretation limits",
        "",
        "- Receive ages start after host decoding, matching the adapter; they exclude capture/network delay.",
        "- Wire source intervals describe observed source timestamps; skipped/conflated messages are included in gaps.",
        f"- Offline moving-target IK: {result.get('offline_ik_enabled', False)}. The probe excludes XR input, hands, arm publication and display.",
        "- Baseline stops this script's subscribers; the remote publisher stays active throughout.",
        "- Sequential phases are not simultaneous controlled trials. Robot CPU, USB capture timing and true one-way latency require robot-side measurements.",
        "- No JPEG pixels are saved. JSON contains offline decode timings, source intervals, CPU and network detail.",
    ]
    if result.get("errors"):
        lines += ["", "## Run errors", "", *result["errors"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unitree_g1_pico.json"))
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--ik", action="store_true", help="Add offline moving-target G1 IK at each tick")
    parser.add_argument("--output", type=Path, default=Path("outputs/camera_diagnostic"))
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.seconds, args.fps)) or args.seconds * args.fps < 3:
        parser.error("seconds and fps must be finite and positive, with at least three probe periods")
    configs = json.loads(args.config.read_text())["robot"].get("cameras", {})
    if not configs or any(c.get("type") != "zmq" for c in configs.values()):
        parser.error("config must contain named ZMQ cameras")
    args.output.mkdir(parents=True, exist_ok=True)
    original_threads = cv2.getNumThreads()
    result = {
        "host": platform.node(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(args.config.resolve()),
        "camera_config": configs,
        "fps": args.fps,
        "logical_cores": psutil.cpu_count(),
        "opencv_version": cv2.__version__,
        "initial_opencv_threads": original_threads,
        "offline_ik_enabled": args.ik,
        "links": {k: v._asdict() for k, v in psutil.net_if_stats().items()},
        "phases": [],
        "errors": [],
    }
    rows = []
    wire_samples = {}
    workload = None
    if args.ik:
        # Offline model only. No UnitreeG1 instance or SDK/control socket is created.
        from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ArmIK

        print("Preparing offline G1 IK workload (cached model, no robot connection)", flush=True)
        solver = G1_29_ArmIK(unitree_xr=True)
        base_q = np.array([0, 0.1, 0, 0.6, 0, 0, 0, 0, -0.1, 0, 0.6, 0, 0, 0])
        targets = []
        for index in range(120):
            q = base_q.copy()
            q[[0, 7]] += 0.1 * math.sin(index * 2 * math.pi / 120)
            targets.append(solver.forward_kinematics(q))

        def workload(index: int) -> None:
            solver.solve_ik(*targets[index % len(targets)], base_q, np.zeros(14))
            if not solver.last_solve_success:
                raise RuntimeError("Offline IK workload did not converge")

        for index in range(10):
            workload(index)
    result["routes"] = {}
    for address in sorted({c["server_address"] for c in configs.values()}):
        try:
            route = subprocess.run(["ip", "route", "get", address], capture_output=True, text=True, timeout=5)
            result["routes"][address] = route.stdout + route.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["routes"][address] = str(exc)
    phases = [
        ("baseline", [], original_threads),
        ("wire_only", [], original_threads),
        ("one_camera", list(configs)[:1], original_threads),
        ("three_cameras", list(configs), original_threads),
        ("three_cameras_cv1", list(configs), 1),
        ("shared_decode", [], original_threads),
        ("baseline_after", [], original_threads),
    ]
    try:
        for phase, names, threads in phases:
            print(
                f"Starting {phase}: {args.seconds:g}s, cameras={names}, OpenCV threads={threads}", flush=True
            )
            cv2.setNumThreads(threads)
            readers, cameras = [], {}
            try:
                with ExitStack() as stack:
                    for name in names:
                        settings = {k: v for k, v in configs[name].items() if k != "type"}
                        camera = MeasuredCamera(ZMQCameraConfig(**settings))
                        camera.connect()
                        stack.callback(camera.disconnect)
                        cameras[name] = camera
                    if phase in ("wire_only", "shared_decode"):
                        for address, port in sorted(
                            {(c["server_address"], c.get("port", 5555)) for c in configs.values()}
                        ):
                            reader = WireReader(
                                f"tcp://{address}:{port}", decode_all=phase == "shared_decode"
                            )
                            stack.callback(reader.close)
                            readers.append(reader)
                        time.sleep(1)  # Subscription handshake, excluded from measurement.
                    pings = []
                    for address in sorted({c["server_address"] for c in configs.values()}):
                        try:
                            ping = subprocess.Popen(
                                [
                                    "ping",
                                    "-n",
                                    "-c",
                                    str(max(1, int(args.seconds * 2))),
                                    "-i",
                                    "0.5",
                                    "-W",
                                    "1",
                                    address,
                                ],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            stack.callback(
                                lambda process=ping: process.kill() if process.poll() is None else None
                            )
                            pings.append((address, ping))
                        except OSError:
                            logging.warning("ping unavailable")
                    monitor = Monitor(phase, rows)
                    start = time.perf_counter()
                    for item in [*cameras.values(), *readers]:
                        item.measure_start = start
                    try:
                        measured = probe(args.seconds, args.fps, cameras, workload)
                    finally:
                        end = time.perf_counter()
                        resources = monitor.finish()
                        for item in [*cameras.values(), *readers]:
                            item.measure_start = float("inf")
                    p = {
                        "name": phase,
                        "opencv_threads": threads,
                        "probe": measured,
                        "resources": resources,
                        "cameras": {},
                        "ping": {},
                    }
                    for address, ping in pings:
                        try:
                            output, _ = ping.communicate(timeout=2)
                        except subprocess.TimeoutExpired:
                            ping.terminate()
                            output, _ = ping.communicate(timeout=2)
                        p["ping"][address] = output
                    for name, camera in cameras.items():
                        times = [t for t in camera.arrivals if start <= t <= end]
                        p["cameras"][name] = {
                            **timing(times, start, end),
                            "shape": camera.shape,
                            "errors": camera.errors,
                        }
                    if readers:
                        p["wire"] = []
                        for reader in readers:
                            reader.close()
                            p["wire"].append(
                                {
                                    "endpoint": reader.endpoint,
                                    "messages": reader.messages,
                                    "payload_mbps": reader.bytes * 8 / (end - start) / 1e6,
                                    "json_parse_ms": distribution(reader.parse_ms),
                                    "decode_all": reader.decode_all,
                                    "decode_ms": distribution(reader.decode_ms),
                                    "errors": reader.errors,
                                }
                            )
                            for name in configs:
                                settings = configs[name]
                                if (
                                    reader.endpoint
                                    != f"tcp://{settings['server_address']}:{settings.get('port', 5555)}"
                                ):
                                    continue
                                p["cameras"][name] = {
                                    **timing(reader.arrivals.get(name, []), start, end),
                                    "base64_bytes": distribution(reader.sizes.get(name, [])),
                                    "source_intervals_ms": distribution(
                                        (np.diff(reader.stamps.get(name, [])) * 1000).tolist()
                                    ),
                                }
                            wire_samples.update(reader.samples)
                    result["phases"].append(p)
                    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
                    print(
                        f"Finished {phase}: probe {measured['hz']:.2f} Hz, CPU {resources['process_cpu_percent']:.1f}%",
                        flush=True,
                    )
            except Exception as exc:
                result["errors"].append(f"{phase}: {type(exc).__name__}: {exc}")
                logging.exception("Phase failed: %s", phase)
        result["offline_decode_benchmark"] = {}
        for threads in sorted({1, original_threads}):
            cv2.setNumThreads(threads)
            result["offline_decode_benchmark"][str(threads)] = decode_benchmark(wire_samples)
    finally:
        cv2.setNumThreads(original_threads)
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "report.md").write_text(report(result))
        if rows:
            with (args.output / "resources.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Reports written to {args.output.resolve()}", flush=True)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
