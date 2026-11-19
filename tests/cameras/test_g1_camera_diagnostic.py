"""Check that camera diagnosis detects silence and exercises real local streams."""

import json
import runpy
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
pytest.importorskip("psutil")

from lerobot.cameras.zmq.image_server import encode_image  # noqa: E402
from lerobot.cameras.zmq.protocol import pack_frame  # noqa: E402

SCRIPT = Path(__file__).parents[2] / "examples/unitree_g1d/measure_cameras.py"


def test_camera_timing_includes_dead_stream_and_trailing_silence():
    timing = runpy.run_path(str(SCRIPT))["timing"]
    empty = timing([], 10.0, 11.0)
    assert empty["fps"] == 0
    assert empty["max_silence_ms"] == 1000
    assert empty["gaps_over_250ms"] == 1
    stalled = timing([10.01, 10.04], 10.0, 11.0)
    assert stalled["max_silence_ms"] == pytest.approx(960)
    assert stalled["gaps_over_250ms"] == 1


def test_diagnostic_cli_measures_three_local_streams(tmp_path, monkeypatch):
    endpoints = Queue()
    stopped = threading.Event()
    names = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    images = {}
    for channel, name in enumerate(names):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, channel] = 255
        images[name] = encode_image(image)

    def publish():
        with zmq.Context() as context, context.socket(zmq.PUB) as socket:
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            endpoints.put(port)
            sequence = 0
            while not stopped.wait(0.02):
                sequence += 1
                # Five-second clock offset plus a varying synthetic camera delay.
                stamp = time.time() - 5 - (sequence % 3 + 1) * 0.01
                for name, jpeg in images.items():
                    socket.send(pack_frame(name, b"a" * 16, sequence, stamp, jpeg))

    publisher = threading.Thread(target=publish)
    publisher.start()
    port = endpoints.get(timeout=5)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "robot": {
                    "cameras": {
                        name: {
                            "type": "zmq",
                            "camera_name": name,
                            "server_address": "127.0.0.1",
                            "port": port,
                            "width": 32,
                            "height": 24,
                            "fps": 50,
                            "warmup_s": 0,
                        }
                        for name in names
                    }
                }
            }
        )
    )
    output = tmp_path / "report"
    original_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--config",
            str(config),
            "--seconds",
            "0.3",
            "--clock-offset-ms",
            "5000",
            "--output",
            str(output),
        ],
    )
    try:
        runpy.run_path(str(SCRIPT))["main"]()
    finally:
        cv2.setNumThreads(original_threads)
        stopped.set()
        publisher.join(timeout=3)
    result = json.loads((output / "report.json").read_text())
    assert result["errors"] == []
    assert len(result["phases"]) == 7
    for phase in result["phases"]:
        for camera in phase["cameras"].values():
            assert camera["frames"] > 3
            assert camera.get("errors", []) == []
            assert camera["latency_ms"]["count"] > 3
            assert 5 < camera["latency_ms"]["mean"] < 200
            assert camera["latency_stddev_ms"] > 1
            assert camera["latency_variance_ms2"] > 1
            assert camera["absolute_delay_change_ms"]["count"] > 2
        for camera in phase["probe"]["frames"].values():
            assert camera["read_errors"] == 0
        for wire in phase.get("wire", []):
            assert wire["errors"] == []
            if phase["name"] == "shared_decode":
                assert wire["decode_ms"]["count"] > 9
    assert (output / "report.md").exists()
    assert "variance" in (output / "report.md").read_text()
    assert "source_wall_s" in (output / "frames.csv").read_text()
    assert not publisher.is_alive()


def test_latency_variance_and_jitter_ignore_constant_offset_and_session_boundary():
    summary = runpy.run_path(str(SCRIPT))["latency_summary"]
    samples = [
        {
            "source_wall_s": 100.0 + i,
            "host_wall_s": 105.0 + i + delay / 1000,
            "session": "first",
            "sequence": sequence,
        }
        for i, delay, sequence in [(0, 10, 1), (1, 20, 2), (2, 30, 4)]
    ]
    result = summary(samples, 5000)
    assert result["latency_ms"]["mean"] == pytest.approx(20)
    assert result["latency_stddev_ms"] == pytest.approx(np.std([10, 20, 30]))
    assert result["latency_variance_ms2"] == pytest.approx(np.var([10, 20, 30]))
    assert result["absolute_delay_change_ms"]["mean"] == pytest.approx(10)
    assert result["skipped_sequences"] == 1
    uncalibrated = summary(samples, None)
    assert uncalibrated["latency_ms"] is None
    assert uncalibrated["latency_variance_ms2"] == result["latency_variance_ms2"]
    samples.append({"source_wall_s": 0.0, "host_wall_s": 105.0, "session": "second", "sequence": 0})
    restarted = summary(samples, None)
    assert restarted["absolute_delay_change_ms"]["count"] == 2
    assert restarted["publisher_sessions"] == 2
    assert summary([], None)["latency_variance_ms2"] is None
