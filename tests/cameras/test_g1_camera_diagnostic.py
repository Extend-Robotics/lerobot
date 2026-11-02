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
            while not stopped.wait(0.02):
                socket.send_json({"images": images, "timestamps": dict.fromkeys(names, time.time())})

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
        sys, "argv", [str(SCRIPT), "--config", str(config), "--seconds", "0.3", "--output", str(output)]
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
        for camera in phase["probe"]["frames"].values():
            assert camera["read_errors"] == 0
        for wire in phase.get("wire", []):
            assert wire["errors"] == []
            if phase["name"] == "shared_decode":
                assert wire["decode_ms"]["count"] > 9
    assert (output / "report.md").exists()
    assert not publisher.is_alive()
