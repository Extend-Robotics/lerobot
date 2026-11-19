"""Statistics and real loopback transport checks; never connects to robot hardware."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import zmq

SCRIPT = Path(__file__).resolve().parents[2] / "examples/unitree_g1d/measure_hands.py"
spec = importlib.util.spec_from_file_location("measure_hands", SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_statistics_and_clock_correction():
    rows = [
        {"event": "feedback", "elapsed_s": t, "duration_ms": 0.1, "bridge_age_ms": 2, "clock_delta_ms": delta}
        for t, delta in zip([0.1, 0.2, 0.4], [102, 104, 106], strict=True)
    ]
    report = diag.summarize(rows, 1.0, 100.0)
    assert report["receive_hz"] == 3
    assert report["receive_interval_ms"]["std"] == pytest.approx(50)
    assert report["interval_jitter_ms"]["mean"] == pytest.approx(100)
    assert report["sample_age_ms_clock_corrected"]["mean"] == pytest.approx(4)
    assert report["max_silence_ms_including_start_and_end"] == pytest.approx(600)
    assert report["silences_over_250ms"] == 1
    assert diag.summarize(rows, 1, None)["sample_age_ms_clock_corrected"] is None


def test_missing_stream_reports_full_silence():
    report = diag.summarize([], 30, None)
    assert report["received"] == 0
    assert report["receive_interval_ms"] is None
    assert report["interval_jitter_ms"] is None
    assert report["max_silence_ms_including_start_and_end"] == 30000


def test_cli_loopback_feedback_and_expired_probe(tmp_path):
    context = zmq.Context()
    state = context.socket(zmq.PUB)
    commands = context.socket(zmq.PULL)
    state_port = state.bind_to_random_port("tcp://127.0.0.1")
    command_port = commands.bind_to_random_port("tcp://127.0.0.1")
    process = None
    probes = []
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--robot-ip",
                "127.0.0.1",
                "--sides",
                "left",
                "--left-state-port",
                str(state_port),
                "--left-command-port",
                str(command_port),
                "--seconds",
                "0.5",
                "--probe-commands",
                "--probe-hz",
                "10",
                "--output",
                str(tmp_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < deadline:
            state.send_json(
                {"side": "left", "positions": [0.2] * 6, "age_s": 0.003, "observed_at": time.time() - 0.003}
            )
            while commands.poll(0):
                probes.append(commands.recv_json())
            time.sleep(0.01)
        stdout, stderr = process.communicate(timeout=2)
        assert process.returncode == 0, stdout + stderr
        report = json.loads((tmp_path / "summary.json").read_text())
        assert 0.5 <= report["duration_s"] < 2
        hand = report["hands"]["left"]
        assert hand["received"] > 3
        assert hand["receive_interval_ms"]["std"] is not None
        assert hand["interval_jitter_ms"]["p99"] is not None
        assert hand["sample_age_ms_clock_corrected"] is None
        assert probes and all(p["expires_at"] == 0 for p in probes)
        assert (tmp_path / "samples.csv").is_file()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        state.close(linger=0)
        commands.close(linger=0)
        context.term()
