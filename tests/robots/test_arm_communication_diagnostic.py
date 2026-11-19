"""Arm diagnostic statistics and loopback checks, without robot connections."""

import json
import runpy
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zmq

SCRIPT = Path(__file__).resolve().parents[2] / "examples/unitree_g1d/measure_arms.py"
diag = runpy.run_path(str(SCRIPT))


def test_statistics_and_missing_stream():
    rows = [{"event": "feedback", "elapsed_s": t, "duration_ms": 0.1} for t in (0.1, 0.2, 0.4)]
    report = diag["summarize"](rows, 1)
    assert report["receive_interval_ms"]["std"] == pytest.approx(50)
    assert report["interval_jitter_ms"]["mean"] == pytest.approx(100)
    assert report["max_silence_ms_including_start_and_end"] == pytest.approx(600)
    empty = diag["summarize"]([], 30)
    assert empty["receive_hz"] == 0
    assert empty["max_silence_ms_including_start_and_end"] == 30000
    assert empty["interval_jitter_ms"] is None


@pytest.mark.parametrize(
    "payload", [b"garbage", b"null", b'{"topic":"rt/lowstate","data":{"motor_state":[]}}']
)
def test_invalid_feedback(payload):
    socket = MagicMock()
    socket.recv.return_value = payload
    assert diag["receive"](socket, time.monotonic())["event"] == "invalid"


@pytest.mark.parametrize("probe", [False, True])
def test_cli_loopback(tmp_path, probe):
    context = zmq.Context()
    state = context.socket(zmq.PUB)
    commands = context.socket(zmq.PULL)
    state_port = state.bind_to_random_port("tcp://127.0.0.1")
    command_port = commands.bind_to_random_port("tcp://127.0.0.1")
    process = None
    probes = []
    try:
        args = [
            sys.executable,
            str(SCRIPT),
            "--robot-ip",
            "127.0.0.1",
            "--state-port",
            str(state_port),
            "--command-port",
            str(command_port),
            "--seconds",
            "0.5",
            "--output",
            str(tmp_path),
        ]
        if probe:
            args += ["--probe-commands", "--probe-hz", "10"]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < deadline:
            state.send_json(
                {"topic": "rt/lowstate", "data": {"motor_state": [{"q": 0, "dq": 0, "tau_est": 0}] * 35}}
            )
            while commands.poll(0):
                probes.append(commands.recv_json())
            time.sleep(0.01)
        stdout, stderr = process.communicate(timeout=2)
        assert process.returncode == 0, stdout + stderr
        report = json.loads((tmp_path / "summary.json").read_text())
        assert report["arms"]["received"] > 3
        assert report["arms"]["interval_jitter_ms"]["p99"] is not None
        assert 0.5 <= report["duration_s"] < 2
        if probe:
            assert probes and all(p["topic"] != "rt/lowcmd" for p in probes)
            assert all(p["data"]["motor_cmd"] == [] for p in probes)
        else:
            assert not probes
            assert report["arms"]["probe_send_ms"] is None
        assert (tmp_path / "samples.csv").exists()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        state.close(linger=0)
        commands.close(linger=0)
        context.term()
