"""HTTP integration with the real controller/DAgger loop and fake hardware."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("datasets")

from fastapi.testclient import TestClient  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.rollout import (
    DAggerStrategyConfig,  # noqa: E402
    server,  # noqa: E402
)
from lerobot.rollout.strategies import DAggerStrategy  # noqa: E402
from tests.test_rollout import _LOOP_FEATURES, _make_loop_ctx  # noqa: E402


@pytest.fixture
def session(tmp_path, monkeypatch, request):
    cfg_strategy = DAggerStrategyConfig(num_episodes=1, smooth_handover=False)
    strategy = DAggerStrategy(cfg_strategy)
    started = Event()
    tick = 0

    def resume():
        nonlocal tick
        tick = 0
        started.set()

    def on_tick(_n):
        nonlocal tick
        tick += 1
        if tick == 1:
            strategy._events.request_transition("pause_resume")
        elif tick in (2, 3):
            strategy._events.request_transition("correction")

    ctx, _ = _make_loop_ctx(200, 1, 10**9, on_tick)
    cfg = ctx.runtime.cfg
    cfg.strategy = cfg_strategy
    cfg.interactive = False
    display_mode = getattr(request, "param", None)
    cfg.display_data = display_mode is not None
    cfg.display_mode = display_mode or "rerun"
    cfg.display_ip = "127.0.0.1"
    cfg.display_port = 8765
    cfg.display_compressed_images = True
    init_display = MagicMock()
    close_display = MagicMock()
    telemetry = MagicMock()
    monkeypatch.setattr(server, "init_visualization", init_display)
    monkeypatch.setattr(server, "shutdown_visualization", close_display)
    monkeypatch.setattr("lerobot.rollout.strategies.core.log_visualization_data", telemetry)
    cfg.autosteer_interval_s = 0
    cfg.dataset.streaming_encoding = False
    cfg.dataset.push_to_hub = False
    engine = ctx.policy.inference
    engine.failed = False
    engine.task = "launch task"
    engine.set_task.side_effect = lambda task: setattr(engine, "task", task) or True
    engine.resume.side_effect = resume
    ctx.hardware.initial_position = {"m.pos": 0.0}
    # Exercise the real return interpolation without its three-second delay.
    monkeypatch.setattr("lerobot.rollout.strategies.core.precise_sleep", lambda _: None)
    ctx.hardware.teleop.feedback_features = {}
    ctx.hardware.teleop.get_action.return_value = {"m.pos": 42.0}
    features = {**_LOOP_FEATURES, **cfg_strategy.extra_dataset_features()}
    dataset = LeRobotDataset.create(repo_id="test/server", fps=200, features=features, root=tmp_path / "ds")
    ctx.data.dataset = dataset
    ctx.data.dataset_features = features

    def build(_cfg, shutdown):
        if cfg.display_data:
            init_display.assert_called_once_with(
                cfg.display_mode, session_name="rollout", ip="127.0.0.1", port=8765
            )
        ctx.runtime.shutdown_event = shutdown
        return ctx

    monkeypatch.setattr(server, "build_rollout_context", build)
    monkeypatch.setattr(server, "create_strategy", lambda _: strategy)
    monkeypatch.setattr("lerobot.rollout.strategies.dagger._init_dagger_keyboard", lambda *_: None)
    return_home = MagicMock(wraps=strategy.return_to_initial_position)
    monkeypatch.setattr(strategy, "return_to_initial_position", return_home)
    with TestClient(server.create_app(cfg)) as client:
        yield client, ctx, dataset, started
        returns_before_teardown = return_home.call_count

        def check_disconnect_order():
            if ctx.hardware.initial_position:
                assert return_home.call_count == returns_before_teardown + 1

        ctx.hardware.robot_wrapper.inner.disconnect.side_effect = check_disconnect_order
    if ctx.hardware.initial_position:
        assert return_home.call_count == returns_before_teardown + 1
    assert dataset._is_finalized
    engine.stop.assert_called_once()
    ctx.hardware.robot_wrapper.inner.disconnect.assert_called_once()
    if cfg.display_data:
        close_display.assert_called_once_with(cfg.display_mode)
        assert telemetry.call_count > 0
        assert all(call.args[0] == cfg.display_mode for call in telemetry.call_args_list)
        assert all(call.kwargs["compress_images"] for call in telemetry.call_args_list)
    else:
        init_display.assert_not_called()
        close_display.assert_not_called()
        telemetry.assert_not_called()


def test_timed_episodes_keep_session_connected(session):
    client, ctx, dataset, _ = session
    engine = ctx.policy.inference
    engine.resume.assert_not_called()
    engine.pause.assert_called()
    for index, task in enumerate(("first task", "second task"), start=1):
        start = time.monotonic()
        response = client.post("/episode", json={"task": task, "timeout_s": 0.08})
        assert response.status_code == 200, response.text
        assert response.json() == {"status": "done"}
        assert time.monotonic() - start >= 0.08
        assert engine.task == "launch task"
        assert ctx.hardware.robot_wrapper.send_action.call_args.args[0] == ctx.hardware.initial_position
        assert dataset.num_episodes == index
        assert not dataset._is_finalized
        ctx.hardware.robot_wrapper.inner.disconnect.assert_not_called()
    assert dataset.meta.tasks.index.tolist() == ["first task", "second task"]
    assert ctx.runtime.cfg.return_to_initial_position is True


def test_overlapping_episode_is_rejected(session):
    client, _, _, started = session
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, "/episode", json={"task": "first", "timeout_s": 0.2})
        assert started.wait(2)
        response = client.post("/episode", json={"task": "second", "timeout_s": 0.01})
        assert response.status_code == 409
        assert pending.result(timeout=3).json() == {"status": "done"}


@pytest.mark.parametrize(
    "body",
    [
        {"task": " ", "timeout_s": 1},
        {"task": "task", "timeout_s": 0},
        {"task": "task", "timeout_s": -1},
        {"task": "task", "timeout_s": "Infinity"},
    ],
)
def test_invalid_episode_does_not_start(session, body):
    client, ctx, _, _ = session
    assert client.post("/episode", json=body).status_code == 422
    ctx.policy.inference.resume.assert_not_called()


def test_engine_failure_does_not_report_done(session):
    client, ctx, _, _ = session

    def fail():
        ctx.policy.inference.failed = True
        ctx.runtime.shutdown_event.set()

    ctx.policy.inference.resume.side_effect = fail
    assert client.post("/episode", json={"task": "task", "timeout_s": 1}).status_code == 503
    assert client.post("/episode", json={"task": "task", "timeout_s": 1}).status_code == 503


def test_interrupted_correction_with_slow_save_is_not_done(session, monkeypatch):
    from lerobot.rollout.strategies.dagger import DAggerEvents, DAggerPhase

    client, _, dataset, _ = session
    consume = DAggerEvents.consume_transition
    save = dataset.save_episode
    interrupted = Event()

    def interrupt(events):
        transition = consume(events)
        if events.phase == DAggerPhase.CORRECTING:
            events.stop_recording.set()  # Same signal as the ESC keyboard dispatcher.
            interrupted.set()
        return transition

    def slow_save():
        time.sleep(0.15)  # Cleanup crosses the request's deadline after ESC has ended execution.
        save()

    # A successful prior run must not leave a stale timeout result behind.
    assert client.post("/episode", json={"task": "first", "timeout_s": 0.08}).status_code == 200
    with monkeypatch.context() as patch:
        patch.setattr(DAggerEvents, "consume_transition", interrupt)
        patch.setattr(dataset, "save_episode", slow_save)
        response = client.post("/episode", json={"task": "interrupted", "timeout_s": 0.1})
    assert interrupted.is_set()
    assert response.status_code == 503
    assert dataset.num_episodes == 2
    assert client.post("/episode", json={"task": "next", "timeout_s": 0.08}).json() == {"status": "done"}


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM"])
def test_server_signal_interrupts_active_request(session, signal_name):
    import signal

    import uvicorn

    client, ctx, _, started = session
    shutdown_event = ctx.runtime.shutdown_event.parent
    http_server = server.RolloutServer(uvicorn.Config(client.app), shutdown_event)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, "/episode", json={"task": "long run", "timeout_s": 30})
        try:
            assert started.wait(2)
            http_server.handle_exit(getattr(signal, signal_name), None)
            assert http_server.should_exit
            assert shutdown_event.is_set()
            assert pending.result(timeout=2).status_code == 503
            assert ctx.policy.inference.pause.call_count >= 2
            assert client.post("/episode", json={"task": "next", "timeout_s": 1}).status_code == 503
        finally:
            shutdown_event.set()  # A regression must not leave the executor waiting for 30 seconds.


def test_programmatic_shutdown_stops_rollout_before_draining_requests(session, monkeypatch):
    import asyncio

    import uvicorn

    client, ctx, _, started = session
    shutdown_event = ctx.runtime.shutdown_event.parent
    http_server = server.RolloutServer(uvicorn.Config(client.app), shutdown_event)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, "/episode", json={"task": "long run", "timeout_s": 30})

        async def drain_requests(_server, sockets):
            assert shutdown_event.is_set()
            response = await asyncio.to_thread(pending.result, timeout=2)
            assert response.status_code == 503

        monkeypatch.setattr(uvicorn.Server, "shutdown", drain_requests)
        try:
            assert started.wait(2)
            asyncio.run(http_server.shutdown())
        finally:
            shutdown_event.set()


@pytest.mark.parametrize("session", ["foxglove", "rerun"], indirect=True)
def test_display_enabled_across_episodes(session):
    client, ctx, dataset, _ = session
    ctx.policy.inference.resume.assert_not_called()
    for task in ("first", "second"):
        response = client.post("/episode", json={"task": task, "timeout_s": 0.08})
        assert response.status_code == 200, response.text
        assert response.json() == {"status": "done"}
    assert dataset.num_episodes == 2


def test_visualization_closes_when_context_startup_fails(monkeypatch):
    ctx, _ = _make_loop_ctx(200, 1, 1)
    cfg = ctx.runtime.cfg
    cfg.strategy = DAggerStrategyConfig()
    cfg.interactive = False
    cfg.display_data = True
    cfg.display_mode = "foxglove"
    init_display = MagicMock()
    close_display = MagicMock()
    monkeypatch.setattr(server, "init_visualization", init_display)
    monkeypatch.setattr(server, "shutdown_visualization", close_display)
    monkeypatch.setattr(server, "build_rollout_context", MagicMock(side_effect=RuntimeError("setup failed")))
    with pytest.raises(RuntimeError, match="setup failed"), TestClient(server.create_app(cfg)):
        pass
    init_display.assert_called_once()
    close_display.assert_called_once_with("foxglove")


@pytest.mark.parametrize("reset_failure", ["failed", "skipped"])
def test_unsuccessful_return_does_not_report_done(session, reset_failure):
    client, ctx, _, _ = session
    # A previous successful reset must not mask this request's result.
    assert client.post("/episode", json={"task": "first", "timeout_s": 0.08}).status_code == 200
    if reset_failure == "skipped":
        ctx.hardware.initial_position = None
    else:
        send_action = ctx.hardware.robot_wrapper.send_action

        def fail_on_return(action):
            if ctx.policy.inference.task == "launch task":
                raise RuntimeError("return motor failure")
            return action

        send_action.side_effect = fail_on_return
    response = client.post("/episode", json={"task": "second", "timeout_s": 0.08})
    assert response.status_code == 503
    assert response.json()["detail"] == "Could not return to the initial position"


@pytest.mark.parametrize("shutdown_during_reset", [False, True])
def test_request_waits_and_rejects_overlap_during_return(session, monkeypatch, shutdown_during_reset):
    client, ctx, _, _ = session
    resetting = Event()
    release = Event()

    def wait_during_return(_duration):
        resetting.set()
        assert release.wait(3)

    monkeypatch.setattr("lerobot.rollout.strategies.core.precise_sleep", wait_during_return)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, "/episode", json={"task": "first", "timeout_s": 0.08})
        try:
            assert resetting.wait(2)
            assert not pending.done()
            assert client.post("/episode", json={"task": "second", "timeout_s": 0.08}).status_code == 409
            if shutdown_during_reset:
                ctx.runtime.shutdown_event.parent.set()
        finally:
            release.set()
        response = pending.result(timeout=3)
        assert response.status_code == (503 if shutdown_during_reset else 200)
