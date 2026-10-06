"""A single-robot API for corrections-only DAgger.

Each request runs until success is detected or its timeout expires, returns the robot to its initial position,
and then responds. Interruption or an unsuccessful return produces HTTP 503.
The timeout limits rollout duration; the request also waits for the return move.
"""

import asyncio
import logging
import time
from contextlib import ExitStack, asynccontextmanager
from threading import Event, Lock, Thread
from typing import Annotated

from lerobot.utils.import_utils import require_package
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

require_package("fastapi", "rollout-server")
require_package("uvicorn", "rollout-server")
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field, StringConstraints  # noqa: E402

from .configs import DAggerStrategyConfig, RolloutConfig  # noqa: E402
from .context import RolloutContext, build_rollout_context  # noqa: E402
from .controller import LinkedEvent, RolloutController, RolloutEvent  # noqa: E402
from .strategies import DAggerStrategy, create_strategy  # noqa: E402
from .success import SuccessDetectionConfig, SuccessMonitor  # noqa: E402

logger = logging.getLogger(__name__)


class RolloutServer(uvicorn.Server):
    """Signal the control loop before Uvicorn waits for active HTTP requests."""

    def __init__(self, config: uvicorn.Config, shutdown_event: Event):
        super().__init__(config)
        self.shutdown_event = shutdown_event

    def handle_exit(self, sig, frame):
        self.shutdown_event.set()
        super().handle_exit(sig, frame)

    async def shutdown(self, sockets=None):
        # Also cover shutdown requested through Uvicorn's should_exit flag.
        self.shutdown_event.set()
        await super().shutdown(sockets)


class EpisodeRequest(BaseModel):
    task: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    timeout_s: float = Field(gt=0, allow_inf_nan=False)


class _RolloutSession:
    """Coordinate HTTP worker threads with one controller thread.

    The episode lock admits one request through rollout and return movement.
    The controller callback publishes a terminal event before waking that request;
    waking means it can inspect the outcome, not necessarily that it succeeded.
    """

    def __init__(
        self,
        ctx: RolloutContext,
        strategy: DAggerStrategy,
        shutdown_event: Event,
        success: SuccessMonitor | None = None,
    ):
        self.success = success
        self.strategy = strategy
        self.shutdown_event = shutdown_event
        self.episode_lock = Lock()
        self.request_finished = Event()
        self.segment_started = Event()
        self.terminal_event: RolloutEvent | None = None
        self.controller = RolloutController(strategy, ctx, on_event=self.on_controller_event)

    def on_controller_event(self, event: RolloutEvent, _payload) -> None:
        """Publish completion or interruption from the controller thread."""
        if event is RolloutEvent.SEGMENT_STARTED:
            self.segment_started.set()
            return
        if event not in (
            RolloutEvent.SEGMENT_ENDED,
            RolloutEvent.RESET_DONE,
            RolloutEvent.RESET_FAILED,
            RolloutEvent.RESET_SKIPPED,
            RolloutEvent.STOPPED,
        ):
            return
        self.terminal_event = event
        self.request_finished.set()
        self.segment_started.set()  # Also unblock startup if the controller stops before starting.

    def _wait_for_decision(self, timeout_s: float) -> str:
        """Keep deadline and success policy here, outside the DAgger loop."""
        deadline = time.monotonic() + timeout_s
        while not self.request_finished.is_set():
            if self.shutdown_event.is_set() or self.controller.failed or self.strategy.stop_requested:
                return "interrupted"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            if self.success is not None:
                try:
                    if self.success.poll() is not None:
                        return "success"
                except Exception:
                    logger.exception("Success classifier failed")
                    return "classifier_error"
            self.request_finished.wait(min(remaining, 0.02))
        return "interrupted"

    def run_episode(self, request: EpisodeRequest) -> dict:
        """Run in an HTTP worker thread, blocking until completion or interruption."""
        if not self.episode_lock.acquire(blocking=False):
            raise HTTPException(409, "An episode is already running")
        try:
            controller = self.controller
            if self.shutdown_event.is_set() or controller.stopped or controller.failed:
                raise HTTPException(503, "Rollout session is unavailable")
            self.request_finished.clear()
            self.segment_started.clear()
            self.terminal_event = None
            if self.success is not None:
                self.success.begin()
            controller.set_task(request.task)
            if not controller.start():
                raise HTTPException(503, "Could not start the episode")
            # Wait until the controller has reset the preceding run's stop state.
            self.segment_started.wait()
            outcome = self._wait_for_decision(request.timeout_s)
            if self.success is not None:
                self.success.end()
            if outcome != "interrupted" and not (
                self.shutdown_event.is_set() or controller.failed or self.strategy.stop_requested
            ):
                # Queue the return move, then wait for its result and correction cleanup.
                controller.reset()
            self.request_finished.wait()
            if (
                self.shutdown_event.is_set()
                or controller.failed
                or controller.stopped
                or self.strategy.stop_requested
                or outcome == "interrupted"
            ):
                raise HTTPException(503, "Episode interrupted before completion")
            if self.terminal_event is not RolloutEvent.RESET_DONE:
                raise HTTPException(503, "Could not return to the initial position")
            if outcome == "classifier_error":
                raise HTTPException(503, "Success classifier failed")
            return {"status": "done", "outcome": outcome}
        finally:
            if self.success is not None:
                self.success.end()
            self.episode_lock.release()


def create_app(
    cfg: RolloutConfig, shutdown_event: Event | None = None, *, success: SuccessDetectionConfig | None = None
) -> FastAPI:
    if not isinstance(cfg.strategy, DAggerStrategyConfig) or cfg.strategy.record_autonomous:
        raise ValueError("The API requires corrections-only DAgger")
    if cfg.interactive:
        raise ValueError("The API requires interactive=false")

    # Disable strategy-owned limits for API sessions: the server decides when to
    # end execution and return home. Standalone rollouts still honour cfg.duration.
    cfg.strategy.num_episodes = 2**63 - 1
    cfg.duration = 0.0
    cfg.return_to_initial_position = True
    session: _RolloutSession | None = None
    if shutdown_event is None:
        shutdown_event = Event()

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal session
        with ExitStack() as cleanup:
            monitor = SuccessMonitor(success) if success is not None else None
            if monitor is not None:
                cleanup.callback(monitor.close)
            if cfg.display_data:
                init_visualization(
                    cfg.display_mode, session_name="rollout", ip=cfg.display_ip, port=cfg.display_port
                )
                cleanup.callback(shutdown_visualization, cfg.display_mode)
            ctx = build_rollout_context(cfg, LinkedEvent(shutdown_event))
            if monitor is not None:
                ctx.runtime.observation_observer = monitor.observe
            strategy = create_strategy(cfg.strategy)
            cleanup.callback(strategy.teardown, ctx)
            thread = None
            try:
                strategy.setup(ctx)
                ctx.policy.inference.pause()
                session = _RolloutSession(ctx, strategy, shutdown_event, monitor)
                thread = Thread(target=session.controller.serve, name="rollout-controller")
                thread.start()
                yield
            finally:
                if session is not None:
                    session.controller.stop()
                if thread is not None:
                    await asyncio.to_thread(thread.join)

    app = FastAPI(lifespan=lifespan)

    @app.post("/episode")
    def episode(request: EpisodeRequest):
        if session is None:
            raise HTTPException(503, "Rollout session is unavailable")
        return session.run_episode(request)

    return app
