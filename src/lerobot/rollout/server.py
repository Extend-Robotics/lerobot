"""A single-robot API for corrections-only DAgger.

Each request runs until its timeout, returns the robot to its initial position,
and then responds. Interruption or an unsuccessful return produces HTTP 503.
The timeout limits rollout duration; the request also waits for the return move.
"""

import asyncio
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
        self, cfg: RolloutConfig, ctx: RolloutContext, strategy: DAggerStrategy, shutdown_event: Event
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.shutdown_event = shutdown_event
        self.episode_lock = Lock()
        self.request_finished = Event()
        self.terminal_event: RolloutEvent | None = None
        self.controller = RolloutController(strategy, ctx, on_event=self.on_controller_event)

    def on_controller_event(self, event: RolloutEvent, _payload) -> None:
        """Run on the controller thread; successful rollout still needs a reset."""
        match event:
            case RolloutEvent.SEGMENT_ENDED:
                if (
                    self.strategy.timed_out
                    and not self.shutdown_event.is_set()
                    and not self.controller.failed
                ):
                    # Queue the return move and keep the request waiting for its result.
                    # reset() returns task-restoration status, not movement success.
                    self.controller.reset()
                    return
                # An interrupted segment finishes the request without returning home.
            case (
                RolloutEvent.RESET_DONE
                | RolloutEvent.RESET_FAILED
                | RolloutEvent.RESET_SKIPPED
                | RolloutEvent.STOPPED
            ):
                pass
            case _:
                return
        self.terminal_event = event
        self.request_finished.set()

    def run_episode(self, request: EpisodeRequest) -> None:
        """Run in an HTTP worker thread, blocking until completion or interruption."""
        if not self.episode_lock.acquire(blocking=False):
            raise HTTPException(409, "An episode is already running")
        try:
            controller = self.controller
            if self.shutdown_event.is_set() or controller.stopped or controller.failed:
                raise HTTPException(503, "Rollout session is unavailable")
            self.request_finished.clear()
            self.terminal_event = None
            self.cfg.duration = request.timeout_s
            controller.set_task(request.task)
            if not controller.start():
                raise HTTPException(503, "Could not start the episode")
            self.request_finished.wait()
            if (
                self.shutdown_event.is_set()
                or controller.failed
                or controller.stopped
                or not self.strategy.timed_out
            ):
                raise HTTPException(503, "Episode interrupted before completion")
            if self.terminal_event is not RolloutEvent.RESET_DONE:
                raise HTTPException(503, "Could not return to the initial position")
        finally:
            self.episode_lock.release()


def create_app(cfg: RolloutConfig, shutdown_event: Event | None = None) -> FastAPI:
    if not isinstance(cfg.strategy, DAggerStrategyConfig) or cfg.strategy.record_autonomous:
        raise ValueError("The API requires corrections-only DAgger")
    if cfg.interactive:
        raise ValueError("The API requires interactive=false")

    # Override the supplied config: effectively disable the correction-count limit
    # so each request's timeout ends the rollout. API success also requires returning home.
    cfg.strategy.num_episodes = 2**63 - 1
    cfg.return_to_initial_position = True
    session: _RolloutSession | None = None
    if shutdown_event is None:
        shutdown_event = Event()

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal session
        with ExitStack() as cleanup:
            if cfg.display_data:
                init_visualization(
                    cfg.display_mode, session_name="rollout", ip=cfg.display_ip, port=cfg.display_port
                )
                cleanup.callback(shutdown_visualization, cfg.display_mode)
            ctx = build_rollout_context(cfg, LinkedEvent(shutdown_event))
            strategy = create_strategy(cfg.strategy)
            cleanup.callback(strategy.teardown, ctx)
            thread = None
            try:
                strategy.setup(ctx)
                ctx.policy.inference.pause()
                session = _RolloutSession(cfg, ctx, strategy, shutdown_event)
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
        session.run_episode(request)
        return {"status": "done"}

    return app
