"""A single-robot API: run a corrections-only DAgger episode until its timeout."""

import asyncio
from contextlib import asynccontextmanager
from threading import Event, Lock, Thread
from typing import Annotated

from lerobot.utils.import_utils import require_package

require_package("fastapi", "rollout-server")
require_package("uvicorn", "rollout-server")
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field, StringConstraints  # noqa: E402

from .configs import DAggerStrategyConfig, RolloutConfig  # noqa: E402
from .context import build_rollout_context  # noqa: E402
from .controller import LinkedEvent, RolloutController, RolloutEvent  # noqa: E402
from .strategies import create_strategy  # noqa: E402


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


def create_app(cfg: RolloutConfig, shutdown_event: Event | None = None) -> FastAPI:
    if not isinstance(cfg.strategy, DAggerStrategyConfig) or cfg.strategy.record_autonomous:
        raise ValueError("The API requires corrections-only DAgger")
    if cfg.interactive or cfg.display_data:
        raise ValueError("The API requires interactive=false and display_data=false")

    # HTTP timeout owns the segment length, not the number of pilot corrections.
    cfg.strategy.num_episodes = 2**63 - 1
    cfg.return_to_initial_position = False
    finished = Event()
    episode_lock = Lock()
    controller = None
    strategy = None
    if shutdown_event is None:
        shutdown_event = Event()

    def on_event(event, _payload):
        if event in (RolloutEvent.SEGMENT_ENDED, RolloutEvent.STOPPED):
            finished.set()

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal controller, strategy
        ctx = build_rollout_context(cfg, LinkedEvent(shutdown_event))
        strategy = create_strategy(cfg.strategy)
        thread = None
        try:
            strategy.setup(ctx)
            ctx.policy.inference.pause()
            controller = RolloutController(strategy, ctx, on_event=on_event)
            thread = Thread(target=controller.serve, name="rollout-controller")
            thread.start()
            yield
        finally:
            if controller is not None:
                controller.stop()
            if thread is not None:
                await asyncio.to_thread(thread.join)
            strategy.teardown(ctx)

    app = FastAPI(lifespan=lifespan)

    @app.post("/episode")
    def episode(request: EpisodeRequest):
        if not episode_lock.acquire(blocking=False):
            raise HTTPException(409, "An episode is already running")
        try:
            if shutdown_event.is_set() or controller is None or controller.stopped or controller.failed:
                raise HTTPException(503, "Rollout session is unavailable")
            finished.clear()
            cfg.duration = request.timeout_s
            controller.set_task(request.task)
            if not controller.start():
                raise HTTPException(503, "Could not start the episode")
            finished.wait()
            if shutdown_event.is_set() or controller.failed or controller.stopped or not strategy.timed_out:
                raise HTTPException(503, "Episode interrupted before completion")
            return {"status": "done"}
        finally:
            episode_lock.release()

    return app
