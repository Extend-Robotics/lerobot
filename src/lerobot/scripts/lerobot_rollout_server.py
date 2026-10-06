"""Serve POST /episode using the usual rollout hardware, policy, and dataset flags."""

from dataclasses import dataclass
from threading import Event

from lerobot.configs import parser
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.success import SuccessDetectionConfig
from lerobot.scripts import lerobot_rollout as _rollout_registrations  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.utils import init_logging


@dataclass
class RolloutServerConfig(RolloutConfig):
    success: SuccessDetectionConfig | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    return_to_initial_position: bool = False


@parser.wrap()
def serve(cfg: RolloutServerConfig):
    require_package("uvicorn", "rollout-server")
    import uvicorn

    from lerobot.rollout.server import RolloutServer, create_app

    init_logging()
    shutdown_event = Event()
    app = create_app(cfg, shutdown_event, success=cfg.success)
    RolloutServer(uvicorn.Config(app, host=cfg.host, port=cfg.port, workers=1), shutdown_event).run()


def main():
    register_third_party_plugins()
    serve()


if __name__ == "__main__":
    main()
