"""Hand-only DDS/ZMQ bridge. Does not create body publishers or switch robot modes.

Run on the robot: python -m lerobot.robots.brainco_revo2.bridge --read-only
"""

import argparse
import logging
import math
import threading
import time
from typing import Literal

from .brainco_revo2 import BraincoRevo2Hand
from .config_brainco_revo2 import BraincoRevo2HandConfig
from .transport import initialize_dds

logger = logging.getLogger(__name__)


def serve_hand(config: BraincoRevo2HandConfig, stop: threading.Event, bind: str = "0.0.0.0") -> None:
    import zmq

    hand = BraincoRevo2Hand(config)
    ctx = zmq.Context.instance()
    state_socket = ctx.socket(zmq.PUB)
    command_socket = None
    try:
        command_port, state_port = config.ports
        state_socket.setsockopt(zmq.LINGER, 0)
        state_socket.bind(f"tcp://{bind}:{state_port}")
        if not config.read_only:
            command_socket = ctx.socket(zmq.PULL)
            command_socket.setsockopt(zmq.LINGER, 0)
            command_socket.setsockopt(zmq.CONFLATE, 1)
            command_socket.bind(f"tcp://{bind}:{command_port}")
        hand.connect()
        logger.info("%s Revo2 feedback connected (read_only=%s)", config.side, config.read_only)
        while not stop.is_set():
            # Preserve the DDS sample age when forwarding cached feedback.
            sample = hand._transport.read()
            age = float("inf") if sample is None else time.monotonic() - sample[1]
            if sample is not None and age <= config.state_timeout_s:
                state_socket.send_json(
                    {
                        "side": config.side,
                        "positions": sample[0],
                        "age_s": age,
                        "observed_at": time.time() - age,
                    }
                )
            if command_socket is not None and command_socket.poll(0):
                try:
                    data = command_socket.recv_json()
                    expires = float(data["expires_at"])
                    remaining = expires - time.time()
                    if not math.isfinite(expires) or not 0 < remaining <= config.state_timeout_s:
                        raise ValueError("Expired command or unsynchronized client/server clocks")
                    speed = float(data["speed"])
                    if not math.isfinite(speed) or not 0 < speed <= 1:
                        raise ValueError("Speed must be normalized in (0, 1]")
                    hand.config.speed = speed
                    q = data["positions"]
                    hand.send_action(dict(zip(hand.action_features, q, strict=True)))
                except (KeyError, TypeError, ValueError, TimeoutError, PermissionError) as exc:
                    logger.warning("Rejected %s hand command: %s", config.side, exc)
            stop.wait(0.01)
    finally:
        hand.disconnect()
        state_socket.close(linger=0)
        if command_socket is not None:
            command_socket.close(linger=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument(
        "--read-only", action="store_true", help="Create state subscribers only; no command DDS publishers"
    )
    parser.add_argument("--network-interface")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    stop = threading.Event()
    # Initialize the process-global DDS factory before starting parallel hand readers.
    initialize_dds(
        BraincoRevo2HandConfig(
            transport="dds", domain_id=args.domain_id, network_interface=args.network_interface
        )
    )
    failures: list[Exception] = []

    def run(side: Literal["left", "right"]) -> None:
        try:
            serve_hand(
                BraincoRevo2HandConfig(
                    side=side,
                    transport="dds",
                    read_only=args.read_only,
                    domain_id=args.domain_id,
                    network_interface=args.network_interface,
                ),
                stop,
                args.bind,
            )
        except Exception as exc:
            failures.append(exc)
            logger.exception("%s hand bridge failed", side)
            stop.set()

    threads = [threading.Thread(target=run, args=(side,)) for side in dict.fromkeys(args.sides)]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            stop.wait(0.1)
    except KeyboardInterrupt:
        stop.set()
    finally:
        for thread in threads:
            thread.join()
    if failures:
        raise RuntimeError("Revo2 bridge failed") from failures[0]


if __name__ == "__main__":
    main()
