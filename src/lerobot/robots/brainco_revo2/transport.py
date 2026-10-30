"""Independent per-hand transports; no body commands or mode changes."""

import math
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import zmq

from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from .config_brainco_revo2 import BraincoRevo2HandConfig

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


_DDS_SETUP_LOCK = threading.RLock()


def initialize_dds(config: BraincoRevo2HandConfig) -> None:
    require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
    with _DDS_SETUP_LOCK:
        ChannelFactoryInitialize(config.domain_id, config.network_interface)


def positions(values: list[float]) -> list[float]:
    """Reject malformed feedback rather than inventing or clipping measurements."""
    # if len(values) != 6 or not all(math.isfinite(q) and 0 <= q <= 1 for q in values):
    #     raise ValueError("Expected six finite normalized Revo2 positions in [0, 1]")
    return values


class DdsHandTransport:
    def __init__(self, config: BraincoRevo2HandConfig):
        self.config = config
        self._lock = threading.Lock()
        self._sample: tuple[list[float], float] | None = None
        self.subscriber: Any = None
        self.publisher: Any = None

    def connect(self) -> None:
        # CycloneDDS lazily builds shared IDL/XTypes metadata during channel creation.
        # Serialize setup across hands so neither sees partially populated types.
        with _DDS_SETUP_LOCK:
            # SDK initialization is process-global and idempotent. All DDS devices in a
            # process must use the same domain/interface; initialize it before connecting.
            initialize_dds(self.config)
            topic = f"rt/brainco/{self.config.side}"
            self.subscriber = ChannelSubscriber(f"{topic}/state", MotorStates_)
            self.subscriber.Init(self._receive, 1)
            if not self.config.read_only:
                self.publisher = ChannelPublisher(f"{topic}/cmd", MotorCmds_)
                self.publisher.Init()

    def _receive(self, msg: Any) -> None:
        try:
            q = positions([float(state.q) for state in msg.states])
        except (AttributeError, TypeError, ValueError):
            return
        with self._lock:
            self._sample = q, time.monotonic()

    def read(self) -> tuple[list[float], float] | None:
        with self._lock:
            return self._sample

    def write(self, q: list[float]) -> None:
        if self.publisher is None:
            raise PermissionError("Hand command publisher is disabled")
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        for cmd, target in zip(msg.cmds, q, strict=True):
            cmd.q, cmd.dq = target, self.config.speed
        if self.publisher.Write(msg) is False:
            raise ConnectionError("Revo2 DDS command write failed")

    def close(self) -> None:
        for channel in (self.subscriber, self.publisher):
            if channel is not None:
                channel.Close()
        self.subscriber = self.publisher = None
        with self._lock:
            self._sample = None


class ZmqHandTransport:
    def __init__(self, config: BraincoRevo2HandConfig):
        self.config = config
        # Composite G1 feedback is read by both IK and the arm publisher.
        self._read_lock = threading.Lock()
        self.subscriber: zmq.Socket | None = None
        self.publisher: zmq.Socket | None = None
        self._sample: tuple[list[float], float] | None = None

    def connect(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        command_port, state_port = self.config.ports
        self.subscriber = ctx.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.LINGER, 0)
        self.subscriber.setsockopt(zmq.CONFLATE, 1)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self.subscriber.connect(f"tcp://{self.config.robot_ip}:{state_port}")
        if not self.config.read_only:
            self.publisher = ctx.socket(zmq.PUSH)
            self.publisher.setsockopt(zmq.LINGER, 0)
            self.publisher.setsockopt(zmq.IMMEDIATE, 1)
            self.publisher.setsockopt(zmq.SNDTIMEO, 100)
            self.publisher.setsockopt(zmq.SNDHWM, 1)
            self.publisher.connect(f"tcp://{self.config.robot_ip}:{command_port}")

    def read(self) -> tuple[list[float], float] | None:
        with self._read_lock:
            return self._read_locked()

    def _read_locked(self) -> tuple[list[float], float] | None:
        import zmq

        if self.subscriber is None:
            return None
        try:
            data = self.subscriber.recv_json(flags=zmq.NOBLOCK)
        except (zmq.Again, ValueError):
            return self._sample
        try:
            age = float(data["age_s"])
            observed_at = float(data["observed_at"])
            if not math.isfinite(observed_at) or observed_at > time.time() + self.config.state_timeout_s:
                return self._sample
            if data["side"] != self.config.side or not math.isfinite(age) or age < 0:
                return self._sample
            q = positions([float(q) for q in data["positions"]])
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._sample
        age = max(age, time.time() - observed_at)
        self._sample = q, time.monotonic() - age
        return self._sample

    def write(self, q: list[float]) -> None:
        if self.publisher is None:
            raise PermissionError("Hand command publisher is disabled")
        # Epoch deadline prevents queued commands being replayed after reconnection.
        self.publisher.send_json(
            {
                "positions": q,
                "speed": self.config.speed,
                "expires_at": time.time() + self.config.state_timeout_s,
            }
        )

    def close(self) -> None:
        for socket in (self.subscriber, self.publisher):
            if socket is not None:
                socket.close(linger=0)
        self.subscriber = self.publisher = None
        self._sample = None
