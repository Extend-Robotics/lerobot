"""Configuration for one BrainCo Revo2 hand using Unitree's DDS driver."""

import math
from dataclasses import dataclass
from typing import Literal

from ..config import RobotConfig


@RobotConfig.register_subclass("brainco_revo2")
@dataclass
class BraincoRevo2HandConfig(RobotConfig):
    side: Literal["left", "right"] = "left"
    transport: Literal["zmq", "dds"] = "zmq"
    robot_ip: str = "192.168.123.164"
    # Defaults: left command/state 6002/6003; right command/state 6004/6005.
    command_port: int | None = None
    state_port: int | None = None
    domain_id: int = 0
    network_interface: str | None = None
    read_only: bool = False
    connect_timeout_s: float = 5.0
    state_timeout_s: float = 0.5
    # Unitree driver positions are normalized: 0 = open, 1 = closed.
    # Neutral names avoid the inconsistent thumb labels in xr_teleoperate.
    motor_names: tuple[str, ...] = ("motor_0", "motor_1", "index", "middle", "ring", "pinky")
    speed: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.side not in ("left", "right") or self.transport not in ("zmq", "dds"):
            raise ValueError("Expected left/right side and zmq/dds transport")
        if len(self.motor_names) != 6 or len(set(self.motor_names)) != 6:
            raise ValueError("Revo2 requires six unique motor names in driver order")
        if any(not name or "." in name for name in self.motor_names):
            raise ValueError("Motor names must be nonempty and contain no dots")
        for value in (self.connect_timeout_s, self.state_timeout_s, self.speed):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timeouts and speed must be finite and positive")
        if self.speed > 1:
            raise ValueError("Normalized speed must be <= 1")
        for port in (self.command_port, self.state_port):
            if port is not None and not 0 < port < 65536:
                raise ValueError("Ports must be between 1 and 65535")

    @property
    def ports(self) -> tuple[int, int]:
        base = 6002 if self.side == "left" else 6004
        return self.command_port or base, self.state_port or base + 1
