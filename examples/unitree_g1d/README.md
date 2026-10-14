# Read G1-D upper-body state

This is the read-only integration milestone. It subscribes directly to `rt/lowstate`
using Unitree SDK2 Python's HG `LowState_` message. It creates no command publisher,
changes no robot modes, and sends no commands on disconnect. `send_action()` raises
`NotImplementedError` and `action_features` is empty.

Install LeRobot and the official [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
in the same environment, following the SDK's CycloneDDS installation instructions.
Run on the robot or a machine with DDS connectivity to it:

```bash
uv run --no-sync python examples/unitree_g1d/read_state.py --network-interface eth0 --duration 10
```

Replace `eth0` with the robot-facing interface. Positions use radians (`.q`) and
velocities use radians/second (`.dq`), with the same joint names as LeRobot's G1.
The default schema covers 14 arm joints (slots 15–28). `--include-waist` also reads
waist yaw/pitch from slots 12/14; verify that your firmware supplies these fields.
Dedicated waist-pitch feedback topics are not implemented in this milestone.

The adapter follows G1's config/class layout and guarded SDK imports. It uses an SDK
subscriber callback instead of G1's polling thread; the SDK owns the reader thread.
State snapshots are copied under a lock, timestamped with a monotonic clock, and
rejected after `state_timeout_s`. Connection waits up to `connect_timeout_s` for a
valid sample. SDK domain/interface initialization is process-global; use a new
process when changing those settings.

```python
from lerobot.robots import make_robot_from_config
from lerobot.robots.unitree_g1d import UnitreeG1DConfig

robot = make_robot_from_config(UnitreeG1DConfig(network_interface="eth0"))
with robot:
    print(robot.get_observation())
```

The class uses the existing dynamic robot factory. Recording/teleoperation CLI
registration is outside this read-only milestone. Hardware validation is still
required; unit tests use mocked SDK subscriptions.
