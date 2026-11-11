# PICO controller teleoperation

Single (`pico4`) and dual-controller (`bi_pico4`) teleoperators ported from
XenseRobotics-AI/lerobot-xense commit
`e1e3bd3a39a77271844d2f6f76f5620d56aa66bc` (Apache-2.0).
Quaternion helpers retain the source attribution. Native binding sources come
from the same checkout; the vendor service/SDK binary is downloaded separately.

## Setup (Ubuntu x86_64)

From the repository root, with the normal LeRobot `.venv` already installed:

```bash
bash scripts/pico4/install_sdk.sh
bash scripts/pico4/start_service.sh
```

The installer builds `xensevr_pc_service_sdk` into `.venv`, using XenseVR PC
Service 0.2.1 extracted under `scripts/pico4/.runtime`. No sudo, conda, spdlog,
or other robot SDK is needed. Prerequisites: uv, curl, tar, dpkg-deb, a C++ compiler.
Use `PICO_PYTHON=/path/to/env/bin/python` to select another Python environment.
Build tools, pybind11 and nlohmann-json are installed/downloaded by the script.
Keep this checkout in place: the extension links to its staged vendor library.
`uv sync` may remove the separately installed SDK; rerun the installer afterwards.
Use `uv run --no-sync` for the commands below.

Only one PC service should run at a time. Stop a previous service before launching
this copy. The service stays in the foreground; Ctrl+C stops it.

On the headset, install XRoboToolkit-PICO-1.1.1.apk from:
https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1

For the current PICO + G1-D setup, use USB for headset input and USB Ethernet
for the robot (`192.168.123.164`, host `192.168.123.10/24`). See the
[complete G1-D setup guide](../unitree_g1d/PICO_G1D_TELEOPERATE_SETUP.md).

Enable USB debugging, connect a data cable, and accept the authorization prompt.
With the PC service running, create the headset-to-host tunnel:

```bash
export PICO_ADB=/home/oleg/Documents/pico_teleop/tools/platform-tools/adb
"$PICO_ADB" devices -l
"$PICO_ADB" reverse tcp:63901 tcp:63901
"$PICO_ADB" reverse --list
```

Set XRoboToolkit **PC Service** to **127.0.0.1**, press **Connect**, enable
headset/controller tracking and **Data & Control → Send**, and wake both
controllers. Recreate forwarding after USB/ADB reconnects. On other hosts,
set `PICO_ADB` to the installed platform-tools executable.

For Wi-Fi instead, put the headset and PC on the same reachable LAN and use
the host's LAN IP in the app. Camera and body tracking are unnecessary.

## Test without a robot

In a second terminal at the repository root:

```bash
uv run --no-sync python examples/pico4/raw_controllers.py
# Stop that test with Ctrl+C before starting the next one.
uv run --no-sync python examples/pico4/teleop_no_robot.py
# Optional bounded run:
uv run --no-sync python examples/pico4/teleop_no_robot.py --seconds 10
```

Hold each controller's side grip to move its target; release to hold the target.
Triggers update gripper fields. The example seeds virtual wrist poses and never
creates a robot object. It prints actions at 5 Hz, polling at 30 Hz.

## API and robot integration

```python
from lerobot.teleoperators.bi_pico4 import BiPico4, BiPico4Config
from lerobot.teleoperators.utils import make_teleoperator_from_config

teleop = make_teleoperator_from_config(BiPico4Config())
# Pose arrays: [x, y, z, qw, qx, qy, qz, gripper].
teleop.connect(left_tcp_pose_quat=left_pose, right_tcp_pose_quat=right_pose)
try:
    action = teleop.get_action()
finally:
    teleop.disconnect()
```

Single-controller API: `Pico4(Pico4Config())`, with
`connect(current_tcp_pose_quat=pose)`.

The teleoperate and record CLIs import the PICO configs for discovery. This does
not automatically make a Cartesian teleoperator compatible with a joint robot.
The output is `tcp.x/y/z`, `tcp.r1` through `tcp.r6` (the first two rotation-matrix
columns), and `gripper.pos`; bimanual fields have `left_` / `right_` prefixes.
Feature metadata uses this branch's flat `{field: float}` contract.

The stock CLI calls `connect()` without measured wrist poses. For physical robots,
use an integration loop/processor which seeds measured TCP poses and maps the
Cartesian outputs to the robot's action interface. G1 needs its existing arm IK
plus joint-name mapping; SO-101 likewise needs its Cartesian-to-joint processor.
The G1 processor integration is documented in
[the PICO-to-G1 guide](../unitree_g1d/TELEOP_PICO4.md). Default identity poses are for testing,
not calibrated physical robot start poses.

The SDK owns a process-global connection; use one Pico4 or BiPico4 instance per
process, not two independently connected single-controller instances.

## Tests

```bash
uv run --no-sync pytest tests/teleoperators/test_pico4.py -q
```

These mock the SDK and cover factory/schema compatibility, one shared SDK
connection, timeout cleanup and independent arm/clutch mapping.
