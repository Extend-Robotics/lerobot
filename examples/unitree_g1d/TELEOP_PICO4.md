# PICO controllers to G1 joint targets

PICO supplies raw controller/headset poses and triggers. A teleoperation processor
applies Unitree XR mapping and IK, producing 14 arm joint targets in radians plus
the configured hand joint targets in normalized [0, 1] units. The G1 robot accepts
joint commands only. Policy state is measured joint positions; policy actions
are joint targets. Cartesian poses and tracking metadata are not dataset features.

The pure mapping and IK retain the pinned upstream conventions documented in
[UNITREE_XR_NOTICES.md](UNITREE_XR_NOTICES.md). The three custom PICO session
classes have been replaced by the standard processor/robot path.

## Setup and preview

The default setup uses **headset USB → host** and **host USB Ethernet → robot**.
The robot is `192.168.123.164`; the host adapter `enx3c18a0d529cb` uses
`192.168.123.10/24` without a gateway. See the [complete setup guide](PICO_G1D_TELEOPERATE_SETUP.md)
for NetworkManager configuration, robot bridges, and USB troubleshooting.

Run the usual G1 body bridge on the robot and start the PICO service:

```bash
# Terminal A: leave the service running.
bash scripts/pico4/start_service.sh
# Terminal B: authorize USB debugging in the headset first.
export PICO_ADB=/home/oleg/Documents/pico_teleop/tools/platform-tools/adb
"$PICO_ADB" devices -l
"$PICO_ADB" reverse tcp:63901 tcp:63901
"$PICO_ADB" reverse --list
# In XRoboToolkit: PC Service = 127.0.0.1, Connect, then Send.
bash scripts/pico4/teleop_g1d.sh --robot-ip 192.168.123.164
```

Recreate the reverse mapping after USB/ADB reconnects. Enable headset and both controller tracking. Preview reads measured joints,
runs the same teleop processor, and prints unbounded IK joint targets without
publishing commands. It does not simulate robot motion.

The launcher uses `../envs/unitree-g1/bin/python`; override with
`PICO_G1_PYTHON=/path/to/python`. The environment needs the Unitree SDK, PICO
SDK, and Pinocchio with CasADi. Install PICO with
`PICO_PYTHON=/path/to/python bash scripts/pico4/install_sdk.sh`.

For execution:

```bash
bash scripts/pico4/teleop_g1d.sh --robot-ip 192.168.123.164 --execute
```

Type `r` then Enter; commands start after two seconds. Ctrl+C stops execution.
There is no supported pause/resume mode: stop and reconnect to begin again.
Each arm follows its absolute head-yaw-relative wrist target only while its controller grip is held. Releasing a grip holds that arm at its measured joint position; pressing again resumes the current absolute target without rebasing. Triggers continue controlling the hands independently.

## Standard scripts

The shared [configuration](../../configs/unitree_g1_pico.json) specifies the
same arm/waist gains as the preview launcher. These commands execute immediately
after connection; they do not use the example's terminal start prompt.

```bash
lerobot-teleoperate --config_path=configs/unitree_g1_pico.json \
  --robot.robot_ip=192.168.123.164 --fps=30

lerobot-record --config_path=configs/unitree_g1_pico.json \
  --robot.robot_ip=192.168.123.164 \
  --dataset.repo_id=<USER/DATASET> --dataset.single_task="<TASK>" \
  --dataset.fps=30 --dataset.num_episodes=1

lerobot-rollout --config_path=configs/unitree_g1_pico.json \
  --robot.robot_ip=192.168.123.164 --policy.path=<JOINT_POLICY> --fps=30
```

The shared configuration enables both Revo2 hands over ZMQ at `192.168.123.164`
with motor speed `0.5`, and PICO input timeout `0.5` seconds. Start the hand
bridge before using it. Cameras are disabled by default. For camera recording,
start its publisher and add `robot.cameras` as described in
[Head camera on a separate machine](HEAD_CAMERA.md). Camera addresses are
independent of `robot.robot_ip`; override the whole camera dictionary when
changing them. For arms without hands, remove `left_hand` and `right_hand`
from a local copy of the JSON.

For policy-only deployment, omit the teleop section from the configuration.
PICO is used only for human input/corrections; policy output bypasses XR mapping
and IK. Configure additional cameras in `robot.cameras` when collecting visual observations.
This repository's collection command is `lerobot-record`.

Arm-only mode advertises exactly 14 arm `.q` features and any configured hand
`.pos` features. The raw observation still contains diagnostic body/velocity/IMU
values, but these are not part of the arm-only policy state. Recording and
rollout retain the joint targets returned by `send_action()`, after command limits.

## Optional Revo2 hands

Run the independent hand bridge on the robot:

```bash
python -m lerobot.robots.brainco_revo2.bridge
bash scripts/pico4/teleop_g1d.sh --robot-ip 192.168.123.164 --hands both
```

Add `--execute` for movement. Use `--hands left` or `right` for one hand.
Each trigger sets all configured joints of its corresponding hand: 0 opens and
1 closes. Custom motor names are preserved. The launcher also accepts
`--hand-transport`, `--hand-network-interface`, `--hand-max-speed`, and
`--hand-motor-speed`. Standard scripts use nested `robot.left_hand` and
`robot.right_hand` configurations and `robot.hand_max_speed`.

## Shared hardware execution

The robot validates finite, complete joint actions and fresh measured feedback.
It scales the 14-arm-joint delta from measured positions to at most
`robot.arm_max_step` (default 0.12 radians). Hand movement is limited by
`robot.hand_max_speed` (default 1 normalized unit/second). A robot-owned publisher
repeats the accepted arm target at the requested 250 Hz; it never advances beyond
that recorded target. Gravity compensation uses measured joints.

This differs from the old example: target clipping happens when a command is
accepted, rather than advancing a hidden target on every publisher tick.
The same limits now apply to teleoperation and policy execution.

The shared JSON sets PICO input freshness to 0.5 seconds; the standalone launcher
uses the PICO class default of 0.25 seconds. Processor sample-age and robot command
deadlines are configured separately and are not extended by the PICO timeout.
A publisher fault or expired command stream passivates arms and waist and latches
a fault; disconnect/reconnect before further execution. Shutdown stops publishing
before sending the passive command. There is no five-second supported hold.
Hand disconnect sends no gesture; the driver's last hand target may remain active.

Episode saving, reset pauses, inference stalls, or interactive pauses longer than
the command deadline also stop the robot. Use a continuous session or reconnect
between such gaps; the one-episode recording example avoids claiming automatic
multi-episode rearming. This is a host watchdog, not a robot-side guarantee after
process kill or network loss.

## Mapping and validation

Raw PICO poses use xyzw quaternions. Mapping changes OpenXR basis into robot axes,
subtracts headset position, removes headset yaw, and adds [0.15, 0, 0.45] metres.
The processor uses the pinned Unitree XR objective and newest-first joint filter,
checks IK success, finite results, joint bounds, and sample age after solving.

```bash
uv run --no-sync pytest tests/robots/test_unitree_g1.py \
  tests/robots/test_pico_g1_teleop.py tests/robots/test_pico_hand_teleop.py \
  tests/robots/test_unitree_xr.py tests/teleoperators/test_pico4.py \
  tests/test_g1_script_integration.py -q
```

These tests use mocks and optional native IK. They do not move hardware.
