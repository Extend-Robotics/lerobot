# Pico + G1-D: robot servers and host teleoperation

This guide uses this checkout's `lerobot-teleoperate` integration: Pico headset/controllers → host XR mapping and arm IK → robot body/hand bridges. Camera images return to the host over ZMQ.

Document path:
`/home/oleg/extend_lerobot/lerobot/examples/unitree_g1d/PICO_G1D_TELEOPERATE_SETUP.md`

## 1. Machines, paths, and prerequisites

| Machine | What runs there | Checkout/environment |
| --- | --- | --- |
| Robot computer | Body/arm bridge, Revo2 hand bridge, camera publisher; existing BrainCo hardware driver | Documented robot checkout: `/home/unitree/Documents/lerobot_unitree/lerobot`; Conda environment `lerobot` |
| Host workstation | Pico PC service, `lerobot-teleoperate`, IK, optional image display | `/home/oleg/extend_lerobot/lerobot`; environment `/home/oleg/extend_lerobot/envs/unitree-g1` |
| Pico headset | XRoboToolkit app, headset and controller tracking | Connects to the host PC service |

The robot paths and Conda name above come from the existing robot README; verify them on your robot. The host paths match this workspace. The working setup uses robot IP `192.168.123.164`, host USB Ethernet address `192.168.123.10/24`, and headset-to-host USB forwarding. The checked-in configuration uses this robot IP for the body and both hands.

Both computers need the updated checkout installed in their selected Python environments, including ZMQ/OpenCV and Unitree SDK2 Python with CycloneDDS. The host additionally needs Pinocchio with CasADi support and `xensevr_pc_service_sdk`. Use the existing Unitree environment; do not replace it with a fresh base LeRobot environment that lacks those native dependencies.

### Host network: USB Ethernet to the robot

Connect the robot Ethernet cable to the host USB Ethernet adapter, currently `enx3c18a0d529cb`. Keep the headset on its separate USB cable. The existing `unitree-g1-ethernet` NetworkManager profile uses static host address `192.168.123.10/24`, with no gateway or DNS; Wi-Fi remains the default route for internet access.

To create this profile on a new host (replace the interface name if necessary):

```bash
sudo nmcli connection add type ethernet ifname enx3c18a0d529cb \
  con-name unitree-g1-ethernet \
  ipv4.method manual ipv4.addresses 192.168.123.10/24 \
  ipv4.never-default yes ipv6.method disabled \
  connection.autoconnect yes connection.autoconnect-priority 50
sudo nmcli connection up unitree-g1-ethernet
```

If the profile already exists, update it instead of adding a duplicate:

```bash
sudo nmcli connection modify unitree-g1-ethernet \
  connection.interface-name enx3c18a0d529cb \
  ipv4.method manual ipv4.addresses 192.168.123.10/24 \
  ipv4.gateway "" ipv4.dns "" ipv4.never-default yes ipv6.method disabled
sudo nmcli connection up unitree-g1-ethernet
ip route get 192.168.123.164
ping -c 30 192.168.123.164
```

The route should use `enx3c18a0d529cb` with source `192.168.123.10`. Check for continuous replies without packet loss before starting teleoperation. The built-in `eno1` repeatedly lost its physical link during diagnosis; the current setup uses the USB adapter instead.

The host must reach the robot's TCP ports:

| Service | Command port | State/image port |
| --- | --- | --- |
| Body/arms | 6000 | 6001 |
| Left Revo2 hand | 6002 | 6003 |
| Right Revo2 hand | 6004 | 6005 |
| Head camera | — | 5555 |

Keep robot and host clocks synchronized through your normal NTP/time-sync service. Hand feedback age and command expiry use wall-clock timestamps. On each machine, `timedatectl status` can help check synchronization.

**Before starting the body bridge:** put the G1-D in its supported operating setup with the workspace clear and the physical stop available. The body bridge calls `MotionSwitcherClient.ReleaseMode()` during startup. Starting it is not a read-only operation. Run only one body command source. The configuration below disables leg commands, holds waist targets, and controls both arms; it is not a walking controller.

## 2. Robot terminal A: start the body/arm server

Open an SSH terminal from the host:

```bash
ssh unitree@192.168.123.164
```

On the robot:

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/unitree/Documents/lerobot_unitree/lerobot
python -m lerobot.robots.unitree_g1.run_g1_server
```

The robot commands use the already-activated Conda Python directly; they do not require `uv` on the robot. Keep this terminal running. Expected message:

```text
bridge running (lowstate -> zmq, lowcmd -> dds)
```

This server bridges body feedback and arm commands on ports 6001/6000. It does not start cameras unless `--camera` is passed. It initializes DDS domain 0 with automatic interface selection; this script has no `--network-interface` argument. If DDS selects the wrong interface, correct the robot's DDS/network configuration rather than passing an unsupported flag.

## 3. Robot terminal B: start hands

These instructions assume **BrainCo Revo2 hands**. The bridge is not a raw serial/CAN driver: the robot's BrainCo hardware driver must already be running and exposing:

- `rt/brainco/left/state` and `rt/brainco/left/cmd`
- `rt/brainco/right/state` and `rt/brainco/right/cmd`

This repository does not include the hardware driver's installation-specific startup command or service name. Start that driver using your robot's installed vendor setup before proceeding. The commands below start the LeRobot DDS/ZMQ bridge.

In a second SSH terminal, on the robot:

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/unitree/Documents/lerobot_unitree/lerobot
python -m lerobot.robots.brainco_revo2.bridge --sides left right
```

Keep it running. Expect feedback-connected messages for both hands. If necessary, add `--network-interface eth0`, replacing `eth0` with the interface carrying the hand driver's DDS traffic. To run one hand only, use `--sides left` or `--sides right` and configure only that hand on the host.

Do not use `--read-only` for execution: that option opens only feedback ports. For initial driver diagnosis, stop the hand bridge and read DDS feedback directly:

```bash
python /home/unitree/Documents/lerobot_unitree/lerobot/examples/unitree_g1d/read_hand_state.py \
  --transport dds --side left
# Ctrl+C, then check the other hand:
python /home/unitree/Documents/lerobot_unitree/lerobot/examples/unitree_g1d/read_hand_state.py \
  --transport dds --side right
```

## 4. Optional robot terminal C: start the head camera

Skip this section for the default configuration, which has no camera. For images, start the publisher and add the camera to a local configuration in section 6. In a third SSH terminal, on the robot:

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/unitree/Documents/lerobot_unitree/lerobot
lerobot-find-cameras opencv
```

Identify the actual head-camera device. Device `4` below is the existing example default, not guaranteed to be the head camera:

```bash
python -m lerobot.cameras.zmq.image_server \
  --device 4 --width 640 --height 480 --fps 30 --fourcc MJPG --port 5555
```

Keep it running. A stable device path can replace `4`, for example `--device /dev/v4l/by-id/ACTUAL_CAMERA_DEVICE`. Use a resolution, frame rate, and capture format supported by the camera. The published stream is named `head_camera` and contains RGB images, not depth. This requires an OpenCV/V4L2-accessible camera; a vendor-only video stream needs a separate capture adapter.

Alternatively, combine arms and camera in terminal A:

```bash
python -m lerobot.robots.unitree_g1.run_g1_server \
  --camera --camera-device 4 --camera-width 640 --camera-height 480 \
  --camera-fps 30 --camera-port 5555
```

Choose either the separate camera publisher or the combined server. Do not run both on the same device/port. Additional cameras require matching publishers and entries in the host's `robot.cameras` dictionary; this guide configures the supported head-camera example.

## 5. Host terminal A: start the Pico service

On the host workstation:

```bash
source /home/oleg/extend_lerobot/activate_unitree_g1.sh
cd /home/oleg/extend_lerobot/lerobot
bash /home/oleg/extend_lerobot/lerobot/scripts/pico4/start_service.sh
```

Keep this terminal running, with only one Pico PC service instance. If the service or Python binding has not been installed, run this once first:

```bash
PICO_PYTHON=/home/oleg/extend_lerobot/envs/unitree-g1/bin/python \
  bash /home/oleg/extend_lerobot/lerobot/scripts/pico4/install_sdk.sh
```

Use USB for the current headset-to-host setup; Wi-Fi is an alternative described below. USB carries the same XRoboToolkit TCP stream through ADB reverse forwarding. It avoids Wi-Fi latency and jitter on this link; its latency still needs to be measured with the stream running. The host-to-robot bridges and camera continue using their configured network addresses.

### 5.1. USB connection

Use a USB data cable between the headset and host. Plugging in the cable alone does not move an existing Wi-Fi stream onto USB.

1. Enable **USB debugging** in the headset's developer settings. If developer settings are hidden, open **Settings → About this device** and tap the software version repeatedly until developer mode is enabled.
2. Connect the cable, put on the headset, and accept **Allow USB debugging** for this host.
3. Keep the Pico PC service from the command above running. In a second host terminal, check ADB and create the tunnel:

```bash
# ADB already available on this workstation:
export PICO_ADB=/home/oleg/Documents/pico_teleop/tools/platform-tools/adb
"$PICO_ADB" devices -l
# Expected: a headset serial followed by "device", not "unauthorized" or "offline".
"$PICO_ADB" reverse tcp:63901 tcp:63901
"$PICO_ADB" reverse --list
```

On another workstation, set `PICO_ADB` to its installed Android platform-tools `adb` executable. With multiple ADB devices attached, add `-s HEADSET_SERIAL` immediately after `"$PICO_ADB"` in each command.

The reverse list must contain:

```text
UsbFfs tcp:63901 tcp:63901
```

The first column can vary; both `tcp:63901` entries must be present. This maps the headset's local TCP port 63901 to the host PC service's port 63901.

4. Open the installed **XRoboToolkit** app (the repository setup uses XRoboToolkit-PICO 1.1.1). Set **PC Service** to exactly **`127.0.0.1`**, without `http://` or a port suffix. Confirm the address entry, then press **Connect** and check for **WORKING**.
5. Enable **Head** and both **Controller** tracking, then **Data & Control → Send**. Wake both controllers and keep the headset app in the foreground. Headset pose is required for this G1 XR mapping; body tracking and headset camera streaming are unnecessary.

**After unplugging/replugging USB, rebooting the headset, or restarting ADB, check `devices -l` and recreate the reverse mapping before pressing Connect again.** A previous successful command does not establish that the tunnel is still present. Repeated USB disconnects were observed during diagnosis and cleared this mapping; if they recur, check the cable and host USB port.

To verify the host service is listening:

```bash
ss -ltnp '( sport = :63901 or sport = :60061 )'
```

Expect port 63901 for the headset stream and local port 60061 for the Python SDK. An ADB device connection, a listening PC service, and the app's WORKING status are separate checks; controller samples must also reach the SDK.

For a bounded stream timing check without connecting to a robot:

```bash
uv run --no-project --python /home/oleg/extend_lerobot/envs/unitree-g1/bin/python \
  python /home/oleg/extend_lerobot/lerobot/examples/pico4/measure_stream.py \
  --seconds 60 --output /tmp/pico-usb-stream.csv
```

Keep the headset awake and move both controllers during the measurement. The script saves polling data to CSV and a summary to `/tmp/pico-usb-stream.json`. Check that `updates` increases and inspect `max_observed_age_s`, update gaps, and polling delays. Zero updates means no advancing source timestamp reached this SDK reader. This diagnostic never creates a robot object; stop other SDK readers before running it.

### 5.2. Wi-Fi connection

Connect the headset and host to the same reachable LAN. In XRoboToolkit, set **PC Service** to the **host PC's LAN IP**, not `127.0.0.1` or the robot IP. Use `hostname -I` on the host to inspect its addresses. Press **Connect**, enable headset and both controller tracking, then **Data & Control → Send**. Wake both controllers and keep tracking active.

To switch from Wi-Fi to USB, disconnect the app session, establish the ADB reverse mapping, change PC Service to `127.0.0.1`, and reconnect. To switch back, disconnect and restore the host's reachable LAN IP.

## 6. Host terminal B: prepare an explicit local configuration

The base JSON is ready for the current setup: arms and both Revo2 hands at `192.168.123.164`, hand motor speed `0.5`, no camera, and PICO `input_timeout_s=0.5`. It can be used directly as the config path in section 8.

For optional head-camera recording, the following creates a separate runtime JSON with the robot/camera IP and both hands at motor speed `0.3`. Start the camera publisher first. This preserves the checked-in base configuration.

```bash
source /home/oleg/extend_lerobot/activate_unitree_g1.sh
cd /home/oleg/extend_lerobot/lerobot
export G1D_ROBOT_IP=192.168.123.164
export G1D_CAMERA_IP="$G1D_ROBOT_IP"

uv run --no-project --python /home/oleg/extend_lerobot/envs/unitree-g1/bin/python python - <<'PY'
import json
import os
from pathlib import Path

source = Path('/home/oleg/extend_lerobot/lerobot/configs/unitree_g1_pico.json')
output = Path('/home/oleg/extend_lerobot/g1d_pico_local.json')
config = json.loads(source.read_text())
robot = config['robot']
robot['robot_ip'] = os.environ['G1D_ROBOT_IP']
robot['cameras'] = {
    'head_camera': {
        'type': 'zmq',
        'server_address': os.environ['G1D_CAMERA_IP'],
        'port': 5555, 'camera_name': 'head_camera',
        'width': 640, 'height': 480, 'fps': 30, 'color_mode': 'rgb',
    },
}
for side in ('left', 'right'):
    robot[f'{side}_hand'] = {
        'side': side,
        'transport': 'zmq',
        'robot_ip': os.environ['G1D_ROBOT_IP'],
        'speed': 0.3,
    }
output.write_text(json.dumps(config, indent=2) + '\n')
print(output)
PY
```

The nested hands use a concrete hand-config type, so these dictionaries do not need a `type` field. `UnitreeG1` assigns each hand its side and the body's `robot_ip` when constructing it. Camera addresses remain independent: changing `--robot.robot_ip` alone does not redirect the camera.

For arms only, omit/remove `left_hand` and `right_hand` from the generated JSON. For operation without a camera, use `--robot.cameras='{}'` on the teleoperation command. To change camera settings through the CLI, replace the whole `--robot.cameras` dictionary; a flag such as `--robot.cameras.head_camera.server_address` is not supported.

## 7. Check feedback and tracking before execution

In host terminal B, with the environment activated, optionally inspect raw Pico input:

```bash
uv run --no-project --python /home/oleg/extend_lerobot/envs/unitree-g1/bin/python \
  python /home/oleg/extend_lerobot/lerobot/examples/pico4/raw_controllers.py
```

Stop this with Ctrl+C before running any other Pico client. The SDK connection is process-global; use one client process at a time.

Preview the arm/hand mapping without sending joint commands:

```bash
bash /home/oleg/extend_lerobot/lerobot/scripts/pico4/teleop_g1d.sh \
  --robot-ip 192.168.123.164 --hands both --duration 10
```

Preview reads robot feedback and prints IK targets; it does not simulate robot motion, exercise the camera, or apply execution-time arm target clipping. The already-started body bridge still has the mode-switching behavior described above.

If using the optional camera configuration, check the camera independently:

```bash
uv run --no-project --python /home/oleg/extend_lerobot/envs/unitree-g1/bin/python python - <<'PY'
import json
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

with open('/home/oleg/extend_lerobot/g1d_pico_local.json') as f:
    settings = json.load(f)['robot']['cameras']['head_camera']
settings.pop('type')
with ZMQCamera(ZMQCameraConfig(**settings)) as camera:
    for _ in range(30):
        frame = camera.async_read(timeout_ms=1000)
        print(frame.shape, frame.dtype)
PY
```

Expect `(480, 640, 3) uint8`. Stop all diagnostic clients before executing teleoperation.

## 8. Run lerobot-teleoperate on the host

With the body and hand services running and the headset/controllers tracked over USB, execute in host terminal B. For the optional camera setup, replace the config path below with `/home/oleg/extend_lerobot/g1d_pico_local.json`:

```bash
source /home/oleg/extend_lerobot/activate_unitree_g1.sh
cd /home/oleg/extend_lerobot/lerobot
/home/oleg/extend_lerobot/envs/unitree-g1/bin/lerobot-teleoperate \
  --config_path=/home/oleg/extend_lerobot/lerobot/configs/unitree_g1_pico.json \
  --fps=30 --display_data=false
```

**This command starts sending commands as soon as connection/setup completes. There is no `r` prompt or two-second countdown in `lerobot-teleoperate`.** Have the operator ready before launching it. Set `--display_data=false` for a host without a graphical display; the configured camera is still consumed. `--teleop_time_s=30` optionally bounds the teleoperation loop to 30 seconds, but still executes physical motion.

Controls and behavior:

- Both arms follow absolute wrist targets relative to headset position/yaw. Grips do **not** clutch or freeze the arms in this XR mode.
- Each trigger commands all six joints of its corresponding hand: 0 opens, 1 closes.
- The processor maps Pico poses to 14 arm joint targets through IK. The robot limits the joint delta from measured positions to `arm_max_step=0.12` radians and hand changes to `hand_max_speed=1.0` normalized units/second by default.
- `speed=0.3` in the hand configs sets the hand-driver motor-speed parameter; it is separate from the host hand-position rate limit.
- Ctrl+C stops teleoperation. To restart, launch a new session; there is no supported pause/resume mode.

For an execution path with a manual start prompt instead of the standard CLI:

```bash
bash /home/oleg/extend_lerobot/lerobot/scripts/pico4/teleop_g1d.sh \
  --robot-ip 192.168.123.164 --hands both --execute
```

This alternative asks for `r` then Enter and waits two seconds. It does not load the local JSON or consume the camera. Never run it concurrently with `lerobot-teleoperate`.

## 9. Stop in order

1. Press Ctrl+C in the host teleoperation terminal while the bridges are still running, allowing disconnect/passivation commands to be sent.
2. Stop the Pico PC service if finished.
3. Stop the robot camera, hand bridge, and body bridge with Ctrl+C in their terminals.
4. Follow the robot's normal hardware shutdown procedure.

The checked-in JSON sets the PICO input timeout to 0.5 seconds. The standalone preview/execution example retains its class default of 0.25 seconds. Robot command expiry and processor sample-age checks have separate deadlines; increasing the PICO timeout does not increase those deadlines. A stale stream or publisher fault passivates arms/waist and latches a fault; fix the cause and reconnect before execution. This host watchdog does not guarantee a stop after a host process kill or network failure. Hand disconnect sends no open/close gesture, and the last hand target may remain active.

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| Ethernet ping drops / Destination Host Unreachable | Check USB adapter detection, cable/ports, and link events with `journalctl -k`; verify the route uses `enx3c18a0d529cb` and source `192.168.123.10`. Repeated carrier drops are a link problem even if the static IP is correct. |
| No body feedback / connection timeout | Body bridge running; correct robot IP; ports 6000/6001 reachable; robot-side DDS interface/domain and `rt/lowstate` available. |
| Hand bridge times out | BrainCo hardware driver running; correct side/interface; six normalized positions arriving on the corresponding DDS state topic. The bridge cannot replace the hardware driver. |
| Hand commands expire or feedback looks stale | Clock synchronization on both computers, network latency, and robot/host logs. |
| Hands do not move | Hands present in the local JSON; command bridge started without `--read-only`; triggers tracked; ports 6002–6005 reachable. |
| Camera timeout or wrong view | Camera IP in local JSON; actual V4L2 device; publisher on port 5555; `head_camera` stream name. |
| Camera shape mismatch | Match publisher width/height and JSON dimensions. For another resolution, update both. |
| Address already in use / camera busy | Duplicate service or combined `--camera` server running alongside the standalone publisher. |
| Pico SDK import/service error | Correct host environment, SDK installed into that environment, staged `.runtime` retained, exactly one PC service running. |
| USB headset missing / unauthorized / offline | Check the USB data cable and port, USB debugging, and the authorization prompt inside the headset. `"$PICO_ADB" devices -l` must show `device`. |
| XRoboToolkit connection error over USB | Confirm the service listens on 63901, `"$PICO_ADB" reverse --list` contains `tcp:63901 tcp:63901`, and PC Service is exactly `127.0.0.1`. Recreate the mapping after every USB/ADB reconnect. |
| USB tunnel verified but app still fails | Close and reopen XRoboToolkit, recheck the address, and retry Connect. Inspect app logs and the PC service terminal; Unity input exceptions were observed during diagnosis, but their relationship to connection failures is unconfirmed. |
| PICO controller stream is stale | The SDK source timestamp stopped advancing beyond `teleop.input_timeout_s` (default 0.25 s; inspect the actual runtime JSON). Check Send, tracking, USB tunnel or Wi-Fi, and PC service forwarding. Use the timing diagnostic in section 5.1; raising the timeout delays stale-input detection and does not restore missing samples. |
| Tracking/IK fault | Headset plus both controllers visible/tracked and data Send enabled; inspect logs, then reconnect. |
| Immediate movement was unexpected | Standard CLI has no arming prompt; use the preview first or the prompted execution example. |

## Source references and validation scope

Commands and field names were checked against this local checkout. During USB diagnosis, the headset was visible through ADB and a TCP probe through the reverse tunnel reached the host service; the operator subsequently confirmed the USB headset / USB Ethernet robot setup works. USB stream latency and physical robot motion were not independently validated in this documentation update. The USB workflow is also described in [XRoboToolkit USB connection instructions](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python/issues/28). No robot services were started and no hardware motion was tested while writing this document. The robot's installed vendor hand-driver startup command remains deployment-specific.

- `/home/oleg/extend_lerobot/lerobot/configs/unitree_g1_pico.json`
- `/home/oleg/extend_lerobot/lerobot/examples/unitree_g1d/TELEOP_PICO4.md`
- `/home/oleg/extend_lerobot/lerobot/examples/unitree_g1d/HEAD_CAMERA.md`
- `/home/oleg/extend_lerobot/lerobot/src/lerobot/robots/unitree_g1/run_g1_server.py`
- `/home/oleg/extend_lerobot/lerobot/src/lerobot/robots/brainco_revo2/bridge.py`
- `/home/oleg/extend_lerobot/lerobot/src/lerobot/scripts/lerobot_teleoperate.py`
