# Three G1-D cameras: robot to host and data collection

The shared [`configs/unitree_g1_pico.json`](../../configs/unitree_g1_pico.json)
now enables `head_camera`, `left_wrist_camera`, and `right_wrist_camera` at
`192.168.123.164:5555`, each RGB, 640×480, 30 FPS. All three must be publishing
before standard teleoperation or recording starts. These are stream names, not
USB device identities: map each physical camera on the robot first.

Images flow from OpenCV/V4L2 on the robot → JPEG/base64 JSON over ZMQ TCP 5555 →
three named ZMQ receivers on the host → LeRobot observations and dataset videos.
The host does not need local `/dev/video*` devices. Capture threads run independently;
there is no depth or hardware synchronization between cameras or joint state.
The robot adapter samples the latest received frames at each recording tick and
rejects images older than one second by host receive time. Source timestamps do
not synchronize the machines.

## 1. Robot: release cameras from Teleimager

Use the updated checkout on both machines. On the robot:

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/unitree/Documents/lerobot_unitree/lerobot
python -m pip install -e '.[pyzmq-dep]'
sudo systemctl stop teleimager.service
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo
sudo udevadm settle
lerobot-find-cameras opencv
ls -l /dev/v4l/by-id/ /dev/v4l/by-path/
```

The installed `teleimager.service` was observed claiming USB camera interfaces
with `Driver=usbfs`, leaving no `/dev/video*` nodes. After stopping it, the
interfaces were unbound; reloading `uvcvideo` restored camera access. Reload only
while camera clients are stopped. This temporarily interrupts Teleimager streams;
it does not disable its startup service. If cameras disappear again, check
`systemctl status teleimager.service` and `lsusb -t`.

Inspect `outputs/captured_images` to identify the head and each wrist. Discovery
may list multiple V4L2 nodes per physical camera; select three working color
capture nodes for three distinct views. Prefer `/dev/v4l/by-id/...` paths. If the
wrist cameras have duplicate IDs, use distinct `/dev/v4l/by-path/...` links tied
to their USB ports. Numeric indices may change after reconnects or reboot.

## 2. Robot: publish all three cameras

The operator confirmed this device mapping on g1d:

| View | Robot capture device | Published stream |
| --- | --- | --- |
| Head | `/dev/video4` | `head_camera` |
| Left hand/wrist | `/dev/video2` | `left_wrist_camera` |
| Right hand/wrist | `/dev/video0` | `right_wrist_camera` |

Start the publisher in the robot's activated `lerobot` environment:

```bash
python -m lerobot.cameras.zmq.image_server \
  --camera head_camera=/dev/video4 \
  --camera left_wrist_camera=/dev/video2 \
  --camera right_wrist_camera=/dev/video0 \
  --width 640 --height 480 --fps 30 --fourcc MJPG --port 5555
```

Numeric indices also work, for example `--camera head_camera=4`. This mapping
reflects the current robot enumeration; verify it after reconnects or reboot, or
replace the device paths with the corresponding stable V4L2 links identified above.
The repeatable `--camera NAME=DEVICE` option publishes all named streams on one
port. Width, height, FPS, and FOURCC apply to all cameras in this command. Use modes
supported by all three devices and update the host config if dimensions/rate change.
The older single-head `--device 4` command remains supported when no `--camera`
options are supplied.

Keep this terminal running. Wait for all three `Camera ... ready` messages. This
publisher starts no motor control. Run the body bridge without `--camera`; do not
also run Teleimager or a second publisher on these devices/port. The Teleimager
wire protocol is not assumed compatible with this checkout's ZMQ receiver.

The publisher binds TCP 5555 on the robot. The host at `192.168.123.10` must be able
to reach `192.168.123.164:5555` through the USB Ethernet link. Body/hand bridges use
their existing ports independently. See the [complete setup guide](PICO_G1D_TELEOPERATE_SETUP.md)
for networking and those bridges.

## 3. Host: verify all views without moving the robot

```bash
source /home/oleg/extend_lerobot/activate_unitree_g1.sh
cd /home/oleg/extend_lerobot/lerobot
uv pip install --python /home/oleg/extend_lerobot/envs/unitree-g1/bin/python -e '.[pyzmq-dep]'

uv run --no-project --python /home/oleg/extend_lerobot/envs/unitree-g1/bin/python python - <<'PYCODE'
import json
from contextlib import ExitStack
from pathlib import Path

import cv2
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

config = json.loads(Path('configs/unitree_g1_pico.json').read_text())
output = Path('outputs/g1_camera_check')
output.mkdir(parents=True, exist_ok=True)
with ExitStack() as stack:
    cameras = {}
    for name, settings in config['robot']['cameras'].items():
        settings = {k: v for k, v in settings.items() if k != 'type'}
        cameras[name] = stack.enter_context(ZMQCamera(ZMQCameraConfig(**settings)))
    for name, camera in cameras.items():
        frame = camera.async_read(timeout_ms=2000)
        destination = output / f'{name}.png'
        if not cv2.imwrite(str(destination), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f'Could not save {destination}')
        print(name, frame.shape, frame.dtype, destination.resolve())
PYCODE
```

Each view must report `(480, 640, 3) uint8`. Open the three saved PNGs and confirm
that left/right/head are correctly assigned before collecting data. A missing
stream times out; another camera is never substituted. A wrong resolution raises
an error instead of silently changing the dataset schema. This check connects
only to cameras, without starting the body bridge, hands, or PICO.

## 4. Host: start data collection with lerobot-record

Start the robot body and hand bridges, then the host PICO PC service and USB/ADB
forwarding as described in [the complete setup guide](PICO_G1D_TELEOPERATE_SETUP.md).
Keep the three-camera publisher running. Stop preview/teleoperation clients before
recording: use one host robot command source and one PICO client at a time.

On the host, with the operator and tracked controllers ready:

```bash
source /home/oleg/extend_lerobot/activate_unitree_g1.sh
cd /home/oleg/extend_lerobot/lerobot
/home/oleg/extend_lerobot/envs/unitree-g1/bin/lerobot-record \
  --config_path=/home/oleg/extend_lerobot/lerobot/configs/unitree_g1_pico.json \
  --dataset.repo_id=local/g1_pico_three_cameras \
  --dataset.root=/home/oleg/extend_lerobot/lerobot/datasets/g1_pico_three_cameras \
  --dataset.no_stamp=true \
  --dataset.single_task="Pick up the object and place it in the box" \
  --dataset.fps=30 --dataset.num_episodes=1 --dataset.episode_time_s=60 \
  --dataset.push_to_hub=false --display_data=true --play_sounds=false
```

Replace the task description with the actual task. Use a new dataset directory
for the first launch; LeRobot creates it. This records locally without a Hub login
or upload. `no_stamp=true` keeps the repo ID fixed for the resume command below.
Use `--display_data=false` on a host without a viewer; all images are still recorded.

**The standard CLI starts robot commands after connection/setup; it has no arming
prompt.** Grips engage each arm's following and triggers control hands, as described
in [the teleoperation guide](TELEOP_PICO4.md). Recording ends after 60 seconds or
Right Arrow / `n`; Escape / `q` ends the session and saves the current episode
through the normal loop. Keep the terminal focused when using terminal keyboard
input. Avoid Ctrl+C as the way to save an in-progress episode: it is not the normal
`save_episode()` path.

The dataset contains joint state/actions plus these RGB video features:

- `observation.images.head_camera`
- `observation.images.left_wrist_camera`
- `observation.images.right_wrist_camera`

Check the saved `meta/info.json`, `data/`, and `videos/` under the specified dataset
root. Confirm three distinct views and sensible joint/action values in the first
episode before collecting more. Monitor the recording timing summary; JPEG capture,
network transfer, IK, display, and encoding all consume resources. If 30 FPS cannot
be sustained, reduce display load or use a supported lower rate consistently on
the publisher, all three camera configs, and dataset.

### Add another episode to the same dataset

The G1 command watchdog can latch a fault during episode-saving or other gaps in
commands. Use **one episode per process** and reconnect between episodes. After
recording exits, reset the scene using your normal procedure, then rerun the same
recording command with the additional flag:

```bash
--resume=true
```

Keep `repo_id`, `root`, task, FPS, and camera schema the same; `num_episodes=1` adds
one more episode on each invocation. Do not run the resume flag as a standalone
shell command. The `r` / Left Arrow re-record control exists in the generic utility,
but its reset pause may trigger this robot's watchdog; reconnecting for each trial
is the documented G1 workflow. Do not extend the watchdog merely to hide recording
pauses.

## 5. Stop and troubleshoot

Stop the host recorder first and allow it to finalize/disconnect while the robot
bridges still run. Stop the camera publisher with Ctrl+C when finished. To restore
the original camera service after releasing all local capture clients:

```bash
sudo systemctl start teleimager.service
```

- No video devices with `Driver=usbfs`: identify the owner using `sudo fuser -v /dev/bus/usb/BBB/DDD` with current addresses from `lsusb`.
- Empty video-interface driver after stopping Teleimager: reload `uvcvideo` as above.
- Device busy or port already in use: check for Teleimager, another publisher, or the body bridge's `--camera` mode.
- Timeout on just one named camera: verify its exact `--camera` name and physical device mapping.
- Shape mismatch: match publisher dimensions with all host camera configs.
- Changed robot address: update all three `server_address` values; changing `robot.robot_ip` alone does not change camera addresses.
- To use the shared config without images, pass `--robot.cameras='{}'`; this affects both recording and standard teleoperation.

The commands and schema are checked against this checkout. Three physical cameras
and end-to-end robot recording require validation on g1d and the host; local tests
use synthetic frames and do not move hardware.
