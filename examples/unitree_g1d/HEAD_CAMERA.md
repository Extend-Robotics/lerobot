# Head camera on a separate machine

Run the camera publisher on the robot computer that has access to the head camera.
Run LeRobot recording on the separate workstation. Images travel as JPEG over ZMQ
TCP port 5555; the workstation does not need a local camera device or DDS camera
subscription. This captures RGB images, without depth or hardware synchronization
with joint state. Recording samples the latest received image on each dataset tick.

This publisher requires a camera accessible through OpenCV/V4L2. Device 4 is the
existing G1 server default, not an automatically detected head-camera identity.
Verify the device on your robot; if its head camera is only exposed through a
vendor streaming service, that service needs a separate capture adapter.

## Robot computer

Install this checkout of LeRobot with its `pyzmq-dep` extra in the robot environment
(OpenCV and NumPy are base dependencies). From the repository root, identify the
camera using `lerobot-find-cameras opencv` or the devices under `/dev/v4l/by-id/`.
Then start the camera-only service:

```bash
python -m lerobot.cameras.zmq.image_server \
  --device 4 --width 640 --height 480 --fps 30 --port 5555
```

Prefer a stable path once the device is identified:

```bash
python -m lerobot.cameras.zmq.image_server \
  --device /dev/v4l/by-id/<HEAD_CAMERA_DEVICE> \
  --width 640 --height 480 --fps 30 --fourcc MJPG --port 5555
```

The service publishes the name `head_camera` and does not initialize motor control.
Use a capture resolution, FPS and FOURCC supported by your camera. It binds port
5555 on the robot's interfaces; allow workstation access on the robot network.
Do not also start the motor bridge with `--camera` on the same device/port. The
existing combined server remains supported if that is how you launch services.

## Recording workstation

Use the same updated checkout and install the `pyzmq-dep` extra here too. Test the
camera independently of the robot and teleoperator first. Replace the IP below
with the address of the computer running the publisher:

```bash
python - <<'PY'
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

config = ZMQCameraConfig(
    server_address="192.168.123.164", port=5555, camera_name="head_camera",
    width=640, height=480, fps=30,
)
with ZMQCamera(config) as camera:
    for _ in range(30):
        image = camera.async_read(timeout_ms=1000)
        print(image.shape, image.dtype)
PY
```

Each image should be RGB `uint8` with shape `(480, 640, 3)`. A missing named stream
times out; another camera is never substituted. A resolution mismatch fails rather
than writing images with an incorrect dataset schema. The robot rejects a latest
image older than one second according to local receive time; source timestamps
are not used to synchronize clocks between computers.

The shared `configs/unitree_g1_pico.json` has cameras disabled by default.
The command below adds the optional head camera at `192.168.123.164`, reached
over the host USB Ethernet adapter. Headset input continues over USB/ADB.
After starting the motor/hand bridges and PICO service as described in the
[teleoperation guide](TELEOP_PICO4.md), record on the workstation:

```bash
lerobot-record --config_path=configs/unitree_g1_pico.json \
  --robot.robot_ip=192.168.123.164 \
  --robot.cameras='{head_camera: {type: zmq, server_address: "192.168.123.164", port: 5555, camera_name: head_camera, width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id=local/g1_head_camera \
  --dataset.single_task="Describe the task" \
  --dataset.fps=30 --dataset.num_episodes=1 \
  --dataset.push_to_hub=false --display_data=true
```

This command starts teleoperation immediately after connection. It records the
camera as `observation.images.head_camera` alongside the existing joint state and
actions. If capture dimensions or rate change, update both publisher flags and
`robot.cameras.head_camera` settings. Keep dataset FPS at or below capture FPS to
reduce repeated frames. Confirm the head view in the display and saved episode
before collecting a dataset. Omit the camera override to use the default
configuration without images. Alternatively, add the camera dictionary to a local
JSON configuration and omit the dictionary override from the command.
The CLI accepts the complete camera dictionary, not flags for individual dictionary entries.
