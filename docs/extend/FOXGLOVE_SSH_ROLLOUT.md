# Viewing LeRobot Rollouts Remotely with SSH and Foxglove

This guide explains how to watch live camera images and robot telemetry from a
LeRobot rollout running on a remote machine.

The important point is that Foxglove receives frames from the rollout process.
It does **not** open the RealSense or another camera for a second time. This
avoids conflicts in which a separate viewer takes ownership of a camera that the
policy needs for control.

## Data flow

```text
Camera -> LeRobot rollout -> Foxglove WebSocket server
                                  |
                              SSH tunnel
                                  |
                         Foxglove on local PC
```

Only the LeRobot rollout acquires the physical camera.

## 1. Enable Foxglove in the rollout

Add the following arguments to the `lerobot-rollout` command running on the
robot computer:

```bash
--display_data=true \
--display_mode=foxglove \
--display_ip=127.0.0.1 \
--display_port=8765 \
--display_compressed_images=true
```

For example:

```bash
lerobot-rollout --config_path=lerobot/configs/rollout_pi05_leyland_merged_corrections.json \
   --display_data=true \
   --display_mode=foxglove \
   --display_ip=127.0.0.1 \
   --display_port=8765 \
   --display_compressed_images=true
```

Keep `display_ip` set to `127.0.0.1`. The WebSocket server will then be
accessible only from the robot computer itself and through the authenticated
SSH tunnel.

Compressed images are recommended over SSH. They use substantially less
bandwidth than raw images, at the cost of a small amount of CPU time for JPEG
encoding.

## 2. Create the SSH tunnel

On the **local computer**, open a separate terminal and run:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@ROBOT_HOST
```

Replace:

- `USER` with the SSH username on the robot computer.
- `ROBOT_HOST` with its hostname or IP address.

For example:

```bash
ssh -N -L 8765:127.0.0.1:8765 oleg@desktop
```

The command normally prints nothing and remains running. Keep this terminal
open while viewing the rollout. Stop the tunnel with `Ctrl+C`.

## 3. Connect Foxglove

1. Open Foxglove on the local computer.
2. Select **Open connection**.
3. Select **Foxglove WebSocket**.
4. Enter:

   ```text
   ws://localhost:8765
   ```

5. Add an **Image** panel.
6. Select the required camera topic. Camera topics are normally named like:

   ```text
   /observation/images/realsense
   /observation/images/front
   ```

The available name depends on the key used in `--robot.cameras`.

Foxglove also exposes robot observations and actions. These can be displayed in
Plot, Raw Messages, or other suitable panels.

## Starting order

Either order works, but the following is convenient:

1. Start the SSH tunnel on the local computer.
2. Start Foxglove and configure the connection.
3. Start the rollout on the robot computer.
4. Foxglove connects when the WebSocket server becomes available.

If Foxglove does not reconnect automatically, reconnect after the rollout has
started.

## Testing the episode API remotely

The API server supports the same Foxglove display flags as the regular rollout.
These steps test remote episode requests, keyboard interventions, and live
telemetry through a single SSH connection.

### Open an SSH session with port forwarding

On the **local computer**, run:

```bash
ssh -t \
  -L 8000:127.0.0.1:8000 \
  -L 8765:127.0.0.1:8765 \
  oleg@desktop
```

Replace `oleg@desktop` with your SSH username and robot host. `-t` allocates a
terminal for keyboard interventions. Port `8000` serves the API and port `8765`
serves Foxglove telemetry. Keep this SSH terminal open.

### Start the server on the robot computer

In the **remote SSH terminal**, run:

```bash
cd /home/oleg/extend_lerobot/lerobot

# Install once in the environment used for the rollout, if needed.
uv pip install -e '.[rollout-server,viz]'

uv run --no-sync python -m lerobot.scripts.lerobot_rollout_server \
  --config_path=configs/rollout_pi05_leyland_merged_corrections.json \
  --host=127.0.0.1 \
  --port=8000 \
  --strategy.input_device=keyboard \
  --interactive=false \
  --display_data=true \
  --display_mode=foxglove \
  --display_ip=127.0.0.1 \
  --display_port=8765 \
  --display_compressed_images=true \
  --dataset.root="/home/oleg/extend_lerobot/lerobot/datasets/api_test_$(date +%Y%m%d_%H%M%S)"
```

Adjust the checkout path and hardware configuration for your setup. Each launch
uses a fresh recording directory. Wait for server startup to complete: hardware
stays connected and policy execution stays paused until an episode request.

### Send an episode request from the local computer

In a **second local terminal**, run:

```bash
curl -i http://127.0.0.1:8000/episode \
  -H 'Content-Type: application/json' \
  -d '{"task":"Pick a fuse and install it on a jig","timeout_s":10}'
```

The request starts execution and waits for the timed segment to finish. Expect
HTTP `200` with `{"status":"done"}` after approximately ten seconds, plus any
recording cleanup time. Inference then pauses while hardware remains connected.
Send another request to test a second episode without reconnecting hardware.

### Test keyboard interventions and shutdown

Send a longer episode, for example with `"timeout_s":30`, then use the server's
keyboard input:

- **Space:** pause or resume autonomous execution.
- **Tab while paused:** start recording a pilot correction.
- **Tab while correcting:** finish and save the correction; remain paused.
- **Esc:** interrupt the episode; the pending request returns HTTP `503`.
- **Ctrl+C in the server terminal:** stop the server and interrupt an active
  episode before waiting for HTTP requests to finish.

Only correction frames are recorded. Finishing a correction does not finish the
HTTP episode. The timeout continues during pauses and corrections; an unfinished
correction is saved when the timeout ends the segment. Saving an interrupted
correction does not change its result to `done`.

On headless SSH or Wayland, the terminal keyboard backend reads keys from the
SSH terminal running the server. With an active X11 keyboard backend, keys are
captured on the robot computer's display instead. Check the server's keyboard
backend messages if SSH keystrokes have no effect.

A second request while an episode is active should return HTTP `409`. Empty tasks
and nonpositive timeouts should return HTTP `422`.

### Connect Foxglove to the API rollout

On the local computer, open a Foxglove WebSocket connection to
`ws://localhost:8765` and select the camera topics described above. The server
initializes visualization once at startup and keeps it available across episode
requests, then closes it during shutdown.

Live frames and actions are published during autonomous execution and pilot
corrections. No fresh camera preview is published while the policy is paused or
while the API is idle between requests; send an episode request to start updates.
Use `--display_data=false` for API-only testing without visualization.

## Troubleshooting

### Connection refused

Confirm that the rollout includes all Foxglove arguments and is still running.
On the robot computer, check whether the server is listening:

```bash
ss -ltn | grep 8765
```

It should show a listener on `127.0.0.1:8765`.

### Address already in use

Another process is using port `8765`. Select another port in both commands:

```bash
# Rollout argument
--display_port=8766

# Local SSH tunnel
ssh -N -L 8766:127.0.0.1:8766 USER@ROBOT_HOST
```

Then connect Foxglove to `ws://localhost:8766`.

### Foxglove connects but no image appears

- Confirm that `--display_data=true` is present.
- Check the topic picker for all topics below `/observation/images/`.
- Confirm that the camera is included in `--robot.cameras`.
- Check the rollout terminal for camera or visualization errors.
- Start the rollout without a second camera program such as FFmpeg,
  `realsense-viewer`, or OpenCV trying to open the same device.

### Rollout becomes slower

Keep `--display_compressed_images=true`. If necessary, reduce the configured
camera resolution or frame rate. For example, use `640x480` at `15` or `30` FPS
instead of `1280x720` at `30` FPS.

### Missing Foxglove Python dependency

If LeRobot reports that `foxglove-sdk` is unavailable, install the LeRobot
visualization extra in the same environment used to run the rollout:

```bash
pip install -e './lerobot[viz]'
```

Use the environment's normal package-management command if this checkout is
managed by `uv` or another tool.

## Recording and teleoperation

The same display arguments also work with `lerobot-record` and
`lerobot-teleoperate`:

```bash
--display_data=true \
--display_mode=foxglove \
--display_ip=127.0.0.1 \
--display_port=8765 \
--display_compressed_images=true
```

The SSH tunnel and Foxglove connection steps remain unchanged.

## Why a separate FFmpeg viewer should not be used

Opening `/dev/video*` or starting another RealSense pipeline may compete with
the rollout for the physical camera, USB bandwidth, or stream configuration.
The rollout can then fail to initialize the camera or lose frames. Publishing
the already-acquired observation through Foxglove provides a shared preview
without acquiring the device again.
