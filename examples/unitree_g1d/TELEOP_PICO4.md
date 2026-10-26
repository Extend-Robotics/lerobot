# PICO controllers or hands to G1-D: Unitree XR mapping and IK

The default launcher now uses the G1-29 controller mapping, IK objective, solver
options, joint filter, gravity feedforward and arm gains from
[`unitreerobotics/xr_teleoperate`](https://github.com/unitreerobotics/xr_teleoperate/tree/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b).
Source revisions and licenses are recorded in [UNITREE_XR_NOTICES.md](UNITREE_XR_NOTICES.md).
The native PICO SDK and the existing robot-side JSON/ZMQ-to-DDS bridge remain the
hardware adapters. No robot-side update or reinstall is needed.

## Launch

Keep the usual robot-side `run_g1_server.py` bridge running. Stop other robot
command writers and headset consumers. Start the PICO service/headset app:

```bash
bash scripts/pico4/start_service.sh
```

Enable headset and both controller tracking in the headset app, then preview
using live robot feedback without publishing any commands, including on exit:

```bash
bash scripts/pico4/teleop_g1d.sh --robot-ip <ROBOT_IP>
```

Preview uses ideal virtual joint feedback and follows both controllers immediately.
Inspect the absolute targets before enabling physical movement:

```bash
bash scripts/pico4/teleop_g1d.sh --robot-ip <ROBOT_IP> --execute
```

Execution waits for `r` followed by Enter, then a two-second delay. `q` followed by Enter or Ctrl+C exits.
Use an interactive terminal for execution. **Starting moves both arms toward the
absolute tracked poses. Grips no longer clutch or rebase an arm.** Triggers and
A/B/X/Y do not control the arms. Looking around changes the head-yaw reference,
as in upstream's default mapping.

The launcher uses `../envs/unitree-g1/bin/python`. Override with
`PICO_G1_PYTHON=/path/to/python` for another environment containing LeRobot,
Unitree SDK, Pinocchio with CasADi, and the PICO SDK. Install the PICO binding with
`PICO_PYTHON=/path/to/python bash scripts/pico4/install_sdk.sh` if needed.

## Bare hands controlling arms

Enable hand tracking and headset tracking in the PICO headset app, and enable
sending data. Controllers are not needed. Only the tracked wrist poses are used;
finger motion, pinch and grasp gestures send no robot hand or gripper commands.
The headset client must send OpenXR-order 26-joint hand arrays in the same world
frame as its headset pose (palm index 0, wrist index 1; metres, xyzw quaternions).

Rebuild the native Python binding once to expose coherent hand packets:

```bash
PICO_PYTHON=../envs/unitree-g1/bin/python bash scripts/pico4/install_sdk.sh
```

Preview, then enable movement after checking alignment:

```bash
bash scripts/pico4/teleop_g1d.sh --robot-ip <ROBOT_IP> --input hands
bash scripts/pico4/teleop_g1d.sh --robot-ip <ROBOT_IP> --input hands --execute
```

Execution requires `r` then Enter, followed by a two-second delay to return your
hands to the tracking area. Both hands must be visible and valid when the delay
ends. `p` then Enter pauses both arms and cancels a pending start; `r` resumes
after the same two-second delay; `q` exits. Repeating `r` restarts the countdown. Preview tries to start immediately;
if tracking is not ready, press `r` once both hands are visible. Tracking recovery
never resumes movement automatically. Hand mode does not wait for controller
poses or read controller buttons. The default remains `--input controllers`.

The existing absolute head-yaw-relative mapping and arm IK are retained, with
televuer's left/right OpenXR hand wrist frame corrections. Preview prints the
mapped wrist target matrices and IK residuals. Verify the left/right sides,
translations and wrist rotations with the installed headset app before execution;
physical wrist alignment and tracking quality have not been hardware validated.
Hand mode uses the same speed clipping as controllers: the upstream 30 rad/s
parameter with a fixed 4 ms period, clipping targets against measured joints.
There is no extra hand-mode speed limit, including at start and resume. Preview
bypasses clipping, as in controller mode. Feedforward uses measured arm joints.

Missing/inactive hands, invalid wrist/head poses, old received packets or stalled
source timestamps pause both arms. Head and both hands must be present in the
same packet; cached skeletons cannot be kept alive by headset-only updates.
Tracking is checked again after IK. During a pause the publisher holds the last
command with gravity support while robot feedback and communication are healthy.
**After five seconds paused, the process exits and existing cleanup makes the
arms and waist passive.** Support the robot accordingly. Feedback, IK or transport
faults still stop execution immediately. This adds no robot-side network watchdog.

## Upstream behavior

Raw SDK poses use `[x,y,z,qx,qy,qz,qw]`; they are converted to SE(3) matrices.
The controller path then applies televuer's OpenXR-to-robot basis change,
subtracts the headset position, removes headset yaw from wrist translation and
rotation, and adds the fixed head-to-waist translation `[0.15, 0, 0.45]` metres.
There is no grip offset, translation/rotation sensitivity scaling, Cartesian
jump filter, or Cartesian output rate limiter in this path. The former
`--position-scale` and `--orientation-scale` options have been removed.

The IK uses the existing G1-29 model with legs, waist and fingers fixed at zero,
and the upstream 5 cm wrist tool frames. Its objective is:

```text
50 * squared_position_error
+ squared_rotation_error
+ 0.02 * squared_joint_angles
+ 0.1 * squared_distance_from_measured_joints
```

The IPOPT settings match G1-29 upstream: 30 iterations, tolerance `1e-4`,
acceptable tolerance `5e-4`, five acceptable iterations, expansion, simple-bound
detection, warm-start option, and exact Jacobians. Each solve starts from measured
joints. The joint filter weights are `[0.4, 0.3, 0.2, 0.1]`, newest first, with
upstream's duplicate-sample behavior. Gravity torques are calculated from the
filtered solution and delivered to the arm motor slots.

IK runs at 30 Hz by default (`--fps` can override it). An independent 250 Hz
publisher clips the target against **measured** joints, scaling the full
14-joint vector with upstream's `30 rad/s` parameter and fixed 4 ms period.
Shoulder/elbow gains are `kp=80, kd=3`; wrist gains are `kp=40, kd=1.5`.
Preview bypasses clipping, matching upstream simulation behavior.

## G1-D and transport adaptations

This ports the upstream G1-29 arm/controller path, not its whole-body operation:
legs remain passive; waist yaw/pitch hold their connect-time targets with
`kp=300, kd=3`; waist roll remains passive. The G1-29 kinematic model is retained;
G1-D geometry, payload and physical tracking still need hardware validation.
The shared default LeRobot IK and legacy clutched session remain available to
other callers; the launcher selects `arm_ik_profile="unitree_xr"`.

The host publishes through the existing bridge, rather than direct DDS, so 250 Hz
is a requested host loop rate, not a hard real-time delivery guarantee. The
publisher is stopped before disconnect sends the existing passive command.

The adapter retains a 0.25 s PICO stream / IK deadline and enables the existing
robot-feedback age check. Invalid tracking, expired targets, failed IK or
publication failure stops execution instead of following invalid solver iterates.
The publisher attempts to make the robot passive on its own failure. Normal
cleanup makes arms and waist passive, removing their support torque. Support
the arms/torso accordingly. There is no robot-side network-loss watchdog,
collision checking, or guarantee of cleanup after process kill/network loss.

## Offline validation

```bash
uv run --no-sync pytest tests/robots/test_unitree_xr.py tests/robots/test_g1_kinematics.py tests/robots/test_pico_g1_teleop.py tests/robots/test_unitree_g1.py tests/robots/test_unitree_g1_utils.py tests/teleoperators/test_pico4.py -q
```

Tests use mocks, synthetic tracking and optional native IK. They do not move hardware.
