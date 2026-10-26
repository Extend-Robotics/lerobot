# PICO controllers to G1-D arms: MVP

This example connects the existing BiPico4 mapping to G1 arm FK/IK and the
existing socket bridge. No robot-side update or reinstall is needed.

## Launch

Keep the usual robot-side `run_g1_server.py` bridge running. Stop other robot
command writers and headset consumers, then start the PICO service/headset app:

```bash
bash scripts/pico4/start_service.sh
```

Preview using live robot feedback without publishing commands, including on exit:

```bash
bash scripts/pico4/teleop_g1d.sh --robot-ip <ROBOT_IP>
```

Enable arm commands for a supervised hardware trial:

```bash
bash scripts/pico4/teleop_g1d.sh --robot-ip <ROBOT_IP> --execute
```

The launcher uses `../envs/unitree-g1/bin/python`. Override with
`PICO_G1_PYTHON=/path/to/python` for another environment containing LeRobot,
Unitree SDK, Pinocchio with CasADi, and the PICO SDK. Install the PICO binding
into a new environment with `PICO_PYTHON=/path/to/python bash scripts/pico4/install_sdk.sh`.
If `.venv` contains all dependencies, run directly with:

```bash
uv run --no-sync python examples/unitree_g1d/teleop_pico4.py --robot-ip <ROBOT_IP>
```

## Controls

Startup waits up to 15 seconds for both controller poses, reporting which side
is missing. The headset can show a connection before tracking data arrives.

Release both grips after startup, then squeeze either grip to move its arm.
Releasing a grip holds the last commanded joints. Re-engagement starts from
that held pose. Triggers and A/B/X/Y are ignored. Ctrl+C exits.

The loop reads inputs, solves IK once, preserves inactive arm joints, applies
a joint speed limit, and optionally publishes. Pico owns grip hysteresis;
there is no second session-level grip state machine. Preview advances virtual
joint commands without requiring the physical robot to follow them.

Faults print a reason and exit through normal cleanup. Restart the script to
retry. There is no automatic recovery or rearming. In execution, cleanup makes
arms and waist passive, removing their support torque. Support the arms/torso
accordingly. During execution the waist holds its connect-time targets and the
legs remain passive, using the same gains as `robot_test.py`.

## Tuning and diagnostics

CLI options are `--robot-ip`, `--execute`, `--duration` (0 means until Ctrl+C),
`--fps` (default 30), `--position-scale`, and `--orientation-scale` (both 0.25).

There are no position or orientation bounds relative to the starting wrist poses.

MVP bounds are named constants in `src/lerobot/robots/unitree_g1/pico_teleop.py`:
0.25 s PICO stream/control-deadline timeout, 0.3 rad/s joint speed, and 0.25 rad
tracking error during execution. IK residuals are printed
as diagnostics rather than causing faults. This path bypasses the IK moving
filter and Pico Cartesian rate limits, using the final joint speed limit.

Targets use the reduced IK model's base frame, with legs and waist fixed at
zero and the existing 5 cm wrist tool frames. Mapped axes are +X forward,
+Y left, +Z up; verify their physical directions in preview.

The Pico MVP uses the latest available robot feedback without rejecting it
based on age. The shared publisher retains age checking by default; this
example explicitly opts out. PICO freshness uses its stream timestamp, which cannot detect a single frozen controller in
otherwise advancing packets. Invalid samples, failed IK, joint-limit violations,
and stale PICO input stop the script.

Cleanup is best-effort over the bridge. There is no robot-side watchdog,
collision checking, or guarantee of stopping on process kill/network loss.
Native calls can delay cleanup until they return. Integrated physical motion
still needs to be verified on hardware.

## Software checks

```bash
uv run --no-sync pytest tests/robots/test_pico_g1_teleop.py tests/robots/test_g1_kinematics.py tests/robots/test_unitree_g1.py tests/robots/test_unitree_g1_utils.py tests/teleoperators/test_pico4.py -q
```

These checks use mocks and synthetic feedback; they do not move hardware.
