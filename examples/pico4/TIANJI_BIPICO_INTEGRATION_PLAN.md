# Tianji + BiPico4 integration plan

**Operator experience.** Use the existing XRoboToolkit headset app and both handheld controllers. Start from the robot's current posture. Squeezing the left or right grip engages only that arm. The first engaged frame holds the measured pose. Releasing freezes that arm at its measured joints and allows the operator to reposition the controller before gripping again. The two arms can engage independently or together. The Tianji driver currently exposes arms only, so triggers generate no robot joint commands.

**Working assumption.** “Trackers” means the two controller poses already available through BiPico4. Wuji also wears two forearm trackers. The controller-only integration can reproduce Wuji's wrist mapping, filtering and IK settings. Elbow posture will use Wuji's configured default direction and solver fallback. Its behavior across our workspace is the principal open question. If forearm trackers are present, the extension described below provides the measured elbow input.

## 1. Establish a reproducible Wuji baseline

Pin source revision `647801345a6a27dec5cbf56280ce63bb8b2f6a32`. Keep the model, transforms and filter values together with their upstream provenance. Use the executed PICO path as the reference when comments, launch defaults and runtime behavior differ. [Comparison report](TIANJI_ARM_TELEOP_COMPARISON.md) · [Robot configuration][config] · [Input configuration][input]

| Quantity | Value and intended use |
|---|---|
| Wrist motion scale | 1 m per metre and 1 rad per rad before filtering. Wuji applies the displacement and rotation directly. Apply no G1 head-to-waist offset or generic Cartesian-mode sensitivity. |
| XR → world | `(x,y,z) → (-z,-x,y)`. World axes are forward, left, up. Establish operator forward during Pico tracking-space setup. Head movement after engagement leaves a stationary wrist target fixed. |
| World → arm chest | Left `(x,-z,y)`, right `(x,z,-y)`. Wuji's ±0.2 m shoulder translations locate the frames for visualization. Displacements use rotation only. |
| Wrist filters | Wuji One-Euro position and quaternion implementations. Minimum cutoff 1 Hz, derivative cutoff 1 Hz, base beta 0.7. Effective beta is 14 for position and 2.8 for orientation. Preserve quaternion hemisphere handling. |
| Processing rate | Wuji input and output default to 90 Hz. Its filter uses a configured sample rate. The G1D CLI workflow commonly uses 30 FPS. Match Wuji on a 90 Hz reference trace and parameterize the filter by its real update cadence. Measure the 30 Hz path separately before choosing the shipped preset. |
| Arm model and units | `ccs_m6.MvKDCfg`, serial 0 for A/left, 1 for B/right. Flange TCP with zero tool-kinematic offset. Tracking and FK-facing processor APIs use metres and rotation matrices. Convert translation to/from millimetres at the Marvin boundary. Marvin and LeRobot Tianji joints both use degrees. |
| Redundancy resolution | First try `zsp_type=1`, `zsp_angle=0°`, `dgr=[5,5,5]°`. Configured plane normals are left `[0,-0.5,-0.3]`, right `[0,+0.5,-0.3]`. Match the live node's default-direction normalization and append three zeros. On IK failure, retry with `zsp_type=0` using the same seed. |
| Operator and driver settings | G1D grip engages above 0.5 and releases at or below 0.3. Preserve BiPico4 source freshness checks and start with the G1D preset's 0.5 s input timeout. Keep Tianji's independent 2° maximum target deviation from measured feedback. Wuji's 60% velocity/acceleration settings, startup move and 0.95 kg Wuji Hand payload belong to its hardware deployment. Keep the installed Tianji driver's settings for commissioning and document the difference. |

## 2. Bring up offline Tianji FK and IK

1. Add a hardware-independent kinematics module beside the Tianji driver. Isolate the Marvin kinematics wrapper, native `libKine.so` and exact M6 model from Wuji's ROS node and robot-connection constructor. Verify the mounted arm variant, library architecture, required symbols and redistribution terms. Use an explicitly configured local SDK path if the binary must be supplied separately. Load the optional library only when the Tianji processor is selected.
2. Initialize both sides from their own model entries, including DH, PNVA and J6/J7 parameters. Expose FK from seven measured degrees to a flange pose and IK from that pose plus a seven-joint seed back to degrees. Reject vendor failure, out-of-range flags, joint-exceeded flags and nonfinite outputs.
3. Run FK on Wuji's reference joints, left `[55,-65,-70,-60,60,0,0]` and right `[-55,-65,70,-60,-60,0,0]`. Compare position and orientation with the rounded fixtures in the pinned YAML. For FK→IK→FK, compare flange poses and also report joint differences to reveal elbow-branch changes. Repeat from recorded local postures. Use the reference joints as offline fixtures only. [Marvin solve and initialization][solver] · [Reference poses][config]

## 3. Implement the clutch-relative wrist processor

Register a Tianji-specific action processor in the existing processor factory for the exact `tianji` + `bi_pico4` pairing with `output_mode="xr"`. It receives a coherent controller sample and measured robot observation. Keep all hardware writes in `Tianji.send_action()`. [BiPico4 input](../../src/lerobot/teleoperators/bi_pico4/bi_pico4.py) · [G1D clutch reference](../../src/lerobot/teleoperators/bi_pico4/g1_processor.py) · [Processor factory](../../src/lerobot/processor/factory.py)

- **Start or reset.** Hold both measured joint vectors. Require a fresh sample with grips released before accepting a new press after reconnect, tracking loss or episode reset. This proposed re-arm rule prevents a held grip from restarting motion.
- **Engage.** Snapshot that controller's pose and the measured Tianji flange pose from FK. Initialize its filters at the flange pose and its IK seed from measured joints. Return exactly those seven joints on the engagement tick.
- **Follow.** Let `B` map XR axes to this arm's chest axes. With controller reference `p0,R0` and measured flange reference `f0,F0`, compute `f = f0 + B(p-p0)` and `F = B(R R0ᵀ)Bᵀ F0`. Apply Wuji's filters to the resulting position and orientation. Rotation in place must produce zero translation. Retain the source timestamp across repeated polls so an old sample cannot refresh the watchdog.
- **Solve and bound.** Start each engagement from measurement and use the preceding valid IK solution as the continuity seed thereafter. Reject a discontinuous candidate before advancing that seed. For reachable motion exceeding the driver's per-command envelope, scale the seven-joint displacement from measurement with one factor per arm. Keep the driver's independent final check.
- **Release or fault.** Latch measured joints on release, discard that side's anchor and filter state, and continue to emit both seven-joint vectors. If both IK modes fail, hold the affected arm and require a new grip engagement. Check input age before and after IK. If input expires, feedback ceases advancing, or dispatch fails, stop the command path and run the driver's idle/disconnect cleanup.

## 4. Wire preview, teleoperation and recording

1. Add an arms-only preview that reads Pico and Tianji feedback, runs the production processor and reports measured joints, wrist targets, unbounded IK, bounded commands, fallback use and tick timing. Preview must never call `enable_position_control()` or send motor commands. Support replaying captured samples and observations through the same processor without a robot.
2. Supply paired presets for the standard `lerobot-teleoperate` and `lerobot-record` entry points. Perform explicit Tianji arming once after both devices are ready and a valid released-grip sample and fresh feedback have been observed. Keep ordinary driver connection read-only. Verify cleanup on partial startup, exceptions and Ctrl+C, including an arming failure after position-follow has been requested. [Driver](../../src/lerobot/robots/tianji/tianji.py) · [Teleoperate loop](../../src/lerobot/scripts/lerobot_teleoperate.py)
3. Record the 14 accepted joint targets in driver order, measured positions in the same degree-valued schema, and G1D-compatible `teleop.grip` and `teleop.clutch` metadata. Verify the recording path uses the action returned by the driver. Clear processor state at episode transitions, resets and reconnects and exercise the recorder's between-episode loop. Use an arms-only recording preset for initial acceptance. [Recorder](../../src/lerobot/scripts/lerobot_record.py)

## 5. Verify before enabling both arms

| Stage | Acceptance evidence |
|---|---|
| Mapping parity, offline | Compare our output against the pinned Wuji wrist implementation on identical traces at 90 Hz and identical initial anchors. Test ± translations and rotations about all three axes on each side, combining rotation and translation and crossing the quaternion sign boundary. A 10 cm upward move must settle at −10 cm left-chest Y and +10 cm right-chest Y. |
| Clutch and fault regressions | With fake FK/IK and a fake robot, assert zero joint step on engagement, measured hold on release, independent side operation, smooth re-grip after moving the free controller, and no motion from head-only movement. Exercise duplicate/stale timestamps, bad quaternions, IK fallback/failure, stale feedback on either arm, dispatch rejection and reconnect with grip held. Strengthen the driver's frame-pair check if one frozen side can pass while its partner advances. |
| Kinematics and cadence | Replay representative reach, lift, rotation, near-extension and return motions from several starting elbows. Compare static plane-normal and type-0 solutions, flange errors, maximum joint jumps, fallback frequency and limiter use. Compare tracking lag and median/p95 tick times at 30 and 90 Hz against their 33.3 and 11.1 ms periods. Record achieved input and feedback rates. A preset is ready when the selected loop rate is sustainable and replay shows continuous elbow motion through the intended tasks. |
| Live preview and execution | First use read-only feedback with an operator facing the tracking-space forward direction. Check side identity, lift, forward reach, yaw, independent clutching and limits. Then commission one engaged arm at a time, followed by bimanual motion. Confirm release, loss of tracking and Ctrl+C against observed robot state. |
| Recording | Record a short two-episode session. Verify feature order, units, measured hold and actual bounded commands in the dataset. Confirm a fresh grip engagement on the second episode. Run existing G1D/Pico regressions alongside the new Tianji checks. |

**If forearm trackers are available.** Add serial-mapped forearm input as a separate, explicitly selected profile. Preserve handheld grips for clutching and document the wrist pose source. Rebase the forearm geometry consistently with the measured robot pose on engagement. Reproduce Wuji's shoulder–wrist projection, side-specific direction clamp, 15 mm singularity threshold, elbow minimum cutoff 0.3 Hz, effective beta 2.8, normalization and sign reversal before constructing the IK plane normal. Compare against the pinned dynamic path with recorded four-tracker data. This profile needs its own input-loss tests because a fresh wrist sample can coexist with stale forearm data. [Incremental controller][mapping] · [Elbow publication][node]

**Implementation order.** Deliver and verify offline FK/IK first, then the pure mapping and clutch processor, then read-only preview. Use preview results to choose the cadence and verify elbow continuity before enabling the CLI hardware and recording paths.

## Running the controller-only implementation

The current Tianji presets use the G1D head-relative controller mapping. Each grip anchors the corresponding mapped pose to measured Tianji flange FK. The local Tianji solver minimizes wrist translation error, orientation error and motion from measured joints, within the model and per-command limits. This default requires no wrist calibration or forearm trackers.

Start the Pico service and stream the head and both controllers as in the G1D setup. From the LeRobot checkout, with the existing environment activated and the Wuji M6 kinematics assets installed at the preset paths, run:

```bash
python examples/pico4/preview_tianji.py --config configs/tianji_pico.json --output "/tmp/tianji-preview-$(date +%s).jsonl"
```

Both execution presets select Marvin joint impedance (state 3, type 1) for startup and tracking. Each arm uses Wuji’s stiffness `[2, 2, 2, 1.6, 1, 1, 1]` and damping `[0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]`. Set `robot.control_mode=position` to select position follow.

The teleoperate and record presets clear latched faults on connection and require the cabinet to confirm idle before motion. A persistent fault aborts startup. Set `robot.clear_faults_on_connect=false` to disable automatic clearing. Both presets set `robot.move_to_start_on_connect=true`. With fresh tracking and both grips released, connecting starts physical motion to the Wuji initial joint angles. Tracking begins once measured joints are within 1° of that pose. Set the flag to false to start in place. The preview explicitly requests a read-only connection. For direction checks keep the headset still while moving one controller at a time. The preview uses measured joints at each tick, so its bounded joint outputs represent the next step a live robot would take. After verifying direction and commissioning, use `lerobot-teleoperate --config_path=configs/tianji_pico.json` or `lerobot-record --config_path=configs/tianji_pico_record.json`.

An optional calibrated tracking-world mode is available for comparison. Run `python examples/pico4/calibrate_tianji.py` and explicitly set `robot.pico_wrist_calibration` to its output path to select the measured pivots, forward stroke and native Marvin IK path. With the supplied presets this option is disabled.

[config]: https://github.com/wuji-technology/wuji-hand-teleop/blob/647801345a6a27dec5cbf56280ce63bb8b2f6a32/src/output_devices/tianji_world_output/config/tianji_robot.yaml
[input]: https://github.com/wuji-technology/wuji-hand-teleop/blob/647801345a6a27dec5cbf56280ce63bb8b2f6a32/src/input_devices/pico_input/config/pico_input.yaml.template
[solver]: https://github.com/wuji-technology/wuji-hand-teleop/blob/647801345a6a27dec5cbf56280ce63bb8b2f6a32/src/output_devices/tianji_world_output/tianji_world_output/cartesian_controller.py
[mapping]: https://github.com/wuji-technology/wuji-hand-teleop/blob/647801345a6a27dec5cbf56280ce63bb8b2f6a32/src/input_devices/pico_input/pico_input/incremental_controller.py
[node]: https://github.com/wuji-technology/wuji-hand-teleop/blob/647801345a6a27dec5cbf56280ce63bb8b2f6a32/src/input_devices/pico_input/pico_input/pico_input_node.py
