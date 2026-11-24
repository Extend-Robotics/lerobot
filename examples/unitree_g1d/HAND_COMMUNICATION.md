# Measure Revo2 hand communication

With the hand DDS/ZMQ bridge running on the robot, run from the LeRobot directory:

```bash
uv run --no-sync python examples/unitree_g1d/measure_hands.py --robot-ip 192.168.123.164
```

The default is a **30-second, read-only test of both hands**. It can run alongside
teleoperation to capture delays under load. It does not initialize DDS or send
motor commands. Use `--sides left` or `--sides right` to measure one hand.

Results are saved to `outputs/hand_communication/summary.json` and `samples.csv`.
Use `--output outputs/hand_test_2` to preserve a previous run. Ctrl+C saves a partial
report. A missing stream is reported as zero receive rate and full-duration silence.

Each hand reports:

- Received messages/second and receive-interval statistics.
- Mean, population standard deviation (`std`), p50/p95/p99 and maximum in milliseconds.
- Interval jitter: absolute difference between successive receive intervals.
- Sample age at the bridge and variation in client-minus-source clock deltas.
- Long gaps, including startup and trailing silence, and invalid messages.
- Local receive/decode duration.

The bridge republishes cached DDS samples, so message rate is **not** the rate of
new hand measurements. SUB conflation matches the normal client and can discard
intermediate messages; this test cannot calculate packet loss. The bridge stops
publishing stale feedback, so stale hardware can appear as silence.

To estimate sample age at the client, supply `--clock-offset-ms 0` if clocks are
synchronized, or the measured offset **client clock minus robot clock** in ms.
Without this option, absolute sample age is omitted; clock deltas and their
variation are still reported. Constant offsets cancel in delay variation, but
clock adjustments and drift affect it. No round-trip or physical motion latency
is measured.

## Commands rejected just above the deadline limit

The client uses its wall clock to set command deadlines, and the robot bridge
checks them against its own wall clock. With the default 500 ms timeout, a small
clock offset can make a fresh command appear 501–503 ms in the future. The bridge
allows 10 ms of clock tolerance on this upper bound by default. Expired commands
are still rejected, and hand feedback freshness checks are unchanged.

Update the bridge code on the robot and restart it to apply this behavior:

```bash
python -m lerobot.robots.brainco_revo2.bridge --sides left right --command-clock-tolerance-ms 10
```

Use `--command-clock-tolerance-ms 0` for the strict upper bound. Both machines
should still synchronize their clocks; this bounded tolerance does not compensate
for large offsets or clock jumps. If deadlines exceed the tolerance, check clock
synchronization and that client and bridge hand timeouts match. This change
addresses command rejection, not intermittent stale feedback or rollout recovery.

## Optional command-send timing

Stop teleoperation and other command publishers first, then run:

```bash
uv run --no-sync python examples/unitree_g1d/measure_hands.py --probe-commands
```

This sends already-expired commands at approximately 1 Hz per hand. The repository's
`brainco_revo2/bridge.py` rejects them before dispatching to DDS, so they request no
motion. Only use this with that bridge's deadline validation. Probes can replace
other queued commands because the bridge conflates its input, and produce rejection
warnings in its logs.

The probe matches the production client's 100 ms send timeout. `probe_send_ms` and
`probe_timeouts` detect local enqueue stalls; successful sends do not establish
server receipt or acceptance. Sends run sequentially in the measurement loop, so
stalls also affect observed feedback gaps. Use the default read-only run as the
feedback baseline before comparing a probe run.

## Direct DDS access from the host

The subscriber-only connectivity check uses the production DDS transport and
counts actual DDS callbacks for both hands (default duration: 10 seconds):

```bash
uv run --no-project --python ../envs/unitree-g1/bin/python python \
  examples/unitree_g1d/check_hand_dds.py \
  --network-interface enx9c69d328e90f --disable-config-trace
```

Select the host interface connected to the robot and the driver's DDS domain
(`--domain-id`, default 0). The script creates no command publishers and exits
with status 0 only if valid feedback from both hands was received without setup
errors. It saves `outputs/hand_dds_check.json`; use `--output` for another path.
This establishes feedback access; command delivery is not tested. Callback
intervals include SDK scheduling and are not one-way network latency.

`--disable-config-trace` is a process-local workaround for a native crash observed
in the installed CycloneDDS configuration-printing function when the SDK enables
`<Verbosity>config</Verbosity>`. It changes that value to `none` in memory for this
check only; no installed SDK files are modified.


Hand DDS initialization now automatically disables the explicit-interface SDK
`config` trace before native initialization. Normal hand teleoperation no longer
needs a Python wrapper or `--disable-config-trace`. The explicit option remains
available in standalone diagnostics, including the arm check which initializes
the SDK directly.
