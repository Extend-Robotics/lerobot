# Measure G1 arm communication

With `run_g1_server.py` running on the robot, run from the LeRobot directory:

```bash
uv run --no-sync python examples/unitree_g1d/measure_arms.py --robot-ip 192.168.123.164
```

The default test runs for **30 seconds**, read-only, and can run alongside
teleoperation. It opens only a ZMQ subscriber to port 6001; it does not initialize
DDS, switch robot modes or send motor commands. Both arms share `rt/lowstate`, so
there is one set of communication statistics, with validation of all 14 arm joints.

Results go to `outputs/arm_communication/summary.json` and `samples.csv`.
Use `--output outputs/arm_baseline_2` to preserve an earlier run, and `--seconds`
to change the duration. Ctrl+C saves a partial report.

The report includes receive rate, interval mean and population standard deviation,
p50/p95/p99/max, jitter (absolute change between consecutive intervals), local
receive/decode/validation time, invalid messages, and gaps over 100/250 ms.
Silence includes startup and trailing outages; an absent stream reports zero rate
and full-duration silence. Missing statistics are `null`, not zero.

The current server publishes no source timestamps or sequence numbers. This test
cannot estimate sample age, end-to-end latency, RTT or packet loss. SUB conflation
matches the production client and may discard intermediate messages. Normal receive
intervals do not prove that the upstream DDS measurements are fresh.

## Optional command-send timing

Stop teleoperation and other command publishers, then run:

```bash
uv run --no-sync python examples/unitree_g1d/measure_arms.py --probe-commands --output outputs/arm_probe
```

This sends `diagnostic/arm_communication` with an empty motor list, approximately
once per second on port 6000. The repository's `run_g1_server.py` forwards only
`rt/lowcmd` to DDS; these diagnostic messages therefore do not command motion.
Use this option only with that server's topic filtering. Unlike the hand bridge,
the arm bridge does not reject commands by expiry, so expired `rt/lowcmd` messages
would not be a safe probe. Probes can replace queued commands due to conflation.

Send settings match the production arm client: CONFLATE and a 100 ms timeout,
without IMMEDIATE. A successful send only measures local enqueue and can occur
without a connected receiver. Probes are smaller than real motor commands and
do not exercise DDS writes, acknowledgement or physical motion. Synchronous send
stalls affect feedback collection, so compare with a read-only baseline.
