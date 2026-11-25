"""Preview Tianji/Pico targets live or replay a saved JSONL trace."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import draccus

from lerobot.processor.factory import make_default_processors
from lerobot.robots.tianji import Tianji
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators.bi_pico4 import BiPico4
from lerobot.utils.cycle_timer import CycleTimer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Tianji teleoperate JSON preset")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--replay", type=Path, help="Prior preview JSONL recording")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL results file")
    args = parser.parse_args()
    with args.config.open() as source:
        config = draccus.load(TeleoperateConfig, source)
    if config.robot.type != "tianji" or config.teleop.type != "bi_pico4":
        parser.error("Expected a Tianji and BiPico4 configuration")
    robot, teleop = Tianji(config.robot), BiPico4(config.teleop)
    pipeline, _, _ = make_default_processors(robot, teleop)
    step = pipeline.steps[0]
    with ExitStack() as stack:
        output = stack.enter_context(args.output.open("x"))

        def preview(action, observation, received_at):
            start = time.perf_counter()
            command = pipeline((action, observation))
            record = {
                "received_at": received_at,
                "input": action,
                "measured_deg": observation,
                "command_deg": command,
                "clutch": step.recording_frame["teleop.clutch"].tolist(),
                "ik": step.diagnostics,
                "processing_ms": 1000 * (time.perf_counter() - start),
            }
            output.write(json.dumps(record) + "\n")

        if args.replay:
            with args.replay.open() as source:
                for line in source:
                    frame = json.loads(line)
                    received_at = frame["received_at"]
                    step.clock = lambda received_at=received_at: received_at
                    preview(frame["input"], frame["measured_deg"], received_at)
            return

        teleop.connect()
        stack.callback(teleop.disconnect)
        robot.connect(read_only=True)
        stack.callback(robot.disconnect)
        timer = CycleTimer(fps=config.fps, records_data=False)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            timer.tick()
            observation = robot.get_observation()
            action = teleop.get_action()
            preview(action, observation, time.monotonic())
            timer.wait()


if __name__ == "__main__":
    main()
