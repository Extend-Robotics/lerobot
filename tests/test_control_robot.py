#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from unittest.mock import patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import RecordConfig, record, record_loop
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def _ticks(summary: str) -> int:
    """Sample size out of a cadence report — every other number is an average over it."""
    return int(re.search(r"(\d+) ticks", summary).group(1))


def _step_calls(summary: str, step: str) -> int:
    """How many ticks ran *step*, off the loop-body breakdown of a run summary."""
    return int(re.search(rf"\n\s+{step}\s+.*· (\d+) calls", summary).group(1))


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate(cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        fps=30,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)

    # A teleop session has no episodes, so there is one cadence block for the whole run,
    # and the steps it names are the ones the loop wraps.
    (summary,) = cadence_log
    assert summary.startswith("Cadence summary — whole run · target 30 Hz (33.3 ms budget per tick):")
    for step in ("observe", "teleop", "send"):
        assert step in summary, step


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


@pytest.mark.parametrize("control", ["right", "left", "esc"])
def test_record_keys_during_save_do_not_skip_next_episode(tmp_path, monkeypatch, control):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.keyboard_input import apply_recording_control

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    monkeypatch.setattr("lerobot.scripts.lerobot_record.init_keyboard_listener", lambda: (None, events))
    original_save = LeRobotDataset.save_episode
    saved_sizes = []

    def save_with_key(dataset, *args, **kwargs):
        saved_sizes.append(dataset.writer.episode_buffer["size"])
        result = original_save(dataset, *args, **kwargs)
        if len(saved_sizes) == 1:
            apply_recording_control(control, events)
        return result

    monkeypatch.setattr(LeRobotDataset, "save_episode", save_with_key)
    dataset = record(
        RecordConfig(
            robot=MockRobotConfig(),
            teleop=MockTeleopConfig(),
            play_sounds=False,
            dataset=DatasetRecordConfig(
                repo_id=DUMMY_REPO_ID,
                root=tmp_path / "save_keys",
                single_task="test save keys",
                num_episodes=2,
                episode_time_s=0.05,
                reset_time_s=0,
                push_to_hub=False,
            ),
        )
    )
    assert dataset.num_episodes == (1 if control == "esc" else 2)
    assert len(saved_sizes) == dataset.num_episodes
    assert all(size > 0 for size in saved_sizes)


@pytest.mark.parametrize("control", ["right", "left", "esc"])
def test_record_handles_key_before_first_frame(tmp_path, monkeypatch, control):
    from lerobot.utils.keyboard_input import apply_recording_control

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    monkeypatch.setattr("lerobot.scripts.lerobot_record.init_keyboard_listener", lambda: (None, events))
    attempts = []

    def early_key_loop(*args, **kwargs):
        attempts.append(kwargs["dataset"].num_episodes)
        if len(attempts) == 1:
            apply_recording_control(control, events)
        return record_loop(*args, **kwargs)

    monkeypatch.setattr("lerobot.scripts.lerobot_record.record_loop", early_key_loop)
    dataset = record(
        RecordConfig(
            robot=MockRobotConfig(),
            teleop=MockTeleopConfig(),
            play_sounds=False,
            dataset=DatasetRecordConfig(
                repo_id=DUMMY_REPO_ID,
                root=tmp_path / "early_key",
                single_task="test early key",
                num_episodes=1,
                episode_time_s=0.05,
                reset_time_s=0,
                push_to_hub=False,
            ),
        )
    )
    assert dataset.num_episodes == (0 if control == "esc" else 1)
    assert attempts == ([0] if control == "esc" else [0, 0])
    assert dataset.num_frames == 0 if control == "esc" else dataset.num_frames > 0


def test_record_and_replay(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)

    # Replay has to hit the dataset's frame rate or the trajectory plays back at the
    # wrong speed, so it reports its cadence like every other loop.  Its block is the
    # last one and names its own steps.
    assert cadence_log[-1].startswith("Cadence summary — whole run · target 30 Hz")
    assert "read_frame" in cadence_log[-1]


def test_record_reports_a_cadence_summary_per_episode_and_for_the_run(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "cadence",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    record(cfg)

    assert len(cadence_log) == 3
    per_episode, run = cadence_log[:2], cadence_log[2]
    assert [m.split(":")[0] for m in per_episode] == ["Cadence (episode 0)", "Cadence (episode 1)"]
    assert run.startswith("Cadence summary — whole run, 2 episodes")
    # Windows partition the session, so the episodes account for every tick of the run...
    assert _ticks(run) == sum(_ticks(m) for m in per_episode)
    # ...and every one of those ticks wrote a frame.  The reset phase paces at the same
    # fps but records nothing, so it runs on its own timer rather than diluting the
    # numbers that answer "did I record at `fps`?".
    assert _step_calls(run, "record") == _step_calls(run, "observe") == _ticks(run)


def test_record_forwards_compressed_images_setting_to_reset_phase(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "compressed_images",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        display_compressed_images=True,
        play_sounds=False,
    )

    with patch("lerobot.scripts.lerobot_record.record_loop", wraps=record_loop) as mock_record_loop:
        record(cfg)

    # Recording episode 0, resetting, then recording episode 1 should all use the same
    # image representation so visualization backends do not receive mixed message types.
    assert [
        call.kwargs.get("display_compressed_images", False) for call in mock_record_loop.call_args_list
    ] == [True, True, True]


def test_record_loop_without_a_teleoperator_paces_and_terminates():
    # Regression: the no-teleop branch used to `continue` past both the pacing sleep and
    # the `timestamp` update, so a reset phase with no teleop device spun as fast as the
    # CPU allowed and never reached `control_time_s` at all.
    robot = make_robot_from_config(MockRobotConfig())
    robot.connect()
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    calls = 0
    real_get_observation = robot.get_observation

    def counted_get_observation():
        nonlocal calls
        calls += 1
        assert calls <= 20, "loop is spinning: 20 iterations of a 0.1 s phase at 30 Hz"
        return real_get_observation()

    robot.get_observation = counted_get_observation

    try:
        record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False, "rerecord_episode": False},
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=None,
            control_time_s=0.1,
        )
    finally:
        robot.disconnect()

    # 0.1 s at 30 Hz is 3 ticks; the upper bound is what proves the phase was paced.
    assert 1 <= calls <= 6


def test_record_holds_through_slow_video_startup_and_resumes(monkeypatch, clock):
    from types import SimpleNamespace

    from tests.mocks.mock_robot import MockRobot
    from tests.mocks.mock_teleop import MockTeleop

    events = {"exit_early": False}
    sends = []
    waits = []

    class HoldingRobot(MockRobot):
        holding = False

        def hold_position(self):
            self.holding = True

        def send_action(self, action):
            if sends:
                assert self.holding, "startup exceeded the active command deadline"
            sends.append(action.copy())
            self.holding = False
            return action

    robot = HoldingRobot(MockRobotConfig())
    teleop = MockTeleop(MockTeleopConfig())
    frames = []

    def add_frame(frame):
        if not frames:
            assert robot.holding
        else:
            assert not robot.holding, "the second command must restore the watchdog"
            events["exit_early"] = True
        frames.append(frame)

    def wait_for_video_encoder():
        assert len(frames) == 1
        assert robot.holding
        clock.advance(0.4)  # a first-frame stall longer than G1's 250ms watchdog
        waits.append(True)

    dataset = SimpleNamespace(
        fps=30,
        features={},
        meta=SimpleNamespace(video_keys=["camera"]),
        add_frame=add_frame,
        wait_for_video_encoder=wait_for_video_encoder,
    )
    monkeypatch.setattr("lerobot.scripts.lerobot_record.build_dataset_frame", lambda *a, **k: {})
    robot.connect()
    teleop.connect()
    try:
        teleop_processor, robot_processor, observation_processor = make_default_processors()
        record_loop(
            robot=robot,
            events=events,
            fps=30,
            dataset=dataset,
            teleop=teleop,
            teleop_action_processor=teleop_processor,
            robot_action_processor=robot_processor,
            robot_observation_processor=observation_processor,
            control_time_s=1.0,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()
    assert len(frames) == len(sends) == 2
    assert waits == [True]


def test_record_holds_during_save_and_resumes_next_episode(tmp_path, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tests.mocks.mock_robot import MockRobot

    class HoldingRobot(MockRobot):
        holding = False
        resumes = 0

        def hold_position(self):
            self.holding = True

        def send_action(self, action):
            if self.holding:
                self.resumes += 1
            result = super().send_action(action)
            self.holding = False
            return result

    robot = HoldingRobot(MockRobotConfig())
    monkeypatch.setattr("lerobot.scripts.lerobot_record.make_robot_from_config", lambda _: robot)
    original_save = LeRobotDataset.save_episode
    saved = []

    def save_in_hold(dataset, *args, **kwargs):
        assert robot.holding, "saving must not leave the active command watchdog armed"
        saved.append(dataset.num_episodes)
        return original_save(dataset, *args, **kwargs)

    monkeypatch.setattr(LeRobotDataset, "save_episode", save_in_hold)
    dataset = record(
        RecordConfig(
            robot=robot.config,
            teleop=MockTeleopConfig(),
            dataset=DatasetRecordConfig(
                repo_id=DUMMY_REPO_ID,
                root=tmp_path / "hold_recording",
                single_task="test hold",
                num_episodes=2,
                episode_time_s=0.1,
                reset_time_s=0.1,
                push_to_hub=False,
            ),
            play_sounds=False,
        )
    )
    assert saved == [0, 1]
    assert dataset.num_episodes == 2
    assert robot.resumes == 2  # enter the reset phase, then enter episode two
