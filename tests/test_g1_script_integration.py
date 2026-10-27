"""G1 joint actions reach the standard scripts without a session-specific loop."""

import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from lerobot.processor import make_default_processors
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.bi_pico4.g1_processor import ARM_KEYS
from tests.robots.test_pico_g1_teleop import pipeline_state as pipeline_state


def test_shared_cli_configuration_decodes_for_teleop_and_record():
    import draccus

    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig

    path = Path(__file__).parents[1] / "configs/unitree_g1_pico.json"
    teleop = draccus.parse(TeleoperateConfig, config_path=path, args=["--fps=30"])
    record = draccus.parse(
        RecordConfig,
        config_path=path,
        args=[
            "--dataset.repo_id=local/test",
            "--dataset.single_task=test",
            "--robot.left_hand.transport=zmq",
        ],
    )
    assert teleop.robot.arm_commands_only and record.robot.arm_commands_only
    assert teleop.teleop.output_mode == record.teleop.output_mode == "xr"
    assert record.robot.left_hand.transport == "zmq"


def test_standard_teleop_does_not_request_pico_feedback(pipeline_state):
    from lerobot.scripts.lerobot_teleoperate import teleop_loop

    s = pipeline_state
    teleop = Mock(spec=Teleoperator)
    teleop.name = "bi_pico4"
    teleop.get_action.return_value = s.action
    teleop.send_feedback.side_effect = NotImplementedError
    robot = Mock()
    robot.name = "unitree_g1"
    robot.action_features = dict.fromkeys(ARM_KEYS, float)
    robot.get_observation.return_value = s.observation
    _, robot_processor, obs_processor = make_default_processors()
    teleop_loop(teleop, robot, 30, s.processor, robot_processor, obs_processor, duration=0)
    teleop.send_feedback.assert_not_called()
    robot.send_action.assert_called_once_with(dict.fromkeys(ARM_KEYS, 0.2))


def test_standard_record_saves_accepted_joint_targets(pipeline_state):
    from lerobot.scripts.lerobot_record import record_loop

    s = pipeline_state
    teleop = Mock(spec=Teleoperator)
    teleop.name = "bi_pico4"
    teleop.get_action.return_value = s.action
    teleop.send_feedback.side_effect = NotImplementedError
    robot = Mock()
    robot.name = "unitree_g1"
    robot.get_observation.return_value = s.observation
    robot.send_action.return_value = dict.fromkeys(ARM_KEYS, 0.12)
    dataset = Mock(fps=30)
    dataset.features = {
        key: {"dtype": "float32", "shape": (14,), "names": list(ARM_KEYS)}
        for key in ("action", "observation.state")
    }
    _, robot_processor, obs_processor = make_default_processors()
    record_loop(
        robot,
        {"exit_early": False},
        30,
        s.processor,
        robot_processor,
        obs_processor,
        dataset=dataset,
        teleop=teleop,
        control_time_s=0.001,
        single_task="test",
    )
    teleop.send_feedback.assert_not_called()
    frame = dataset.add_frame.call_args.args[0]
    np.testing.assert_allclose(frame["action"], np.full(14, 0.12))
    np.testing.assert_array_equal(frame["observation.state"], np.zeros(14))


def test_rollout_keeps_g1_joint_state_actions_and_order(monkeypatch):
    import lerobot.rollout.context as context
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    class JointRobot(MockRobot):
        @property
        def action_features(self):
            return dict.fromkeys((*ARM_KEYS, "left_hand.index.pos"), float)

        @property
        def observation_features(self):
            return {**self.action_features, "kLeftElbow.dq": float, "kLeftElbow.tau": float}

        def get_observation(self):
            return {**dict.fromkeys(self.observation_features, 0.0), "kLeftHipPitch.q": 0.0}

    cfg = RolloutConfig(
        robot=MockRobotConfig(),
        device="cpu",
        policy=ACTConfig(device="cpu", pretrained_path=Path("unused-checkpoint")),
    )
    robot = JointRobot(cfg.robot)
    monkeypatch.setattr(context, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(context, "_load_pretrained_policy", lambda _: torch.nn.Linear(15, 15))
    monkeypatch.setattr(
        context,
        "make_pre_post_processors",
        lambda **kwargs: (
            PolicyProcessorPipeline(steps=[]),
            PolicyProcessorPipeline(steps=[]),
        ),
    )
    try:
        ctx = context.build_rollout_context(cfg, threading.Event())
        expected = list(robot.action_features)
        assert ctx.data.ordered_action_keys == expected
        assert ctx.data.dataset_features["action"]["names"] == expected
        assert ctx.data.dataset_features["observation.state"]["names"] == expected
        assert set(ctx.hardware.initial_position) == set(expected)
    finally:
        robot.disconnect()
