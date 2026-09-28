"""Weights-only initialization must rebuild processors, but resume must restore them."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts import lerobot_train


@pytest.mark.parametrize(
    ("load_saved", "resume", "expect_saved"),
    [(True, False, True), (False, False, False), (False, True, True)],
)
def test_training_processor_source(monkeypatch, load_saved, resume, expect_saved):
    policy_cfg = PI05Config(pretrained_path="checkpoint", device="cpu")
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test"),
        policy=policy_cfg,
        load_pretrained_processors=load_saved,
        resume=resume,
    )
    stats = {"action": {"q01": [0.0], "q99": [1.0]}}
    dataset = SimpleNamespace(meta=SimpleNamespace(stats=stats))
    factory = Mock(return_value=("pre", "post"))
    monkeypatch.setattr(lerobot_train, "make_pre_post_processors", factory)

    assert lerobot_train._make_training_processors(
        cfg, SimpleNamespace(config=policy_cfg), dataset, torch.device("cpu")
    ) == ("pre", "post")
    kwargs = factory.call_args.kwargs
    assert kwargs["pretrained_path"] == ("checkpoint" if expect_saved else None)
    assert policy_cfg.pretrained_path == "checkpoint"
    if not expect_saved:
        assert kwargs["dataset_stats"] == stats
        assert "preprocessor_overrides" not in kwargs
        assert "postprocessor_overrides" not in kwargs
    elif resume:
        assert "dataset_stats" not in kwargs
        assert "stats" not in kwargs["preprocessor_overrides"]["normalizer_processor"]
        assert "stats" not in kwargs["postprocessor_overrides"]["unnormalizer_processor"]
    else:
        assert kwargs["preprocessor_overrides"]["normalizer_processor"]["stats"] == stats
