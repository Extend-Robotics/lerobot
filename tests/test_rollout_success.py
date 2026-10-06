"""Classifier input compatibility and asynchronous observation lifetime."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.rollout import success


@pytest.fixture
def monitor(monkeypatch):
    predictor = MagicMock(return_value=0.9)
    monkeypatch.setattr(success, "SuccessClassifier", lambda _: predictor)
    now = [0.0]
    monkeypatch.setattr(success.time, "monotonic", lambda: now[0])
    instance = success.SuccessMonitor(success.SuccessDetectionConfig("unused"))
    instance.begin()
    try:
        yield instance, predictor, now
    finally:
        instance.close()


def test_only_autonomous_images_are_sampled_at_half_second_intervals(monitor):
    instance, predictor, now = monitor
    obs = {"front": np.zeros((4, 4, 3), dtype=np.uint8)}
    instance.observe(obs, True)
    predictor.assert_not_called()
    now[0] = 0.5
    instance.observe(obs, False)
    predictor.assert_not_called()
    instance.observe(obs, True)
    instance._future.result(timeout=2)
    assert instance.poll() == 0.9
    instance.observe(obs, True)
    assert predictor.call_count == 1
    now[0] = 1.0
    instance.observe(obs, True)
    instance._future.result(timeout=2)
    assert instance.poll() == 0.9
    assert predictor.call_count == 2


@pytest.mark.parametrize("invalidate", ["phase", "episode"])
def test_discard_inflight_results_after_phase_or_episode_change(monitor, invalidate):
    instance, predictor, now = monitor
    entered, release = Event(), Event()

    def predict(obs):
        entered.set()
        assert release.wait(2)
        assert obs["front"].max() == 0  # Camera buffer was copied.
        return 0.9

    predictor.side_effect = predict
    obs = {"front": np.zeros((4, 4, 3), dtype=np.uint8)}
    now[0] = 0.5
    instance.observe(obs, True)
    try:
        assert entered.wait(2)
        obs["front"][:] = 255
        if invalidate == "phase":
            instance.observe(None, False)
            instance.observe(None, True)
        else:
            instance.end()
            instance.begin()
            instance.observe(None, True)
        now[0] = 2.0
        instance.observe(obs, True)
        assert predictor.call_count == 1  # No backlog behind slow inference.
    finally:
        release.set()
    instance._future.result(timeout=2)
    assert instance.poll() is None


def test_classifier_matches_training_preprocessing_and_camera_order():
    classifier = success.SuccessClassifier.__new__(success.SuccessClassifier)
    classifier.device = "cpu"
    classifier.image_features = {
        "observation.images.front": SimpleNamespace(shape=(3, 128, 128)),
        "observation.images.realsense": SimpleNamespace(shape=(3, 128, 128)),
    }
    classifier.model = MagicMock()
    classifier.model.predict.return_value.probabilities = torch.tensor([0.8])
    probability = classifier(
        {
            "realsense": np.zeros((16, 24, 3), dtype=np.uint8),
            "front": np.full((8, 12, 3), 255, dtype=np.uint8),
        }
    )
    assert probability == pytest.approx(0.8)
    front, realsense = classifier.model.predict.call_args.args[0]
    assert front.shape == realsense.shape == (1, 3, 128, 128)
    assert torch.all(front == 1)
    assert torch.all(realsense == 0)
    with pytest.raises(KeyError, match="realsense"):
        classifier({"front": np.zeros((8, 12, 3), dtype=np.uint8)})


@pytest.mark.parametrize("kwargs", [{"interval_s": 0}, {"interval_s": float("inf")}, {"threshold": 2}])
def test_invalid_success_config(kwargs):
    with pytest.raises(ValueError):
        success.SuccessDetectionConfig("unused", **kwargs)
