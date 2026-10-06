"""Server-owned success detection using the custom success reward classifier."""

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as functional

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.classifier import Classifier, RewardClassifierConfig


@dataclass
class SuccessDetectionConfig:
    pretrained_path: str
    interval_s: float = 0.5
    threshold: float = 0.5
    device: str = "cuda"

    def __post_init__(self):
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError("Success detection interval must be finite and positive")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Success threshold must be between 0 and 1")


class SuccessClassifier:
    """Match examples/dataset/custom/train_success_reward_classifier.py preprocessing."""

    def __init__(self, cfg: SuccessDetectionConfig):
        config = RewardModelConfig.from_pretrained(cfg.pretrained_path)
        if not isinstance(config, RewardClassifierConfig):
            raise ValueError("Expected a reward classifier checkpoint")
        config.device = cfg.device
        self.model = Classifier.from_pretrained(cfg.pretrained_path, config=config, strict=True).eval()
        if config.num_classes != 2:
            raise ValueError("Success detection requires a binary classifier")
        self.image_features = {
            k: v for k, v in config.input_features.items() if k.startswith("observation.images.")
        }
        if not self.image_features:
            raise ValueError("Success classifier has no camera inputs")
        self.device = cfg.device

    def __call__(self, observation: dict) -> float:
        images = []
        for key, feature in self.image_features.items():
            camera = key.removeprefix("observation.images.")
            image = observation[camera]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected uint8 HWC RGB image for {camera}")
            tensor = torch.from_numpy(image).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255
            images.append(
                functional.interpolate(tensor, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            )
        with torch.inference_mode():
            probability = float(self.model.predict(images).probabilities.item())
        if not math.isfinite(probability):
            raise ValueError("Success classifier returned a non-finite probability")
        return probability


class SuccessMonitor:
    """Sample autonomous observations without blocking the control loop on inference.

    One inference may be in flight. A phase change or a new request invalidates its
    result; old work is drained rather than queued ahead of fresh observations.
    """

    def __init__(self, cfg: SuccessDetectionConfig):
        self.cfg = cfg
        self.classifier = SuccessClassifier(cfg)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-success")
        self._lock = Lock()
        self._generation = 0
        self._active = False
        self._autonomous = False
        self._next_check = 0.0
        self._future: Future | None = None
        self._future_generation = 0

    def begin(self) -> None:
        with self._lock:
            self._generation += 1
            self._active = True
            self._autonomous = False
            self._next_check = time.monotonic() + self.cfg.interval_s

    def end(self) -> None:
        with self._lock:
            self._active = False
            self._generation += 1

    def observe(self, observation: dict | None, autonomous: bool) -> None:
        """Called by the control loop; None announces a phase change or loop exit."""
        with self._lock:
            if autonomous != self._autonomous:
                self._generation += 1
                self._autonomous = autonomous
            if not self._active or not autonomous or observation is None:
                return
            if self._future is not None or time.monotonic() < self._next_check:
                return
            # Copy camera buffers before the robot can reuse them on the next tick.
            snapshot = {k: v.copy() for k, v in observation.items() if isinstance(v, np.ndarray)}
            self._future_generation = self._generation
            self._future = self._executor.submit(self.classifier, snapshot)
            self._next_check = time.monotonic() + self.cfg.interval_s

    def poll(self) -> float | None:
        """Return a current success probability, or raise a current inference error."""
        with self._lock:
            if self._future is None or not self._future.done():
                return None
            future, self._future = self._future, None
            if not self._active or not self._autonomous or self._future_generation != self._generation:
                return None
            probability = future.result()
            return probability if probability > self.cfg.threshold else None

    def close(self) -> None:
        self.end()
        self._executor.shutdown(wait=True, cancel_futures=True)
