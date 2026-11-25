"""Handheld controller pivot and tracking-space heading calibration (metres)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def fit_wrist_offset(poses):
    """Fit the constant controller-local offset whose world position is stationary."""
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 20:
        raise ValueError("Collect at least 20 distinct tracking samples")
    rotations = poses[:, :3, :3]
    positions = poses[:, :3, 3]
    matrix = (rotations - rotations.mean(axis=0)).reshape(-1, 3)
    offset, _, _, singular = np.linalg.lstsq(matrix, (positions.mean(axis=0) - positions).ravel(), rcond=None)
    if singular[-1] / np.sqrt(len(poses)) < 0.08:
        raise ValueError("Rotate about two different axes while holding the wrist still")
    wrists = positions + rotations @ offset
    rms = np.sqrt(np.mean(np.sum((wrists - wrists.mean(axis=0)) ** 2, axis=1)))
    if rms > 0.015 or np.linalg.norm(offset) > 0.35:
        raise ValueError(f"Keep wrist centre stationary and repeat (fit residual {rms * 1000:.1f} mm)")
    return offset, rms


@dataclass
class WristCalibration:
    offsets: np.ndarray
    forward: np.ndarray

    def __post_init__(self):
        self.offsets = np.asarray(self.offsets, dtype=float)
        self.forward = np.array(self.forward, dtype=float, copy=True)
        if (
            self.offsets.shape != (2, 3)
            or not np.isfinite(self.offsets).all()
            or np.any(np.linalg.norm(self.offsets, axis=1) > 0.35)
        ):
            raise ValueError("Invalid controller-local wrist offsets")
        if self.forward.shape != (3,) or not np.isfinite(self.forward).all():
            raise ValueError("Invalid XR forward direction")
        self.forward[1] = 0  # XR gravity-up is +Y.
        length = np.linalg.norm(self.forward)
        if length < 0.08:
            raise ValueError("Forward stroke must travel at least 8 cm horizontally")
        self.forward /= length

    @property
    def world_from_xr(self):
        up = np.array([0, 1, 0])
        return np.stack((self.forward, np.cross(up, self.forward), up))

    @classmethod
    def load(cls, path: Path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing wrist calibration {path}. From the LeRobot checkout run "
                f"python examples/pico4/calibrate_tianji.py --output {path}"
            )
        data = json.loads(path.read_text())
        return cls(data["controller_to_wrist_m"], data["xr_forward"])
