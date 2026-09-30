"""Compute per-timestamp action normalization statistics and print JSON.

Run from the workspace root (dataset paths are relative to the working directory):
    uv run --extra scipy-dep src/lerobot/scripts/action_chunk_stats.py \
        --config_path=configs/pi05_leyland_merged.json

The iterator yields unnormalized actions shaped [horizon, action_dim] and a
boolean padding mask shaped [horizon]. True means a repeated boundary value.
Only training episodes and eligible training start frames are used, once each.
No images are decoded, model weights loaded, or dataset statistics overwritten.

Uses the per-timestamp recipe from IliaLarchenko/lehome_solution's
scripts/compute_norm_stats.py: linear mean, shifted-sqrt std, and rescaled
1st/99th percentiles. Unlike that reference, padded entries are excluded.
Exact percentiles require keeping all action chunks and masks in CPU memory.
All statistics retain shape [horizon, action_dim]; count has shape [horizon].
"""

import json
import math
import warnings
from collections.abc import Iterator

import numpy as np
import torch

import lerobot.policies  # noqa: F401 — register policy config types for CLI parsing
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _scipy_available, register_third_party_plugins, require_package

if _scipy_available:
    from scipy.optimize import minimize


def iter_action_chunks(
    cfg: TrainPipelineConfig, dataset: LeRobotDataset
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (actions, action_is_pad), retaining each chunk's horizon axis.

    Pass the training dataset returned by make_train_eval_datasets(cfg).
    Action offsets come from the policy, not its inference execution length.
    Boundary requests repeat the first/last action within the same episode.
    Dropping final start frames does NOT remove those actions from earlier chunks.

    Relative actions, when enabled, subtract the anchor frame's state from every
    action in the chunk, respecting relative_exclude_joints. Other policy-specific
    representation transforms are not applied; values are never normalized.
    """
    policy = cfg.policy
    if policy is None or cfg.reward_model is not None:
        raise ValueError("An action policy config is required.")
    offsets = policy.action_delta_indices
    if offsets is None or len(offsets) == 0:
        raise ValueError("The policy must define nonempty action_delta_indices.")
    offsets = torch.tensor(offsets, dtype=torch.long)
    drop_last = getattr(policy, "drop_n_last_frames", 0)
    if drop_last < 0:
        raise ValueError("drop_n_last_frames must be nonnegative.")

    def source_key(policy_key: str) -> str:
        matches = [key for key in dataset.features if cfg.rename_map.get(key, key) == policy_key]
        if len(matches) != 1:
            raise ValueError(f"Expected one dataset feature mapped to {policy_key}, got {matches}.")
        return matches[0]

    action_key = source_key(ACTION)
    relative = getattr(policy, "use_relative_actions", False)
    state_key = source_key(OBS_STATE) if relative else None
    converter = RelativeActionsProcessorStep(
        enabled=relative,
        exclude_joints=getattr(policy, "relative_exclude_joints", []),
        action_names=dataset.features[action_key].get("names"),
    )
    columns = [action_key] + ([state_key] if state_key else [])
    table = dataset.hf_dataset.select_columns(columns)
    index_map = dataset.absolute_to_relative_idx
    episodes = dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes)
    yielded = False

    for episode_index in episodes:
        episode = dataset.meta.episodes[episode_index]
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        length = end - start  # end is exclusive
        if length <= drop_last:
            continue
        # Episode filtering changes table row positions, but not metadata indices.
        rows = [index_map[i] if index_map is not None else i for i in range(start, end)]
        values = table[rows]
        actions = torch.stack([torch.as_tensor(value) for value in values[action_key]])
        states = values[state_key] if state_key else None

        for anchor in range(length - drop_last):
            requested = anchor + offsets
            is_pad = (requested < 0) | (requested >= length)
            chunk = actions[requested.clamp(0, length - 1)]
            if states is not None:
                transition = {
                    TransitionKey.ACTION: chunk.unsqueeze(0),
                    TransitionKey.OBSERVATION: {OBS_STATE: torch.as_tensor(states[anchor]).unsqueeze(0)},
                }
                chunk = converter(transition)[TransitionKey.ACTION].squeeze(0)
            yielded = True
            yield chunk, is_pad

    if not yielded:
        raise ValueError("No eligible training chunks remain after episode selection and frame dropping.")


def fit_per_timestamp(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean=a+b*t and std=a+s*sqrt(t+e), independently per action dimension.

    t is the zero-based chunk position, as in the reference. Missing positions
    stay NaN and do not influence the fits. With fewer than three observations,
    fix e=1 because three free parameters cannot be determined.
    """
    require_package("scipy", "scipy-dep")
    t = np.arange(len(mean), dtype=np.float64)
    valid = np.isfinite(mean).all(axis=1) & np.isfinite(std).all(axis=1)
    if not valid.any():
        raise ValueError("No valid action positions available for fitting.")
    t = t[valid]
    linear_basis = np.column_stack([np.ones(len(t)), t])
    sqrt_basis = np.column_stack([np.ones(len(t)), np.sqrt(t + 1)])
    fitted_mean = np.full_like(mean, np.nan)
    fitted_std = np.full_like(std, np.nan)
    fitted_mean[valid] = linear_basis @ np.linalg.lstsq(linear_basis, mean[valid], rcond=None)[0]
    warm_start = np.linalg.lstsq(sqrt_basis, std[valid], rcond=None)[0]

    for dimension in range(std.shape[1]):
        target = std[valid, dimension]
        if np.ptp(target) < 1e-12:
            fitted_std[valid, dimension] = target.mean()
            continue

        def predict(params):
            offset, scale, log_e = params
            return offset + scale * np.sqrt(t + np.exp(log_e))

        initial = [*warm_start[:, dimension], 0.0]
        if len(t) < 3:
            prediction = predict(initial)
        else:
            result = minimize(
                lambda params: np.sum((target - predict(params)) ** 2),
                initial,
                method="L-BFGS-B",
                # Keep exp(log_e) finite while allowing a broad range of curvature.
                bounds=[(None, None), (None, None), (-20, 20)],
            )
            if not result.success or not np.isfinite(result.fun):
                raise ValueError(f"Std fit failed for action dimension {dimension}: {result.message}")
            prediction = predict(result.x)
        fitted_std[valid, dimension] = prediction

    fitted_std = np.maximum(fitted_std, 1e-6)
    std_ratio = fitted_std[valid] / np.maximum(std[valid], 1e-12)
    if np.any(std_ratio < 0.1):
        raise ValueError("Fitted std is below 0.1 times raw std; normalization would be unstable.")
    mean_ratio = np.abs(fitted_mean[valid]) / np.maximum(np.abs(mean[valid]), 1e-12)
    mean_off = (np.abs(mean[valid]) > 0.01) & ((mean_ratio > 2) | (mean_ratio < 0.5))
    std_off = (std[valid] > 1e-6) & ((std_ratio > 2) | (std_ratio < 0.5))
    if mean_off.any() or std_off.any():
        warnings.warn("Some fitted values differ from raw statistics by more than 2x.", stacklevel=2)
    return fitted_mean, fitted_std


def compute_stats(chunks: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, np.ndarray]:
    """Compute exact masked statistics, then the reference's smooth normalization.

    mean/std/q01/q99 are raw per-timestamp statistics (population std, ddof=0).
    per_timestamp_* are the fitted values to use for normalization. Positions
    without valid samples remain NaN, serialized as null rather than extrapolated.
    """
    require_package("scipy", "scipy-dep")
    samples = list(chunks)
    if not samples:
        raise ValueError("No chunks supplied.")
    actions = np.stack([action.cpu().numpy() for action, _ in samples])
    valid = ~np.stack([is_pad.cpu().numpy() for _, is_pad in samples])
    del samples
    if actions.ndim != 3 or valid.shape != actions.shape[:2] or valid.dtype != np.bool_:
        raise ValueError("Expected actions [N,H,D] and boolean padding masks [N,H].")
    shape = actions.shape[1:]
    stats = {name: np.full(shape, np.nan) for name in ("mean", "std", "q01", "q99")}
    stats["count"] = valid.sum(axis=0)
    for timestamp in range(shape[0]):
        values = actions[valid[:, timestamp], timestamp].astype(np.float64)
        if not len(values):
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite action at chunk position {timestamp}.")
        stats["mean"][timestamp] = values.mean(axis=0)
        stats["std"][timestamp] = values.std(axis=0)
        stats["q01"][timestamp], stats["q99"][timestamp] = np.percentile(values, [1, 99], axis=0)

    mean, std = fit_per_timestamp(stats["mean"], stats["std"])
    stats["per_timestamp_mean"] = mean
    stats["per_timestamp_std"] = std
    ratio = std / np.maximum(stats["std"], 1e-8)
    for name in ("q01", "q99"):
        stats[f"per_timestamp_{name}"] = mean + (stats[name] - stats["mean"]) * ratio
    return stats


@parser.wrap()
def main(cfg: TrainPipelineConfig) -> None:
    # Resolve policy.path without training validation (which checks output dirs).
    cfg._resolve_pretrained_from_cli()
    if cfg.dataset.streaming or not isinstance(cfg.dataset.repo_id, str):
        raise ValueError("This script requires one non-streaming LeRobot dataset.")
    dataset, _ = make_train_eval_datasets(cfg)
    stats = compute_stats(iter_action_chunks(cfg, dataset))
    # JSON null denotes a horizon position with no valid observations.
    result = {"action_delta_indices": cfg.policy.action_delta_indices, "count": stats["count"].tolist()}
    for name in stats:
        if name == "count":
            continue
        result[name] = [[v if math.isfinite(v) else None for v in row] for row in stats[name].tolist()]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    register_third_party_plugins()
    main()
