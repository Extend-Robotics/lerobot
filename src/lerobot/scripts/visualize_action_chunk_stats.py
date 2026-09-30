"""Plot per-timestamp statistics with 100 uniformly sampled training chunks.

From the lerobot directory:
    uv run --extra scipy-dep --extra matplotlib-dep \
        src/lerobot/scripts/visualize_action_chunk_stats.py \
        --config_path=configs/pi05_leyland_merged.json \
        --plot_output=action_chunk_stats.png

Optionally pass --stats_path=stats.json to load action_chunk_stats.py's JSON
instead of recomputing statistics. Use the same dataset and policy config that
produced that file. Without it, statistics use ALL eligible chunks, not just
the 100 displayed samples. Sampling is uniform over chunk starts, without
replacement; short datasets display all available chunks.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.scripts.action_chunk_stats import compute_stats, iter_action_chunks
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins, require_package


@dataclass
class PlotConfig(TrainPipelineConfig):
    stats_path: Path | None = None
    plot_output: Path = Path("action_chunk_stats.png")
    sample_seed: int = 42


def sample_chunks(
    chunks: Iterator[tuple[torch.Tensor, torch.Tensor]],
    samples: list[np.ndarray],
    seed: int = 42,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Pass through all chunks while keeping a uniform reservoir of at most 100.

    Samples are independent copies with padding replaced by NaN for plotting.
    The original actions and masks are yielded unchanged for statistics.
    """
    rng = np.random.default_rng(seed)
    for index, (actions, is_pad) in enumerate(chunks):
        slot = index if index < 100 else int(rng.integers(index + 1))
        if slot < 100:
            sample = actions.cpu().numpy().astype(np.float64, copy=True)
            sample[is_pad.cpu().numpy()] = np.nan
            if index < 100:
                samples.append(sample)
            else:
                samples[slot] = sample
        yield actions, is_pad


def plot_stats(
    stats: dict[str, np.ndarray],
    samples: list[np.ndarray],
    offsets: list[int],
    action_names: list[str],
    output: Path,
) -> None:
    """One row per action dimension: action distributions and std-fit comparison."""
    require_package("matplotlib", "matplotlib-dep")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not samples:
        raise ValueError("No action chunks available to plot.")
    chunks = np.stack(samples)
    horizon, dimensions = chunks.shape[1:]
    for key in ("mean", "std", "q01", "q99", "per_timestamp_mean", "per_timestamp_std",
                "per_timestamp_q01", "per_timestamp_q99"):
        if key not in stats or stats[key].shape != (horizon, dimensions):
            raise ValueError(f"Statistics field {key!r} must have shape {(horizon, dimensions)}.")
    if len(offsets) != horizon:
        raise ValueError("Action offsets do not match the sampled horizon.")

    figure = Figure(figsize=(15, 3 * dimensions + 1.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(dimensions, 2, squeeze=False, sharex=True)
    for dimension, (action_ax, std_ax) in enumerate(axes):
        action_ax.plot(offsets, chunks[:, :, dimension].T, color="0.5", alpha=0.16, linewidth=0.7)
        # A single legend entry for the entire collection of sampled trajectories.
        action_ax.plot([], [], color="0.5", alpha=0.6, label=f"{len(samples)} sampled chunks")
        raw_mean = stats["mean"][:, dimension]
        mean = stats["per_timestamp_mean"][:, dimension]
        std = stats["per_timestamp_std"][:, dimension]
        action_ax.fill_between(offsets, mean - std, mean + std, color="tab:orange", alpha=0.18,
                               label="Fitted mean ± std")
        action_ax.plot(offsets, raw_mean, color="tab:blue", linestyle="--", label="Raw mean")
        action_ax.plot(offsets, mean, color="tab:orange", linewidth=2, label="Fitted mean")
        for quantile in ("q01", "q99"):
            action_ax.plot(offsets, stats[quantile][:, dimension], color="tab:blue", linestyle=":",
                           label="Raw q01/q99" if quantile == "q01" else None)
            action_ax.plot(offsets, stats[f"per_timestamp_{quantile}"][:, dimension], color="tab:orange",
                           linestyle=":", label="Adjusted q01/q99" if quantile == "q01" else None)
        std_ax.plot(offsets, stats["std"][:, dimension], color="tab:blue", linestyle="--", label="Raw std")
        std_ax.plot(offsets, std, color="tab:orange", linewidth=2, label="Fitted std")
        name = action_names[dimension] if dimension < len(action_names) else f"Action {dimension}"
        action_ax.set_title(name)
        std_ax.set_title(f"{name} — standard deviation")
        action_ax.set_ylabel("Action value (dataset units)")
        std_ax.set_ylabel("Standard deviation")
        for axis in (action_ax, std_ax):
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Action offset from chunk anchor (frames)")
    axes[0, 1].legend(fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=9)
    figure.suptitle("Per-timestamp action statistics · padded positions excluded", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)


@parser.wrap()
def main(cfg: PlotConfig) -> None:
    require_package("matplotlib", "matplotlib-dep")
    cfg._resolve_pretrained_from_cli()
    if cfg.dataset.streaming or not isinstance(cfg.dataset.repo_id, str):
        raise ValueError("This script requires one non-streaming LeRobot dataset.")
    dataset, _ = make_train_eval_datasets(cfg)
    samples = []
    chunks = sample_chunks(iter_action_chunks(cfg, dataset), samples, cfg.sample_seed)
    offsets = cfg.policy.action_delta_indices
    if cfg.stats_path is None:
        stats = compute_stats(chunks)
    else:
        with cfg.stats_path.open() as file:
            saved = json.load(file)
        if saved.get("action_delta_indices") != offsets:
            raise ValueError("Saved statistics and config have different action offsets.")
        stats = {key: np.asarray(value, dtype=np.float64) for key, value in saved.items()}
        for _ in chunks:
            pass  # Traverse every eligible start to obtain an unbiased reservoir.

    action_key = next(key for key in dataset.features if cfg.rename_map.get(key, key) == ACTION)
    names = dataset.features[action_key].get("names") or []
    plot_stats(stats, samples, offsets, names, cfg.plot_output)
    print(f"Saved {cfg.plot_output.resolve()} with {len(samples)} sampled chunks (seed={cfg.sample_seed}).")


if __name__ == "__main__":
    register_third_party_plugins()
    main()
