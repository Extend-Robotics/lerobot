#!/usr/bin/env python
"""Report local v3 episode outcomes and optionally create a reward-labeled copy.

Success is inferred from episode length, not visually verified. The default
failure threshold includes the 445/446-frame timeout cluster in Leyland rollouts.
Only metadata-designated rows are copied; stale data fragments are reported.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
REWARD = "next.reward"


def reward_values(length: int, success: bool, num_samples: int) -> np.ndarray:
    values = np.zeros(length, dtype=np.float32)
    if success:
        values[max(0, length - num_samples) :] = 1
    return values


def reward_stats(values: np.ndarray) -> dict:
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.astype(np.float64).mean())],
        "std": [float(values.astype(np.float64).std())],
        "count": [len(values)],
        **{f"q{q:02d}": [float(np.quantile(values, q / 100))] for q in (1, 10, 50, 90, 99)},
    }


def inspect_dataset(root: Path, failure_min_frames: int):
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("This script requires a local LeRobot v3 dataset.")
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    episodes = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    episodes = episodes.sort_values("episode_index")
    if episodes.episode_index.duplicated().any() or len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata count or indices are inconsistent.")
    tables = {p.relative_to(root): pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))}
    selected = {p: [] for p in tables}
    physical_counts = {}
    for table in tables.values():
        for ep, count in table.column("episode_index").to_pandas().value_counts().items():
            physical_counts[int(ep)] = physical_counts.get(int(ep), 0) + int(count)
    rows = []
    for _, episode in episodes.iterrows():
        ep, length = int(episode.episode_index), int(episode.length)
        path = Path(info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"])
        ))
        table = tables[path]
        indices = table.column("index").to_numpy()
        mask = (table.column("episode_index").to_numpy() == ep)
        mask &= (indices >= int(episode.dataset_from_index)) & (indices < int(episode.dataset_to_index))
        positions = np.flatnonzero(mask)
        data = table.take(pa.array(positions))
        if length <= 0 or len(data) != length:
            raise ValueError(f"Episode {ep}: metadata length does not match designated data.")
        if not np.array_equal(data.column("frame_index").to_numpy(), np.arange(length)):
            raise ValueError(f"Episode {ep}: frame indices are not contiguous.")
        if not np.array_equal(data.column("index").to_numpy(), np.arange(
            int(episode.dataset_from_index), int(episode.dataset_to_index)
        )):
            raise ValueError(f"Episode {ep}: global indices are not contiguous.")
        selected[path].extend(positions.tolist())
        timestamps = data.column("timestamp").to_numpy()
        success = length < failure_min_frames
        rows.append({
            "episode_index": ep, "length_frames": length,
            "duration_s": length / info["fps"],
            "first_timestamp_s": float(timestamps[0]), "last_timestamp_s": float(timestamps[-1]),
            "success": success, "outcome": "successful" if success else "unsuccessful",
            "label_basis": "length_below_timeout" if success else "length_at_or_above_timeout",
            "failure_min_frames": failure_min_frames, "data_file": str(path),
            "physical_rows": physical_counts[ep], "stale_rows": physical_counts[ep] - length,
        })
    report = pd.DataFrame(rows)
    if report.length_frames.sum() != info["total_frames"]:
        raise ValueError("Metadata total_frames does not match episode lengths.")
    canonical = {p: tables[p].take(pa.array(sorted(positions), type=pa.int64())) for p, positions in selected.items()}
    indices = np.concatenate([t.column("index").to_numpy() for t in canonical.values()])
    if not np.array_equal(np.sort(indices), np.arange(info["total_frames"])):
        raise ValueError("Canonical dataset indices have gaps or duplicates.")
    return info, episode_files, canonical, report


def annotate(root, output, info, episode_files, tables, report, num_samples):
    if output.exists() or output == root or root in output.parents or output in root.parents:
        raise ValueError("Output must be a new directory outside the source dataset.")
    if REWARD in info["features"]:
        raise ValueError(f"Source already has {REWARD}.")
    rewards = {int(row.episode_index): reward_values(int(row.length_frames), bool(row.success), num_samples)
               for row in report.itertuples()}
    stats = {ep: reward_stats(values) for ep, values in rewards.items()}
    shutil.copytree(root, output)
    for path, table in tables.items():
        values = np.array([rewards[int(ep)][int(frame)] for ep, frame in zip(
            table.column("episode_index").to_numpy(), table.column("frame_index").to_numpy(), strict=True
        )], dtype=np.float32)
        table = table.append_column(REWARD, pa.array(values))
        # Avoid stale Hugging Face schema metadata omitting the added column.
        metadata = dict(table.schema.metadata or {})
        metadata.pop(b"huggingface", None)
        pq.write_table(table.replace_schema_metadata(metadata), output / path)
    for path in episode_files:
        table = pq.read_table(path)
        for key in next(iter(stats.values())):
            table = table.append_column(f"stats/{REWARD}/{key}", pa.array([
                stats[int(ep)][key] for ep in table.column("episode_index").to_pylist()
            ]))
        pq.write_table(table, output / path.relative_to(root))
    info["features"][REWARD] = {"dtype": "float32", "shape": [1], "names": None}
    (output / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    global_stats = json.loads((root / "meta/stats.json").read_text())
    global_stats[REWARD] = reward_stats(np.concatenate(list(rewards.values())))
    (output / "meta/stats.json").write_text(json.dumps(global_stats, indent=4) + "\n")
    print(f"Created {output}; positive reward frames: {sum(int(v.sum()) for v in rewards.values())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets/rollout_leyland_merged_500")
    parser.add_argument("--failure-min-frames", type=int, default=445,
                        help="Episodes at or above this length are unsuccessful (default: 445).")
    parser.add_argument("--num-reward-samples", type=int, default=5)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--annotate", action="store_true", help="Also create an annotated dataset copy.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.failure_min_frames <= 0 or args.num_reward_samples <= 0:
        parser.error("Frame threshold and reward sample count must be positive.")
    root = args.dataset_root.resolve()
    report_dir = args.report_dir or REPO_ROOT / "outputs" / f"{root.name}_episode_success"
    info, episode_files, tables, report = inspect_dataset(root, args.failure_min_frames)
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_dir / "episodes.csv", index=False)
    for outcome in ("successful", "unsuccessful"):
        report[report.outcome == outcome].to_csv(report_dir / f"{outcome}_episodes.csv", index=False)
    summary = report.groupby("outcome").agg(
        episodes=("episode_index", "count"), total_frames=("length_frames", "sum"),
        min_duration_s=("duration_s", "min"), mean_duration_s=("duration_s", "mean"),
        max_duration_s=("duration_s", "max"), stale_rows=("stale_rows", "sum"),
    )
    summary.to_csv(report_dir / "summary.csv")
    print(summary.to_string())
    print(f"Reports: {report_dir}. Labels are inferred from duration.")
    print(f"Physical rows: {report.physical_rows.sum()}; canonical frames: {info['total_frames']}; "
          f"stale rows excluded from annotation: {report.stale_rows.sum()}")
    if args.annotate:
        output = (args.output_dir or root.with_name(root.name + "_with_success_reward")).resolve()
        annotate(root, output, info, episode_files, tables, report, args.num_reward_samples)


if __name__ == "__main__":
    main()
