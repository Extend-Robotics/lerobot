"""Build the local G1/PICO + eight Unitree BrainCo datasets for relative-action training.

Run from the repository root with ``uv run --no-sync python
examples/dataset/build_g1_pico_brainco_relative.py`` after downloading the three
selected camera streams into datasets/.g1_brainco_sources/<task>.
Original recordings are never modified. Stored actions remain absolute.
"""

import copy
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import RunningQuantileStats, _compute_relative_chunk_batch
from lerobot.datasets.feature_utils import canonicalize_depth_marker, get_hf_features_from_features
from lerobot.datasets.io_utils import load_tasks, to_parquet_one_row_group_per_episode, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames, get_video_duration_in_s, get_video_info

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
NAME = "g1_pico_brainco_combined_relative"
OUTPUT = DATASETS / NAME
PARTIAL = DATASETS / f".{NAME}.building"
SOURCES = DATASETS / ".g1_brainco_sources"
STAGING = SOURCES / "normalized"
TASKS = [
    "GraspOreo",
    "GraspRubiksCube",
    "PickApple",
    "PickCharger",
    "PickDoll",
    "PickDrink",
    "PickTissues",
    "PickToothpaste",
]
CAMERAS = {
    "observation.images.cam_left_high": "observation.images.head_camera",
    "observation.images.cam_left_wrist": "observation.images.left_wrist_camera",
    "observation.images.cam_right_wrist": "observation.images.right_wrist_camera",
}
CHUNK_SIZE = 50
EXCLUDE = ["left_hand.", "right_hand."]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.stat().st_size == source.stat().st_size
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def rename_column(column: str) -> str | None:
    if "teleop.grip" in column or "teleop.clutch" in column or "cam_right_high" in column:
        return None
    for old, new in CAMERAS.items():
        column = column.replace(old, new)
    return column


def normalize(source: Path, destination: Path, common: dict, remote: bool) -> dict:
    """Materialize matching schemas, preserving original numeric values and video bytes."""
    info = read_json(source / "meta/info.json")
    assert info["codebase_version"] == "v3.0" and info["fps"] == 30
    assert info["features"]["action"]["shape"] == [26]
    assert info["features"]["observation.state"]["shape"] == [26]
    if remote:
        local_names = common["action"]["names"]
        for key in ("action", "observation.state"):
            names = info["features"][key]["names"]
            if len(names) == 1 and isinstance(names[0], list):
                names = names[0]
            assert [name + ".q" for name in names[:14]] == local_names[:14]
            for side, start in (("Left", 14), ("Right", 20)):
                assert names[start : start + 6] == [
                    f"k{side}Hand{joint}"
                    for joint in ("Thumb", "ThumbAux", "Index", "Middle", "Ring", "Pinky")
                ]

    # Match feature names, shapes, layout labels and scalar dtypes to the local schema.
    features = copy.deepcopy(common)
    for old, new in CAMERAS.items():
        original_key = old if remote else new
        features[new]["info"] = copy.deepcopy(info["features"][original_key]["info"])
        canonicalize_depth_marker(features[new])
    info["features"] = features
    info["robot_type"] = "unitree_g1"
    info["fps"] = 30
    write_json(destination / "meta/info.json", info)

    schema_features = {key: {**value, "shape": tuple(value["shape"])} for key, value in features.items()}
    schema = get_hf_features_from_features(schema_features).arrow_schema
    task_table = load_tasks(source)
    assert len(task_table) == 1 and int(task_table.iloc[0]["task_index"]) == 0
    task = task_table.index[0]
    for path in sorted((source / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=schema.names).cast(schema)
        assert set(table["task_index"].to_pylist()) == {0}
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        to_parquet_one_row_group_per_episode(table.to_pandas(), target)

    repaired_task_labels = 0
    repaired_metadata_pointers = 0
    for path in sorted((source / "meta/episodes").glob("*/*.parquet")):
        frame = pd.read_parquet(path)
        keep = {column: rename_column(column) for column in frame.columns}
        frame = frame[[column for column, renamed in keep.items() if renamed is not None]]
        frame = frame.rename(
            columns={column: renamed for column, renamed in keep.items() if renamed is not None}
        )
        # Previously merged datasets can retain the source metadata-file index.
        # These pointers must describe the physical normalized metadata shard.
        chunk_index = int(path.parent.name.split("-")[1])
        file_index = int(path.stem.split("-")[1])
        repaired_metadata_pointers += int(
            (
                (frame["meta/episodes/chunk_index"] != chunk_index)
                | (frame["meta/episodes/file_index"] != file_index)
            ).sum()
        )
        frame["meta/episodes/chunk_index"] = chunk_index
        frame["meta/episodes/file_index"] = file_index
        # Some remote episode summaries contain a copied red-cup instruction.
        # Their single-task table and per-frame task_index identify the actual task.
        repaired_task_labels += sum(list(value) != [task] for value in frame["tasks"])
        frame["tasks"] = [[task] for _ in range(len(frame))]
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)

    shutil.copy2(source / "meta/tasks.parquet", destination / "meta/tasks.parquet")
    stats = {
        rename_column(key): value
        for key, value in read_json(source / "meta/stats.json").items()
        if rename_column(key) in features
    }
    write_json(destination / "meta/stats.json", stats)
    mapping = CAMERAS if remote else {name: name for name in CAMERAS.values()}
    for old, new in mapping.items():
        for path in sorted((source / "videos" / old).glob("*/*.mp4")):
            target = destination / "videos" / new / path.parent.name / path.name
            link_or_copy(path, target)
    provenance = (
        read_json(source / "download_provenance.json")
        if remote
        else {
            "repo_id": "local/g1_pico_combined",
            "info_sha256": hashlib.sha256((source / "meta/info.json").read_bytes()).hexdigest(),
        }
    )
    provenance.update(episodes=info["total_episodes"], frames=info["total_frames"])
    provenance["metadata_repairs"] = {
        "episode_task_labels": repaired_task_labels,
        "episode_metadata_file_pointers": repaired_metadata_pointers,
    }
    return provenance


def numeric_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["action", "observation.state", "episode_index"])
        arrays.append(
            (
                np.stack(table["action"].to_pylist()).astype(np.float32),
                np.stack(table["observation.state"].to_pylist()).astype(np.float32),
                table["episode_index"].to_numpy(),
            )
        )
    return tuple(np.concatenate([entry[i] for entry in arrays]) for i in range(3))


def recompute(root: Path) -> dict:
    actions, states, episodes = numeric_arrays(root)
    assert np.isfinite(actions).all() and np.isfinite(states).all()
    candidates = np.arange(len(episodes) - CHUNK_SIZE + 1)
    starts = candidates[episodes[candidates] == episodes[candidates + CHUNK_SIZE - 1]]
    mask = np.array([True] * 14 + [False] * 12, dtype=np.float32)
    running = RunningQuantileStats()
    sums = np.zeros(26, dtype=np.float64)
    squares = np.zeros(26, dtype=np.float64)
    for offset in range(0, len(starts), 4096):
        batch = _compute_relative_chunk_batch(
            starts[offset : offset + 4096], actions, states, CHUNK_SIZE, mask
        )
        # Float64 moments avoid large-batch float32 reduction error.
        batch64 = batch.astype(np.float64)
        running.update(batch64)
        sums += batch64.sum(axis=0)
        squares += np.einsum("ij,ij->j", batch64, batch64)
        if offset % (4096 * 20) == 0:
            logging.info("Relative action chunks %s / %s", min(offset + 4096, len(starts)), len(starts))
    action_stats = running.get_statistics()
    count = len(starts) * CHUNK_SIZE
    independent_mean = sums / count
    independent_std = np.sqrt(np.maximum(squares / count - independent_mean**2, 0))
    np.testing.assert_allclose(action_stats["mean"], independent_mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(action_stats["std"], independent_std, rtol=1e-8, atol=1e-9)

    # Exact global state quantiles; camera statistics remain aggregated from source metadata.
    state64 = states.astype(np.float64)
    state_stats = {
        "min": state64.min(0),
        "max": state64.max(0),
        "mean": state64.mean(0),
        "std": state64.std(0),
        "count": np.array([len(states)]),
    }
    for q, value in zip(
        ("q01", "q10", "q50", "q90", "q99"),
        np.quantile(state64, [0.01, 0.1, 0.5, 0.9, 0.99], axis=0),
        strict=True,
    ):
        state_stats[q] = value
    stats = {
        key: value
        for key, value in read_json(root / "meta/stats.json").items()
        if key in {"action", "observation.state", *CAMERAS.values()}
    }
    stats["action"] = action_stats
    stats["observation.state"] = state_stats
    write_stats(stats, root)
    return {
        "action_stats_chunks": len(starts),
        "action_stats_count": count,
        "action_mean_max_validation_error": float(np.max(np.abs(action_stats["mean"] - independent_mean))),
        "action_std_max_validation_error": float(np.max(np.abs(action_stats["std"] - independent_std))),
    }


def validate(root: Path, provenance: list[dict]) -> dict:
    """Check every episode reference and decode samples spanning all source datasets."""
    dataset = LeRobotDataset(
        repo_id=f"local/{NAME}",
        root=root,
        video_backend="pyav",
        delta_timestamps={"action": [i / 30 for i in range(CHUNK_SIZE)]},
    )
    metadata = (
        pd.concat(
            [pd.read_parquet(path) for path in sorted((root / "meta/episodes").glob("*/*.parquet"))],
            ignore_index=True,
        )
        .sort_values("episode_index")
        .reset_index(drop=True)
    )
    assert len(metadata) == sum(source["episodes"] for source in provenance)
    assert dataset.num_frames == sum(source["frames"] for source in provenance)
    np.testing.assert_array_equal(metadata.episode_index, np.arange(len(metadata)))
    np.testing.assert_array_equal(
        metadata.dataset_from_index.to_numpy()[1:], metadata.dataset_to_index.to_numpy()[:-1]
    )
    assert int(metadata.dataset_to_index.iloc[-1]) == dataset.num_frames
    assert dataset.meta.info.total_tasks == 9
    tables = {
        (int(path.parent.name.split("-")[1]), int(path.stem.split("-")[1])): pq.read_table(path)
        for path in sorted((root / "data").glob("*/*.parquet"))
    }
    durations = {}
    for path in sorted((root / "videos").glob("*/*/*.mp4")):
        info = get_video_info(path)
        assert info["video.width"] == 640 and info["video.height"] == 480
        assert abs(info["video.fps"] - 30) < 1e-6
    for row in metadata.to_dict("records"):
        table = tables[(int(row["data/chunk_index"]), int(row["data/file_index"]))]
        selected = table.filter(pa.compute.equal(table["episode_index"], row["episode_index"]))
        assert selected.num_rows == row["length"]
        np.testing.assert_array_equal(selected["frame_index"].to_numpy(), np.arange(row["length"]))
        np.testing.assert_array_equal(
            selected["index"].to_numpy(), np.arange(row["dataset_from_index"], row["dataset_to_index"])
        )
        np.testing.assert_allclose(
            selected["timestamp"].to_numpy(), np.arange(row["length"]) / 30, atol=1e-4, rtol=0
        )
        for key in CAMERAS.values():
            prefix = f"videos/{key}/"
            path = (
                root
                / "videos"
                / key
                / f"chunk-{int(row[prefix + 'chunk_index']):03d}"
                / f"file-{int(row[prefix + 'file_index']):03d}.mp4"
            )
            assert path.is_file()
            if path not in durations:
                durations[path] = get_video_duration_in_s(path)
            assert row[prefix + "from_timestamp"] >= 0
            assert row[prefix + "to_timestamp"] <= durations[path] + 0.1

    # Check every stored action and state against the original recordings.
    combined = pa.concat_tables(list(tables.values()))
    offset = 0
    originals = [DATASETS / "g1_pico_combined"] + [SOURCES / task for task in TASKS]
    for original in originals:
        for path in sorted((original / "data").glob("*/*.parquet")):
            source_table = pq.read_table(path, columns=["action", "observation.state"])
            destination_table = combined.slice(offset, len(source_table))
            for key in ("action", "observation.state"):
                np.testing.assert_array_equal(
                    np.stack(destination_table[key].to_pylist()), np.stack(source_table[key].to_pylist())
                )
            offset += len(source_table)
    assert offset == dataset.num_frames
    source_boundaries = np.cumsum([0] + [source["episodes"] for source in provenance])
    decoded = 0
    for start, end in zip(source_boundaries[:-1], source_boundaries[1:], strict=True):
        for ep_index in (int(start), int(end - 1)):
            row = metadata.iloc[ep_index]
            # Exercise the real LeRobot loader and a full action chunk.
            sample = dataset[int(row.dataset_from_index)]
            assert tuple(sample["action"].shape) == (CHUNK_SIZE, 26)
            assert "task" in sample
            for key in CAMERAS.values():
                assert tuple(sample[key].shape) == (3, 480, 640)
                prefix = f"videos/{key}/"
                path = (
                    root
                    / "videos"
                    / key
                    / f"chunk-{int(row[prefix + 'chunk_index']):03d}"
                    / f"file-{int(row[prefix + 'file_index']):03d}.mp4"
                )
                timestamp = float(row[prefix + "from_timestamp"]) + (int(row.length) - 1) / 30
                frames = decode_video_frames(path, [timestamp], tolerance_s=1e-3, backend="pyav")
                assert tuple(frames.shape) == (1, 3, 480, 640)
                decoded += 2
    return {
        "episodes_checked": len(metadata),
        "video_files_checked": len(durations),
        "decoded_frame_checks": decoded,
        "training_samples_checked": (len(source_boundaries) - 1) * 2,
        "action_and_state_rows_compared_to_originals": offset,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    local = DATASETS / "g1_pico_combined"
    common = {
        key: value
        for key, value in read_json(local / "meta/info.json")["features"].items()
        if not key.startswith("teleop.")
    }
    entries = [(local, STAGING / "pico", False)] + [(SOURCES / task, STAGING / task, True) for task in TASKS]
    provenance = []
    for source, destination, remote in entries:
        if remote and not (source / "download_provenance.json").exists():
            raise FileNotFoundError(f"Download not complete: {source}")
        logging.info("Normalizing %s", source.name)
        provenance.append(normalize(source, destination, common, remote))
    if not (PARTIAL / "aggregation_complete.json").exists():
        if PARTIAL.exists():
            # Only our explicitly named, incomplete build directory may be replaced.
            shutil.rmtree(PARTIAL)
        aggregate_datasets(
            repo_ids=[source["repo_id"] for source in provenance],
            roots=[entry[1] for entry in entries],
            aggr_repo_id=f"local/{NAME}",
            aggr_root=PARTIAL,
            concatenate_videos=False,
            concatenate_data=False,
        )
        write_json(PARTIAL / "aggregation_complete.json", {"sources": provenance})
    logging.info("Recomputing relative-action statistics")
    stats_report = recompute(PARTIAL)
    logging.info("Validating combined dataset")
    validation = validate(PARTIAL, provenance)
    manifest = {
        "repo_id": f"local/{NAME}",
        "sources": provenance,
        "episodes": 1663,
        "frames": 1723669,
        "fps": 30,
        "stored_actions": "absolute",
        "use_relative_actions": True,
        "relative_exclude_joints": EXCLUDE,
        "chunk_size": CHUNK_SIZE,
        "camera_mapping": CAMERAS,
        "omitted_camera": "observation.images.cam_right_high",
        "omitted_features": ["teleop.grip", "teleop.clutch"],
        "thumb_mapping": "driver order confirmed by user",
        "statistics": "All valid within-episode action chunks; arm targets minus chunk-start state; hands absolute; float64 moments; LeRobot histogram action quantiles; exact global state statistics; source-aggregated camera statistics",
        "validation": validation,
        **stats_report,
    }
    write_json(PARTIAL / "meta/combination.json", manifest)
    for task in TASKS:
        card = PARTIAL / "meta/source_cards" / f"G1_Brainco_{task}_Dataset.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCES / task / "README.md", card)
    (PARTIAL / "aggregation_complete.json").unlink()
    (PARTIAL / "README.md").write_text(
        f"# {NAME}\n\n1,663 episodes, 1,723,669 frames, 30 FPS, three cameras, nine tasks.\n\n"
        "Combines g1_pico_combined with all eight Unitree G1 BrainCo collection datasets. "
        "Remote cam_left_high is head_camera; both wrist streams are mapped by side. "
        "cam_right_high and local teleoperation diagnostics are omitted.\n\n"
        "## Training\n\nStored actions remain **absolute**. Enable the LeRobot training processor:\n\n"
        '```json\n{"use_relative_actions": true, "relative_exclude_joints": ["left_hand.", "right_hand."], "chunk_size": 50}\n```\n\n'
        "meta/stats.json contains chunk-relative arm action statistics and absolute hand statistics. "
        "Each arm target in a chunk is relative to the observation at the chunk start. "
        "Do not subtract state again in stored recordings. Changing chunk size requires recalculating action statistics.\n\n"
        "Episode metadata retains source statistics for absolute stored recordings. Global action statistics "
        "were recalculated over all valid within-episode chunks; state statistics use all frames. "
        "Camera statistics are aggregated from source metadata. Videos are copied without re-encoding.\n\n"
        "See meta/combination.json for pinned remote revisions, source episode/frame counts, mappings, and validation. "
        "The eight Unitree source datasets declare Apache-2.0; retain their attribution when redistributing. "
        "Local recordings account for about 3.9% of frames; choose training sampling accordingly.\n"
    )
    config = read_json(ROOT / "configs/pi05_local_g1d.json")
    config["dataset"].update(repo_id=f"local/{NAME}", root=f"datasets/{NAME}")
    config["output_dir"] = f"outputs/pi05_{NAME}"
    config["job_name"] = f"pi05_{NAME}"
    config["policy"].update(use_relative_actions=True, relative_exclude_joints=EXCLUDE, chunk_size=CHUNK_SIZE)
    PARTIAL.rename(OUTPUT)
    write_json(ROOT / "configs/pi05_local_g1_pico_brainco_relative.json", config)
    logging.info("COMPLETE %s\n%s", OUTPUT, json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
