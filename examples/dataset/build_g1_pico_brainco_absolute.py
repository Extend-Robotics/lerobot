"""Create the absolute-action variant and its Flux cloud training config.

Run from the LeRobot checkout with:
uv run --no-sync python examples/dataset/build_g1_pico_brainco_absolute.py
"""

import copy
import hashlib
import json
import shutil

import numpy as np
import pyarrow.parquet as pq
from build_g1_pico_brainco_relative import DATASETS, ROOT, read_json, validate, write_json

from lerobot.datasets.io_utils import write_stats

NAME = "g1_pico_brainco_combined_absolute"
SOURCE = DATASETS / "g1_pico_brainco_combined_relative"
OUTPUT = DATASETS / NAME
PARTIAL = DATASETS / f".{NAME}.building"
CONFIG = ROOT / "configs/flux_cloud_g1_pico_brainco_absolute.json"


def main() -> None:
    if OUTPUT.exists() or PARTIAL.exists() or CONFIG.exists():
        raise FileExistsError("Refusing to overwrite an existing absolute dataset, partial build, or config")
    print("Copying combined recordings without changing action values", flush=True)
    # Independent copies allow either dataset to be edited without changing the other.
    shutil.copytree(SOURCE, PARTIAL, ignore=shutil.ignore_patterns(".cache", "__pycache__"))
    actions = np.concatenate(
        [
            np.stack(pq.read_table(path, columns=["action"])["action"].to_pylist()).astype(np.float64)
            for path in sorted((PARTIAL / "data").glob("*/*.parquet"))
        ]
    )
    assert actions.shape == (1723669, 26) and np.isfinite(actions).all()
    action_stats = {
        "min": actions.min(axis=0),
        "max": actions.max(axis=0),
        "mean": actions.mean(axis=0),
        "std": actions.std(axis=0),
        "count": np.array([len(actions)]),
    }
    for key, value in zip(
        ("q01", "q10", "q50", "q90", "q99"),
        np.quantile(actions, [0.01, 0.1, 0.5, 0.9, 0.99], axis=0),
        strict=True,
    ):
        action_stats[key] = value
    # Check moments with an independent sum-of-squares reduction.
    second_moment = np.einsum("ij,ij->j", actions, actions) / len(actions)
    independent_std = np.sqrt(np.maximum(second_moment - action_stats["mean"] ** 2, 0))
    np.testing.assert_allclose(action_stats["std"], independent_std, rtol=1e-8, atol=1e-9)
    del actions
    stats = read_json(PARTIAL / "meta/stats.json")
    stats["action"] = action_stats
    write_stats(stats, PARTIAL)

    print("Checking copied payloads and dataset references", flush=True)
    files_checked = 0
    for subdir in ("data", "videos", "meta/episodes"):
        for source in sorted((SOURCE / subdir).rglob("*")):
            if not source.is_file():
                continue
            target = PARTIAL / source.relative_to(SOURCE)
            assert source.stat().st_size == target.stat().st_size
            assert not source.samefile(target)
            if source.suffix == ".parquet":
                assert (
                    hashlib.sha256(source.read_bytes()).digest()
                    == hashlib.sha256(target.read_bytes()).digest()
                )
            files_checked += 1
    manifest = read_json(SOURCE / "meta/combination.json")
    validation = validate(PARTIAL, manifest["sources"])
    validation["copied_payload_files_checked"] = files_checked
    for key in ("action_stats_chunks", "action_mean_max_validation_error", "action_std_max_validation_error"):
        manifest.pop(key, None)
    manifest.update(
        repo_id=f"local/{NAME}",
        derived_from="local/g1_pico_brainco_combined_relative",
        stored_actions="absolute",
        use_relative_actions=False,
        relative_exclude_joints=[],
        action_stats_count=int(action_stats["count"][0]),
        statistics="Exact global absolute-action and state statistics and quantiles over all frames; float64 action moments; source-aggregated camera statistics",
        validation=validation,
    )
    write_json(PARTIAL / "meta/combination.json", manifest)
    (PARTIAL / "README.md").write_text(
        f"# {NAME}\n\n"
        "1,663 episodes, 1,723,669 frames, nine tasks, three cameras, 30 FPS.\n\n"
        "Independent copy of g1_pico_brainco_combined_relative with absolute-action normalization statistics. "
        "All 26 stored action channels are unchanged absolute arm/hand targets. "
        "meta/stats.json contains exact global action quantiles and float64 moments over all frames. "
        "State and camera statistics, task labels, and episode/video references are preserved.\n\n"
        "Train Flux with `use_relative_actions=false`, `conditioning=frame`, "
        "`action_representation=absolute`, and QUANTILES action/state normalization. "
        "Use configs/flux_cloud_g1_pico_brainco_absolute.json. No relative-to-absolute reconstruction is needed. "
        "Absolute-action statistics do not depend on the training chunk size; the supplied recipe predicts 50 frames.\n\n"
        "The current Flux action-selection implementation accepts this absolute-action representation. "
        "Training quality, inference latency, and live G1D behavior still require evaluation.\n\n"
        "See meta/combination.json for pinned source revisions, metadata repairs, and validation; "
        "Unitree source cards and Apache-2.0 declarations are retained in meta/source_cards.\n"
    )
    config = copy.deepcopy(read_json(ROOT / "configs/flux_cloud_g1_pico_brainco_relative.json"))
    config["dataset"].update(repo_id=f"local/{NAME}", root=f"/data/datasets/{NAME}")
    config["policy"].update(use_relative_actions=False, relative_exclude_joints=[])
    config.update(job_name=f"flux3_{NAME}", output_dir=f"/data/outputs/flux3_{NAME}")
    PARTIAL.rename(OUTPUT)
    write_json(CONFIG, config)
    print(json.dumps({"dataset": str(OUTPUT), "config": str(CONFIG), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
