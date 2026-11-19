#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export episode preview images and a CSV of episode lengths and stationary percentages.

Run from the LeRobot checkout with its dependencies installed:

    uv run examples/dataset/inspect_dataset.py /path/to/dataset /path/to/images
    uv run examples/dataset/inspect_dataset.py user/dataset ./images \
        --camera observation.images.front

Output: <output_dir>/<camera_key>/episode_000000.png, ...
Each PNG has twice the original frame width and a header showing the episode
length in frames and seconds, plus stationary percentage, above the original-height frames.
Also writes <output_dir>/episode_lengths.csv with episode_index,length,stationary_percent
sorted by length in frames (shortest first), then by episode index.
Stationary samples have all observation.state changes <= --motion-threshold
in the dataset's state units. The first sample is excluded; single-frame episodes
have a blank stationary percentage because motion cannot be measured.
Requires a dataset format supported by the installed LeRobot version.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset


def inspect_dataset(
    dataset: LeRobotDataset,
    output_dir: Path,
    camera: str | None = None,
    overwrite: bool = False,
    motion_threshold: float = 0.001,
) -> int:
    """Export first/last frame pairs and episode statistics; return the image count."""
    if not math.isfinite(motion_threshold) or motion_threshold < 0:
        raise ValueError("Motion threshold must be finite and nonnegative.")
    if "observation.state" not in dataset.features:
        raise ValueError("The dataset needs observation.state to measure motion.")
    state_data = dataset.select_columns("observation.state")
    camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise ValueError("The dataset has no camera images.")
    if camera is not None:
        if camera not in camera_keys:
            raise ValueError(f"Unknown camera {camera!r}. Available cameras: {', '.join(camera_keys)}")
        camera_keys = [camera]

    for key in camera_keys:
        # Camera keys become directory names, so reject path components.
        if Path(key).name != key or key in {".", ".."}:
            raise ValueError(f"Camera key cannot be used as a directory name: {key!r}")
        (output_dir / key).mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "episode_lengths.csv"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"{csv_path} already exists. Use --overwrite to replace it.")

    episode_lengths: list[tuple[int, int, str]] = []
    saved = 0
    for episode in tqdm(dataset.meta.episodes, desc="Exporting first/last frames"):
        episode_index = int(episode["episode_index"])
        paths = {key: output_dir / key / f"episode_{episode_index:06d}.png" for key in camera_keys}
        if not overwrite:
            existing = next((path for path in paths.values() if path.exists()), None)
            if existing is not None:
                raise FileExistsError(f"{existing} already exists. Use --overwrite to replace images.")

        first_index = int(episode["dataset_from_index"])
        # The metadata's end index is exclusive.
        last_index = int(episode["dataset_to_index"]) - 1
        episode_length = last_index - first_index + 1
        states = np.asarray(state_data[first_index : last_index + 1]["observation.state"], dtype=np.float64)
        if not np.isfinite(states).all():
            raise ValueError(f"Episode {episode_index} contains non-finite observation.state values.")
        stationary_percent = ""
        if episode_length > 1:
            stationary = (np.abs(np.diff(states, axis=0)) <= motion_threshold).all(axis=1)
            stationary_percent = f"{100 * stationary.mean():.2f}"
        episode_lengths.append((episode_index, episode_length, stationary_percent))
        length_label = f"Episode {episode_index}: {episode_length} frames ({episode_length / dataset.fps:.2f} s)"
        stationary_label = f"Stationary: {stationary_percent}%" if stationary_percent else "Stationary: N/A"
        first_frame = dataset[first_index]
        last_frame = dataset[last_index] if last_index != first_index else first_frame
        for key, path in paths.items():
            # LeRobot returns RGB tensors in (C, H, W) layout with values in [0, 1].
            first_image = to_pil_image(first_frame[key]).convert("RGB")
            last_image = to_pil_image(last_frame[key]).convert("RGB")
            header_height = 44
            combined = Image.new("RGB", (first_image.width * 2, first_image.height + header_height))
            combined.paste(first_image, (0, header_height))
            combined.paste(last_image, (first_image.width, header_height))
            draw = ImageDraw.Draw(combined)
            draw.text((8, 8), length_label, fill="white")
            draw.text((8, 24), stationary_label, fill="white")
            combined.save(path)
            saved += 1

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode_index", "length", "stationary_percent"])
        writer.writerows(sorted(episode_lengths, key=lambda item: (item[1], item[0])))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Local dataset directory or Hugging Face dataset ID (user/name).")
    parser.add_argument("output_dir", type=Path, help="Directory in which to save PNG images.")
    parser.add_argument("--camera", help="Camera feature key. By default, export every camera.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output images and CSV.")
    parser.add_argument("--video-backend", help="Video decoder backend, e.g. pyav or torchcodec.")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.001,
        help="Maximum per-sample change in every state dimension counted as stationary (default: 0.001).",
    )
    args = parser.parse_args()

    local_path = Path(args.dataset).expanduser()
    if local_path.exists():
        if not (local_path / "meta" / "info.json").is_file():
            parser.error(f"{local_path} is not a LeRobot dataset directory (missing meta/info.json).")
        dataset = LeRobotDataset(
            repo_id=local_path.resolve().name,
            root=local_path.resolve(),
            video_backend=args.video_backend,
        )
    else:
        if args.dataset.startswith(("/", ".", "~")):
            parser.error(f"Local dataset directory does not exist: {local_path}")
        dataset = LeRobotDataset(repo_id=args.dataset, video_backend=args.video_backend)

    saved = inspect_dataset(
        dataset, args.output_dir.expanduser(), args.camera, args.overwrite, args.motion_threshold
    )
    print(f"Saved {saved} images to {args.output_dir.expanduser().resolve()}")
    print(f"Saved episode lengths to {args.output_dir.expanduser().resolve() / 'episode_lengths.csv'}")


if __name__ == "__main__":
    main()
