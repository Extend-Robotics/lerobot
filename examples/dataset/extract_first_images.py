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

"""Save each episode's first (left) and last (right) frames in one PNG per camera.

Run from the LeRobot checkout with its dependencies installed:

    uv run examples/dataset/extract_first_images.py /path/to/dataset /path/to/images
    uv run examples/dataset/extract_first_images.py user/dataset ./images \
        --camera observation.images.front

Output: <output_dir>/<camera_key>/episode_000000.png, ...
Each PNG has the original frame height and twice the original frame width.
Requires a dataset format supported by the installed LeRobot version.
"""

import argparse
from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset


def extract_first_images(
    dataset: LeRobotDataset,
    output_dir: Path,
    camera: str | None = None,
    overwrite: bool = False,
) -> int:
    """Export first/last frame pairs from a complete dataset; return the image count."""
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
        first_frame = dataset[first_index]
        last_frame = dataset[last_index] if last_index != first_index else first_frame
        for key, path in paths.items():
            # LeRobot returns RGB tensors in (C, H, W) layout with values in [0, 1].
            first_image = to_pil_image(first_frame[key]).convert("RGB")
            last_image = to_pil_image(last_frame[key]).convert("RGB")
            combined = Image.new("RGB", (first_image.width * 2, first_image.height))
            combined.paste(first_image, (0, 0))
            combined.paste(last_image, (first_image.width, 0))
            combined.save(path)
            saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Local dataset directory or Hugging Face dataset ID (user/name).")
    parser.add_argument("output_dir", type=Path, help="Directory in which to save PNG images.")
    parser.add_argument("--camera", help="Camera feature key. By default, export every camera.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output images.")
    parser.add_argument("--video-backend", help="Video decoder backend, e.g. pyav or torchcodec.")
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

    saved = extract_first_images(dataset, args.output_dir.expanduser(), args.camera, args.overwrite)
    print(f"Saved {saved} images to {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
