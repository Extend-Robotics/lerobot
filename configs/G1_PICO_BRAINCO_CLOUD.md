# G1D cloud training

Both recipes use `g1_pico_brainco_combined_relative` at
`/data/datasets/g1_pico_brainco_combined_relative`. Copy the completed dataset
there before training. Store checkpoints under `/data/outputs` on persistent storage.
Run commands from the cloud checkout containing these configs; the checkout must
include the same custom LeRobot code used to create the dataset.

| Recipe | GPUs | Batch per GPU | Accumulation | Effective batch | Updates |
| --- | --- | --- | --- | --- | --- |
| Pi0.5, full fine-tuning | 8, DDP | 8 | 4 | 256 | 50,000 |
| Flux3, full fine-tuning with fresh G1 heads | 8, FSDP | 1 | 8 | 64 | 80,000 |

These are starting recipes for the eight-H100 cloud setup used by the existing
cloud configs; GPU memory and throughput have not been measured for these recipes.
Both save every 5,000 updates. W&B logging is enabled in project `pickplace`;
provide credentials through the cloud runtime or pass `--wandb.enable=false`.
Model downloads may require a Hugging Face account with access to the base weights.
Flux additionally requires the Flux3 dependencies and the torch-matched NATTEN
installation described in `docs/source/flux3.mdx`; a Pi0.5-only image is insufficient.

```bash
uv run --no-sync accelerate launch --multi_gpu --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --config_path=configs/pi05_cloud_g1_pico_brainco_relative.json

uv run --no-sync accelerate launch --multi_gpu --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --config_path=configs/flux_cloud_g1_pico_brainco_relative.json
```

Run the recipes separately. Use a new `--output_dir` for each fresh experiment.
These commands do not provision cloud resources or upload the dataset.

## Shared data and deployment contract

Both policies use 26 state/action channels, the three named cameras, a 50-frame
prediction chunk and 10 executed actions per inference. Arms use offsets from the
chunk-start observation; all 12 hand channels remain absolute. Dataset actions
remain stored as absolute targets. Both configs use the dataset's matching
relative-action quantiles. Keep `chunk_size=50`; recalculate action statistics if
you change it. Cloud video decoding uses `torchcodec` (override
`--dataset.video_backend=pyav` if needed).

Pi0.5 uses its base weights, all three cameras, and full fine-tuning with the
vision encoder unfrozen. This differs from the local LoRA recipe.

Flux uses the base action trunk rather than the SO-101 policy/normalizers, so it
can initialize new 26-channel G1 heads. Frame conditioning supports the same
state-relative action representation as Pi0.5; history conditioning owns a
different command-delta representation and cannot be combined with these stats.
The `absolute` action-representation setting disables that additional command-delta
transform; `use_relative_actions=true` still makes the arm training targets relative.
The three cameras are packed in head/left-wrist/right-wrist order into a 384x512
grid; no camera is renamed or discarded.

**Flux deployment limitation:** this checkout's `Flux3Policy.select_action()`
rejects `use_relative_actions=true`, and Flux RTC execution is not implemented.
This config supports training and offline `predict_action_chunk()` evaluation;
live G1D rollout requires implementing and validating the relative-action execution
path first. Pi0.5 already has the required deployment conversion and chunk anchoring.

Local bottle demonstrations represent approximately 3.9% of frames. These configs
retain ordinary sampling across the mixture. A subsequent local-only fine-tune can
focus performance on the G1D workspace; neither recipe has been hardware evaluated.
