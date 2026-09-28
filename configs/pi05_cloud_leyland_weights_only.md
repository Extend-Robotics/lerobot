# Leyland PI05 weights-only comparison

Start from `pi05_cloud_leyland_merged.json` and compare these two initializations:

| Config | Initial weights |
| --- | --- |
| `pi05_cloud_leyland_merged_so100_weights_only.json` | `hqfang/pi05-so100_101` at `d3204e03ae84d232e2493a933fd460515904eb58` |
| `pi05_cloud_leyland_merged_base_weights_only.json` | `lerobot/pi05_base` at `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` (the SO100 run's original base revision) |

Both configs use full fine-tuning, batch size 32, 50,000 steps, BF16, gradient
checkpointing, and the cloud config's optimizer/scheduler. They have separate
output directories and W&B run names. Neither uses LoRA.

The experiment changes normalization to Leyland q01/q99 statistics (`QUANTILES`),
disables training-time RTC, explicitly keeps absolute actions and 50-action chunks,
and sets `load_pretrained_processors: false`. This loads weights into the local
policy definition and constructs fresh processors using the standard PaliGemma
tokenizer. It does not load the source checkpoint's config or saved processors.
`n_action_steps` remains 10, as in the cloud config.

Camera meaning and order:

1. `observation.images.front`: **wrist camera**, despite its name.
2. `observation.images.realsense`: **scene camera**.
3. One masked empty camera, retained from the cloud config.

No camera rename is needed: fresh input features come from the dataset. Keep the
same physical camera mapping at rollout. The SO100 checkpoint trained with
randomized camera assignments; its generic camera names do not fix physical roles.

Run from the `lerobot` repository directory (where `datasets/` is located):

```bash
uv run lerobot-train --config_path=configs/pi05_cloud_leyland_merged_so100_weights_only.json
```

Run the matched baseline separately:

```bash
uv run lerobot-train --config_path=configs/pi05_cloud_leyland_merged_base_weights_only.json
```

Do not add `--policy.path`: it would replace the explicit local policy config with
the source checkpoint's config. The standard Google tokenizer must be cached or
accessible through your Hugging Face account on the training machine.

When resuming an experiment checkpoint, `resume: true` always restores that run's
saved processors/statistics, even though its initial config set
`load_pretrained_processors: false`. For deployment, load the resulting fine-tuned
checkpoint and its saved processors together.

Compare the two runs on the same held-out episodes and robot tasks, with the same
checkpoint-selection rule and rollout settings. These configs retain the cloud
config's evaluation defaults; they do not create a held-out split automatically.
The old cloud run is not a matched baseline because it used MEAN_STD and RTC.

## Validation

Validated locally on 2026-09-28:

- Both configs decode and validate; they differ only in checkpoint source/revision
  and output/job names.
- 28 targeted tests pass, covering processor selection, resume statistics,
  strict weight-loading failures, revision forwarding, tokenizer configuration,
  batch preprocessing, and training config behavior.
- All 813 SO100 checkpoint tensor names/shapes match this branch. The downloaded
  file matches the author's SHA256, and strict loading succeeds.
- Real dataset frames 0, 100, 70,587 and 141,173 pass preprocessing and action
  normalization/unnormalization round trips. The standard tokenizer matches the
  bundled SentencePiece tokenizer on these prompts, including BOS, padding and masks.
- A real batch-size-one forward/backward pass with the SO100 weights and BF16
  autocast produces loss 0.348119 and finite gradients for 805 parameter tensors.
  Peak allocated GPU memory is 18.28 GiB on an RTX 5090. This checks execution,
  not task success or convergence; no optimizer update or full training was run.
- The configured batch size 32 has not been validated on the local GPU. It is
  retained from the cloud config and needs sufficient training hardware.
