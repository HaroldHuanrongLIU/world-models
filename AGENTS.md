# Repository Guidance

## Project Shape

This repository is a local clone of the PyTorch World Models baseline from
`ctallec/world-models`. The original code targets CarRacing/Gym:

- `trainvae.py`, `trainmdrnn.py`, `traincontroller.py`
- `models/vae.py`, `models/mdrnn.py`, `models/controller.py`
- `data/carracing.py`, `data/loaders.py`
- `utils/misc.py`

Keep those original files readable and avoid unrelated rewrites. New
SurgWMBench work should live under `surgwm_worldmodels/`, `tools/`, `configs/`,
and focused tests unless a task explicitly asks to modernize original files.

## SurgWMBench Contract

The real dataset root on this machine is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Treat the dataset README at that root as canonical. Important constraints:

- Use official manifests: `manifests/train.jsonl`, `val.jsonl`, `test.jsonl`,
  and `all.jsonl`.
- Do not create random train/val/test splits.
- Sparse human anchors are the primary target; every clip has exactly 20
  anchors.
- Clips are variable length. Never assume a clip has 20 frames.
- Dense interpolation coordinates are auxiliary pseudo labels, not human
  ground truth.
- Keep pixel coordinates for metrics and normalized coordinates for training.
- Do not infer difficulty from folder names.
- Do not use `old_frame_idx` as a dense local frame index.

## Current SurgWMBench Code

First-pass data support is in:

- `surgwm_worldmodels/data/surgwmbench.py`
- `surgwm_worldmodels/data/collate.py`
- `surgwm_worldmodels/data/transforms.py`
- `surgwm_worldmodels/evaluation/metrics.py`
- `tools/make_toy_surgwmbench.py`
- `tools/validate_surgwmbench_loader.py`

This first pass intentionally does not edit VAE, MDN-RNN, controller, rollout,
training, or evaluation model code.

## Development Commands

Focused tests for the current SurgWMBench data foundation:

```bash
pytest tests/test_surgwmbench_dataset.py tests/test_collate.py tests/test_metrics.py
```

Create and validate a synthetic toy dataset:

```bash
python -m tools.make_toy_surgwmbench --output /tmp/SurgWMBench_toy --num-clips 2
python -m tools.validate_surgwmbench_loader \
  --dataset-root /tmp/SurgWMBench_toy \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 2
```

Validate a small slice of the real dataset:

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 8
```

## Coding Notes

- Use Python 3.10+ style type hints and pathlib.
- Prefer PyTorch 2.x APIs for new code.
- Use PIL for extracted frame loading; OpenCV should remain optional.
- Keep dense pseudo losses and metrics clearly labeled as pseudo-coordinate
  results in later work.
- Avoid TensorFlow and internet-dependent workflows.
- Do not modify SurgWMBench annotations, manifests, or interpolation files.
