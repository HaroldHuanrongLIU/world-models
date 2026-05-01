# PyTorch World Models with SurgWMBench data support

Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631. For a quick summary of the paper and some additional experiments, visit the [github page](https://ctallec.github.io/world-models/).

This repository started as the PyTorch implementation of the classic
`VAE + MDN-RNN + Controller` World Models baseline. It now also contains a
first-pass SurgWMBench data foundation under `surgwm_worldmodels/`.

The original CarRacing/Gym scripts are still present for reference. The
SurgWMBench path does not use Gym rollouts or random train/val/test splits.


## Prerequisites

The SurgWMBench utilities are written for Python 3.10+ and PyTorch 2.x. Install
PyTorch for your CUDA/CPU environment from the [PyTorch website](https://pytorch.org),
then install the repository requirements:

```bash
pip install -r requirements.txt
```

The legacy CarRacing/Gym code has extra optional dependencies:

```bash
pip install -r requirements-carracing.txt
```

## SurgWMBench support

The local SurgWMBench dataset root used on this machine is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Treat that dataset's `README.md` as the canonical dataset contract.

Current implemented SurgWMBench scope:

- Final-layout manifest/annotation/interpolation loaders.
- Sparse 20-anchor human-label mode.
- Dense pseudo-coordinate mode with source labels and masks.
- Frame-level dataset for VAE pretraining on extracted clip frames.
- Collators for sparse anchors, dense variable-length clips, VAE frames, and
  video windows.
- Sparse/dense trajectory metrics.
- Synthetic toy dataset generator and read-only validation tool.

Not implemented yet:

- SurgWMBench VAE model/training script.
- SurgWMBench MDN-RNN model/training script.
- Offline controller/policy/planner training.
- Sparse/dense model evaluation scripts and rollout visualization.

### Dataset rules

- Use official manifests: `manifests/train.jsonl`, `manifests/val.jsonl`,
  `manifests/test.jsonl`, and `manifests/all.jsonl`.
- Do not create random train/val/test splits.
- Every clip has exactly 20 sparse human anchors, but clips are variable length.
- Sparse human anchors are the primary benchmark target.
- Dense interpolation coordinates are auxiliary pseudo labels, not human ground
  truth.
- Coordinates are loaded in both pixel `[x, y]` and normalized `[0, 1]` forms.
- Do not use `old_frame_idx` as a dense local frame index.
- Do not infer difficulty from folder names.

### Validate the real dataset

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 8
```

### Create and validate a toy dataset

```bash
python -m tools.make_toy_surgwmbench \
  --output /tmp/SurgWMBench_toy \
  --num-clips 2

python -m tools.validate_surgwmbench_loader \
  --dataset-root /tmp/SurgWMBench_toy \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 2
```

### Run the current SurgWMBench tests

```bash
pytest tests/test_surgwmbench_dataset.py tests/test_collate.py tests/test_metrics.py
```

### Minimal loader example

```python
from torch.utils.data import DataLoader

from surgwm_worldmodels.data import SurgWMBenchClipDataset, collate_sparse_anchors

root = "/mnt/hdd1/neurips2026_dataset_track/SurgWMBench"
dataset = SurgWMBenchClipDataset(
    dataset_root=root,
    manifest="manifests/train.jsonl",
    frame_sampling="sparse_anchors",
    image_size=128,
)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_sparse_anchors)
batch = next(iter(loader))

print(batch["frames"].shape)            # [B, 20, 3, 128, 128]
print(batch["actions_delta_dt"].shape)  # [B, 19, 3], action=[dx_norm, dy_norm, dt]
```

## Legacy CarRacing World Models

The model is composed of three parts:

  1. A Variational Auto-Encoder (VAE), whose task is to compress the input images into a compact latent representation.
  2. A Mixture-Density Recurrent Network (MDN-RNN), trained to predict the latent encoding of the next frame given past latent encodings and actions.
  3. A linear Controller (C), which takes both the latent encoding of the current frame, and the hidden state of the MDN-RNN given past latents and actions as input and outputs an action. It is trained to maximize the cumulated reward using the Covariance-Matrix Adaptation Evolution-Strategy ([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)) from the `cma` python package.

In the given code, all three sections are trained separately, using the scripts `trainvae.py`, `trainmdrnn.py` and `traincontroller.py`.

Training scripts take as argument:
* **--logdir** : The directory in which the models will be stored. If the logdir specified already exists, it loads the old model and continues the training.
* **--noreload** : If you want to override a model in *logdir* instead of reloading it, add this option.

### 1. CarRacing data generation
Before launching the VAE and MDN-RNN training scripts, you need to generate a dataset of random rollouts and place it in the `datasets/carracing` folder.

Data generation is handled through the `data/generation_script.py` script, e.g.
```bash
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
```

Rollouts are generated using a *brownian* random policy, instead of the *white noise* random `action_space.sample()` policy from gym, providing more consistent rollouts.

### 2. Training the CarRacing VAE
The VAE is trained using the `trainvae.py` file, e.g.
```bash
python trainvae.py --logdir exp_dir
```

### 3. Training the CarRacing MDN-RNN
The MDN-RNN is trained using the `trainmdrnn.py` file, e.g.
```bash
python trainmdrnn.py --logdir exp_dir
```
A VAE must have been trained in the same `exp_dir` for this script to work.
### 4. Training and testing the CarRacing Controller
Finally, the controller is trained using CMA-ES, e.g.
```bash
python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```
You can test the obtained policy with `test_controller.py` e.g.
```bash
python test_controller.py --logdir exp_dir
```

### Notes
When running on a headless server, you will need to use `xvfb-run` to launch the controller training script. For instance,
```bash
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```
If you do not have a display available and you launch `traincontroller` without
`xvfb-run`, the script will fail silently (but logs are available in
`logdir/tmp`).

Be aware that `traincontroller` requires heavy gpu memory usage when launched
on gpus. To reduce the memory load, you can directly modify the maximum number
of workers by specifying the `--max-workers` argument.

If you have several GPUs available, `traincontroller` will take advantage of
all gpus specified by `CUDA_VISIBLE_DEVICES`.

## Authors

* **Corentin Tallec** - [ctallec](https://github.com/ctallec)
* **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
* **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
