from __future__ import annotations

import torch

from surgwm_worldmodels.data.collate import collate_dense_variable_length, collate_frame_vae, collate_sparse_anchors
from surgwm_worldmodels.data.surgwmbench import SurgWMBenchClipDataset, SurgWMBenchFrameDataset
from tests.surgwmbench_test_utils import make_surgwmbench_root


def test_collate_sparse_returns_shapes_and_delta_dt_actions(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )

    batch = collate_sparse_anchors([dataset[0], dataset[1]])

    assert batch["frames"].shape == (2, 20, 3, 32, 32)
    assert batch["coords_norm"].shape == (2, 20, 2)
    assert batch["coords_px"].shape == (2, 20, 2)
    assert batch["sampled_indices"].shape == (2, 20)
    assert batch["human_anchor_mask"].shape == (2, 20)
    assert batch["anchor_dt"].shape == (2, 19)
    assert batch["actions_delta"].shape == (2, 19, 2)
    assert batch["actions_delta_dt"].shape == (2, 19, 3)
    assert batch["difficulty"] == ["low", "medium"]
    assert batch["human_anchor_mask"].all()

    expected_delta = batch["coords_norm"][:, 1:] - batch["coords_norm"][:, :-1]
    expected_dt = (batch["sampled_indices"][:, 1:] - batch["sampled_indices"][:, :-1]).float()
    expected_dt = expected_dt / torch.clamp(batch["num_frames"].float() - 1.0, min=1.0).unsqueeze(1)
    assert torch.allclose(batch["actions_delta"], expected_delta)
    assert torch.allclose(batch["actions_delta_dt"][..., :2], expected_delta)
    assert torch.allclose(batch["actions_delta_dt"][..., 2], expected_dt)


def test_collate_dense_pads_variable_length_clips_and_masks(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )

    batch = collate_dense_variable_length([dataset[0], dataset[1]])

    assert batch["frames"].shape == (2, 31, 3, 32, 32)
    assert batch["coords_norm"].shape == (2, 31, 2)
    assert batch["coords_px"].shape == (2, 31, 2)
    assert batch["frame_mask"].shape == (2, 31)
    assert batch["action_mask"].shape == (2, 30)
    assert int(batch["frame_mask"][0].sum()) == 25
    assert int(batch["frame_mask"][1].sum()) == 31
    assert int(batch["action_mask"][0].sum()) == 24
    assert int(batch["action_mask"][1].sum()) == 30
    assert not batch["frame_mask"][0, 25:].any()
    assert torch.all(batch["frame_indices"][0, 25:] == -1)
    assert torch.all(batch["coord_source"][0, 25:] == 0)
    assert torch.all(batch["label_weight"][0, 25:] == 0)
    assert torch.all(batch["confidence"][0, 25:] == 0)

    expected_delta = batch["coords_norm"][1, 1:] - batch["coords_norm"][1, :-1]
    assert torch.allclose(batch["actions_delta"][1], expected_delta)
    assert torch.allclose(batch["actions_delta_dt"][1, :, :2], expected_delta)


def test_collate_frame_vae(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchFrameDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        max_frames_per_clip=1,
    )

    batch = collate_frame_vae([dataset[0], dataset[1]])

    assert batch["images"].shape == (2, 3, 32, 32)
    assert batch["metadata"][0]["local_frame_idx"] == 0
    assert batch["metadata"][1]["local_frame_idx"] == 0
