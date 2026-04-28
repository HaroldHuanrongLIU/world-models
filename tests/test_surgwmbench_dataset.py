from __future__ import annotations

import pytest
import torch

from surgwm_worldmodels.data.surgwmbench import SOURCE_TO_CODE, SurgWMBenchClipDataset, SurgWMBenchFrameDataset
from tests.surgwmbench_test_utils import make_surgwmbench_root, remove_interpolation_file, set_dataset_version
from tools.validate_surgwmbench_loader import validate_surgwmbench


def test_sparse_dataset_loads_exactly_20_anchors(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )

    sample = dataset[0]

    assert sample["frames"].shape == (20, 3, 32, 32)
    assert sample["sampled_indices"].shape == (20,)
    assert sample["human_anchor_coords_px"].shape == (20, 2)
    assert sample["human_anchor_coords_norm"].shape == (20, 2)
    assert sample["selected_coords_norm"].shape == (20, 2)
    assert torch.equal(sample["frame_indices"], sample["human_anchor_local_indices"])
    assert torch.all(sample["selected_coord_sources"] == SOURCE_TO_CODE["human"])
    assert sample["dense_coords_norm"] is None


def test_dense_dataset_loads_variable_length_coordinates_and_sources(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )

    sample = dataset[0]

    assert sample["frames"].shape == (25, 3, 32, 32)
    assert sample["dense_coords_norm"].shape == (25, 2)
    assert sample["selected_coords_px"].shape == (25, 2)
    assert int((sample["selected_coord_sources"] == SOURCE_TO_CODE["human"]).sum()) == 20
    assert int((sample["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"]).sum()) == 5
    assert torch.all(
        sample["selected_label_weights"][sample["selected_coord_sources"] == SOURCE_TO_CODE["human"]] == 1.0
    )
    assert torch.all(
        sample["selected_label_weights"][sample["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"]] == 0.5
    )
    assert sample["difficulty"] == "low"


def test_window_dataset_is_deterministic_prefix(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="window",
        max_frames=7,
        interpolation_method="linear",
    )

    sample = dataset[0]

    assert sample["frames"].shape == (7, 3, 32, 32)
    assert sample["frame_indices"].tolist() == list(range(7))


def test_interpolation_method_switching_loads_selected_file(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    linear = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )[0]
    pchip = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="pchip",
    )[0]

    non_anchor = torch.nonzero(linear["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"])[0].item()
    assert linear["interpolation_method"] == "linear"
    assert pchip["interpolation_method"] == "pchip"
    assert not torch.allclose(linear["selected_coords_px"][non_anchor], pchip["selected_coords_px"][non_anchor])


def test_strict_loader_rejects_missing_interpolation_file(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    missing = remove_interpolation_file(root, "pchip")
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="pchip",
        strict=True,
    )

    with pytest.raises(FileNotFoundError, match=str(missing)):
        _ = dataset[0]


def test_strict_loader_rejects_wrong_dataset_version_unless_allowed(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    set_dataset_version(root, "SurgWMBench-v0")

    with pytest.raises(ValueError, match="dataset_version"):
        SurgWMBenchClipDataset(
            dataset_root=root,
            manifest="manifests/train.jsonl",
            image_size=32,
            frame_sampling="sparse_anchors",
            strict=True,
        )

    with pytest.warns(UserWarning, match="legacy dataset_version"):
        dataset = SurgWMBenchClipDataset(
            dataset_root=root,
            manifest="manifests/train.jsonl",
            image_size=32,
            frame_sampling="sparse_anchors",
            strict=True,
            allow_legacy_version=True,
        )
    with pytest.warns(UserWarning, match="legacy dataset_version"):
        sample = dataset[0]
    assert sample["frames"].shape[0] == 20


def test_frame_dataset_returns_images_and_metadata(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchFrameDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
    )

    image, metadata = dataset[0]

    assert image.shape == (3, 32, 32)
    assert metadata["trajectory_id"] == "traj_001"
    assert metadata["local_frame_idx"] == 0
    assert metadata["frame_path"].endswith("000000.jpg")


def test_dataset_loader_does_not_create_random_splits(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    manifests_dir = root / "manifests"
    before = sorted(path.name for path in manifests_dir.iterdir())

    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )
    _ = dataset[0]

    after = sorted(path.name for path in manifests_dir.iterdir())
    assert after == before == ["all.jsonl", "test.jsonl", "train.jsonl", "val.jsonl"]


def test_validation_tool_passes_and_reports_file_errors(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    assert validate_surgwmbench(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        interpolation_method="linear",
        check_files=True,
        num_samples=2,
    ) == []

    missing = remove_interpolation_file(root, "akima")
    errors = validate_surgwmbench(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        interpolation_method="linear",
        check_files=True,
        num_samples=1,
    )
    assert any(str(missing) in error for error in errors)
