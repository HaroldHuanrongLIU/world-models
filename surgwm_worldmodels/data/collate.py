"""Collate functions for SurgWMBench datasets."""

from __future__ import annotations

from typing import Any

import torch

from .surgwmbench import _metadata_from_sample


def _difficulty(batch: list[dict[str, Any]]) -> list[str | None]:
    return [item.get("difficulty") for item in batch]


def _metadata(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_metadata_from_sample(item) for item in batch]


def collate_sparse_anchors(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate ``frame_sampling='sparse_anchors'`` samples."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if any(item["frames"] is None for item in batch):
        raise ValueError("collate_sparse_anchors requires return_images=True samples")

    frames = torch.stack([item["frames"] for item in batch], dim=0)
    coords_norm = torch.stack([item["selected_coords_norm"] for item in batch], dim=0)
    coords_px = torch.stack([item["selected_coords_px"] for item in batch], dim=0)
    sampled_indices = torch.stack([item["sampled_indices"] for item in batch], dim=0)
    frame_indices = torch.stack([item["frame_indices"] for item in batch], dim=0)
    coord_source = torch.stack([item["selected_coord_sources"] for item in batch], dim=0)
    label_weight = torch.stack([item["selected_label_weights"] for item in batch], dim=0)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    if frames.shape[1] != 20:
        raise ValueError(f"Sparse SurgWMBench batches must have 20 frames, got {frames.shape[1]}")
    if coords_norm.shape[1] != 20 or sampled_indices.shape[1] != 20:
        raise ValueError("Sparse SurgWMBench batches must have 20 coordinates and sampled indices")

    actions_delta = coords_norm[:, 1:] - coords_norm[:, :-1]
    denom = torch.clamp(num_frames.to(torch.float32) - 1.0, min=1.0).unsqueeze(1)
    anchor_dt = (sampled_indices[:, 1:] - sampled_indices[:, :-1]).to(torch.float32) / denom
    actions_delta_dt = torch.cat([actions_delta, anchor_dt.unsqueeze(-1)], dim=-1)
    human_anchor_mask = torch.ones(coords_norm.shape[:2], dtype=torch.bool)

    return {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "sampled_indices": sampled_indices,
        "frame_indices": frame_indices,
        "human_anchor_mask": human_anchor_mask,
        "mask": human_anchor_mask,
        "num_frames": num_frames,
        "anchor_dt": anchor_dt,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "coord_source": coord_source,
        "label_weight": label_weight,
        "difficulty": _difficulty(batch),
        "metadata": _metadata(batch),
    }


def collate_dense_variable_length(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate dense or windowed variable-length samples with padding."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if any(item["frames"] is None for item in batch):
        raise ValueError("collate_dense_variable_length requires return_images=True samples")

    batch_size = len(batch)
    max_t = max(int(item["frames"].shape[0]) for item in batch)
    channels, height, width = batch[0]["frames"].shape[1:]

    frames = torch.zeros(batch_size, max_t, channels, height, width, dtype=torch.float32)
    coords_norm = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    coords_px = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    coord_source = torch.zeros(batch_size, max_t, dtype=torch.long)
    label_weight = torch.zeros(batch_size, max_t, dtype=torch.float32)
    confidence = torch.zeros(batch_size, max_t, dtype=torch.float32)
    frame_indices = torch.full((batch_size, max_t), -1, dtype=torch.long)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    action_t = max(max_t - 1, 0)
    actions_delta = torch.zeros(batch_size, action_t, 2, dtype=torch.float32)
    actions_delta_dt = torch.zeros(batch_size, action_t, 3, dtype=torch.float32)
    action_mask = torch.zeros(batch_size, action_t, dtype=torch.bool)

    for row, item in enumerate(batch):
        t = int(item["frames"].shape[0])
        frames[row, :t] = item["frames"]
        coords_norm[row, :t] = item["selected_coords_norm"]
        coords_px[row, :t] = item["selected_coords_px"]
        frame_mask[row, :t] = True
        coord_source[row, :t] = item["selected_coord_sources"]
        label_weight[row, :t] = item["selected_label_weights"]
        confidence[row, :t] = item["selected_confidence"]
        frame_indices[row, :t] = item["frame_indices"]

        if t > 1:
            delta = item["selected_coords_norm"][1:] - item["selected_coords_norm"][:-1]
            denom = max(float(item["num_frames"] - 1), 1.0)
            dt = (item["frame_indices"][1:] - item["frame_indices"][:-1]).to(torch.float32) / denom
            actions_delta[row, : t - 1] = delta
            actions_delta_dt[row, : t - 1] = torch.cat([delta, dt.unsqueeze(-1)], dim=-1)
            action_mask[row, : t - 1] = True

    return {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "frame_mask": frame_mask,
        "coord_source": coord_source,
        "label_weight": label_weight,
        "confidence": confidence,
        "frame_indices": frame_indices,
        "num_frames": num_frames,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "action_mask": action_mask,
        "difficulty": _difficulty(batch),
        "metadata": _metadata(batch),
    }


def collate_frame_vae(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> dict[str, Any]:
    """Collate frame-level VAE samples."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    images, metadata = zip(*batch)
    return {
        "images": torch.stack(list(images), dim=0),
        "metadata": [dict(item) for item in metadata],
    }


def collate_video_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate video/window samples into a padded batch."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_t = max(int(item["frames"].shape[0]) for item in batch)
    batch_size = len(batch)
    channels, height, width = batch[0]["frames"].shape[1:]
    frames = torch.zeros(batch_size, max_t, channels, height, width, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    frame_indices = torch.full((batch_size, max_t), -1, dtype=torch.long)
    metadata: list[dict[str, Any]] = []

    for row, item in enumerate(batch):
        t = int(item["frames"].shape[0])
        frames[row, :t] = item["frames"]
        frame_mask[row, :t] = True
        if "frame_indices" in item:
            frame_indices[row, :t] = item["frame_indices"]
        metadata.append({key: value for key, value in item.items() if key not in {"frames", "frame_indices"}})

    return {
        "frames": frames,
        "frame_mask": frame_mask,
        "frame_indices": frame_indices,
        "metadata": metadata,
    }


__all__ = [
    "collate_dense_variable_length",
    "collate_frame_vae",
    "collate_sparse_anchors",
    "collate_video_windows",
]
