"""Trajectory metrics for SurgWMBench sparse and dense trajectories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _prep_coords(pred: Any, target: Any | None = None, mask: Any | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    pred_t = _to_tensor(pred)
    if pred_t.ndim == 2:
        pred_t = pred_t.unsqueeze(0)
    if pred_t.ndim != 3 or pred_t.shape[-1] != 2:
        raise ValueError(f"coordinates must have shape [T,2] or [B,T,2], got {tuple(pred_t.shape)}")

    target_t = None
    if target is not None:
        target_t = _to_tensor(target)
        if target_t.ndim == 2:
            target_t = target_t.unsqueeze(0)
        if target_t.shape != pred_t.shape:
            raise ValueError(f"pred and target shapes differ: {tuple(pred_t.shape)} vs {tuple(target_t.shape)}")

    if mask is None:
        mask_t = torch.ones(pred_t.shape[:2], dtype=torch.bool)
    else:
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(0)
        if mask_t.shape != pred_t.shape[:2]:
            raise ValueError(f"mask shape {tuple(mask_t.shape)} incompatible with coords {tuple(pred_t.shape)}")
    return pred_t, target_t, mask_t


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    mean_value = values.to(dtype=torch.float32).mean()
    if torch.isnan(mean_value):
        return 0.0
    return float(mean_value.item())


def _per_batch_metric(
    pred: Any,
    target: Any,
    mask: Any | None,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
) -> float:
    pred_t, target_t, mask_t = _prep_coords(pred, target, mask)
    assert target_t is not None
    values: list[float] = []
    for batch_idx in range(pred_t.shape[0]):
        valid = mask_t[batch_idx]
        if bool(valid.any()):
            values.append(metric_fn(pred_t[batch_idx, valid], target_t[batch_idx, valid]))
    return float(np.mean(values)) if values else 0.0


def ade(pred: Any, target: Any, mask: Any | None = None) -> float:
    pred_t, target_t, mask_t = _prep_coords(pred, target, mask)
    assert target_t is not None
    dist = torch.linalg.norm(pred_t - target_t, dim=-1)
    return _safe_mean(dist[mask_t])


def fde(pred: Any, target: Any, mask: Any | None = None) -> float:
    pred_t, target_t, mask_t = _prep_coords(pred, target, mask)
    assert target_t is not None
    values: list[torch.Tensor] = []
    for batch_idx in range(pred_t.shape[0]):
        valid_idx = torch.nonzero(mask_t[batch_idx], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        last = valid_idx[-1]
        values.append(torch.linalg.norm(pred_t[batch_idx, last] - target_t[batch_idx, last], dim=-1))
    return _safe_mean(torch.stack(values)) if values else 0.0


def _discrete_frechet_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    n_points, m_points = pred.shape[0], target.shape[0]
    if n_points == 0 or m_points == 0:
        return 0.0
    cache = torch.full((n_points, m_points), -1.0, dtype=torch.float32)

    def recurse(i: int, j: int) -> torch.Tensor:
        if cache[i, j] > -0.5:
            return cache[i, j]
        dist = torch.linalg.norm(pred[i] - target[j])
        if i == 0 and j == 0:
            cache[i, j] = dist
        elif i > 0 and j == 0:
            cache[i, j] = torch.maximum(recurse(i - 1, 0), dist)
        elif i == 0 and j > 0:
            cache[i, j] = torch.maximum(recurse(0, j - 1), dist)
        else:
            cache[i, j] = torch.maximum(
                torch.minimum(torch.minimum(recurse(i - 1, j), recurse(i - 1, j - 1)), recurse(i, j - 1)),
                dist,
            )
        return cache[i, j]

    return float(recurse(n_points - 1, m_points - 1).item())


def discrete_frechet(pred: Any, target: Any, mask: Any | None = None) -> float:
    return _per_batch_metric(pred, target, mask, _discrete_frechet_single)


def _symmetric_hausdorff_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0 or target.numel() == 0:
        return 0.0
    dist = torch.cdist(pred, target)
    pred_to_target = dist.min(dim=1).values.max()
    target_to_pred = dist.min(dim=0).values.max()
    return float(torch.maximum(pred_to_target, target_to_pred).item())


def symmetric_hausdorff(pred: Any, target: Any, mask: Any | None = None) -> float:
    return _per_batch_metric(pred, target, mask, _symmetric_hausdorff_single)


def endpoint_error(pred: Any, target: Any, mask: Any | None = None) -> float:
    return fde(pred, target, mask)


def trajectory_length(coords: Any, mask: Any | None = None) -> float:
    coords_t, _, mask_t = _prep_coords(coords, None, mask)
    values: list[torch.Tensor] = []
    for batch_idx in range(coords_t.shape[0]):
        valid = coords_t[batch_idx, mask_t[batch_idx]]
        if valid.shape[0] < 2:
            values.append(torch.tensor(0.0))
        else:
            values.append(torch.linalg.norm(valid[1:] - valid[:-1], dim=-1).sum())
    return _safe_mean(torch.stack(values)) if values else 0.0


def trajectory_length_error(pred: Any, target: Any, mask: Any | None = None) -> float:
    pred_t, target_t, mask_t = _prep_coords(pred, target, mask)
    assert target_t is not None
    values: list[torch.Tensor] = []
    for batch_idx in range(pred_t.shape[0]):
        pred_valid = pred_t[batch_idx, mask_t[batch_idx]]
        target_valid = target_t[batch_idx, mask_t[batch_idx]]
        if pred_valid.shape[0] < 2:
            values.append(torch.tensor(0.0))
            continue
        pred_len = torch.linalg.norm(pred_valid[1:] - pred_valid[:-1], dim=-1).sum()
        target_len = torch.linalg.norm(target_valid[1:] - target_valid[:-1], dim=-1).sum()
        values.append(torch.abs(pred_len - target_len))
    return _safe_mean(torch.stack(values)) if values else 0.0


def trajectory_smoothness(coords: Any, mask: Any | None = None) -> float:
    coords_t, _, mask_t = _prep_coords(coords, None, mask)
    values: list[torch.Tensor] = []
    for batch_idx in range(coords_t.shape[0]):
        valid = coords_t[batch_idx, mask_t[batch_idx]]
        if valid.shape[0] < 3:
            values.append(torch.tensor(0.0))
        else:
            second_diff = valid[2:] - 2.0 * valid[1:-1] + valid[:-2]
            values.append(second_diff.square().sum(dim=-1).mean())
    return _safe_mean(torch.stack(values)) if values else 0.0


def error_by_horizon(pred: Any, target: Any, horizons: Sequence[int], mask: Any | None = None) -> dict[str, float]:
    pred_t, target_t, mask_t = _prep_coords(pred, target, mask)
    assert target_t is not None
    dist = torch.linalg.norm(pred_t - target_t, dim=-1)
    out: dict[str, float] = {}
    for horizon in horizons:
        idx = min(max(int(horizon), 1), pred_t.shape[1]) - 1
        valid = mask_t[:, idx]
        out[str(int(horizon))] = _safe_mean(dist[:, idx][valid])
    return out


__all__ = [
    "ade",
    "discrete_frechet",
    "endpoint_error",
    "error_by_horizon",
    "fde",
    "symmetric_hausdorff",
    "trajectory_length",
    "trajectory_length_error",
    "trajectory_smoothness",
]
