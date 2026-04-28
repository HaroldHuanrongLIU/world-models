from __future__ import annotations

import math

import numpy as np
import torch

from surgwm_worldmodels.evaluation.metrics import (
    ade,
    discrete_frechet,
    endpoint_error,
    error_by_horizon,
    fde,
    symmetric_hausdorff,
    trajectory_length,
    trajectory_length_error,
    trajectory_smoothness,
)


def test_basic_distance_metrics_on_simple_trajectory():
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pred = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

    assert math.isclose(ade(pred, target), 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(fde(pred, target), 0.0, abs_tol=1e-6)
    assert math.isclose(endpoint_error(pred, target), 0.0, abs_tol=1e-6)
    assert math.isclose(discrete_frechet(pred, target), 1.0, rel_tol=1e-6)
    assert math.isclose(symmetric_hausdorff(pred, target), 1.0, rel_tol=1e-6)


def test_length_smoothness_and_horizon_metrics():
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]])
    mask = torch.tensor([[True, True, True]])

    assert math.isclose(trajectory_length(target, mask), 2.0, rel_tol=1e-6)
    assert trajectory_length_error(pred, target, mask) > 0.0
    assert trajectory_smoothness(target, mask) == 0.0
    horizons = error_by_horizon(pred, target, [1, 2, 3, 20], mask)
    assert horizons["1"] == 0.0
    assert horizons["2"] == 1.0
    assert horizons["3"] == 0.0
    assert horizons["20"] == 0.0


def test_metrics_accept_numpy_and_avoid_nan_when_mask_empty():
    pred = np.zeros((1, 3, 2), dtype=np.float32)
    target = np.ones((1, 3, 2), dtype=np.float32)
    mask = np.zeros((1, 3), dtype=bool)

    assert ade(pred, target, mask) == 0.0
    assert fde(pred, target, mask) == 0.0
    assert discrete_frechet(pred, target, mask) == 0.0
    assert symmetric_hausdorff(pred, target, mask) == 0.0
