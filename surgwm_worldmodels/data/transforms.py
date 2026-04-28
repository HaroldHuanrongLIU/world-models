"""Image transform helpers for SurgWMBench frame loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _resample_bilinear() -> int:
    resampling = getattr(Image, "Resampling", Image)
    return int(resampling.BILINEAR)


def target_size_hw(image_size: int | tuple[int, int] | None) -> tuple[int, int] | None:
    """Normalize an image size setting to ``(height, width)``."""

    if image_size is None or image_size == 0:
        return None
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    if len(image_size) != 2:
        raise ValueError(f"image_size tuple must contain two values, got {image_size!r}")
    return int(image_size[0]), int(image_size[1])


def pil_to_float_tensor(image: Image.Image) -> torch.FloatTensor:
    """Convert an RGB PIL image to ``FloatTensor[3,H,W]`` in ``[0, 1]``."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_rgb_frame(path: str | Path, image_size: int | tuple[int, int] | None = 128) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an RGB frame and return ``(tensor, original_size_hw)``."""

    frame_path = Path(path).expanduser()
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame image not found: {frame_path}")
    with Image.open(frame_path) as image:
        image = image.convert("RGB")
        original_size_hw = (int(image.height), int(image.width))
        size_hw = target_size_hw(image_size)
        if size_hw is not None and (image.height, image.width) != size_hw:
            image = image.resize((size_hw[1], size_hw[0]), _resample_bilinear())
        return pil_to_float_tensor(image), original_size_hw


def normalize_tensor(image: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    """Normalize a ``[3,H,W]`` or ``[...,3,H,W]`` image tensor."""

    mean_t = torch.as_tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.as_tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    if image.ndim > 3:
        view_shape = (1,) * (image.ndim - 3) + (3, 1, 1)
        mean_t = mean_t.view(view_shape)
        std_t = std_t.view(view_shape)
    return (image - mean_t) / std_t
