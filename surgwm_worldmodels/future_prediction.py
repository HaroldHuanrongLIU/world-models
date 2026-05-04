from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break

import torch
from torch import nn

from surgwm_worldmodels.adapter import SurgWMBenchVaeMdrnn
from surgwmbench_benchmark.future_model_helpers import normalized_context_time, normalized_future_time
from surgwmbench_benchmark.future_prediction import FutureProtocolConfig, main


class WorldModelsFuturePredictionModel(nn.Module):
    """Future-prediction wrapper around the VAE + MDRNN world-model core."""

    def __init__(self, config: FutureProtocolConfig) -> None:
        super().__init__()
        self.core = SurgWMBenchVaeMdrnn(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim)
        self.hidden_to_latent = nn.Linear(config.hidden_dim, config.latent_dim)

    def _encode_context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        frames = batch["context_frames"]
        bsz, context, channels, height, width = frames.shape
        flat = frames.reshape(bsz * context, channels, height, width)
        z, _, _ = self.core.encoder(flat)
        z_seq = z.view(bsz, context, -1)
        context_input = torch.cat([z_seq, normalized_context_time(batch)], dim=-1)
        _, hidden = self.core.mdrnn(context_input)
        return z_seq[:, -1], hidden

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["context_frames"]
        _, _, _, height, width = frames.shape
        last_z, hidden = self._encode_context(batch)
        future_input = torch.cat(
            [last_z.unsqueeze(1).expand(-1, batch["future_frame_indices"].shape[1], -1), normalized_future_time(batch)],
            dim=-1,
        )
        hidden_seq, _ = self.core.mdrnn(future_input, hidden)
        pred_coords = torch.sigmoid(self.core.coord_head(hidden_seq))
        z_future = self.hidden_to_latent(hidden_seq)
        pred_frames = self.core.decoder(z_future.reshape(-1, z_future.shape[-1]), (height, width))
        pred_frames = pred_frames.view(z_future.shape[0], z_future.shape[1], 3, height, width)
        return {"pred_frames": pred_frames, "pred_coords_norm": pred_coords}


def make_model(config: FutureProtocolConfig) -> nn.Module:
    return WorldModelsFuturePredictionModel(config)


if __name__ == "__main__":
    raise SystemExit(main("world_models", "WorldModelsFuturePredictionCore", "surgwm_worldmodels.data.surgwmbench", make_model))
