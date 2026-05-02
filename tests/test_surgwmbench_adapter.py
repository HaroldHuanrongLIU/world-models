from __future__ import annotations

from types import SimpleNamespace

from surgwm_worldmodels.adapter import eval_adapter, train_adapter
from tests.surgwmbench_test_utils import make_surgwmbench_root


def test_world_models_adapter_train_eval_smoke(tmp_path):
    root = make_surgwmbench_root(tmp_path, num_clips=1)
    output_dir = tmp_path / "run"
    train_result = train_adapter(
        SimpleNamespace(
            dataset_root=str(root),
            manifest="manifests/train.jsonl",
            train_manifest="manifests/train.jsonl",
            val_manifest="manifests/val.jsonl",
            target="sparse_20_anchor",
            interpolation_method="linear",
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            image_size=32,
            latent_dim=8,
            hidden_dim=16,
            recon_weight=0.01,
            kl_weight=1e-5,
            max_clips=1,
            max_frames=None,
            num_workers=0,
            device="cpu",
            seed=7,
        )
    )
    checkpoint = train_result["checkpoint"]
    result = eval_adapter(
        SimpleNamespace(
            dataset_root=str(root),
            manifest="manifests/test.jsonl",
            checkpoint=checkpoint,
            target="sparse_20_anchor",
            interpolation_method="linear",
            output=str(output_dir / "metrics.json"),
            batch_size=1,
            image_size=None,
            max_clips=1,
            max_frames=None,
            num_workers=0,
            device="cpu",
        )
    )
    assert result["baseline"] == "world_models"
    assert result["experiment_target"] == "sparse_20_anchor"
    assert "ade" in result["metrics_overall"]
