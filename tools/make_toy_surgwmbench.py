"""Create a synthetic SurgWMBench dataset for tests and smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

METHODS = ("linear", "pchip", "akima", "cubic_spline")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _sampled_indices(num_frames: int) -> list[int]:
    if num_frames < 20:
        raise ValueError("Toy SurgWMBench clips must have at least 20 frames")
    indices = np.linspace(0, num_frames - 1, 20).round().astype(int).tolist()
    indices[0] = 0
    indices[-1] = num_frames - 1
    for pos in range(1, len(indices)):
        if indices[pos] <= indices[pos - 1]:
            indices[pos] = indices[pos - 1] + 1
    for pos in range(len(indices) - 2, -1, -1):
        if indices[pos] >= indices[pos + 1]:
            indices[pos] = indices[pos + 1] - 1
    return [int(value) for value in indices]


def _coord_for_frame(frame_idx: int, num_frames: int, width: int, height: int, offset: float = 0.0) -> tuple[float, float]:
    denom = max(num_frames - 1, 1)
    x = 5.0 + (width - 10.0) * frame_idx / denom
    y = 6.0 + (height - 12.0) * ((frame_idx + offset) % num_frames) / denom
    return float(x), float(y)


def create_toy_surgwmbench(root: str | Path, num_clips: int = 2) -> Path:
    """Create a final-layout toy SurgWMBench root and return its path."""

    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    (root / "videos" / "video_01").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "video_01" / "video_left.avi").write_bytes(b"synthetic")

    rows: list[dict[str, Any]] = []
    lengths = [25, 31, 28, 33]
    difficulties = ["low", "medium", "high", None]
    width, height = 64, 48
    source_video_id = "video_01"
    source_video_path = "videos/video_01/video_left.avi"

    for clip_idx in range(num_clips):
        patient_id = f"patient_{clip_idx + 1:03d}"
        trajectory_id = f"traj_{clip_idx + 1:03d}"
        num_frames = lengths[clip_idx % len(lengths)]
        difficulty = difficulties[clip_idx % len(difficulties)]
        frames_dir_rel = f"clips/{patient_id}/{trajectory_id}/frames"
        annotation_rel = f"clips/{patient_id}/{trajectory_id}/annotation.json"
        sampled = _sampled_indices(num_frames)
        sampled_set = set(sampled)

        frame_records: list[dict[str, Any]] = []
        for frame_idx in range(num_frames):
            x, y = _coord_for_frame(frame_idx, num_frames, width, height, offset=clip_idx)
            image = Image.new("RGB", (width, height), color=(20 + clip_idx * 20, 20, 35))
            draw = ImageDraw.Draw(image)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 80, 40))
            frame_rel = f"{frames_dir_rel}/{frame_idx:06d}.jpg"
            (root / frame_rel).parent.mkdir(parents=True, exist_ok=True)
            image.save(root / frame_rel)
            frame_records.append(
                {
                    "local_frame_idx": frame_idx,
                    "source_frame_idx": 1000 + frame_idx,
                    "path": frame_rel,
                }
            )

        human_anchors: list[dict[str, Any]] = []
        for anchor_idx, local_idx in enumerate(sampled):
            x, y = _coord_for_frame(local_idx, num_frames, width, height, offset=clip_idx)
            human_anchors.append(
                {
                    "anchor_idx": anchor_idx,
                    "local_frame_idx": int(local_idx),
                    "source_frame_idx": 1000 + int(local_idx),
                    "old_frame_idx": anchor_idx,
                    "coord_px": [x, y],
                    "coord_norm": [x / width, y / height],
                }
            )

        interpolation_files = {
            method: f"interpolations/{patient_id}/{trajectory_id}.{method}.json" for method in METHODS
        }
        anchor_by_frame = {anchor["local_frame_idx"]: anchor for anchor in human_anchors}
        for method_idx, method in enumerate(METHODS):
            coords: list[dict[str, Any]] = []
            for frame_idx in range(num_frames):
                if frame_idx in sampled_set:
                    anchor = anchor_by_frame[frame_idx]
                    coord_px = anchor["coord_px"]
                    coord_norm = anchor["coord_norm"]
                    source = "human"
                    anchor_idx = anchor["anchor_idx"]
                    confidence = 1.0
                    label_weight = 1.0
                else:
                    x, y = _coord_for_frame(frame_idx, num_frames, width, height, offset=clip_idx + method_idx * 0.25)
                    coord_px = [x, y]
                    coord_norm = [x / width, y / height]
                    source = "interpolated"
                    anchor_idx = None
                    confidence = 0.6
                    label_weight = 0.5
                coords.append(
                    {
                        "local_frame_idx": frame_idx,
                        "coord_px": coord_px,
                        "coord_norm": coord_norm,
                        "source": source,
                        "anchor_idx": anchor_idx,
                        "confidence": confidence,
                        "label_weight": label_weight,
                        "is_out_of_bounds": False,
                    }
                )
            _write_json(root / interpolation_files[method], {"coordinates": coords})

        annotation = {
            "dataset_version": "SurgWMBench",
            "patient_id": patient_id,
            "source_video_id": source_video_id,
            "source_video_path": source_video_path,
            "trajectory_id": trajectory_id,
            "difficulty": difficulty,
            "num_frames": num_frames,
            "frames": frame_records,
            "human_anchors": human_anchors,
            "sampled_indices": sampled,
            "interpolation_files": interpolation_files,
            "default_interpolation_method": "linear",
            "image_size": {"width": width, "height": height},
        }
        _write_json(root / annotation_rel, annotation)

        rows.append(
            {
                "dataset_version": "SurgWMBench",
                "patient_id": patient_id,
                "source_video_id": source_video_id,
                "source_video_path": source_video_path,
                "trajectory_id": trajectory_id,
                "difficulty": difficulty,
                "num_frames": num_frames,
                "annotation_path": annotation_rel,
                "frames_dir": frames_dir_rel,
                "interpolation_files": interpolation_files,
                "default_interpolation_method": "linear",
                "num_human_anchors": 20,
                "sampled_indices": sampled,
            }
        )

    for split in ("train", "val", "test", "all"):
        _write_jsonl(root / "manifests" / f"{split}.jsonl", rows)

    _write_json(root / "metadata" / "source_videos.json", [{"source_video_id": source_video_id}])
    _write_json(root / "metadata" / "validation_report.json", {"ok": True, "num_clips": len(rows)})
    _write_json(root / "metadata" / "dataset_stats.json", {"num_clips": len(rows)})
    _write_json(root / "metadata" / "difficulty_rubric.json", {})
    _write_json(root / "metadata" / "interpolation_config.json", {"default": "linear", "methods": list(METHODS)})
    (root / "README.md").write_text(
        "# SurgWMBench Toy Dataset\n\n"
        "Synthetic final-layout SurgWMBench fixture for loader and smoke tests.\n\n"
        "- dataset_version: SurgWMBench\n"
        "- human anchors per clip: 20\n"
        "- dense coordinates are pseudo coordinates except at human anchors\n",
        encoding="utf-8",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-clips", type=int, default=2)
    args = parser.parse_args()
    root = create_toy_surgwmbench(args.output, num_clips=args.num_clips)
    print(f"Created toy SurgWMBench dataset at {root}")


if __name__ == "__main__":
    main()
