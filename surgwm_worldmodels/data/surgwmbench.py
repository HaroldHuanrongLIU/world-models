"""SurgWMBench final-layout datasets.

The loader follows the final SurgWMBench manifest contract and does not create
or infer train/val/test splits. Sparse human anchors remain the primary target;
dense interpolation coordinates are exposed as auxiliary pseudo labels.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import load_rgb_frame

FrameSampling = Literal["sparse_anchors", "dense", "all", "window"]

DATASET_VERSION = "SurgWMBench"
NUM_HUMAN_ANCHORS = 20
INTERPOLATION_METHODS = ("linear", "pchip", "akima", "cubic_spline")
SOURCE_TO_CODE = {"unlabeled": 0, "human": 1, "interpolated": 2}
CODE_TO_SOURCE = {value: key for key, value in SOURCE_TO_CODE.items()}


def read_jsonl_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read an official SurgWMBench JSONL manifest."""

    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() != ".jsonl":
        raise ValueError(f"SurgWMBench manifests must be JSONL files: {manifest_path}")

    entries: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {manifest_path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Manifest entry at {manifest_path}:{line_no} must be an object.")
            entries.append(item)
    if not entries:
        raise ValueError(f"Manifest contains no entries: {manifest_path}")
    return entries


def load_json(path: str | Path) -> Any:
    json_path = Path(path).expanduser()
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset_path(dataset_root: str | Path, value: str | Path | None) -> Path | None:
    """Resolve a dataset-relative path while preserving the stored path value."""

    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(dataset_root).expanduser() / path


def _parse_image_size(value: Any) -> tuple[int, int]:
    """Return image size as ``(height, width)``.

    Dict forms are unambiguous. Two-element sequences are accepted for common
    datasets that store either ``[height, width]`` or landscape ``[width,
    height]``.
    """

    if isinstance(value, dict):
        width = value.get("width", value.get("w"))
        height = value.get("height", value.get("h"))
        if width is None or height is None:
            nested = value.get("size", value.get("image_size"))
            if nested is not None:
                return _parse_image_size(nested)
        if width is None or height is None:
            raise ValueError(f"image_size dict must contain width/height: {value!r}")
        return int(height), int(width)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = int(value[0]), int(value[1])
        if first > second:
            return second, first
        return first, second
    raise ValueError(f"Unsupported image_size value: {value!r}")


def _coord_from_item(item: dict[str, Any], image_size_hw: tuple[int, int]) -> tuple[list[float], list[float]]:
    coord_px = (
        item.get("coord_px")
        or item.get("coordinate_px")
        or item.get("coordinates_px")
        or item.get("coord")
        or item.get("xy")
    )
    coord_norm = item.get("coord_norm") or item.get("coordinate_norm") or item.get("coordinates_norm")

    height, width = image_size_hw
    scale = np.asarray([width, height], dtype=np.float32)
    if coord_px is None and coord_norm is None:
        raise ValueError(f"Coordinate entry has neither coord_px nor coord_norm: {item!r}")

    if coord_px is not None:
        px = np.asarray(coord_px, dtype=np.float32)
        if px.shape != (2,):
            raise ValueError(f"coord_px must have shape [2], got {px.shape}")
    else:
        norm = np.asarray(coord_norm, dtype=np.float32)
        if norm.shape != (2,):
            raise ValueError(f"coord_norm must have shape [2], got {norm.shape}")
        px = norm * scale

    if coord_norm is not None:
        norm = np.asarray(coord_norm, dtype=np.float32)
        if norm.shape != (2,):
            raise ValueError(f"coord_norm must have shape [2], got {norm.shape}")
    else:
        norm = px / scale
    return px.astype(np.float32).tolist(), norm.astype(np.float32).tolist()


def _source_code(value: Any) -> int:
    if value is None:
        return SOURCE_TO_CODE["unlabeled"]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized not in SOURCE_TO_CODE:
            raise ValueError(f"Unknown coordinate source: {value!r}")
        return SOURCE_TO_CODE[normalized]
    if isinstance(value, (int, np.integer)):
        code = int(value)
        if code not in CODE_TO_SOURCE:
            raise ValueError(f"Unknown coordinate source code: {code}")
        return code
    raise ValueError(f"Unsupported coordinate source value: {value!r}")


def _frame_local_index(frame: Any, fallback: int) -> int:
    if isinstance(frame, dict):
        value = frame.get("local_frame_idx", frame.get("frame_idx", frame.get("index", fallback)))
        return int(value)
    return fallback


def _frame_path_value(frame: Any) -> str | None:
    if isinstance(frame, str):
        return frame
    if isinstance(frame, dict):
        for key in ("path", "frame_path", "file_path", "relative_path"):
            value = frame.get(key)
            if value is not None:
                return str(value)
        for key in ("filename", "file_name", "name"):
            value = frame.get(key)
            if value is not None:
                return str(value)
    return None


def _metadata_from_sample(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "patient_id",
        "source_video_id",
        "source_video_path",
        "trajectory_id",
        "difficulty",
        "num_frames",
        "image_size_original",
        "annotation_path",
        "interpolation_path",
        "interpolation_method",
    )
    return {key: item.get(key) for key in keys}


class SurgWMBenchClipDataset(Dataset):
    """Load SurgWMBench clips from official final-layout manifests."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        interpolation_method: str | None = None,
        image_size: int | tuple[int, int] = 128,
        frame_sampling: FrameSampling = "sparse_anchors",
        max_frames: int | None = None,
        use_dense_pseudo: bool = False,
        return_images: bool = True,
        normalize_coords: bool = True,
        cache_annotations: bool = True,
        strict: bool = True,
        allow_legacy_version: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        manifest_path = Path(manifest).expanduser()
        self.manifest_path = manifest_path if manifest_path.is_absolute() else self.dataset_root / manifest_path
        self.interpolation_method = interpolation_method
        self.image_size = image_size
        self.frame_sampling = frame_sampling
        self.max_frames = max_frames
        self.use_dense_pseudo = use_dense_pseudo
        self.return_images = return_images
        self.normalize_coords = normalize_coords
        self.cache_annotations = cache_annotations
        self.strict = strict
        self.allow_legacy_version = allow_legacy_version
        self._annotation_cache: dict[Path, dict[str, Any]] = {}

        if frame_sampling not in ("sparse_anchors", "dense", "all", "window"):
            raise ValueError(f"Unsupported frame_sampling={frame_sampling!r}")
        if interpolation_method is not None and interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(f"Unsupported interpolation_method={interpolation_method!r}")

        self.entries = read_jsonl_manifest(self.manifest_path)
        if self.strict:
            for index, entry in enumerate(self.entries):
                self._validate_manifest_entry(entry, index)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        if self.strict:
            self._validate_manifest_entry(entry, index)
        annotation_path = self._annotation_path(entry)
        annotation = self._load_annotation(annotation_path)
        if self.strict:
            self._validate_annotation(entry, annotation, annotation_path)

        method = self._selected_interpolation_method(entry, annotation)
        interpolation_path = self._interpolation_path(entry, annotation, method)
        image_size_hw = self._annotation_image_size(annotation)
        human_anchors = self._human_anchors(annotation)
        sampled_indices = self._sampled_indices(entry, annotation)
        frame_records = self._frame_records(entry, annotation)
        num_frames = int(entry.get("num_frames", annotation.get("num_frames", len(frame_records))))

        anchor_local_indices, anchor_coords_px, anchor_coords_norm = self._anchor_arrays(human_anchors, image_size_hw)
        selected_indices = self._select_indices(num_frames, anchor_local_indices)
        selected_frame_paths = self._paths_for_indices(entry, frame_records, selected_indices)
        frames = self._load_frames(selected_frame_paths) if self.return_images else None

        dense = None
        if self.frame_sampling in ("dense", "all", "window") or self.use_dense_pseudo:
            dense = self._load_dense_coordinates(interpolation_path, image_size_hw, num_frames)
            if self.strict:
                self._validate_dense_anchors(dense, anchor_local_indices, anchor_coords_px, interpolation_path)

        if self.frame_sampling == "sparse_anchors":
            selected_coords_px = anchor_coords_px
            selected_coords_norm = anchor_coords_norm
            selected_sources = torch.full((NUM_HUMAN_ANCHORS,), SOURCE_TO_CODE["human"], dtype=torch.long)
            selected_label_weights = torch.ones(NUM_HUMAN_ANCHORS, dtype=torch.float32)
            selected_confidence = torch.ones(NUM_HUMAN_ANCHORS, dtype=torch.float32)
        else:
            if dense is None:
                dense = self._load_dense_coordinates(interpolation_path, image_size_hw, num_frames)
                if self.strict:
                    self._validate_dense_anchors(dense, anchor_local_indices, anchor_coords_px, interpolation_path)
            positions = torch.as_tensor(selected_indices, dtype=torch.long)
            selected_coords_px = dense["coords_px"][positions]
            selected_coords_norm = dense["coords_norm"][positions]
            selected_sources = dense["sources"][positions]
            selected_label_weights = dense["label_weights"][positions]
            selected_confidence = dense["confidence"][positions]

        return {
            "patient_id": str(entry.get("patient_id", annotation.get("patient_id", ""))),
            "source_video_id": str(entry.get("source_video_id", annotation.get("source_video_id", ""))),
            "source_video_path": str(entry.get("source_video_path", annotation.get("source_video_path", ""))),
            "trajectory_id": str(entry.get("trajectory_id", annotation.get("trajectory_id", ""))),
            "difficulty": entry.get("difficulty", annotation.get("difficulty")),
            "num_frames": num_frames,
            "image_size_original": image_size_hw,
            "frames": frames,
            "frame_paths": [str(path) for path in selected_frame_paths],
            "frame_indices": torch.as_tensor(selected_indices, dtype=torch.long),
            "sampled_indices": torch.as_tensor(sampled_indices, dtype=torch.long),
            "human_anchor_coords_px": anchor_coords_px,
            "human_anchor_coords_norm": anchor_coords_norm,
            "human_anchor_local_indices": torch.as_tensor(anchor_local_indices, dtype=torch.long),
            "dense_coords_px": None if dense is None else dense["coords_px"],
            "dense_coords_norm": None if dense is None else dense["coords_norm"],
            "dense_coord_sources": None if dense is None else dense["sources"],
            "dense_label_weights": None if dense is None else dense["label_weights"],
            "dense_confidence": None if dense is None else dense["confidence"],
            "selected_coords_norm": selected_coords_norm,
            "selected_coords_px": selected_coords_px,
            "selected_coord_sources": selected_sources,
            "selected_label_weights": selected_label_weights,
            "selected_confidence": selected_confidence,
            "interpolation_method": method,
            "annotation_path": str(annotation_path),
            "interpolation_path": str(interpolation_path),
        }

    def _check_version(self, version: Any, where: str) -> None:
        if version == DATASET_VERSION:
            return
        message = f"{where} dataset_version={version!r}, expected {DATASET_VERSION!r}"
        if self.allow_legacy_version:
            warnings.warn(f"Allowing legacy dataset_version: {message}", UserWarning, stacklevel=3)
            return
        raise ValueError(message)

    def _validate_manifest_entry(self, entry: dict[str, Any], index: int) -> None:
        self._check_version(entry.get("dataset_version"), f"Manifest entry {index}")
        required = (
            "dataset_version",
            "patient_id",
            "source_video_id",
            "source_video_path",
            "trajectory_id",
            "difficulty",
            "num_frames",
            "annotation_path",
            "frames_dir",
            "interpolation_files",
            "default_interpolation_method",
            "num_human_anchors",
            "sampled_indices",
        )
        missing = [field for field in required if field not in entry]
        if missing:
            raise ValueError(f"Manifest entry {index} is missing required fields: {missing}")
        if int(entry["num_human_anchors"]) != NUM_HUMAN_ANCHORS:
            raise ValueError(f"Manifest entry {index} num_human_anchors must be {NUM_HUMAN_ANCHORS}")
        if not isinstance(entry["sampled_indices"], list) or len(entry["sampled_indices"]) != NUM_HUMAN_ANCHORS:
            raise ValueError(f"Manifest entry {index} sampled_indices must contain {NUM_HUMAN_ANCHORS} values")
        if not isinstance(entry["interpolation_files"], dict):
            raise ValueError(f"Manifest entry {index} interpolation_files must be a mapping")

    def _validate_annotation(self, entry: dict[str, Any], annotation: dict[str, Any], annotation_path: Path) -> None:
        self._check_version(annotation.get("dataset_version", entry.get("dataset_version")), f"Annotation {annotation_path}")
        frames = annotation.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Annotation {annotation_path} must contain non-empty frames[]")
        human_anchors = annotation.get("human_anchors")
        if not isinstance(human_anchors, list):
            raise ValueError(f"Annotation {annotation_path} must contain human_anchors[]")
        if len(human_anchors) != NUM_HUMAN_ANCHORS:
            raise ValueError(
                f"Annotation {annotation_path} must contain exactly {NUM_HUMAN_ANCHORS} human anchors, got {len(human_anchors)}"
            )
        sampled = self._sampled_indices(entry, annotation)
        if len(sampled) != NUM_HUMAN_ANCHORS:
            raise ValueError(f"Annotation {annotation_path} sampled_indices must contain {NUM_HUMAN_ANCHORS} values")
        if list(map(int, entry.get("sampled_indices", sampled))) != sampled:
            raise ValueError(f"Annotation {annotation_path} sampled_indices differ from manifest")
        if len(frames) != int(entry.get("num_frames", annotation.get("num_frames", len(frames)))):
            raise ValueError(f"Annotation {annotation_path} frames[] length differs from num_frames")
        _ = self._annotation_image_size(annotation)

    def _annotation_path(self, entry: dict[str, Any]) -> Path:
        path = resolve_dataset_path(self.dataset_root, entry.get("annotation_path"))
        if path is None:
            raise ValueError("Manifest entry is missing annotation_path")
        if not path.exists():
            raise FileNotFoundError(f"Annotation not found: {path}")
        return path

    def _load_annotation(self, path: Path) -> dict[str, Any]:
        if self.cache_annotations and path in self._annotation_cache:
            return self._annotation_cache[path]
        annotation = load_json(path)
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation must be a JSON object: {path}")
        if self.cache_annotations:
            self._annotation_cache[path] = annotation
        return annotation

    def _selected_interpolation_method(self, entry: dict[str, Any], annotation: dict[str, Any]) -> str:
        method = self.interpolation_method or entry.get("default_interpolation_method") or annotation.get(
            "default_interpolation_method"
        )
        if method is None:
            method = "linear"
        if method not in INTERPOLATION_METHODS:
            raise ValueError(f"Unsupported interpolation method: {method!r}")
        return str(method)

    def _interpolation_path(self, entry: dict[str, Any], annotation: dict[str, Any], method: str) -> Path:
        interpolation_files = entry.get("interpolation_files") or annotation.get("interpolation_files")
        if not isinstance(interpolation_files, dict):
            raise ValueError("interpolation_files must be a mapping from method name to relative path")
        if method not in interpolation_files:
            raise FileNotFoundError(f"Interpolation method {method!r} is not listed for trajectory")
        path = resolve_dataset_path(self.dataset_root, interpolation_files[method])
        if path is None or not path.exists():
            raise FileNotFoundError(f"Interpolation file not found for method {method!r}: {path}")
        return path

    def _annotation_image_size(self, annotation: dict[str, Any]) -> tuple[int, int]:
        if "image_size" not in annotation:
            raise ValueError("Annotation is missing image_size")
        return _parse_image_size(annotation["image_size"])

    def _human_anchors(self, annotation: dict[str, Any]) -> list[dict[str, Any]]:
        anchors = annotation.get("human_anchors")
        if not isinstance(anchors, list):
            raise ValueError("Annotation is missing human_anchors[]")
        if len(anchors) != NUM_HUMAN_ANCHORS:
            raise ValueError(f"Expected {NUM_HUMAN_ANCHORS} human anchors, got {len(anchors)}")
        if not all(isinstance(anchor, dict) for anchor in anchors):
            raise ValueError("Every human_anchors[] entry must be an object")
        return sorted(anchors, key=lambda item: int(item.get("anchor_idx", NUM_HUMAN_ANCHORS)))

    def _sampled_indices(self, entry: dict[str, Any], annotation: dict[str, Any]) -> list[int]:
        sampled = annotation.get("sampled_indices", entry.get("sampled_indices"))
        if not isinstance(sampled, list):
            raise ValueError("sampled_indices must be a list")
        return [int(value) for value in sampled]

    def _frame_records(self, entry: dict[str, Any], annotation: dict[str, Any]) -> list[Any]:
        frames = annotation.get("frames")
        if isinstance(frames, list) and frames:
            return frames
        raise ValueError("Annotation must contain a non-empty frames[] list")

    def _anchor_arrays(
        self, anchors: list[dict[str, Any]], image_size_hw: tuple[int, int]
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        local_indices: list[int] = []
        coords_px: list[list[float]] = []
        coords_norm: list[list[float]] = []
        for anchor_pos, anchor in enumerate(anchors):
            local_indices.append(int(anchor.get("local_frame_idx", anchor.get("frame_idx", anchor_pos))))
            coord_px, coord_norm = _coord_from_item(anchor, image_size_hw)
            coords_px.append(coord_px)
            coords_norm.append(coord_norm)
        return (
            local_indices,
            torch.as_tensor(coords_px, dtype=torch.float32),
            torch.as_tensor(coords_norm, dtype=torch.float32),
        )

    def _load_dense_coordinates(
        self, interpolation_path: Path, image_size_hw: tuple[int, int], num_frames: int
    ) -> dict[str, torch.Tensor]:
        interpolation = load_json(interpolation_path)
        coordinates = interpolation.get("coordinates") if isinstance(interpolation, dict) else None
        if not isinstance(coordinates, list):
            raise ValueError(f"Interpolation file must contain coordinates[]: {interpolation_path}")
        if self.strict and len(coordinates) != num_frames:
            raise ValueError(
                f"Interpolation file {interpolation_path} has {len(coordinates)} coordinates; expected {num_frames}"
            )

        coords_px = torch.zeros(num_frames, 2, dtype=torch.float32)
        coords_norm = torch.zeros(num_frames, 2, dtype=torch.float32)
        sources = torch.zeros(num_frames, dtype=torch.long)
        label_weights = torch.zeros(num_frames, dtype=torch.float32)
        confidence = torch.zeros(num_frames, dtype=torch.float32)
        seen: set[int] = set()

        for fallback_idx, item in enumerate(coordinates):
            if not isinstance(item, dict):
                raise ValueError(f"Interpolation coordinate entry must be an object: {interpolation_path}")
            local_idx = int(item.get("local_frame_idx", fallback_idx))
            if local_idx < 0 or local_idx >= num_frames:
                if self.strict:
                    raise ValueError(f"local_frame_idx={local_idx} outside [0, {num_frames}) in {interpolation_path}")
                continue
            if self.strict and local_idx in seen:
                raise ValueError(f"Duplicate local_frame_idx={local_idx} in {interpolation_path}")
            seen.add(local_idx)
            coord_px, coord_norm = _coord_from_item(item, image_size_hw)
            coords_px[local_idx] = torch.as_tensor(coord_px, dtype=torch.float32)
            coords_norm[local_idx] = torch.as_tensor(coord_norm, dtype=torch.float32)
            sources[local_idx] = _source_code(item.get("source"))
            label_weights[local_idx] = float(item.get("label_weight", 0.0))
            confidence[local_idx] = float(item.get("confidence", 0.0))

        if self.strict and len(seen) != num_frames:
            missing = sorted(set(range(num_frames)) - seen)
            raise ValueError(f"Interpolation file {interpolation_path} is missing local_frame_idx values: {missing[:5]}")

        return {
            "coords_px": coords_px,
            "coords_norm": coords_norm,
            "sources": sources,
            "label_weights": label_weights,
            "confidence": confidence,
        }

    def _validate_dense_anchors(
        self,
        dense: dict[str, torch.Tensor],
        anchor_indices: list[int],
        anchor_coords_px: torch.Tensor,
        interpolation_path: Path,
    ) -> None:
        anchor_set = set(anchor_indices)
        for anchor_pos, local_idx in enumerate(anchor_indices):
            if int(dense["sources"][local_idx].item()) != SOURCE_TO_CODE["human"]:
                raise ValueError(f"{interpolation_path}: anchor local_frame_idx={local_idx} source is not human")
            if not torch.allclose(dense["coords_px"][local_idx], anchor_coords_px[anchor_pos], atol=1e-5):
                raise ValueError(f"{interpolation_path}: anchor local_frame_idx={local_idx} coordinate differs from human label")
            if float(dense["confidence"][local_idx].item()) != 1.0:
                raise ValueError(f"{interpolation_path}: anchor local_frame_idx={local_idx} confidence is not 1.0")
            if float(dense["label_weights"][local_idx].item()) != 1.0:
                raise ValueError(f"{interpolation_path}: anchor local_frame_idx={local_idx} label_weight is not 1.0")

        for local_idx in range(int(dense["sources"].shape[0])):
            if local_idx in anchor_set:
                continue
            if int(dense["sources"][local_idx].item()) != SOURCE_TO_CODE["interpolated"]:
                raise ValueError(f"{interpolation_path}: non-anchor local_frame_idx={local_idx} source is not interpolated")
            if not np.isclose(float(dense["confidence"][local_idx].item()), 0.6):
                raise ValueError(f"{interpolation_path}: non-anchor local_frame_idx={local_idx} confidence is not 0.6")
            if not np.isclose(float(dense["label_weights"][local_idx].item()), 0.5):
                raise ValueError(f"{interpolation_path}: non-anchor local_frame_idx={local_idx} label_weight is not 0.5")

    def _select_indices(self, num_frames: int, anchor_indices: list[int]) -> list[int]:
        if self.frame_sampling == "sparse_anchors":
            return [int(index) for index in anchor_indices]
        if self.frame_sampling in ("dense", "all"):
            return list(range(num_frames))
        if self.frame_sampling == "window":
            if self.max_frames is None or self.max_frames <= 0 or num_frames <= self.max_frames:
                return list(range(num_frames))
            return list(range(int(self.max_frames)))
        raise ValueError(f"Unsupported frame_sampling={self.frame_sampling!r}")

    def _paths_for_indices(self, entry: dict[str, Any], frames: list[Any], indices: list[int]) -> list[Path]:
        frames_dir = resolve_dataset_path(self.dataset_root, entry.get("frames_dir"))
        if frames_dir is None:
            raise ValueError("Manifest entry is missing frames_dir")

        by_index: dict[int, Path] = {}
        for fallback_idx, frame in enumerate(frames):
            local_idx = _frame_local_index(frame, fallback_idx)
            value = _frame_path_value(frame)
            if value is None:
                path = frames_dir / f"{local_idx:06d}.jpg"
            else:
                candidate = Path(value)
                path = candidate if candidate.is_absolute() else self.dataset_root / candidate
                if not path.exists() and not candidate.is_absolute():
                    path = frames_dir / value
            by_index[local_idx] = path

        paths: list[Path] = []
        for local_idx in indices:
            if local_idx not in by_index:
                by_index[local_idx] = frames_dir / f"{local_idx:06d}.jpg"
            path = by_index[local_idx]
            if self.strict and self.return_images and not path.exists():
                raise FileNotFoundError(f"No frame image found for local_frame_idx={local_idx}: {path}")
            paths.append(path)
        return paths

    def _load_frames(self, paths: list[Path]) -> torch.Tensor:
        if not paths:
            raise ValueError("Cannot load a sample with no frames")
        frames = [load_rgb_frame(path, self.image_size)[0] for path in paths]
        return torch.stack(frames, dim=0)


class SurgWMBenchFrameDataset(Dataset):
    """Frame-level dataset for VAE training on extracted clip frames."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        image_size: int | tuple[int, int] = 128,
        cache_annotations: bool = True,
        strict: bool = True,
        allow_legacy_version: bool = False,
        max_frames_per_clip: int | None = None,
    ) -> None:
        self.clip_dataset = SurgWMBenchClipDataset(
            dataset_root=dataset_root,
            manifest=manifest,
            image_size=image_size,
            frame_sampling="dense",
            return_images=False,
            cache_annotations=cache_annotations,
            strict=strict,
            allow_legacy_version=allow_legacy_version,
        )
        self.image_size = image_size
        self.records: list[dict[str, Any]] = []
        for clip_idx, entry in enumerate(self.clip_dataset.entries):
            annotation_path = self.clip_dataset._annotation_path(entry)
            annotation = self.clip_dataset._load_annotation(annotation_path)
            frame_records = self.clip_dataset._frame_records(entry, annotation)
            paths = self.clip_dataset._paths_for_indices(
                entry, frame_records, list(range(int(entry.get("num_frames", len(frame_records)))))
            )
            if max_frames_per_clip is not None:
                paths = paths[: max(int(max_frames_per_clip), 0)]
            for fallback_idx, path in enumerate(paths):
                frame_record = frame_records[fallback_idx]
                local_idx = _frame_local_index(frame_record, fallback_idx)
                self.records.append(
                    {
                        "clip_idx": clip_idx,
                        "patient_id": str(entry.get("patient_id", annotation.get("patient_id", ""))),
                        "source_video_id": str(entry.get("source_video_id", annotation.get("source_video_id", ""))),
                        "trajectory_id": str(entry.get("trajectory_id", annotation.get("trajectory_id", ""))),
                        "difficulty": entry.get("difficulty", annotation.get("difficulty")),
                        "local_frame_idx": int(local_idx),
                        "frame_path": str(path),
                        "annotation_path": str(annotation_path),
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        metadata = dict(self.records[index])
        image, _ = load_rgb_frame(metadata["frame_path"], self.image_size)
        return image, metadata


__all__ = [
    "CODE_TO_SOURCE",
    "DATASET_VERSION",
    "INTERPOLATION_METHODS",
    "NUM_HUMAN_ANCHORS",
    "SOURCE_TO_CODE",
    "SurgWMBenchClipDataset",
    "SurgWMBenchFrameDataset",
    "_coord_from_item",
    "_frame_local_index",
    "_frame_path_value",
    "_metadata_from_sample",
    "_parse_image_size",
    "load_json",
    "read_jsonl_manifest",
    "resolve_dataset_path",
]
