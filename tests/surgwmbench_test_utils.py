from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.make_toy_surgwmbench import create_toy_surgwmbench


def make_surgwmbench_root(tmp_path: Path, *, num_clips: int = 2) -> Path:
    return create_toy_surgwmbench(tmp_path / "SurgWMBench", num_clips=num_clips)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def set_dataset_version(root: Path, version: str) -> None:
    manifest = root / "manifests" / "train.jsonl"
    rows = read_jsonl(manifest)
    for row in rows:
        row["dataset_version"] = version
        annotation_path = root / row["annotation_path"]
        annotation = read_json(annotation_path)
        annotation["dataset_version"] = version
        write_json(annotation_path, annotation)
    for split in ("train", "val", "test", "all"):
        split_path = root / "manifests" / f"{split}.jsonl"
        split_rows = read_jsonl(split_path)
        for row in split_rows:
            row["dataset_version"] = version
        write_jsonl(split_path, split_rows)


def remove_interpolation_file(root: Path, method: str) -> Path:
    rows = read_jsonl(root / "manifests" / "train.jsonl")
    missing = root / rows[0]["interpolation_files"][method]
    missing.unlink()
    return missing
