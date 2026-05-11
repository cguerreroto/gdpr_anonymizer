from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_dataset_config(dataset_dir: Path) -> tuple[dict[str, Path], dict[int, str]]:
    """Load YOLO dataset metadata and split directories."""
    config_path = dataset_dir / "dataset.yaml"
    if not config_path.exists():
        msg = f"Missing dataset.yaml in {dataset_dir}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    splits: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        entry = config.get(split)
        if not entry:
            continue
        resolved = Path(entry)
        if not resolved.is_absolute():
            resolved = (dataset_dir / resolved).resolve()
        splits[split] = resolved

    raw_names = config.get("names", {})
    names: dict[int, str] = {}
    for key, value in raw_names.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        names[idx] = str(value)

    return splits, names


def derive_label_dir(image_dir: Path) -> Path:
    """Guess matching labels directory from an images directory."""
    parts = image_dir.parts
    try:
        idx = parts.index("images")
    except ValueError:
        return image_dir

    candidate = Path(*parts[:idx], "labels", *parts[idx + 1 :])
    return candidate if candidate.exists() else image_dir


def find_image_path(label_path: Path, image_dir: Path) -> Path | None:
    """Match a label file with its companion image."""
    stem = label_path.stem
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def parse_label_line(line: str, width: int, height: int) -> tuple[int, np.ndarray]:
    """Parse a YOLO segmentation line into absolute polygon coordinates."""
    tokens = line.strip().split()
    if len(tokens) < 7:
        msg = "Segmentation entry requires class id and at least three points"
        raise ValueError(msg)

    class_id = int(float(tokens[0]))
    coords = np.asarray([float(value) for value in tokens[1:]], dtype=np.float32)
    if coords.size % 2 != 0:
        msg = "Segmentation coordinates must be in x/y pairs"
        raise ValueError(msg)

    xs = coords[0::2] * width
    ys = coords[1::2] * height
    polygon = np.stack((xs, ys), axis=1)
    if polygon.shape[0] < 3:
        msg = "Segmentation polygons require at least 3 points"
        raise ValueError(msg)

    return class_id, polygon


def sanitize_class_name(name: str) -> str:
    """Produce filesystem-friendly class name tokens."""
    safe = name.lower().replace(" ", "-")
    return "".join(char for char in safe if char.isalnum() or char in {"-", "_"})


__all__ = [
    "IMAGE_EXTENSIONS",
    "derive_label_dir",
    "find_image_path",
    "load_dataset_config",
    "parse_label_line",
    "sanitize_class_name",
]
