from __future__ import annotations

import argparse
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import yaml

from yolo_raw_extractor.dataset_utils import IMAGE_EXTENSIONS, find_image_path

LOGGER = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
DEFAULT_PREFIX = "imported_face_"
SPLIT_ALIASES = {"valid": "val", "validation": "val"}
CONFIG_NAMES = ("data.yaml", "dataset.yaml")
PERSON_CLASS_ALIASES = frozenset(
    {
        "face",
        "faces",
        "person",
        "people",
        "mask",
        "no-mask",
        "no_mask",
        "nomask",
        "with_mask",
        "without_mask",
    }
)


@dataclass
class ImportStats:
    images_copied: int = 0
    labels_written: int = 0
    objects_imported: int = 0
    objects_skipped: int = 0
    splits: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.splits is None:
            self.splits = {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolo-bbox-import",
        description=(
            "Import a YOLO object-detection dataset and convert bbox labels into "
            "a GDPR segmentation dataset (class 0 = Person polygons)."
        ),
    )
    parser.add_argument(
        "bbox_src_dir",
        type=Path,
        help="Root of the bbox source dataset (contains data.yaml and split folders).",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Target dataset root (e.g. data/dataset_raw) with images/<split>/.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Filename prefix to avoid collisions (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Do not add a filename prefix (use when dataset_raw contains only this import).",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=PERSON_CLASS_ID,
        help="Target class index for imported faces (default: 0 = Person).",
    )
    parser.add_argument(
        "--map-all-classes",
        action="store_true",
        help="Map every annotated class to the person class id (ignore name filter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination image/label pairs with the same stem.",
    )
    return parser.parse_args(argv)


def _load_bbox_src_config(bbox_src_dir: Path) -> dict:
    for name in CONFIG_NAMES:
        path = bbox_src_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def _normalize_split_name(name: str) -> str:
    lowered = name.strip().lower()
    return SPLIT_ALIASES.get(lowered, lowered)


def _resolve_bbox_src_path(bbox_src_dir: Path, entry: str | Path) -> Path:
    candidate = Path(entry)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (bbox_src_dir / candidate).resolve()
    if resolved.is_dir():
        return resolved
    # Some YOLO detection exports use ../train/images relative to the source root.
    parts = candidate.parts
    if parts and parts[0] == "..":
        trimmed = Path(*parts[1:])
        fallback = (bbox_src_dir / trimmed).resolve()
        if fallback.is_dir():
            return fallback
    return resolved


def _split_image_label_dirs(split_root: Path) -> tuple[Path, Path] | None:
    """Return image and label directories for a standard YOLO detection split."""
    images = split_root / "images"
    labels = split_root / "labels"
    if images.is_dir() and labels.is_dir():
        return images, labels

    if split_root.name in {"images", "labels"}:
        return None

    nested_images = split_root / "images"
    nested_labels = split_root / "labels"
    if nested_images.is_dir() and nested_labels.is_dir():
        return nested_images, nested_labels

    if any(split_root.glob(f"*{ext}") for ext in IMAGE_EXTENSIONS):
        return split_root, split_root

    return None


def discover_bbox_splits(bbox_src_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Resolve bbox source split directories to (images_dir, labels_dir) pairs."""
    bbox_src_dir = bbox_src_dir.resolve()
    config = _load_bbox_src_config(bbox_src_dir)
    discovered: dict[str, tuple[Path, Path]] = {}

    for key in ("train", "val", "valid", "test"):
        entry = config.get(key)
        if not entry:
            continue
        split_name = _normalize_split_name(key)
        image_dir = _resolve_bbox_src_path(bbox_src_dir, entry)
        if not image_dir.is_dir():
            fallback = _split_image_label_dirs(bbox_src_dir / split_name)
            if fallback is None and split_name == "val":
                fallback = _split_image_label_dirs(bbox_src_dir / "valid")
            if fallback is None:
                msg = f"Configured split path does not exist: {image_dir}"
                raise FileNotFoundError(msg)
            image_dir, label_dir = fallback
        else:
            label_dir = _guess_label_dir(image_dir)
        discovered[split_name] = (image_dir, label_dir)

    if discovered:
        return discovered

    for child in sorted(bbox_src_dir.iterdir()):
        if not child.is_dir():
            continue
        split_name = _normalize_split_name(child.name)
        if split_name not in {"train", "val", "test"}:
            continue
        pair = _split_image_label_dirs(child)
        if pair is not None:
            discovered[split_name] = pair

    if not discovered:
        msg = (
            f"No YOLO detection splits found under {bbox_src_dir}. "
            "Expected data.yaml with train/val/test paths or train/, valid/, test/ folders."
        )
        raise FileNotFoundError(msg)

    return discovered


def _guess_label_dir(image_dir: Path) -> Path:
    parts = image_dir.parts
    if "images" in parts:
        idx = parts.index("images")
        candidate = Path(*parts[:idx], "labels", *parts[idx + 1 :])
        if candidate.is_dir():
            return candidate
    sibling = image_dir.parent / "labels"
    if sibling.is_dir():
        return sibling
    return image_dir


def build_person_class_map(
    names: dict[int, str],
    *,
    map_all_classes: bool,
    person_class_id: int,
) -> dict[int, int]:
    """Map source class indices to the target person class id."""
    if map_all_classes:
        return {idx: person_class_id for idx in names}

    mapping: dict[int, int] = {}
    for idx, label in names.items():
        token = label.strip().lower().replace(" ", "-")
        if token in PERSON_CLASS_ALIASES:
            mapping[idx] = person_class_id
    return mapping


def load_bbox_class_names(bbox_src_dir: Path) -> dict[int, str]:
    config = _load_bbox_src_config(bbox_src_dir)
    raw_names = config.get("names", {})
    names: dict[int, str] = {}
    if isinstance(raw_names, dict):
        for key, value in raw_names.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw_names, list):
        for idx, value in enumerate(raw_names):
            names[idx] = str(value)
    return names


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def bbox_to_polygon_tokens(cx: float, cy: float, w: float, h: float) -> list[str]:
    """Convert YOLO bbox (cx, cy, w, h) to four normalized polygon corners."""
    half_w = w / 2.0
    half_h = h / 2.0
    corners = (
        (cx - half_w, cy - half_h),
        (cx + half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        (cx - half_w, cy + half_h),
    )
    tokens: list[str] = []
    for x, y in corners:
        tokens.append(f"{_clamp01(x):.6f}")
        tokens.append(f"{_clamp01(y):.6f}")
    return tokens


def convert_label_line(
    line: str,
    class_map: dict[int, int],
    *,
    person_class_id: int,
) -> str | None:
    """Convert one YOLO detection or segmentation line to a Person polygon line."""
    stripped = line.strip()
    if not stripped:
        return None

    tokens = stripped.split()
    if len(tokens) < 5:
        return None

    source_class = int(float(tokens[0]))
    target_class = class_map.get(source_class)
    if target_class is None:
        return None

    coords = [float(value) for value in tokens[1:]]
    if len(coords) == 4:
        polygon_tokens = bbox_to_polygon_tokens(*coords)
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        polygon_tokens = [f"{_clamp01(value):.6f}" for value in coords]
    else:
        return None

    return f"{target_class} " + " ".join(polygon_tokens)


def convert_label_file(
    source: Path,
    class_map: dict[int, int],
    *,
    person_class_id: int,
) -> tuple[list[str], int, int]:
    """Return converted lines plus imported/skipped object counts."""
    imported = 0
    skipped = 0
    converted: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = convert_label_line(
            raw_line,
            class_map,
            person_class_id=person_class_id,
        )
        if line is None:
            if raw_line.strip():
                skipped += 1
            continue
        converted.append(line)
        imported += 1
    return converted, imported, skipped


def _safe_stem(stem: str, prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not cleaned:
        cleaned = "image"
    if not prefix or cleaned.startswith(prefix):
        return cleaned
    return f"{prefix}{cleaned}"


def write_dataset_yaml(dataset_dir: Path, *, person_class_id: int = PERSON_CLASS_ID) -> Path:
    """Write dataset.yaml for a Person-only segmentation dataset."""
    config = {
        "name": dataset_dir.name,
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {person_class_id: "Person"},
    }
    path = dataset_dir / "dataset.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def import_bbox_to_segment(
    bbox_src_dir: Path,
    dataset_dir: Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    person_class_id: int = PERSON_CLASS_ID,
    map_all_classes: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> ImportStats:
    """Copy bbox source images and YOLO Person segmentation labels into dataset_dir."""
    bbox_src_dir = bbox_src_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    splits = discover_bbox_splits(bbox_src_dir)
    class_names = load_bbox_class_names(bbox_src_dir)
    class_map = build_person_class_map(
        class_names,
        map_all_classes=map_all_classes,
        person_class_id=person_class_id,
    )
    if not class_map:
        labels = ", ".join(f"{idx}={name}" for idx, name in sorted(class_names.items()))
        msg = (
            "No source classes matched Person/face aliases. "
            f"Found: {labels or '(none)'}. "
            "Use --map-all-classes to import every class as Person."
        )
        raise ValueError(msg)

    LOGGER.info(
        "Class mapping %s -> person id %s",
        {class_names.get(k, k): v for k, v in class_map.items()},
        person_class_id,
    )

    stats = ImportStats()
    for split_name, (image_dir, label_dir) in sorted(splits.items()):
        dest_dir = dataset_dir / "images" / split_name
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(label_dir.glob("*.txt"))
        split_count = 0
        for label_path in label_files:
            image_path = find_image_path(label_path, image_dir)
            if image_path is None:
                LOGGER.warning("Skipping %s: no matching image in %s", label_path, image_dir)
                continue

            converted, imported, skipped = convert_label_file(
                label_path,
                class_map,
                person_class_id=person_class_id,
            )
            if not converted:
                LOGGER.debug("Skipping empty label file %s", label_path)
                continue

            dest_stem = _safe_stem(label_path.stem, prefix)
            dest_image = dest_dir / f"{dest_stem}{image_path.suffix.lower()}"
            dest_label = dest_dir / f"{dest_stem}.txt"

            if dest_image.exists() or dest_label.exists():
                if not overwrite:
                    LOGGER.warning(
                        "Skipping %s: destination exists (use --overwrite)",
                        dest_stem,
                    )
                    continue

            if dry_run:
                LOGGER.info(
                    "Would import %s -> %s (%s objects)",
                    image_path.name,
                    dest_image.relative_to(dataset_dir),
                    len(converted),
                )
            else:
                shutil.copy2(image_path, dest_image)
                dest_label.write_text("\n".join(converted) + "\n", encoding="utf-8")
                stats.images_copied += 1
                stats.labels_written += 1

            stats.objects_imported += imported
            stats.objects_skipped += skipped
            split_count += 1

        stats.splits[split_name] = split_count
        LOGGER.info(
            "Split %s: imported %s labeled images from %s",
            split_name,
            split_count,
            image_dir,
        )

    if not dry_run and stats.images_copied > 0:
        yaml_path = write_dataset_yaml(dataset_dir, person_class_id=person_class_id)
        LOGGER.info("Wrote %s", yaml_path)

    return stats


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )


def _run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging()

    try:
        prefix = "" if args.no_prefix else args.prefix
        stats = import_bbox_to_segment(
            args.bbox_src_dir,
            args.dataset_dir,
            prefix=prefix,
            person_class_id=args.person_class_id,
            map_all_classes=args.map_all_classes,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Done. images=%s labels=%s objects=%s skipped=%s splits=%s",
        stats.images_copied,
        stats.labels_written,
        stats.objects_imported,
        stats.objects_skipped,
        stats.splits,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(_run(argv))


__all__ = [
    "ImportStats",
    "bbox_to_polygon_tokens",
    "build_person_class_map",
    "convert_label_line",
    "discover_bbox_splits",
    "import_bbox_to_segment",
    "load_bbox_class_names",
    "main",
    "write_dataset_yaml",
]
