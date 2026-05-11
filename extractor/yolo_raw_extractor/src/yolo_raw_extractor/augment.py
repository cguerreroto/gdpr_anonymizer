from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from yolo_raw_extractor.dataset_utils import (
    derive_label_dir,
    find_image_path,
    load_dataset_config,
    parse_label_line,
    sanitize_class_name,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class LabelEntry:
    class_id: int
    polygon: np.ndarray  # absolute coordinates (pixels)


@dataclass
class SegmentAsset:
    class_id: int
    class_name: str
    image: np.ndarray  # BGRA
    points: np.ndarray  # polygon points relative to asset image


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolo-augmentor",
        description="Generate deterministic augmentation variants for a labeled YOLO dataset.",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the labeled YOLO dataset (must contain dataset.yaml).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will receive the augmented YOLO dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (written to seed.txt).",
    )
    return parser.parse_args(argv)


def load_label_entries(
    label_path: Path, width: int, height: int
) -> list[LabelEntry]:
    with label_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    entries: list[LabelEntry] = []
    for line in lines:
        class_id, polygon = parse_label_line(line, width, height)
        entries.append(LabelEntry(class_id=class_id, polygon=polygon))
    return entries


def write_label_file(path: Path, entries: list[LabelEntry], width: int, height: int) -> None:
    lines: list[str] = []
    for entry in entries:
        coords: list[str] = []
        normalized = entry.polygon.astype(np.float32).copy()
        normalized[:, 0] /= width
        normalized[:, 1] /= height
        for point in normalized:
            coords.append(f"{point[0]:.6f}")
            coords.append(f"{point[1]:.6f}")
        line = f"{entry.class_id} " + " ".join(coords)
        lines.append(line)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def _uniform_with_gap(
    rng: random.Random,
    lower_range: tuple[float, float] = (0.5, 0.8),
    upper_range: tuple[float, float] = (1.2, 1.5),
) -> float:
    if rng.random() < 0.5:
        return rng.uniform(*lower_range)
    return rng.uniform(*upper_range)


def adjust_saturation_exposure(image: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_factor = _uniform_with_gap(rng)
    exp_factor = _uniform_with_gap(rng)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * exp_factor, 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted


def occlude_segments(
    image: np.ndarray,
    entries: list[LabelEntry],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    if not entries:
        return image.copy()

    result = image.copy()
    max_regions = min(len(entries), 3)
    selected = rng.sample(entries, k=max(1, min(max_regions, rng.randint(1, max_regions + 1))))
    for entry in selected:
        polygon = np.round(entry.polygon).astype(np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        x, y, w, h = cv2.boundingRect(polygon)
        if w <= 1 or h <= 1:
            continue
        occ_w = max(1, int(w * rng.uniform(0.3, 0.7)))
        occ_h = max(1, int(h * rng.uniform(0.3, 0.7)))
        max_x = max(0, w - occ_w)
        max_y = max(0, h - occ_h)
        offset_x = x + (rng.randint(0, max_x) if max_x > 0 else 0)
        offset_y = y + (rng.randint(0, max_y) if max_y > 0 else 0)

        noise = np_rng.integers(0, 256, size=(occ_h, occ_w, 3), dtype=np.uint8)
        roi = result[offset_y : offset_y + occ_h, offset_x : offset_x + occ_w]
        sub_mask = mask[offset_y : offset_y + occ_h, offset_x : offset_x + occ_w]
        if roi.size == 0 or not np.any(sub_mask):
            continue

        blend_mask = sub_mask > 0
        roi[blend_mask] = noise[blend_mask]

    return result


def rotate_with_bounds(patch: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = patch.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    rotated = cv2.warpAffine(
        patch,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated, matrix


def apply_affine_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.transform(pts, matrix.astype(np.float32))
    return transformed.reshape(-1, 2)


def place_random_segment(
    canvas: np.ndarray, assets: list[SegmentAsset], rng: random.Random
) -> tuple[bool, LabelEntry | None]:
    asset = rng.choice(assets)
    if asset.image.shape[2] < 4 or asset.points.size == 0:
        return False, None

    scale = rng.uniform(0.6, 1.3)
    scaled = cv2.resize(
        asset.image,
        (0, 0),
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )
    scaled_points = asset.points * scale
    angle = rng.uniform(0, 360)
    rotated, matrix = rotate_with_bounds(scaled, angle)
    rotated_points = apply_affine_transform(scaled_points, matrix)

    patch_height, patch_width = rotated.shape[:2]
    canvas_height, canvas_width = canvas.shape[:2]
    if patch_height <= 1 or patch_width <= 1:
        return False, None
    if patch_height >= canvas_height or patch_width >= canvas_width:
        return False, None

    max_x = canvas_width - patch_width
    max_y = canvas_height - patch_height
    pos_x = rng.randint(0, max_x) if max_x > 0 else 0
    pos_y = rng.randint(0, max_y) if max_y > 0 else 0

    alpha = rotated[:, :, 3].astype(np.float32) / 255.0
    if not np.any(alpha > 0):
        return False, None

    roi = canvas[pos_y : pos_y + patch_height, pos_x : pos_x + patch_width]
    fg = rotated[:, :, :3].astype(np.float32)
    bg = roi.astype(np.float32)
    blended = fg * alpha[..., None] + bg * (1.0 - alpha[..., None])
    roi[:] = blended.astype(np.uint8)

    polygon = rotated_points + np.array([[pos_x, pos_y]], dtype=np.float32)
    return True, LabelEntry(asset.class_id, polygon)


def insert_segments(
    image: np.ndarray,
    entries: list[LabelEntry],
    assets: list[SegmentAsset],
    rng: random.Random,
) -> tuple[np.ndarray, bool, list[LabelEntry]]:
    if not assets:
        return image.copy(), False, entries

    canvas = image.copy()
    augmented_entries = clone_entries(entries)
    placements = rng.randint(1, min(4, len(assets)))

    placed = 0
    attempts = 0
    max_attempts = placements * 5
    while placed < placements and attempts < max_attempts:
        attempts += 1
        success, entry = place_random_segment(canvas, assets, rng)
        if not success or entry is None:
            continue
        augmented_entries.append(entry)
        placed += 1

    return canvas, placed > 0, augmented_entries


def add_offwhite_rectangle(
    image: np.ndarray,
    entries: list[LabelEntry],
    rng: random.Random,
) -> np.ndarray | None:
    if not entries:
        return None

    candidate = max(
        entries,
        key=lambda entry: cv2.contourArea(entry.polygon.astype(np.float32)),
        default=None,
    )
    if candidate is None:
        return None

    x, y, w, h = cv2.boundingRect(candidate.polygon.astype(np.float32))
    if w <= 1 or h <= 1:
        return None

    color_val = int(rng.uniform(0.7, 0.9) * 255)
    patch = np.zeros((h, w, 4), dtype=np.uint8)
    patch[:, :, :3] = color_val
    patch[:, :, 3] = 255

    angle = rng.uniform(0, 360)
    rotated, _ = rotate_with_bounds(patch, angle)
    patch_height, patch_width = rotated.shape[:2]

    canvas_height, canvas_width = image.shape[:2]
    if patch_height >= canvas_height or patch_width >= canvas_width:
        return None

    max_x = canvas_width - patch_width
    max_y = canvas_height - patch_height
    pos_x = rng.randint(0, max_x) if max_x > 0 else 0
    pos_y = rng.randint(0, max_y) if max_y > 0 else 0

    alpha = rotated[:, :, 3].astype(np.float32) / 255.0
    if not np.any(alpha > 0):
        return None

    result = image.copy()
    roi = result[pos_y : pos_y + patch_height, pos_x : pos_x + patch_width]
    fg = rotated[:, :, :3].astype(np.float32)
    bg = roi.astype(np.float32)
    blended = fg * alpha[..., None] + bg * (1.0 - alpha[..., None])
    roi[:] = blended.astype(np.uint8)
    return result


def ensure_output_layout(output_dir: Path) -> Path:
    images_root = output_dir / "images"
    for split in ("train", "val", "test"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
    return images_root


def load_segment_assets(
    dataset_dir: Path, class_names: dict[int, str]
) -> list[SegmentAsset]:
    base_dir = dataset_dir / "segments"
    if not base_dir.exists():
        LOGGER.warning("Segments directory not found at %s; insertion variants disabled.", base_dir)
        return []

    sanitized_to_id = {
        sanitize_class_name(name): class_id for class_id, name in class_names.items()
    }

    assets: list[SegmentAsset] = []
    for split_dir in sorted(base_dir.glob("*")):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.glob("*")):
            if not class_dir.is_dir():
                continue
            class_id = sanitized_to_id.get(class_dir.name)
            if class_id is None:
                continue
            class_name = class_names.get(class_id, class_dir.name)
            for asset_path in sorted(class_dir.glob("*.png")):
                metadata_path = asset_path.with_suffix(".json")
                if not metadata_path.exists():
                    continue
                image = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
                if image is None or image.shape[2] < 4:
                    continue
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                points = np.asarray(metadata.get("points", []), dtype=np.float32)
                if points.size == 0:
                    continue
                assets.append(SegmentAsset(class_id, class_name, image, points))

    if not assets:
        LOGGER.warning("No segment assets loaded; insertion variants disabled.")
    else:
        LOGGER.info("Loaded %s reusable segments for insertion.", len(assets))

    return assets


def clone_entries(entries: list[LabelEntry]) -> list[LabelEntry]:
    return [LabelEntry(entry.class_id, entry.polygon.copy()) for entry in entries]


def save_variant(
    image: np.ndarray,
    entries: list[LabelEntry],
    width: int,
    height: int,
    stem: str,
    suffix: str,
    extension: str,
    output_dir: Path,
) -> None:
    image_path = output_dir / f"{stem}_{suffix}{extension}"
    label_path = image_path.with_suffix(".txt")
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Failed to write augmented image {image_path}")
    write_label_file(label_path, entries, width, height)


def augment_image(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    segments: list[SegmentAsset],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        LOGGER.warning("Unable to read image: %s", image_path)
        return

    height, width = image.shape[:2]
    if not label_path.exists():
        LOGGER.warning("Label file missing for %s", image_path)
        return

    entries = load_label_entries(label_path, width, height)
    stem = image_path.stem
    extension = image_path.suffix or ".jpg"

    base_entries = clone_entries(entries)
    save_variant(image, base_entries, width, height, stem, "orig", extension, output_dir)

    for idx in range(4):
        adjusted = adjust_saturation_exposure(image, rng)
        save_variant(
            adjusted,
            clone_entries(entries),
            width,
            height,
            stem,
            f"col{idx}",
            extension,
            output_dir,
        )

    for idx in range(2):
        occluded = occlude_segments(image, entries, rng, np_rng)
        save_variant(
            occluded,
            clone_entries(entries),
            width,
            height,
            stem,
            f"occ{idx}",
            extension,
            output_dir,
        )

    for idx in range(4):
        augmented_image, inserted, updated_entries = insert_segments(
            image, entries, segments, rng
        )
        if not inserted:
            LOGGER.info("Insertion skipped for %s_%s (no segments available).", stem, idx)
            augmented_image = image.copy()
            updated_entries = clone_entries(entries)
        save_variant(
            augmented_image,
            updated_entries,
            width,
            height,
            stem,
            f"ins{idx}",
            extension,
            output_dir,
        )

    rectangle_variant = add_offwhite_rectangle(image, entries, rng)
    if rectangle_variant is not None:
        save_variant(
            rectangle_variant,
            clone_entries(entries),
            width,
            height,
            stem,
            "rect",
            extension,
            output_dir,
        )


def copy_dataset_metadata(dataset_dir: Path, output_dir: Path) -> None:
    config_path = dataset_dir / "dataset.yaml"
    if not config_path.exists():
        msg = f"Missing dataset.yaml in {dataset_dir}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["path"] = "."
    config["train"] = "images/train"
    config["val"] = "images/val"
    config["test"] = "images/test"
    if "name" in config:
        config["name"] = f"{config['name']}_augmented"

    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def augment_dataset(dataset_dir: Path, output_dir: Path, seed: int) -> None:
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "seed.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(seed) + "\n")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    splits, class_names = load_dataset_config(dataset_dir)
    images_out_root = ensure_output_layout(output_dir)
    segments = load_segment_assets(dataset_dir, class_names)

    for split, image_dir in splits.items():
        split_output_dir = images_out_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        labels_dir = derive_label_dir(image_dir)
        if not labels_dir.exists():
            LOGGER.warning("Missing labels directory for split %s: %s", split, labels_dir)
            continue

        label_files = sorted(labels_dir.glob("*.txt"))
        if not label_files:
            LOGGER.info("No labels found for split %s", split)
            continue

        for label_path in label_files:
            image_path = find_image_path(label_path, image_dir)
            if image_path is None:
                LOGGER.warning("Skipping %s (missing image match).", label_path)
                continue
            augment_image(
                image_path,
                label_path,
                split_output_dir,
                segments,
                rng,
                np_rng,
            )

    copy_dataset_metadata(dataset_dir, output_dir)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )


def _run(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging()

    LOGGER.info("Augmenting dataset %s -> %s (seed=%s)", args.dataset_dir, args.output_dir, args.seed)
    try:
        augment_dataset(args.dataset_dir, args.output_dir, args.seed)
    except Exception as exc:  # noqa: BLE001 - propagate as exit code
        LOGGER.error("Augmentation failed: %s", exc)
        return 1

    LOGGER.info("Augmented dataset written to %s", args.output_dir)
    return 0


def main(argv: Iterable[str] | None = None) -> None:
    raise SystemExit(_run(argv))


__all__ = [
    "augment_dataset",
    "main",
    "parse_args",
]
