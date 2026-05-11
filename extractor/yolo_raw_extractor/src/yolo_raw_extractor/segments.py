from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from yolo_raw_extractor.dataset_utils import (
    derive_label_dir,
    find_image_path,
    load_dataset_config,
    parse_label_line,
    sanitize_class_name,
)

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolo-segment-extractor",
        description=(
            "Extract labeled positives from a YOLO segmentation dataset into "
            "PNG files with alpha channels."
        ),
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the labeled YOLO dataset (must contain dataset.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (default: <dataset>/segments).",
    )
    return parser.parse_args(argv)


def build_alpha_segment(
    image: np.ndarray, polygon: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    polygon_int = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [polygon_int], 255)

    x, y, w, h = cv2.boundingRect(polygon_int)
    if w == 0 or h == 0:
        msg = "Degenerate bounding box for polygon"
        raise ValueError(msg)

    region = image[y : y + h, x : x + w]
    mask_region = mask[y : y + h, x : x + w]

    segment = cv2.cvtColor(region, cv2.COLOR_BGR2BGRA)
    segment[:, :, 3] = mask_region

    return segment, (x, y, w, h)


def write_metadata(
    image_path: Path,
    class_id: int,
    class_label: str,
    polygon: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> None:
    x, y, w, h = bbox
    relative_polygon = polygon - np.array([[x, y]], dtype=np.float32)
    payload = {
        "class_id": class_id,
        "class_name": class_label,
        "points": relative_polygon.tolist(),
        "crop": {"x": x, "y": y, "width": w, "height": h},
        "source_image": image_path.name,
    }
    with image_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def extract_segments(dataset_dir: Path, output_dir: Path | None = None) -> int:
    dataset_dir = dataset_dir.expanduser().resolve()
    splits, class_names = load_dataset_config(dataset_dir)

    output_root = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else dataset_dir / "segments"
    )

    extracted = 0
    for split, image_dir in splits.items():
        if not image_dir.exists():
            LOGGER.warning("Image directory missing for split %s: %s", split, image_dir)
            continue

        labels_dir = derive_label_dir(image_dir)
        if not labels_dir.exists():
            LOGGER.warning("Label directory missing for split %s: %s", split, labels_dir)
            continue

        destination_split = output_root / split
        destination_split.mkdir(parents=True, exist_ok=True)

        label_files = sorted(labels_dir.glob("*.txt"))
        if not label_files:
            LOGGER.info("No labels found for split %s in %s", split, labels_dir)
            continue

        for label_path in label_files:
            image_path = find_image_path(label_path, image_dir)
            if image_path is None:
                LOGGER.warning(
                    "Skipping %s (missing matching image in %s)", label_path, image_dir
                )
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("Unable to read image: %s", image_path)
                continue

            height, width = image.shape[:2]
            with label_path.open("r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]

            if not lines:
                continue

            for entry_idx, line in enumerate(lines):
                try:
                    class_id, polygon = parse_label_line(line, width, height)
                    segment, bbox = build_alpha_segment(image, polygon)
                except Exception as exc:  # noqa: BLE001 - record and keep going
                    LOGGER.warning("Failed to process %s line %s: %s", label_path, entry_idx, exc)
                    continue

                class_label = class_names.get(class_id, f"class-{class_id}")
                safe_class = sanitize_class_name(class_label)
                class_dir = destination_split / safe_class
                class_dir.mkdir(parents=True, exist_ok=True)

                output_name = (
                    f"{image_path.stem}_cls{class_id:02d}_obj{entry_idx:02d}.png"
                )
                output_path = class_dir / output_name
                if not cv2.imwrite(str(output_path), segment):
                    LOGGER.warning("Failed to write segment to %s", output_path)
                    continue

                write_metadata(output_path, class_id, class_label, polygon, bbox)
                extracted += 1

    return extracted


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )


def _run(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    LOGGER.info("Extracting labeled segments from %s", dataset_dir)
    try:
        count = extract_segments(dataset_dir, output_dir)
    except Exception as exc:  # noqa: BLE001 - surface to CLI
        LOGGER.error("Segment extraction failed: %s", exc)
        return 1

    target = output_dir if output_dir else dataset_dir / "segments"
    LOGGER.info("Wrote %s segments to %s", count, target)
    return 0


def main(argv: Iterable[str] | None = None) -> None:
    raise SystemExit(_run(argv))


__all__ = [
    "extract_segments",
    "main",
    "parse_args",
]
