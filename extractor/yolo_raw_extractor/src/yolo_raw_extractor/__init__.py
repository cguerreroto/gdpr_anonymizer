from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2

LOGGER = logging.getLogger(__name__)
FRAME_STRIDE = 100
SPLIT_PATTERN: tuple[str, ...] = ("train",) * 10 + ("val",) + ("test",)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolo-raw-extractor",
        description=(
            "Extract deterministic frame subsets from a video and place them "
            "into YOLO-style image directories."
        ),
    )
    parser.add_argument("video_path", type=Path, help="Video file to sample.")
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Target directory that will contain the YOLO dataset.",
    )
    return parser.parse_args(argv)


def ensure_split_dirs(dataset_dir: Path) -> dict[str, Path]:
    root = dataset_dir / "images"
    splits: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = split_dir
    return splits


def extract_frames(
    video_path: Path, dataset_dir: Path, frame_stride: int = FRAME_STRIDE
) -> int:
    if frame_stride <= 0:
        msg = f"Frame stride must be positive, got {frame_stride}"
        raise ValueError(msg)

    splits = ensure_split_dirs(dataset_dir)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        msg = f"Unable to open video file: {video_path}"
        raise FileNotFoundError(msg)

    saved = 0
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % frame_stride == 0:
                split_name = SPLIT_PATTERN[saved % len(SPLIT_PATTERN)]
                split_dir = splits[split_name]
                filename = f"{video_path.stem}_frame_{frame_index:09d}.jpg"
                destination = split_dir / filename
                if not cv2.imwrite(str(destination), frame):
                    msg = f"Failed to write frame to {destination}"
                    raise RuntimeError(msg)
                saved += 1
            frame_index += 1
    finally:
        cap.release()

    return saved


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )


def _run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging()

    video_path = args.video_path.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()

    LOGGER.info("Extracting frames from %s", video_path)
    try:
        saved = extract_frames(video_path, dataset_dir)
    except Exception as exc:  # noqa: BLE001 - surface meaningful log
        LOGGER.error("Extraction failed: %s", exc)
        return 1

    LOGGER.info(
        "Wrote %s frames into %s/images/{train,val,test}",
        saved,
        dataset_dir,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(_run(argv))


__all__ = [
    "extract_frames",
    "main",
    "parse_args",
]
