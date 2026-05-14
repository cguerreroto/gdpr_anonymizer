"""Tests for yolo_raw_extractor package entry (frame extraction CLI helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_raw_extractor import ensure_split_dirs, extract_frames, parse_args


def test_parse_args_minimal() -> None:
    args = parse_args(["/tmp/v.mp4", "/tmp/out"])
    assert args.video_path == Path("/tmp/v.mp4")
    assert args.dataset_dir == Path("/tmp/out")


def test_ensure_split_dirs(tmp_path: Path) -> None:
    splits = ensure_split_dirs(tmp_path)
    for name in ("train", "val", "test"):
        assert name in splits
        assert splits[name].is_dir()
        assert splits[name] == tmp_path / "images" / name


def test_extract_frames_invalid_stride(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Frame stride"):
        extract_frames(tmp_path / "fake.mp4", tmp_path, frame_stride=0)


def test_extract_frames_missing_video(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent_video.mp4"
    with pytest.raises(FileNotFoundError, match="Unable to open"):
        extract_frames(missing, tmp_path)
