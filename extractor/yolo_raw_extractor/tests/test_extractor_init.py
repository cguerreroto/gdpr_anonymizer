"""Tests for yolo_raw_extractor package entry (frame extraction CLI helpers)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pytest

import yolo_raw_extractor as extractor
from yolo_raw_extractor import ensure_split_dirs, extract_frames, parse_args


class _FakeCapture:
    def __init__(self, frames: Iterable[np.ndarray] | None) -> None:
        self._frames = list(frames) if frames is not None else None
        self._index = 0
        self._released = False

    def isOpened(self) -> bool:
        return self._frames is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frames is None or self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self._released = True


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


def test_extract_frames_writes_split_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(15)]
    capture = _FakeCapture(frames)
    monkeypatch.setattr(extractor.cv2, "VideoCapture", lambda _path: capture)

    written: list[str] = []

    def fake_imwrite(path: str, _frame: np.ndarray) -> bool:
        written.append(path)
        Path(path).write_bytes(b"")
        return True

    monkeypatch.setattr(extractor.cv2, "imwrite", fake_imwrite)

    saved = extract_frames(tmp_path / "video.mp4", tmp_path, frame_stride=5)

    assert saved == 3
    assert capture._released is True
    assert all(path.endswith(".jpg") for path in written)
    train_dir = tmp_path / "images" / "train"
    val_dir = tmp_path / "images" / "val"
    test_dir = tmp_path / "images" / "test"
    assert sorted(p.name for p in train_dir.iterdir()) == [
        "video_frame_000000000.jpg",
        "video_frame_000000005.jpg",
        "video_frame_000000010.jpg",
    ]
    assert list(val_dir.iterdir()) == []
    assert list(test_dir.iterdir()) == []


def test_extract_frames_imwrite_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _FakeCapture([np.zeros((2, 2, 3), dtype=np.uint8)])
    monkeypatch.setattr(extractor.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(extractor.cv2, "imwrite", lambda _path, _frame: False)

    with pytest.raises(RuntimeError, match="Failed to write frame"):
        extract_frames(tmp_path / "video.mp4", tmp_path, frame_stride=1)


def test_run_returns_zero_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extractor, "extract_frames", lambda _v, _d: 7)

    result = extractor._run([str(tmp_path / "video.mp4"), str(tmp_path)])

    assert result == 0


def test_run_returns_one_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("decode failed")

    monkeypatch.setattr(extractor, "extract_frames", boom)

    result = extractor._run([str(tmp_path / "video.mp4"), str(tmp_path)])

    assert result == 1


def test_main_raises_systemexit_with_run_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extractor, "_run", lambda _argv: 0)

    with pytest.raises(SystemExit) as info:
        extractor.main([str(tmp_path / "video.mp4"), str(tmp_path)])

    assert info.value.code == 0
