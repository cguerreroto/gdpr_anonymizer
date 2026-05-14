"""Tests for label load/write helpers in yolo_raw_extractor.augment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yolo_raw_extractor.augment import LabelEntry, load_label_entries, write_label_file


def test_write_label_file_round_trip(tmp_path: Path) -> None:
    width, height = 100, 50
    original = [
        LabelEntry(
            class_id=1,
            polygon=np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 50.0]], dtype=np.float32),
        )
    ]
    path = tmp_path / "labels" / "img.txt"
    path.parent.mkdir(parents=True)
    write_label_file(path, original, width, height)
    loaded = load_label_entries(path, width, height)
    assert len(loaded) == 1
    assert loaded[0].class_id == 1
    np.testing.assert_allclose(loaded[0].polygon, original[0].polygon, rtol=1e-5)
