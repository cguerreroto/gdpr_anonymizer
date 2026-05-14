"""Tests for yolo_raw_extractor.segments."""

from __future__ import annotations

import numpy as np
import pytest

import yolo_raw_extractor.segments as segments_mod
from yolo_raw_extractor.segments import build_alpha_segment


def test_build_alpha_segment_rectangle() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    polygon = np.array(
        [[5.0, 5.0], [25.0, 5.0], [25.0, 15.0], [5.0, 15.0]], dtype=np.float32
    )
    segment, bbox = build_alpha_segment(image, polygon)
    x, y, w, h = bbox
    assert w > 0 and h > 0
    assert segment.shape == (h, w, 4)
    assert segment.dtype == np.uint8
    assert np.any(segment[:, :, 3] > 0)


def test_build_alpha_segment_degenerate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the w/h guard; OpenCV's boundingRect rarely returns zero for polygons."""
    monkeypatch.setattr(segments_mod.cv2, "boundingRect", lambda _pts: (0, 0, 0, 0))
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    polygon = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Degenerate bounding box"):
        build_alpha_segment(image, polygon)
