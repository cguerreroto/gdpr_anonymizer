"""Smoke tests for shared video test fakes."""

from __future__ import annotations

import pytest
from fakes import (
    FakeArray,
    FakeCapture,
    FakeWriter,
    block_ultralytics_import,
    fake_segmentation_result,
    install_fake_cv2_numpy,
    install_fake_ultralytics,
    make_fake_cv2_module,
    make_fake_numpy_module,
)

from ml_pipeline.video import _apply_blur, _result_classes, _union_mask
from ml_pipeline.video import VideoBlurConfig


def test_fake_capture_reads_configured_frames() -> None:
    frames = [FakeArray.zeros((4, 4, 3)), FakeArray.zeros((4, 4, 3))]
    capture = FakeCapture("clip.mp4", frames=frames, width=4, height=4, fps=10.0)

    ok1, frame1 = capture.read()
    ok2, frame2 = capture.read()
    ok3, _ = capture.read()

    assert ok1 and frame1 is frames[0]
    assert ok2 and frame2 is frames[1]
    assert ok3 is False
    assert capture.get(5) == pytest.approx(10.0)


def test_fake_writer_records_frames() -> None:
    writer = FakeWriter("out.mp4", 0, 10.0, (8, 8))
    frame = FakeArray.zeros((8, 8, 3))
    writer.write(frame)
    assert writer.frames_written == [frame]


def test_install_fake_cv2_numpy_allows_apply_blur(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_cv2_numpy(monkeypatch)
    frame = FakeArray.zeros((8, 8, 3))
    config = VideoBlurConfig(
        weights=__import__("pathlib").Path("w.pt"),
        source=__import__("pathlib").Path("in.mp4"),
        output=__import__("pathlib").Path("out.mp4"),
        blur_method="gaussian",
    )
    blurred = _apply_blur(frame, config)
    assert blurred is frame


def test_fake_segmentation_result_classes_and_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_cv2_numpy(monkeypatch)
    result = fake_segmentation_result(classes=[0, 1], height=4, width=4)
    assert _result_classes(result) == [0, 1]

    union = _union_mask(result, [0], height=4, width=4, dilate_px=0)
    assert isinstance(union, FakeArray)
    assert union.shape == (4, 4)


def test_make_fake_modules_are_independent() -> None:
    cv2_a = make_fake_cv2_module()
    cv2_b = make_fake_cv2_module()
    np_a = make_fake_numpy_module()
    np_b = make_fake_numpy_module()
    assert cv2_a is not cv2_b
    assert np_a is not np_b


def test_install_fake_ultralytics_registers_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    refs: list[str] = []
    install_fake_ultralytics(
        monkeypatch,
        yolo_factory=lambda ref: refs.append(ref) or __import__("types").SimpleNamespace(ref=ref),
    )
    from ultralytics import YOLO

    model = YOLO("demo.pt")
    assert refs == ["demo.pt"]
    assert model.ref == "demo.pt"
