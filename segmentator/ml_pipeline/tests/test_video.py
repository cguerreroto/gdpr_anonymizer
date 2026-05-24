"""Tests for ml_pipeline.video and ml_pipeline.cli_video.

A stub pipeline runner replaces the OpenCV-backed default so the
orchestration is covered without OpenCV, NumPy, or a real video file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from ml_pipeline.cli_video import main as cli_main
from ml_pipeline.video import (
    FrameProgress,
    FrameStats,
    VideoBlurConfig,
    _apply_blur,
    _default_video_model_factory,
    _default_video_pipeline_runner,
    _union_mask,
    aggregate_video_metrics,
    blurred_video_filename,
    build_predict_kwargs,
    log_video_status,
    resolve_blurred_output_path,
    run_video_blur,
    validate_video_inputs,
)
from fakes import (
    FakeArray,
    FakeCapture,
    FakeTensor,
    FakeWriter,
    block_ultralytics_import,
    fake_segmentation_result,
    install_fake_cv2_numpy,
    install_fake_ultralytics,
    make_fake_cv2_module,
)


class _StubModel:
    def __init__(self, ref: str) -> None:
        self.ref = ref


def _make_config(tmp_path: Path) -> VideoBlurConfig:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"")
    output = tmp_path / "blurred.mp4"
    return VideoBlurConfig(weights=weights, source=source, output=output)


class _PredictStubModel:
    """Model stub returning a preset sequence of predict results."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._call = 0

    def predict(self, _frame: Any, **_kwargs: Any) -> Any:
        if self._call >= len(self._responses):
            return []
        response = self._responses[self._call]
        self._call += 1
        return response


class _NonSubscriptableResults:
    """Iterable predict output that rejects ``results[0]``."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __bool__(self) -> bool:
        return bool(self._items)

    def __getitem__(self, _idx: int) -> Any:
        raise TypeError("not subscriptable")

    def __iter__(self):
        return iter(self._items)


def _install_pipeline_cv2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capture: FakeCapture | None = None,
    writer: FakeWriter | None = None,
    capture_factory: Any = None,
    writer_factory: Any = None,
) -> tuple[FakeCapture | None, FakeWriter | None]:
    """Install fake cv2 with optional fixed capture/writer instances."""
    holder: dict[str, FakeCapture | FakeWriter | None] = {
        "capture": capture,
        "writer": writer,
    }

    def _capture_factory(path: str) -> FakeCapture:
        if capture_factory is not None:
            cap = capture_factory(path)
        elif holder["capture"] is not None:
            cap = holder["capture"]
        else:
            cap = FakeCapture(path)
        holder["capture"] = cap
        return cap

    def _writer_factory(
        path: str, fourcc: int, fps: float, size: tuple[int, int]
    ) -> FakeWriter:
        if writer_factory is not None:
            wr = writer_factory(path, fourcc, fps, size)
        elif holder["writer"] is not None:
            wr = holder["writer"]
        else:
            wr = FakeWriter(path, fourcc, fps, size)
        holder["writer"] = wr
        return wr

    cv2_mod = make_fake_cv2_module(
        capture_factory=_capture_factory,
        writer_factory=_writer_factory,
    )
    install_fake_cv2_numpy(monkeypatch, cv2_module=cv2_mod)
    return holder["capture"], holder["writer"]


def test_frame_progress_renders_percent_bar() -> None:
    import io

    stream = io.StringIO()
    progress = FrameProgress(total=100, stream=stream, width=10)
    progress.update(50)
    progress.close()
    line = stream.getvalue()
    assert "Blurring frames:" in line
    assert "50/100 (50%)" in line
    assert "|=====" in line


def test_frame_progress_without_total_shows_frame_count() -> None:
    import io

    stream = io.StringIO()
    progress = FrameProgress(total=None, stream=stream)
    progress.update(12)
    progress.close()
    assert "frame 12" in stream.getvalue()


def test_frame_progress_disabled_writes_nothing() -> None:
    import io

    stream = io.StringIO()
    progress = FrameProgress(total=100, stream=stream, enabled=False)
    progress.update(5)
    progress.close()
    assert stream.getvalue() == ""


def test_result_classes_handles_missing_boxes() -> None:
    from ml_pipeline.video import _result_classes

    class _Empty:
        pass

    assert _result_classes(_Empty()) == []


def test_result_classes_handles_missing_cls() -> None:
    from ml_pipeline.video import _result_classes

    class _Boxes:
        pass

    class _Result:
        boxes = _Boxes()

    assert _result_classes(_Result()) == []


def test_result_classes_uses_tolist_when_present() -> None:
    from ml_pipeline.video import _result_classes

    class _ClsTensor:
        @staticmethod
        def tolist() -> list[float]:
            return [0.0, 1.0, 1.0]

    class _Boxes:
        cls = _ClsTensor()

    class _Result:
        boxes = _Boxes()

    assert _result_classes(_Result()) == [0, 1, 1]


def test_result_classes_falls_back_to_iter() -> None:
    from ml_pipeline.video import _result_classes

    class _Boxes:
        cls = [0, 2, 1]

    class _Result:
        boxes = _Boxes()

    assert _result_classes(_Result()) == [0, 2, 1]


def test_apply_blur_gaussian_uses_odd_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels: list[tuple[int, int]] = []

    def gaussian_blur(frame: FakeArray, kernel: tuple[int, int], _sigma: float) -> FakeArray:
        kernels.append(kernel)
        return frame

    cv2_mod = make_fake_cv2_module()
    cv2_mod.GaussianBlur = gaussian_blur
    install_fake_cv2_numpy(monkeypatch, cv2_module=cv2_mod)

    config = _make_config(tmp_path)
    config.blur_method = "gaussian"
    config.blur_kernel = 50
    frame = FakeArray.zeros((8, 8, 3))

    result = _apply_blur(frame, config)

    assert result is frame
    assert kernels == [(51, 51)]


def test_apply_blur_pixelate_returns_resized_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[tuple[int, int]] = []

    def resize(
        _image: FakeArray,
        size: tuple[int, int],
        *,
        interpolation: int = 0,
    ) -> FakeArray:
        sizes.append(size)
        width, height = size
        return FakeArray.zeros((height, width, 3))

    cv2_mod = make_fake_cv2_module()
    cv2_mod.resize = resize
    install_fake_cv2_numpy(monkeypatch, cv2_module=cv2_mod)

    config = _make_config(tmp_path)
    config.blur_method = "pixelate"
    config.pixelate_block = 4
    frame = FakeArray.zeros((8, 8, 3))

    blurred = _apply_blur(frame, config)

    assert blurred.shape == (8, 8, 3)
    assert sizes[0] == (2, 2)
    assert sizes[1] == (8, 8)


def test_apply_blur_invalid_method_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_cv2_numpy(monkeypatch)
    config = _make_config(tmp_path)
    config.blur_method = "watercolor"
    frame = FakeArray.zeros((4, 4, 3))

    with pytest.raises(ValueError, match="Unsupported blur method"):
        _apply_blur(frame, config)


def test_union_mask_returns_zeros_when_masks_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_cv2_numpy(monkeypatch)

    class _NoMasks:
        pass

    union = _union_mask(_NoMasks(), [0], height=4, width=4, dilate_px=0)

    assert isinstance(union, FakeArray)
    assert union.shape == (4, 4)
    assert union.sum() == 0


def test_union_mask_returns_zeros_when_mask_data_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_cv2_numpy(monkeypatch)

    class _EmptyMasks:
        masks = type("M", (), {"data": None})()

    union = _union_mask(_EmptyMasks(), [0], height=4, width=4, dilate_px=0)

    assert union.sum() == 0


def test_union_mask_resizes_mismatched_mask_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resize_calls: list[tuple[int, int]] = []

    def resize(
        _image: FakeArray,
        size: tuple[int, int],
        *,
        interpolation: int = 0,
    ) -> FakeArray:
        resize_calls.append(size)
        width, height = size
        grid = [[1.0] * width for _ in range(height)]
        return FakeArray.from_scalar_grid(grid, dtype="float32")

    cv2_mod = make_fake_cv2_module()
    cv2_mod.resize = resize
    install_fake_cv2_numpy(monkeypatch, cv2_module=cv2_mod)

    small = FakeArray.from_scalar_grid([[1.0, 0.0], [0.0, 0.0]], dtype="float32")
    result = fake_segmentation_result(classes=[0], masks=[small], height=4, width=4)

    union = _union_mask(result, [0], height=4, width=4, dilate_px=0)

    assert resize_calls == [(4, 4)]
    assert union.sum() > 0


def test_union_mask_applies_dilation_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dilate_calls: list[tuple[int, int]] = []

    def dilate(image: FakeArray, kernel: tuple[int, int]) -> FakeArray:
        dilate_calls.append(kernel)
        return image

    cv2_mod = make_fake_cv2_module()
    cv2_mod.dilate = dilate
    install_fake_cv2_numpy(monkeypatch, cv2_module=cv2_mod)

    mask = FakeArray.from_scalar_grid(
        [[1.0, 0.0], [0.0, 0.0]],
        dtype="float32",
    )
    result = fake_segmentation_result(classes=[0], masks=[mask], height=2, width=2)

    _union_mask(result, [0], height=2, width=2, dilate_px=2)

    assert dilate_calls == [(5, 5)]


def test_union_mask_uses_cpu_and_numpy_on_tensor_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_cv2_numpy(monkeypatch)

    grid = FakeArray.from_scalar_grid([[0.8, 0.0], [0.0, 0.0]], dtype="float32")
    tensor = FakeTensor(grid)
    result = fake_segmentation_result(classes=[0], masks=[tensor], height=2, width=2)

    union = _union_mask(result, [0], height=2, width=2, dilate_px=0)

    assert union.sum() > 0


def test_union_mask_skips_out_of_range_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_cv2_numpy(monkeypatch)
    result = fake_segmentation_result(classes=[0], height=2, width=2)

    union = _union_mask(result, [5], height=2, width=2, dilate_px=0)

    assert union.sum() == 0


def test_default_video_pipeline_runner_processes_frames_with_detection_on_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    config.show_progress = False
    height, width = 48, 64
    frames = [
        FakeArray.zeros((height, width, 3)),
        FakeArray.zeros((height, width, 3)),
    ]
    capture = FakeCapture(
        str(config.source),
        frames=frames,
        width=width,
        height=height,
        fps=10.0,
    )
    holder: dict[str, FakeWriter | None] = {"writer": None}
    _install_pipeline_cv2(
        monkeypatch,
        capture=capture,
        writer_factory=lambda path, fourcc, fps, size: holder.update(
            writer=FakeWriter(path, fourcc, fps, size)
        )
        or holder["writer"],
    )

    detection = fake_segmentation_result(classes=[0], height=height, width=width)
    model = _PredictStubModel([[], [detection]])
    predict_kwargs = build_predict_kwargs(config)

    stats = _default_video_pipeline_runner(model, config, predict_kwargs)

    assert len(stats) == 2
    assert stats[0].detections == 0
    assert stats[0].mask_pixels == 0
    assert stats[1].detections == 1
    assert stats[1].detection_classes == [0]
    assert stats[1].mask_pixels > 0
    writer = holder["writer"]
    assert writer is not None
    assert len(writer.frames_written) == 2
    assert capture._released
    assert writer._released


def test_default_video_pipeline_runner_falls_back_when_results_not_indexable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    config.show_progress = False
    height, width = 48, 64
    capture = FakeCapture(
        str(config.source),
        frames=[FakeArray.zeros((height, width, 3))],
        width=width,
        height=height,
    )
    holder: dict[str, FakeWriter | None] = {"writer": None}
    _install_pipeline_cv2(
        monkeypatch,
        capture=capture,
        writer_factory=lambda path, fourcc, fps, size: holder.update(
            writer=FakeWriter(path, fourcc, fps, size)
        )
        or holder["writer"],
    )

    detection = fake_segmentation_result(classes=[1], height=height, width=width)
    model = _PredictStubModel([_NonSubscriptableResults([detection])])

    stats = _default_video_pipeline_runner(
        model, config, build_predict_kwargs(config)
    )

    assert len(stats) == 1
    assert stats[0].detections == 1
    assert stats[0].detection_classes == [1]
    assert holder["writer"] is not None
    assert len(holder["writer"].frames_written) == 1


def test_default_video_pipeline_runner_raises_when_capture_not_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    _install_pipeline_cv2(
        monkeypatch,
        capture_factory=lambda path: FakeCapture(path, opened=False),
    )
    model = _PredictStubModel([])

    with pytest.raises(RuntimeError, match="Cannot open video for reading"):
        _default_video_pipeline_runner(model, config, build_predict_kwargs(config))


def test_default_video_pipeline_runner_raises_when_writer_not_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    capture = FakeCapture(str(config.source), width=64, height=48)
    _install_pipeline_cv2(
        monkeypatch,
        capture=capture,
        writer_factory=lambda path, fourcc, fps, size: FakeWriter(
            path, fourcc, fps, size, opened=False
        ),
    )
    model = _PredictStubModel([])

    with pytest.raises(RuntimeError, match="Cannot open video for writing"):
        _default_video_pipeline_runner(model, config, build_predict_kwargs(config))

    assert capture._released


def test_default_video_pipeline_runner_passthrough_when_class_filter_excludes_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    config.show_progress = False
    config.classes = [99]
    height, width = 48, 64
    frame = FakeArray.zeros((height, width, 3))
    capture = FakeCapture(
        str(config.source),
        frames=[frame],
        width=width,
        height=height,
    )
    holder: dict[str, FakeWriter | None] = {"writer": None}
    _install_pipeline_cv2(
        monkeypatch,
        capture=capture,
        writer_factory=lambda path, fourcc, fps, size: holder.update(
            writer=FakeWriter(path, fourcc, fps, size)
        )
        or holder["writer"],
    )

    detection = fake_segmentation_result(classes=[0], height=height, width=width)
    model = _PredictStubModel([[detection]])

    stats = _default_video_pipeline_runner(
        model, config, build_predict_kwargs(config)
    )

    assert len(stats) == 1
    assert stats[0].detections == 0
    assert stats[0].detection_classes == []
    assert holder["writer"] is not None
    assert holder["writer"].frames_written == [frame]


def test_default_video_model_factory_loads_fake_yolo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs: list[str] = []
    install_fake_ultralytics(
        monkeypatch,
        yolo_factory=lambda ref: refs.append(ref) or _StubModel(ref),
    )

    model = _default_video_model_factory("weights/best.pt")

    assert refs == ["weights/best.pt"]
    assert isinstance(model, _StubModel)
    assert model.ref == "weights/best.pt"


def test_default_video_model_factory_surfaces_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_ultralytics_import(monkeypatch)

    with pytest.raises(RuntimeError, match="uv sync --extra train"):
        _default_video_model_factory("best.pt")


def test_log_video_status_respects_enabled_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_video_status("hello", enabled=True)
    assert "hello" in capsys.readouterr().err
    log_video_status("hidden", enabled=False)
    assert capsys.readouterr().err == ""


def test_run_video_blur_logs_status_with_stub_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(tmp_path)
    cfg.show_progress = True

    monkeypatch.setattr(
        "ml_pipeline.video._default_video_model_factory",
        lambda ref: _StubModel(ref),
    )

    def runner(model, config, predict_kwargs):
        config.output.write_bytes(b"fake-video")
        return []

    monkeypatch.setattr("ml_pipeline.video._default_video_pipeline_runner", runner)

    run_video_blur(cfg)
    err = capsys.readouterr().err
    assert "Loading model:" in err
    assert "Source:" in err
    assert "Done." in err


def test_blurred_video_filename_preserves_suffix() -> None:
    source = Path("/tmp/2025_0802_075419_097F.MP4")
    assert blurred_video_filename(source) == "2025_0802_075419_097F_blurred.MP4"


def test_resolve_blurred_output_path_default_next_to_source() -> None:
    source = Path("/data/raw/clip.mp4")
    assert resolve_blurred_output_path(source) == Path("/data/raw/clip_blurred.mp4")


def test_resolve_blurred_output_path_output_directory(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    out_dir = tmp_path / "blur"
    out_dir.mkdir()
    assert resolve_blurred_output_path(source, out_dir) == out_dir / "clip_blurred.mp4"


def test_resolve_blurred_output_path_output_directory_not_yet_created(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.mp4"
    out_dir = tmp_path / "blur"
    assert resolve_blurred_output_path(source, out_dir) == out_dir / "clip_blurred.mp4"


def test_resolve_blurred_output_path_explicit_file(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    explicit = tmp_path / "custom_name.mp4"
    assert resolve_blurred_output_path(source, explicit) == explicit


def test_build_predict_kwargs_defaults(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    kwargs = build_predict_kwargs(cfg)
    assert kwargs["task"] == "segment"
    assert kwargs["verbose"] is False
    assert kwargs["save"] is False
    assert kwargs["conf"] == 0.25
    assert kwargs["iou"] == 0.7
    assert kwargs["imgsz"] == 640
    assert "device" not in kwargs
    assert "classes" not in kwargs


def test_build_predict_kwargs_with_overrides(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.classes = [0, 1]
    cfg.device = "cpu"
    cfg.conf = 0.4
    cfg.iou = 0.6
    cfg.extra = {"agnostic_nms": True}
    kwargs = build_predict_kwargs(cfg)
    assert kwargs["classes"] == [0, 1]
    assert kwargs["device"] == "cpu"
    assert kwargs["conf"] == 0.4
    assert kwargs["iou"] == 0.6
    assert kwargs["agnostic_nms"] is True


def test_validate_video_inputs_rejects_missing_weights(tmp_path: Path) -> None:
    src = tmp_path / "video.mp4"
    src.write_bytes(b"")
    cfg = VideoBlurConfig(
        weights=tmp_path / "missing.pt",
        source=src,
        output=tmp_path / "out.mp4",
    )
    with pytest.raises(FileNotFoundError, match="Missing weights"):
        validate_video_inputs(cfg)


def test_validate_video_inputs_rejects_missing_source(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.source = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError, match="Video source"):
        validate_video_inputs(cfg)


def test_validate_video_inputs_rejects_non_video(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    bad = tmp_path / "input.txt"
    bad.write_text("not a video")
    cfg.source = bad
    with pytest.raises(ValueError, match="not look like a video"):
        validate_video_inputs(cfg)


def test_validate_video_inputs_rejects_same_source_and_output(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.output = cfg.source
    with pytest.raises(ValueError, match="must differ"):
        validate_video_inputs(cfg)


def test_validate_video_inputs_rejects_unknown_blur(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.blur_method = "watercolor"
    with pytest.raises(ValueError, match="Unsupported blur method"):
        validate_video_inputs(cfg)


def test_validate_video_inputs_rejects_even_kernel(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.blur_kernel = 50
    with pytest.raises(ValueError, match="must be odd"):
        validate_video_inputs(cfg)


def test_validate_video_inputs_rejects_bad_thresholds(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.conf = 1.2
    with pytest.raises(ValueError, match="--conf"):
        validate_video_inputs(cfg)
    cfg.conf = 0.25
    cfg.iou = -0.1
    with pytest.raises(ValueError, match="--iou"):
        validate_video_inputs(cfg)


def test_validate_video_inputs_rejects_negative_dilate(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.mask_dilate = -1
    with pytest.raises(ValueError, match="--mask-dilate"):
        validate_video_inputs(cfg)


def test_aggregate_video_metrics_empty() -> None:
    summary = aggregate_video_metrics([])
    assert summary == {
        "total_frames": 0,
        "frames_with_detections": 0,
        "frames_with_detections_pct": 0.0,
        "total_detections": 0,
        "detections_by_class": {},
        "average_mask_area_fraction": 0.0,
    }


def test_aggregate_video_metrics_counts_classes() -> None:
    stats = [
        FrameStats(detections=2, detection_classes=[0, 1], mask_pixels=100, frame_pixels=1000),
        FrameStats(detections=0, detection_classes=[], mask_pixels=0, frame_pixels=1000),
        FrameStats(detections=1, detection_classes=[1], mask_pixels=200, frame_pixels=1000),
    ]
    summary = aggregate_video_metrics(stats)
    assert summary["total_frames"] == 3
    assert summary["frames_with_detections"] == 2
    assert summary["frames_with_detections_pct"] == pytest.approx(2 / 3)
    assert summary["total_detections"] == 3
    assert summary["detections_by_class"] == {"0": 1, "1": 2}
    assert summary["average_mask_area_fraction"] == pytest.approx((0.1 + 0.0 + 0.2) / 3)


def test_run_video_blur_dry_run_without_existing_source(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    cfg = VideoBlurConfig(
        weights=weights,
        source=tmp_path / "not-yet-downloaded.mp4",
        output=tmp_path / "out.mp4",
    )
    report = run_video_blur(cfg, dry_run=True)
    assert report["dry_run"] is True
    assert "summary" not in report


def test_run_video_blur_dry_run_skips_factory(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    called = {"factory": False, "runner": False}

    def factory(_ref: str) -> _StubModel:
        called["factory"] = True
        return _StubModel(_ref)

    def runner(_model, _cfg, _kwargs):
        called["runner"] = True
        return []

    report = run_video_blur(
        cfg, model_factory=factory, pipeline_runner=runner, dry_run=True
    )
    assert called["factory"] is False
    assert called["runner"] is False
    assert report["dry_run"] is True
    assert "summary" not in report
    assert report["predict_kwargs"]["task"] == "segment"


def test_run_video_blur_with_stub_runner(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.classes = [0, 1]
    seen: dict[str, Any] = {}

    def factory(ref: str) -> _StubModel:
        seen["ref"] = ref
        return _StubModel(ref)

    def runner(model, config, predict_kwargs):
        seen["predict_kwargs"] = predict_kwargs
        cfg.output.write_bytes(b"fake-video")
        return [
            FrameStats(detections=2, detection_classes=[0, 1], mask_pixels=120, frame_pixels=1000),
            FrameStats(detections=0, detection_classes=[], mask_pixels=0, frame_pixels=1000),
        ]

    report = run_video_blur(cfg, model_factory=factory, pipeline_runner=runner)

    assert seen["ref"] == str(cfg.weights)
    assert seen["predict_kwargs"]["classes"] == [0, 1]
    assert report["dry_run"] is False
    assert report["output"] == str(cfg.output)
    assert report["output_exists"] is True
    assert report["summary"]["total_frames"] == 2
    assert report["summary"]["frames_with_detections"] == 1
    assert report["summary"]["detections_by_class"] == {"0": 1, "1": 1}


def test_run_video_blur_marks_missing_output_file(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)

    def runner(model, config, predict_kwargs):
        return []

    report = run_video_blur(
        cfg,
        model_factory=lambda _ref: _StubModel(_ref),
        pipeline_runner=runner,
    )
    assert report["output_exists"] is False


def test_cli_dry_run_prints_kwargs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"")

    code = cli_main(
        [
            str(weights),
            "--source",
            str(source),
            "--classes",
            "0,1",
            "--blur-method",
            "pixelate",
            "--pixelate-block",
            "24",
            "--dry-run",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["config"]["output"] == str(tmp_path / "video_blurred.mp4")
    assert payload["predict_kwargs"]["classes"] == [0, 1]
    assert payload["config"]["blur_method"] == "pixelate"
    assert payload["config"]["pixelate_block"] == 24


def test_cli_writes_report_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"")
    out_dir = tmp_path / "blur"
    out_dir.mkdir()

    monkeypatch.setattr(
        "ml_pipeline.video._default_video_model_factory",
        lambda ref: _StubModel(ref),
    )

    def fake_runner(model, config, predict_kwargs):
        config.output.write_bytes(b"fake-video")
        return [
            FrameStats(
                detections=1,
                detection_classes=[0],
                mask_pixels=10,
                frame_pixels=100,
            )
        ]

    monkeypatch.setattr(
        "ml_pipeline.video._default_video_pipeline_runner", fake_runner
    )

    report_path = tmp_path / "report.json"
    code = cli_main(
        [
            str(weights),
            "--source",
            str(source),
            "--output",
            str(out_dir),
            "--report-json",
            str(report_path),
        ]
    )
    assert code == 0
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["summary"]["total_frames"] == 1
    assert saved["output"] == str(out_dir / "video_blurred.mp4")
    assert saved["output_exists"] is True
