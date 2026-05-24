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
    aggregate_video_metrics,
    blurred_video_filename,
    build_predict_kwargs,
    log_video_status,
    resolve_blurred_output_path,
    run_video_blur,
    validate_video_inputs,
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
