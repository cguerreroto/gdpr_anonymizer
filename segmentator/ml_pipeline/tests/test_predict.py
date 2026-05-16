"""Tests for ml_pipeline.predict and ml_pipeline.cli_predict.

A stub factory replaces Ultralytics so the predict driver and CLI can be
unit tested without GPU, network access, or model weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from ml_pipeline.cli_predict import main as cli_main
from ml_pipeline.predict import (
    PredictConfig,
    build_predict_kwargs,
    list_source_images,
    run_predict,
    summarize_results,
    validate_predict_inputs,
    write_polygons_for_result,
)


class _StubArray:
    """Minimal array stand-in that mimics numpy's flatten/tolist/iter."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def flatten(self) -> "_StubArray":
        return self

    def tolist(self) -> list[float]:
        return list(self._values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class _StubMasks:
    def __init__(self, polygons: list[list[float]] | None) -> None:
        if polygons is None:
            self.xyn = None
        else:
            self.xyn = [_StubArray(p) for p in polygons]


class _StubBoxes:
    def __init__(self, classes: list[int]) -> None:
        self.cls = _StubArray([float(c) for c in classes])


class _StubResult:
    def __init__(
        self,
        path: str,
        save_dir: Path,
        classes: list[int],
        polygons: list[list[float]] | None,
    ) -> None:
        self.path = path
        self.save_dir = save_dir
        self.boxes = _StubBoxes(classes)
        self.masks = _StubMasks(polygons)


class _StubModel:
    def __init__(self, ref: str) -> None:
        self.ref = ref
        self.predict_kwargs: dict[str, Any] | None = None
        self.results: list[_StubResult] = []

    def predict(self, **kwargs: Any) -> list[_StubResult]:
        self.predict_kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        source = Path(kwargs["source"])
        if source.is_file():
            paths = [source]
        else:
            paths = sorted(p for p in source.rglob("*") if p.is_file())
        results = []
        for idx, path in enumerate(paths):
            classes = [0, 1] if idx == 0 else [1]
            polygons = [
                [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4],
                [0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9],
            ][: len(classes)]
            results.append(
                _StubResult(
                    path=str(path),
                    save_dir=save_dir,
                    classes=classes,
                    polygons=polygons,
                )
            )
        self.results = results
        return results


def _make_config(tmp_path: Path) -> PredictConfig:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    source = tmp_path / "frames"
    source.mkdir()
    (source / "a.jpg").write_bytes(b"img-a")
    (source / "b.png").write_bytes(b"img-b")
    return PredictConfig(
        weights=weights,
        source=source,
        project=tmp_path / "runs",
        name="yolo26n_seg_predict",
    )


def test_list_source_images_directory(tmp_path: Path) -> None:
    src = tmp_path / "frames"
    src.mkdir()
    (src / "img1.jpg").write_bytes(b"")
    (src / "img2.png").write_bytes(b"")
    (src / "ignored.txt").write_text("not an image")
    nested = src / "nested"
    nested.mkdir()
    (nested / "img3.jpeg").write_bytes(b"")

    paths = list_source_images(src)

    assert [p.name for p in paths] == ["img1.jpg", "img2.png", "img3.jpeg"]


def test_list_source_images_single_file(tmp_path: Path) -> None:
    img = tmp_path / "single.jpg"
    img.write_bytes(b"")
    assert list_source_images(img) == [img]


def test_list_source_images_rejects_non_image(tmp_path: Path) -> None:
    txt = tmp_path / "note.txt"
    txt.write_text("hi")
    with pytest.raises(ValueError, match="not a supported image"):
        list_source_images(txt)


def test_list_source_images_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list_source_images(tmp_path / "nope")


def test_validate_predict_inputs_rejects_missing_weights(tmp_path: Path) -> None:
    src = tmp_path / "frames"
    src.mkdir()
    cfg = PredictConfig(
        weights=tmp_path / "missing.pt",
        source=src,
    )
    with pytest.raises(FileNotFoundError, match="Missing weights"):
        validate_predict_inputs(cfg)


def test_validate_predict_inputs_rejects_missing_source(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.source = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="Predict source"):
        validate_predict_inputs(cfg)


def test_validate_predict_inputs_rejects_bad_thresholds(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.conf = 1.5
    with pytest.raises(ValueError, match="--conf"):
        validate_predict_inputs(cfg)
    cfg.conf = 0.25
    cfg.iou = -0.1
    with pytest.raises(ValueError, match="--iou"):
        validate_predict_inputs(cfg)


def test_build_predict_kwargs_defaults(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    kwargs = build_predict_kwargs(cfg)
    assert kwargs["task"] == "segment"
    assert kwargs["save"] is True
    assert kwargs["save_txt"] is False
    assert kwargs["exist_ok"] is False
    assert "device" not in kwargs
    assert "classes" not in kwargs


def test_build_predict_kwargs_with_overrides(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.device = "cpu"
    cfg.classes = [0]
    cfg.exist_ok = True
    cfg.save_overlays = False
    cfg.extra = {"agnostic_nms": True}
    kwargs = build_predict_kwargs(cfg)
    assert kwargs["device"] == "cpu"
    assert kwargs["classes"] == [0]
    assert kwargs["exist_ok"] is True
    assert kwargs["save"] is False
    assert kwargs["agnostic_nms"] is True


def test_run_predict_dry_run(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    report = run_predict(cfg, dry_run=True)
    assert report["dry_run"] is True
    assert "summary" not in report
    assert report["predict_kwargs"]["task"] == "segment"


def test_run_predict_invokes_factory_and_summarizes(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    captured: dict[str, _StubModel] = {}

    def factory(ref: str) -> _StubModel:
        model = _StubModel(ref)
        captured["model"] = model
        return model

    report = run_predict(cfg, model_factory=factory)
    assert captured["model"].predict_kwargs is not None
    assert report["dry_run"] is False
    assert report["summary"]["image_count"] == 2
    assert report["summary"]["total_detections"] == 3
    assert report["summary"]["detections_by_class"] == {"0": 1, "1": 2}


def test_run_predict_save_polygons(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.save_polygons = True

    def factory(ref: str) -> _StubModel:
        return _StubModel(ref)

    report = run_predict(cfg, model_factory=factory)
    polygons_dir = Path(report["polygons_dir"])
    assert polygons_dir.is_dir()
    txt_files = sorted(polygons_dir.glob("*.txt"))
    assert len(txt_files) == 2
    first = txt_files[0].read_text(encoding="utf-8")
    assert first.splitlines()[0].startswith("0 ")


def test_summarize_results_empty() -> None:
    summary = summarize_results([])
    assert summary == {
        "image_count": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_by_class": {},
        "images": [],
    }


def test_write_polygons_for_result_skips_when_no_masks(tmp_path: Path) -> None:
    result = _StubResult(
        path=str(tmp_path / "img.jpg"),
        save_dir=tmp_path,
        classes=[],
        polygons=None,
    )
    out = write_polygons_for_result(result, tmp_path / "polygons")
    assert out is None
    assert not (tmp_path / "polygons").exists()


def test_cli_dry_run_prints_kwargs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    src = tmp_path / "frames"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"")

    code = cli_main(
        [
            str(weights),
            "--source",
            str(src),
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "preview",
            "--dry-run",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["predict_kwargs"]["name"] == "preview"


def test_cli_writes_report_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    src = tmp_path / "frames"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"")
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        "ml_pipeline.predict._default_predict_model_factory",
        lambda ref: _StubModel(ref),
    )

    code = cli_main(
        [
            str(weights),
            "--source",
            str(src),
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "preview",
            "--report-json",
            str(report_path),
        ]
    )
    assert code == 0
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["summary"]["image_count"] == 1
