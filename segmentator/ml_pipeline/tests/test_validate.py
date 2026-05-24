"""Tests for ml_pipeline.validate and ml_pipeline.cli_validate.

A stub factory replaces Ultralytics so the validation driver and CLI can be
unit tested without GPU, network access, or model weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from ml_pipeline.cli_validate import main as cli_main
from ml_pipeline.validate import (
    ValidateConfig,
    _default_val_model_factory,
    build_val_kwargs,
    extract_metrics,
    run_validation,
    validate_inputs,
)
from fakes import block_ultralytics_import, install_fake_ultralytics


class _FakeBox:
    def __init__(self, ap: float, ap50: float, maps: list[float]) -> None:
        self.map = ap
        self.map50 = ap50
        self.maps = maps


class _FakeSeg(_FakeBox):
    def __init__(
        self, ap: float, ap50: float, ap75: float, maps: list[float]
    ) -> None:
        super().__init__(ap, ap50, maps)
        self.map75 = ap75


class _FakeResults:
    def __init__(
        self,
        save_dir: Path,
        seg: _FakeSeg | None = None,
        box: _FakeBox | None = None,
        names: dict[int, str] | None = None,
    ) -> None:
        self.save_dir = save_dir
        self.seg = seg
        self.box = box
        self.names = names or {}


class _StubModel:
    def __init__(self, ref: str) -> None:
        self.ref = ref
        self.val_kwargs: dict[str, Any] | None = None
        self.results = _FakeResults(
            save_dir=Path("/tmp/unused"),
            seg=_FakeSeg(0.42, 0.61, 0.45, [0.40, 0.44]),
            box=_FakeBox(0.55, 0.74, [0.52, 0.58]),
            names={0: "Person", 1: "Car"},
        )

    def val(self, **kwargs: Any) -> _FakeResults:
        self.val_kwargs = kwargs
        project = Path(kwargs["project"])
        self.results.save_dir = project / kwargs["name"]
        return self.results


def _make_config(tmp_path: Path) -> ValidateConfig:
    (tmp_path / "dataset.yaml").write_text(
        "path: .\nval: images/val\n", encoding="utf-8"
    )
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    return ValidateConfig(
        weights=weights,
        dataset_yaml=tmp_path / "dataset.yaml",
        project=tmp_path / "runs",
    )


def test_default_val_model_factory_loads_fake_yolo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs: list[str] = []
    install_fake_ultralytics(
        monkeypatch,
        yolo_factory=lambda ref: refs.append(ref) or _StubModel(ref),
    )

    model = _default_val_model_factory("weights/best.pt")

    assert refs == ["weights/best.pt"]
    assert isinstance(model, _StubModel)
    assert model.ref == "weights/best.pt"


def test_default_val_model_factory_surfaces_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_ultralytics_import(monkeypatch)

    with pytest.raises(RuntimeError, match="uv sync --extra train"):
        _default_val_model_factory("best.pt")


def test_build_val_kwargs_defaults(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    kwargs = build_val_kwargs(cfg)
    assert kwargs["task"] == "segment"
    assert kwargs["split"] == "val"
    assert kwargs["save_json"] is False
    assert "device" not in kwargs
    assert "conf" not in kwargs
    assert "iou" not in kwargs


def test_build_val_kwargs_thresholds_and_extra(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.conf = 0.25
    cfg.iou = 0.6
    cfg.device = "cpu"
    cfg.save_json = True
    cfg.extra = {"plots": False}
    kwargs = build_val_kwargs(cfg)
    assert kwargs["conf"] == 0.25
    assert kwargs["iou"] == 0.6
    assert kwargs["device"] == "cpu"
    assert kwargs["save_json"] is True
    assert kwargs["plots"] is False


def test_validate_inputs_rejects_missing_files(tmp_path: Path) -> None:
    cfg = ValidateConfig(
        weights=tmp_path / "missing.pt",
        dataset_yaml=tmp_path / "missing.yaml",
    )
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        validate_inputs(cfg)
    (tmp_path / "missing.yaml").write_text("path: .\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing weights"):
        validate_inputs(cfg)


def test_validate_inputs_lists_checkpoints_under_project(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
    project = tmp_path / "runs"
    best = project / "yolo26n_seg_v1-3" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"")
    cfg = ValidateConfig(
        weights=project / "yolo26n_seg_v1" / "weights" / "best.pt",
        dataset_yaml=tmp_path / "dataset.yaml",
        project=project,
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_inputs(cfg)
    message = str(exc_info.value)
    assert "yolo26n_seg_v1-3" in message
    assert "validate_weights" in message


def test_validate_inputs_rejects_unknown_split(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.split = "foo"
    with pytest.raises(ValueError, match="Unknown split"):
        validate_inputs(cfg)


def test_extract_metrics_handles_full_results() -> None:
    results = _FakeResults(
        save_dir=Path("/tmp/unused"),
        seg=_FakeSeg(0.42, 0.61, 0.45, [0.40, 0.44]),
        box=_FakeBox(0.55, 0.74, [0.52, 0.58]),
        names={0: "Person", 1: "Car"},
    )
    metrics = extract_metrics(results)
    assert metrics["mask"]["map"] == pytest.approx(0.42)
    assert metrics["mask"]["map50"] == pytest.approx(0.61)
    assert metrics["mask"]["map75"] == pytest.approx(0.45)
    assert metrics["box"]["map"] == pytest.approx(0.55)
    assert metrics["per_class"]["Person"]["mask_map"] == pytest.approx(0.40)
    assert metrics["per_class"]["Car"]["mask_map"] == pytest.approx(0.44)
    assert metrics["per_class"]["Car"]["box_map"] == pytest.approx(0.58)


def test_extract_metrics_tolerates_missing_seg_or_box() -> None:
    results = _FakeResults(save_dir=Path("/tmp/unused"))
    metrics = extract_metrics(results)
    assert metrics["mask"] == {}
    assert metrics["box"] == {}
    assert metrics["per_class"] == {}


def test_run_validation_dry_run_skips_factory(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    called: list[str] = []

    def _factory(ref: str) -> _StubModel:
        called.append(ref)
        return _StubModel(ref)

    report = run_validation(cfg, model_factory=_factory, dry_run=True)
    assert called == []
    assert report["dry_run"] is True
    assert "metrics" not in report
    assert report["val_kwargs"]["task"] == "segment"


def test_run_validation_invokes_factory_and_extracts_metrics(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    last_model: dict[str, _StubModel] = {}

    def _factory(ref: str) -> _StubModel:
        m = _StubModel(ref)
        last_model["m"] = m
        return m

    report = run_validation(cfg, model_factory=_factory, dry_run=False)
    assert last_model["m"].val_kwargs is not None
    resolved = Path(report["resolved_dataset_yaml"])
    assert resolved.is_file()
    assert last_model["m"].val_kwargs["data"] == str(resolved)
    assert report["metrics"]["mask"]["map"] == pytest.approx(0.42)
    assert report["interpretation"]["assessment"]["band"] == "usable"
    assert report["interpretation"]["assessment"]["ready_for_video_blur"] is True
    assert report["save_dir"].endswith("yolo26n_seg_val")


def test_run_validation_omits_interpretation_when_disabled(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)

    def _factory(ref: str) -> _StubModel:
        return _StubModel(ref)

    report = run_validation(
        cfg, model_factory=_factory, dry_run=False, interpret=False
    )
    assert "interpretation" not in report


def test_cli_dry_run_prints_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    code = cli_main(
        [
            str(tmp_path),
            "--weights",
            str(weights),
            "--dry-run",
            "--imgsz",
            "320",
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "demo_val",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["dry_run"] is True
    assert report["val_kwargs"]["imgsz"] == 320
    assert report["val_kwargs"]["name"] == "demo_val"


def test_cli_writes_report_json(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    out_path = tmp_path / "out" / "val_report.json"
    code = cli_main(
        [
            str(tmp_path),
            "--weights",
            str(weights),
            "--dry-run",
            "--report-json",
            str(out_path),
            "--project",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 0
    assert out_path.is_file()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["val_kwargs"]["task"] == "segment"
