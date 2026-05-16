"""Tests for ml_pipeline.train and ml_pipeline.cli_train.

Ultralytics is never imported here: a stub factory is injected so the
training driver can be unit tested without GPU, network, or model downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
import yaml

from ml_pipeline.cli_train import _resolve_dataset_yaml, main as cli_main
from ml_pipeline.train import (
    TrainConfig,
    build_train_kwargs,
    materialize_resolved_dataset_yaml,
    resolve_model_reference,
    run_training,
    validate_dataset_yaml,
)


class _StubResults:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir


class _StubModel:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self.train_kwargs: dict[str, Any] | None = None

    def train(self, **kwargs: Any) -> _StubResults:
        self.train_kwargs = kwargs
        project = Path(kwargs["project"])
        return _StubResults(project / kwargs["name"])


def _make_config(tmp_path: Path) -> TrainConfig:
    (tmp_path / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )
    return TrainConfig(
        dataset_yaml=tmp_path / "dataset.yaml",
        project=tmp_path / "runs",
    )


def test_build_train_kwargs_defaults(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    kwargs = build_train_kwargs(config)
    assert kwargs["task"] == "segment"
    assert kwargs["data"].endswith("dataset.yaml")
    assert kwargs["epochs"] == 100
    assert kwargs["imgsz"] == 640
    assert kwargs["name"] == "yolo26n_seg"
    assert "device" not in kwargs


def test_build_train_kwargs_respects_device_and_extra(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.device = "cpu"
    config.extra = {"lr0": 0.001, "cos_lr": True}
    kwargs = build_train_kwargs(config)
    assert kwargs["device"] == "cpu"
    assert kwargs["lr0"] == 0.001
    assert kwargs["cos_lr"] is True


def test_resolve_model_reference_prefers_weights(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    assert resolve_model_reference(config) == "yolo26n-seg.pt"
    config.weights = "/some/best.pt"
    assert resolve_model_reference(config) == "/some/best.pt"


def test_validate_dataset_yaml_missing(tmp_path: Path) -> None:
    config = TrainConfig(dataset_yaml=tmp_path / "absent.yaml")
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        validate_dataset_yaml(config)


def test_run_training_dry_run_skips_factory(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    called: list[str] = []

    def _factory(ref: str) -> _StubModel:
        called.append(ref)
        return _StubModel(ref)

    report = run_training(config, model_factory=_factory, dry_run=True)
    assert report["dry_run"] is True
    assert report["model_reference"] == "yolo26n-seg.pt"
    assert called == []
    assert report["train_kwargs"]["task"] == "segment"


def test_run_training_invokes_factory_and_train(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    last_model: dict[str, _StubModel] = {}

    def _factory(ref: str) -> _StubModel:
        model = _StubModel(ref)
        last_model["m"] = model
        return model

    report = run_training(config, model_factory=_factory, dry_run=False)
    assert last_model["m"].train_kwargs is not None
    assert last_model["m"].train_kwargs["task"] == "segment"
    assert report["save_dir"].endswith("yolo26n_seg")
    resolved = Path(report["resolved_dataset_yaml"])
    assert resolved.is_file()
    assert last_model["m"].train_kwargs["data"] == str(resolved)


def test_materialize_resolved_dataset_yaml_rewrites_paths(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    resolved = materialize_resolved_dataset_yaml(
        config.dataset_yaml, config.project / config.name
    )
    assert resolved.is_file()
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert data["path"] == str(tmp_path.resolve())
    assert data["train"] == str((tmp_path / "images" / "train").resolve())
    assert data["val"] == str((tmp_path / "images" / "val").resolve())
    original = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
    assert "path: .\n" in original


def test_materialize_resolved_dataset_yaml_handles_missing_path_field(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs" / "out"
    resolved = materialize_resolved_dataset_yaml(
        tmp_path / "dataset.yaml", output_dir
    )
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert data["path"] == str(tmp_path.resolve())
    assert data["val"] == str((tmp_path / "images" / "val").resolve())


def test_run_training_uses_warm_start_weights(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    config.weights = str(weights)
    seen: dict[str, str] = {}

    def _factory(ref: str) -> _StubModel:
        seen["ref"] = ref
        return _StubModel(ref)

    run_training(config, model_factory=_factory, dry_run=False)
    assert seen["ref"] == str(weights)


def test_cli_dry_run_prints_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
    code = cli_main(
        [
            str(tmp_path),
            "--dry-run",
            "--epochs",
            "5",
            "--imgsz",
            "320",
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "demo",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["dry_run"] is True
    assert report["train_kwargs"]["epochs"] == 5
    assert report["train_kwargs"]["imgsz"] == 320
    assert report["train_kwargs"]["name"] == "demo"


def test_cli_writes_report_json(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
    out_path = tmp_path / "out" / "train_report.json"
    code = cli_main(
        [
            str(tmp_path),
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
    assert data["train_kwargs"]["task"] == "segment"


def test_resolve_dataset_yaml_accepts_file(tmp_path: Path) -> None:
    yaml_file = tmp_path / "dataset.yaml"
    yaml_file.write_text("path: .\n", encoding="utf-8")
    assert _resolve_dataset_yaml(yaml_file) == yaml_file.resolve()
    assert _resolve_dataset_yaml(tmp_path) == yaml_file.resolve()
