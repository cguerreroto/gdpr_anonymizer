"""Tests for ml_pipeline.iterate and ml_pipeline.cli_iterate.

A pair of stub runners replaces the real Ultralytics-backed
:func:`run_training` and :func:`run_validation` so the orchestration is
exercised without GPU, network, or model downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml_pipeline.cli_iterate import main as cli_main
from ml_pipeline.iterate import (
    IterateConfig,
    compare_validation_reports,
    derive_next_run_name,
    resolve_previous_weights,
    run_iteration,
    validate_iterate_inputs,
)
from ml_pipeline.train import TrainConfig
from ml_pipeline.validate import ValidateConfig


def _make_previous_run(tmp_path: Path, name: str = "yolo26n_seg_v1") -> Path:
    run_dir = tmp_path / "runs" / name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"")
    return run_dir


def _make_dataset(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )
    return yaml_path


def _make_config(tmp_path: Path, **overrides: Any) -> IterateConfig:
    previous = _make_previous_run(tmp_path)
    dataset = _make_dataset(tmp_path)
    return IterateConfig(
        previous_run=previous,
        dataset_yaml=dataset,
        project=tmp_path / "runs",
        **overrides,
    )


def test_derive_next_run_name_appends_v2_when_no_version() -> None:
    assert derive_next_run_name("yolo26n_seg") == "yolo26n_seg_v2"
    assert derive_next_run_name("experiment_a") == "experiment_a_v2"


def test_derive_next_run_name_increments_existing_version() -> None:
    assert derive_next_run_name("yolo26n_seg_v1") == "yolo26n_seg_v2"
    assert derive_next_run_name("yolo26n_seg_v9") == "yolo26n_seg_v10"


def test_derive_next_run_name_strips_ultralytics_auto_suffix() -> None:
    assert derive_next_run_name("yolo26n_seg_v1-3") == "yolo26n_seg_v2"
    assert derive_next_run_name("yolo26n_seg-2") == "yolo26n_seg_v2"


def test_validate_iterate_inputs_missing_run(tmp_path: Path) -> None:
    config = IterateConfig(
        previous_run=tmp_path / "absent",
        dataset_yaml=_make_dataset(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="Previous run directory"):
        validate_iterate_inputs(config)


def test_validate_iterate_inputs_missing_dataset(tmp_path: Path) -> None:
    config = IterateConfig(
        previous_run=_make_previous_run(tmp_path),
        dataset_yaml=tmp_path / "absent.yaml",
    )
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        validate_iterate_inputs(config)


def test_validate_iterate_inputs_unknown_split(tmp_path: Path) -> None:
    config = _make_config(tmp_path, val_split="weird")
    with pytest.raises(ValueError, match="Unknown split"):
        validate_iterate_inputs(config)


def test_validate_iterate_inputs_missing_compare_to(tmp_path: Path) -> None:
    config = _make_config(tmp_path, compare_to=tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="--compare-to file"):
        validate_iterate_inputs(config)


def test_resolve_previous_weights_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No weights/best.pt"):
        resolve_previous_weights(run_dir)


def test_resolve_previous_weights_returns_path(tmp_path: Path) -> None:
    previous = _make_previous_run(tmp_path)
    weights = resolve_previous_weights(previous)
    assert weights == previous / "weights" / "best.pt"


def test_compare_validation_reports_improvement() -> None:
    previous = {"metrics": {"mask": {"map": 0.30, "map50": 0.50}, "box": {}, "per_class": {}}}
    current = {"metrics": {"mask": {"map": 0.45, "map50": 0.65}, "box": {}, "per_class": {}}}
    comparison = compare_validation_reports(previous, current)
    assert comparison["overall"]["mask_map"]["verdict"] == "improved"
    assert comparison["overall"]["mask_map"]["delta"] == pytest.approx(0.15)
    assert comparison["overall_verdict"] == "improved"
    assert "improved" in comparison["summary"]
    assert "0.300" in comparison["summary"] and "0.450" in comparison["summary"]


def test_compare_validation_reports_regression() -> None:
    previous = {"mask": {"map": 0.50}}
    current = {"mask": {"map": 0.30}}
    comparison = compare_validation_reports(previous, current)
    assert comparison["overall_verdict"] == "regressed"
    assert comparison["overall"]["mask_map"]["delta"] == pytest.approx(-0.20)


def test_compare_validation_reports_unchanged_within_threshold() -> None:
    previous = {"mask": {"map": 0.40}}
    current = {"mask": {"map": 0.402}}
    comparison = compare_validation_reports(previous, current)
    assert comparison["overall_verdict"] == "unchanged"


def test_compare_validation_reports_unknown_metric() -> None:
    previous = {"mask": {}}
    current = {"mask": {}}
    comparison = compare_validation_reports(previous, current)
    assert comparison["overall_verdict"] == "unknown"
    assert comparison["summary"].startswith("Mask mAP@0.5:0.95 not available")


def test_compare_validation_reports_per_class_deltas() -> None:
    previous = {
        "metrics": {
            "mask": {"map": 0.30},
            "per_class": {
                "Person": {"mask_map": 0.40, "box_map": 0.45},
                "Car": {"mask_map": 0.20, "box_map": 0.25},
            },
        }
    }
    current = {
        "metrics": {
            "mask": {"map": 0.45},
            "per_class": {
                "Person": {"mask_map": 0.50, "box_map": 0.55},
                "Car": {"mask_map": 0.40, "box_map": 0.42},
            },
        }
    }
    comparison = compare_validation_reports(previous, current)
    person = comparison["per_class"]["Person"]
    car = comparison["per_class"]["Car"]
    assert person["mask_map"]["verdict"] == "improved"
    assert car["mask_map"]["delta"] == pytest.approx(0.20)
    assert car["box_map"]["verdict"] == "improved"


def test_run_iteration_dry_run_skips_runners(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    train_called: list[TrainConfig] = []
    val_called: list[ValidateConfig] = []

    report = run_iteration(
        config,
        train_runner=lambda c: (train_called.append(c) or {}),
        val_runner=lambda c: (val_called.append(c) or {}),
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["next_run_name"] == "yolo26n_seg_v2"
    assert report["train_kwargs"]["task"] == "segment"
    assert report["val_kwargs"]["task"] == "segment"
    assert report["val_kwargs"]["split"] == "val"
    assert "weights_placeholder" in report["val_kwargs"]
    assert train_called == []
    assert val_called == []


def test_run_iteration_dry_run_skip_validation_omits_val_kwargs(tmp_path: Path) -> None:
    config = _make_config(tmp_path, skip_validation=True)
    report = run_iteration(config, dry_run=True)
    assert "val_kwargs" not in report
    assert report["train_kwargs"]["task"] == "segment"


def test_run_iteration_invokes_train_and_val(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    seen_train: dict[str, TrainConfig] = {}
    seen_val: dict[str, ValidateConfig] = {}

    new_weights = tmp_path / "runs" / "yolo26n_seg_v2" / "weights" / "best.pt"

    def train_runner(c: TrainConfig) -> dict[str, Any]:
        seen_train["c"] = c
        new_weights.parent.mkdir(parents=True, exist_ok=True)
        new_weights.write_bytes(b"")
        return {
            "save_dir": str(new_weights.parent.parent),
            "validate_weights": str(new_weights),
        }

    def val_runner(c: ValidateConfig) -> dict[str, Any]:
        seen_val["c"] = c
        return {
            "metrics": {
                "mask": {"map": 0.42, "map50": 0.65, "map75": 0.50},
                "box": {"map": 0.50, "map50": 0.70},
                "per_class": {},
            }
        }

    report = run_iteration(config, train_runner=train_runner, val_runner=val_runner)
    assert seen_train["c"].weights == str(config.previous_run / "weights" / "best.pt")
    assert seen_train["c"].name == "yolo26n_seg_v2"
    assert seen_val["c"].weights == new_weights
    assert seen_val["c"].name == "yolo26n_seg_v2_val"
    assert report["train_report"]["validate_weights"] == str(new_weights)
    assert report["validate_report"]["metrics"]["mask"]["map"] == pytest.approx(0.42)
    assert "comparison" not in report


def test_run_iteration_compares_with_previous_report(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    prev_report_path = tmp_path / "previous_metrics.json"
    prev_report_path.write_text(
        json.dumps({"metrics": {"mask": {"map": 0.30}, "box": {}, "per_class": {}}}),
        encoding="utf-8",
    )
    config.compare_to = prev_report_path

    new_weights = tmp_path / "runs" / "yolo26n_seg_v2" / "weights" / "best.pt"

    def train_runner(_: TrainConfig) -> dict[str, Any]:
        new_weights.parent.mkdir(parents=True, exist_ok=True)
        new_weights.write_bytes(b"")
        return {"validate_weights": str(new_weights)}

    def val_runner(_: ValidateConfig) -> dict[str, Any]:
        return {"metrics": {"mask": {"map": 0.45}, "box": {}, "per_class": {}}}

    report = run_iteration(config, train_runner=train_runner, val_runner=val_runner)
    assert report["comparison"]["overall_verdict"] == "improved"
    assert report["comparison"]["overall"]["mask_map"]["delta"] == pytest.approx(0.15)


def test_run_iteration_skip_validation(tmp_path: Path) -> None:
    config = _make_config(tmp_path, skip_validation=True)

    def train_runner(_: TrainConfig) -> dict[str, Any]:
        return {"validate_weights": str(tmp_path / "fake.pt")}

    def val_runner(_: ValidateConfig) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("val should not run when skip_validation=True")

    report = run_iteration(config, train_runner=train_runner, val_runner=val_runner)
    assert "validate_report" not in report
    assert "comparison" not in report


def test_run_iteration_marks_skip_when_train_omits_weights(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    def train_runner(_: TrainConfig) -> dict[str, Any]:
        return {"save_dir": "/somewhere"}

    report = run_iteration(config, train_runner=train_runner, val_runner=lambda c: {})
    assert "validate_report" not in report
    assert "did not include 'validate_weights'" in report["validation_skipped_reason"]


def test_run_iteration_explicit_name_overrides_derivation(tmp_path: Path) -> None:
    config = _make_config(tmp_path, name="custom_run")
    report = run_iteration(config, dry_run=True)
    assert report["next_run_name"] == "custom_run"
    assert report["val_kwargs"]["name"] == "custom_run_val"


def test_cli_dry_run_prints_kwargs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = _make_previous_run(tmp_path)
    _make_dataset(tmp_path)

    code = cli_main(
        [
            str(previous),
            str(tmp_path),
            "--project",
            str(tmp_path / "runs"),
            "--epochs",
            "5",
            "--dry-run",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["next_run_name"] == "yolo26n_seg_v2"
    assert payload["train_kwargs"]["epochs"] == 5
    assert payload["val_kwargs"]["task"] == "segment"


def test_cli_writes_report_json(tmp_path: Path) -> None:
    previous = _make_previous_run(tmp_path)
    _make_dataset(tmp_path)
    out_path = tmp_path / "out" / "iterate_report.json"

    code = cli_main(
        [
            str(previous),
            str(tmp_path),
            "--project",
            str(tmp_path / "runs"),
            "--dry-run",
            "--skip-validation",
            "--report-json",
            str(out_path),
        ]
    )
    assert code == 0
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["dry_run"] is True
    assert saved["next_run_name"] == "yolo26n_seg_v2"
    assert "val_kwargs" not in saved


def test_cli_dry_run_includes_compare_to_path(tmp_path: Path) -> None:
    previous = _make_previous_run(tmp_path)
    _make_dataset(tmp_path)
    prev_metrics = tmp_path / "prev.json"
    prev_metrics.write_text(
        json.dumps({"metrics": {"mask": {"map": 0.3}}}), encoding="utf-8"
    )

    code = cli_main(
        [
            str(previous),
            str(tmp_path),
            "--project",
            str(tmp_path / "runs"),
            "--dry-run",
            "--compare-to",
            str(prev_metrics),
        ]
    )
    assert code == 0
