"""Tests for ml_pipeline.cli (gdpr-yolo-normalize-labels entrypoint)."""

from __future__ import annotations

import json
from pathlib import Path
from ml_pipeline.cli import main


def test_main_success_without_strict(sample_dataset: Path) -> None:
    code = main([str(sample_dataset)])
    assert code == 0


def test_main_strict_fails_when_missing_label(sample_dataset: Path) -> None:
    code = main([str(sample_dataset), "--strict"])
    assert code == 1


def test_main_dry_run_does_not_write_labels(sample_dataset: Path) -> None:
    code = main([str(sample_dataset), "--dry-run"])
    assert code == 0
    assert not (sample_dataset / "labels").exists()


def test_main_writes_report_json(sample_dataset: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    code = main([str(sample_dataset), "--dry-run", "--report-json", str(report_path)])
    assert code == 0
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["dataset_root"] == str(sample_dataset.resolve())
    assert "train" in data["splits"]
