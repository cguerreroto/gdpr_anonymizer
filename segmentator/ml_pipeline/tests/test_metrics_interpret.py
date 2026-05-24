"""Tests for ml_pipeline.metrics_interpret."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from ml_pipeline.cli_validate import main as cli_validate_main
from ml_pipeline.metrics_interpret import (
    MASK_MAP_HIGH,
    MASK_MAP_LOW,
    assess_metrics,
    classify_mask_map,
    interpret_validation_report,
)


def test_classify_mask_map_bands() -> None:
    assert classify_mask_map(0.10) == "below_threshold"
    assert classify_mask_map(MASK_MAP_LOW - 0.01) == "below_threshold"
    assert classify_mask_map(0.35) == "usable"
    assert classify_mask_map(MASK_MAP_HIGH - 0.01) == "usable"
    assert classify_mask_map(0.55) == "comfortable"
    assert classify_mask_map(None) == "unknown"


def test_assess_metrics_below_threshold_recommends_more_data() -> None:
    metrics = {
        "mask": {"map": 0.107, "map50": 0.218},
        "per_class": {
            "Person": {"mask_map": 0.051},
            "Car": {"mask_map": 0.163},
        },
    }
    out = assess_metrics(metrics)
    assert out["assessment"]["band"] == "below_threshold"
    assert out["assessment"]["ready_for_video_blur"] is False
    assert any("below 0.30" in line for line in out["recommendations"])
    assert any("Person" in line for line in out["recommendations"])


def test_assess_metrics_usable_band() -> None:
    metrics = {"mask": {"map": 0.35, "map50": 0.40}}
    out = assess_metrics(metrics)
    assert out["assessment"]["band"] == "usable"
    assert out["assessment"]["ready_for_video_blur"] is True


def test_assess_metrics_comfortable_band() -> None:
    metrics = {"mask": {"map": 0.55, "map50": 0.60}}
    out = assess_metrics(metrics)
    assert out["assessment"]["band"] == "comfortable"
    assert out["assessment"]["ready_for_video_blur"] is True


def test_interpret_validation_report_accepts_full_report() -> None:
    report = {
        "metrics": {
            "mask": {"map": 0.42},
            "per_class": {"Person": {"mask_map": 0.40}},
        }
    }
    out = interpret_validation_report(report)
    assert out["assessment"]["band"] == "usable"


def test_cli_interpret_report_reads_saved_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "metrics.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "mask": {"map": 0.11, "map50": 0.22},
                    "per_class": {"Person": {"mask_map": 0.05}},
                }
            }
        ),
        encoding="utf-8",
    )
    code = cli_validate_main(["--interpret-report", str(report_path)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["assessment"]["band"] == "below_threshold"


def test_assess_metrics_unknown_band_when_mask_map_missing() -> None:
    out = assess_metrics({})
    assert out["assessment"]["band"] == "unknown"
    assert out["assessment"]["ready_for_video_blur"] is False
    assert any("Could not read" in r for r in out["recommendations"])


def test_assess_metrics_class_gap_recommendation() -> None:
    metrics = {
        "mask": {"map": 0.45, "map50": 0.50},
        "per_class": {
            "Person": {"mask_map": 0.20},
            "Car": {"mask_map": 0.55},
        },
    }
    out = assess_metrics(metrics)
    assert any("Largest gap" in line for line in out["recommendations"])


def test_assess_metrics_map50_far_above_map() -> None:
    metrics = {"mask": {"map": 0.20, "map50": 0.45}}
    out = assess_metrics(metrics)
    assert any("loosely aligned" in line for line in out["recommendations"])


def test_rank_classes_handles_invalid_per_class_value() -> None:
    metrics = {
        "mask": {"map": 0.40, "map50": 0.50},
        "per_class": {"weird": "not-a-dict"},
    }
    out = assess_metrics(metrics)
    assert out["assessment"]["band"] == "usable"
