"""Tests for ml_pipeline.errors and ml_pipeline.cli_errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml_pipeline.errors import (
    ErrorConfig,
    aggregate_errors,
    analyze_image_errors,
    build_predict_kwargs,
    compute_mask_iou,
    parse_yolo_label,
    run_error_analysis,
    validate_error_inputs,
)


def test_error_config_defaults():
    """ErrorConfig should set reasonable defaults."""
    cfg = ErrorConfig(
        weights=Path("weights.pt"),
        dataset_yaml=Path("data.yaml"),
    )
    assert cfg.imgsz == 640
    assert cfg.conf == 0.25
    assert cfg.iou_threshold == 0.5
    assert cfg.device is None
    assert cfg.name == "error_analysis"


def test_error_config_paths_resolved():
    """ErrorConfig should resolve string paths to Path objects."""
    cfg = ErrorConfig(
        weights="weights.pt",
        dataset_yaml="data.yaml",
        project="runs",
        report_json="report.json",
    )
    assert isinstance(cfg.weights, Path)
    assert isinstance(cfg.dataset_yaml, Path)
    assert isinstance(cfg.project, Path)
    assert isinstance(cfg.report_json, Path)


def test_validate_error_inputs_missing_weights(tmp_path: Path):
    """validate_error_inputs should raise if weights missing."""
    yaml_file = tmp_path / "dataset.yaml"
    yaml_file.write_text("nc: 2\n")
    cfg = ErrorConfig(
        weights=tmp_path / "missing.pt",
        dataset_yaml=yaml_file,
    )
    with pytest.raises(FileNotFoundError, match="Missing weights"):
        validate_error_inputs(cfg)


def test_validate_error_inputs_missing_yaml(tmp_path: Path):
    """validate_error_inputs should raise if dataset.yaml missing."""
    weights_file = tmp_path / "best.pt"
    weights_file.write_text("fake weights")
    cfg = ErrorConfig(
        weights=weights_file,
        dataset_yaml=tmp_path / "missing.yaml",
    )
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        validate_error_inputs(cfg)


def test_build_predict_kwargs(tmp_path: Path):
    """build_predict_kwargs should construct correct kwargs for YOLO predict."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.write_text("nc: 2\n")
    
    cfg = ErrorConfig(
        weights=Path("weights.pt"),
        dataset_yaml=dataset_yaml,
        imgsz=512,
        conf=0.3,
        device="cpu",
        project=Path("runs"),
        name="test_errors",
    )
    kwargs = build_predict_kwargs(cfg)
    
    assert kwargs["source"] == dataset_root / "images" / "val"
    assert kwargs["imgsz"] == 512
    assert kwargs["conf"] == 0.3
    assert kwargs["device"] == "cpu"
    assert kwargs["project"] == "runs"
    assert kwargs["name"] == "test_errors"
    assert kwargs["save"] is False


def test_compute_mask_iou_perfect_match():
    """compute_mask_iou should return 1.0 for identical polygons."""
    poly = [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]
    iou = compute_mask_iou(poly, poly)
    assert iou == 1.0


def test_compute_mask_iou_no_overlap():
    """compute_mask_iou should return 0.0 for non-overlapping polygons."""
    poly1 = [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]
    poly2 = [0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9]
    iou = compute_mask_iou(poly1, poly2)
    assert iou == 0.0


def test_compute_mask_iou_partial_overlap():
    """compute_mask_iou should return value between 0 and 1 for partial overlap."""
    poly1 = [0.1, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6]
    poly2 = [0.4, 0.4, 0.9, 0.4, 0.9, 0.9, 0.4, 0.9]
    iou = compute_mask_iou(poly1, poly2)
    assert 0.0 < iou < 1.0


def test_compute_mask_iou_empty_polygon():
    """compute_mask_iou should return 0.0 for empty polygons."""
    poly1 = [0.1, 0.1, 0.4, 0.4]
    poly2 = []
    assert compute_mask_iou(poly1, poly2) == 0.0
    assert compute_mask_iou([], poly1) == 0.0


def test_analyze_image_errors_perfect_predictions():
    """analyze_image_errors should report zero errors for perfect predictions."""
    gt = [
        {"class": 0, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
        {"class": 1, "polygon": [0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9]},
    ]
    preds = [
        {"class": 0, "confidence": 0.9, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
        {"class": 1, "confidence": 0.85, "polygon": [0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9]},
    ]
    
    result = analyze_image_errors(Path("test.jpg"), preds, gt, iou_threshold=0.5)
    
    assert result["false_negatives"] == {}
    assert result["false_positives"] == {}
    assert result["gt_count"] == 2
    assert result["pred_count"] == 2


def test_analyze_image_errors_missed_detection():
    """analyze_image_errors should report FN when GT object not detected."""
    gt = [
        {"class": 0, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
        {"class": 1, "polygon": [0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9]},
    ]
    preds = [
        {"class": 0, "confidence": 0.9, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
    ]
    
    result = analyze_image_errors(Path("test.jpg"), preds, gt, iou_threshold=0.5)
    
    assert result["false_negatives"] == {1: 1}
    assert result["false_positives"] == {}


def test_analyze_image_errors_false_positive():
    """analyze_image_errors should report FP when prediction has no matching GT."""
    gt = [
        {"class": 0, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
    ]
    preds = [
        {"class": 0, "confidence": 0.9, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
        {"class": 1, "confidence": 0.3, "polygon": [0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9]},
    ]
    
    result = analyze_image_errors(Path("test.jpg"), preds, gt, iou_threshold=0.5)
    
    assert result["false_negatives"] == {}
    assert result["false_positives"] == {1: 1}


def test_analyze_image_errors_class_mismatch():
    """analyze_image_errors should report FN+FP when pred class differs from GT."""
    gt = [
        {"class": 0, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
    ]
    preds = [
        {"class": 1, "confidence": 0.9, "polygon": [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]},
    ]
    
    result = analyze_image_errors(Path("test.jpg"), preds, gt, iou_threshold=0.5)
    
    assert result["false_negatives"] == {0: 1}
    assert result["false_positives"] == {1: 1}


def test_aggregate_errors_empty():
    """aggregate_errors should handle empty input gracefully."""
    result = aggregate_errors([])
    assert result["total_false_negatives_by_class"] == {}
    assert result["total_false_positives_by_class"] == {}
    assert result["total_images_analyzed"] == 0
    assert result["images_with_errors"] == 0
    assert result["worst_images"] == []


def test_aggregate_errors_sum_by_class():
    """aggregate_errors should sum FN and FP counts per class."""
    image_errors = [
        {
            "image": "img1.jpg",
            "false_negatives": {0: 2, 1: 1},
            "false_positives": {0: 1},
            "gt_count": 5,
            "pred_count": 4,
        },
        {
            "image": "img2.jpg",
            "false_negatives": {0: 1},
            "false_positives": {1: 2},
            "gt_count": 3,
            "pred_count": 4,
        },
    ]
    
    result = aggregate_errors(image_errors)
    
    assert result["total_false_negatives_by_class"] == {0: 3, 1: 1}
    assert result["total_false_positives_by_class"] == {0: 1, 1: 2}
    assert result["total_images_analyzed"] == 2
    assert result["images_with_errors"] == 2


def test_aggregate_errors_worst_images_sorted():
    """aggregate_errors should sort worst_images by total error count descending."""
    image_errors = [
        {
            "image": "low_error.jpg",
            "false_negatives": {0: 1},
            "false_positives": {},
            "gt_count": 2,
            "pred_count": 1,
        },
        {
            "image": "high_error.jpg",
            "false_negatives": {0: 3},
            "false_positives": {1: 2},
            "gt_count": 5,
            "pred_count": 2,
        },
        {
            "image": "med_error.jpg",
            "false_negatives": {0: 2},
            "false_positives": {},
            "gt_count": 3,
            "pred_count": 1,
        },
    ]
    
    result = aggregate_errors(image_errors)
    
    assert len(result["worst_images"]) == 3
    assert result["worst_images"][0]["image"] == "high_error.jpg"
    assert result["worst_images"][0]["errors"] == 5
    assert result["worst_images"][1]["image"] == "med_error.jpg"
    assert result["worst_images"][1]["errors"] == 2


def test_parse_yolo_label_valid(tmp_path: Path):
    """parse_yolo_label should parse valid YOLO segmentation format."""
    label_file = tmp_path / "img.txt"
    label_file.write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
        "1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
    )
    
    objects = parse_yolo_label(label_file)
    
    assert len(objects) == 2
    assert objects[0]["class"] == 0
    assert objects[0]["polygon"] == [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]
    assert objects[1]["class"] == 1


def test_parse_yolo_label_missing_file(tmp_path: Path):
    """parse_yolo_label should return empty list for missing file."""
    objects = parse_yolo_label(tmp_path / "nonexistent.txt")
    assert objects == []


def test_parse_yolo_label_malformed_lines(tmp_path: Path):
    """parse_yolo_label should skip malformed lines."""
    label_file = tmp_path / "img.txt"
    label_file.write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
        "invalid line\n"
        "1\n"
        "1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
    )
    
    objects = parse_yolo_label(label_file)
    
    assert len(objects) == 2
    assert objects[0]["class"] == 0
    assert objects[1]["class"] == 1


def _make_test_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal test dataset with images and labels."""
    dataset_root = tmp_path / "dataset"
    images_val = dataset_root / "images" / "val"
    labels_val = dataset_root / "labels" / "val"
    images_val.mkdir(parents=True)
    labels_val.mkdir(parents=True)
    
    (images_val / "img1.jpg").write_bytes(b"fake image 1")
    (images_val / "img2.jpg").write_bytes(b"fake image 2")
    
    (labels_val / "img1.txt").write_text("0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n")
    (labels_val / "img2.txt").write_text("1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n")
    
    yaml_file = dataset_root / "dataset.yaml"
    yaml_file.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names: ['Person', 'Car']\n"
    )
    
    weights_file = tmp_path / "best.pt"
    weights_file.write_text("fake weights")
    
    return dataset_root, weights_file


class _StubResult:
    """Stub for Ultralytics Results object."""

    def __init__(self, path: str, masks_data: list[list[float]] | None, classes: list[int], confs: list[float]):
        self.path = path
        if masks_data is not None:
            self.masks = _StubMasks(masks_data)
            self.boxes = _StubBoxes(classes, confs)
        else:
            self.masks = None
            self.boxes = None


class _StubArray:
    """Minimal array stand-in (avoids numpy when train extra is not installed)."""

    def __init__(self, values: list[float]):
        self._values = values

    def flatten(self):
        return self

    def tolist(self) -> list[float]:
        return list(self._values)

    def item(self) -> float:
        return float(self._values[0])


class _StubMasks:
    """Stub for Ultralytics masks."""

    def __init__(self, xy_data: list[list[float]]):
        self.xy = [_StubArray(poly) for poly in xy_data]


class _StubBoxes:
    """Stub for Ultralytics boxes."""

    def __init__(self, classes: list[int], confs: list[float]):
        self.data = list(zip(classes, confs))

    def __iter__(self):
        for cls, conf in self.data:
            yield _StubBox(_StubArray([float(cls)]), _StubArray([float(conf)]))


class _StubBox:
    """Stub for a single box."""

    def __init__(self, cls_array: _StubArray, conf_array: _StubArray):
        self.cls = cls_array
        self.conf = conf_array


class _StubModel:
    """Stub YOLO model for testing."""

    def __init__(self, weights_path: str):
        self.weights_path = weights_path

    def predict(self, **kwargs) -> list[_StubResult]:
        """Return stub predictions."""
        source = Path(kwargs["source"])
        images = sorted(source.glob("*.jpg"))
        
        results = []
        for img in images:
            if img.stem == "img1":
                results.append(
                    _StubResult(
                        str(img),
                        [[0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]],
                        [0],
                        [0.9],
                    )
                )
            elif img.stem == "img2":
                results.append(
                    _StubResult(
                        str(img),
                        [[0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9]],
                        [1],
                        [0.85],
                    )
                )
        return results


def test_run_error_analysis_dry_run(tmp_path: Path):
    """run_error_analysis should return config only in dry_run mode."""
    dataset_root, weights_file = _make_test_dataset(tmp_path)
    
    config = ErrorConfig(
        weights=weights_file,
        dataset_yaml=dataset_root / "dataset.yaml",
        imgsz=640,
        conf=0.25,
        iou_threshold=0.5,
        project=tmp_path / "runs",
        name="test_errors",
    )
    
    report = run_error_analysis(config, dry_run=True)
    
    assert report["dry_run"] is True
    assert "summary" not in report
    assert "config" in report


def test_run_error_analysis_with_stub_model(tmp_path: Path):
    """run_error_analysis should analyze errors using stub model."""
    dataset_root, weights_file = _make_test_dataset(tmp_path)
    
    config = ErrorConfig(
        weights=weights_file,
        dataset_yaml=dataset_root / "dataset.yaml",
        imgsz=640,
        conf=0.25,
        iou_threshold=0.5,
        project=tmp_path / "runs",
        name="test_errors",
    )
    
    report = run_error_analysis(config, dry_run=False, model_factory=_StubModel)
    
    assert report["dry_run"] is False
    assert "summary" in report
    assert report["summary"]["total_images_analyzed"] == 2
    assert report["summary"]["total_false_negatives_by_class"] == {}
    assert report["summary"]["total_false_positives_by_class"] == {}


def test_run_error_analysis_saves_json(tmp_path: Path):
    """run_error_analysis should save JSON report when requested."""
    dataset_root, weights_file = _make_test_dataset(tmp_path)
    report_json = tmp_path / "error_report.json"
    
    config = ErrorConfig(
        weights=weights_file,
        dataset_yaml=dataset_root / "dataset.yaml",
        project=tmp_path / "runs",
        name="test_errors",
        report_json=report_json,
    )
    
    run_error_analysis(config, dry_run=False, model_factory=_StubModel)
    
    assert report_json.exists()
    with open(report_json, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
    assert "summary" in saved_report


def test_cli_errors_dry_run(tmp_path: Path, capsys):
    """CLI should support --dry-run mode."""
    from ml_pipeline.cli_errors import main
    
    dataset_root, weights_file = _make_test_dataset(tmp_path)
    
    import sys
    sys.argv = [
        "gdpr-yolo-analyze-errors",
        str(dataset_root),
        "--weights", str(weights_file),
        "--dry-run",
    ]
    
    code = main()
    
    assert code == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["dry_run"] is True


def test_cli_errors_missing_dataset(tmp_path: Path, capsys):
    """CLI should fail gracefully when dataset.yaml missing."""
    from ml_pipeline.cli_errors import main
    
    weights_file = tmp_path / "best.pt"
    weights_file.write_text("fake weights")
    
    import sys
    sys.argv = [
        "gdpr-yolo-analyze-errors",
        str(tmp_path / "nonexistent"),
        "--weights", str(weights_file),
    ]
    
    code = main()
    
    assert code == 1
    captured = capsys.readouterr()
    assert "dataset.yaml not found" in captured.err
