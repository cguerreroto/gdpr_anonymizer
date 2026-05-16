"""
Error analysis for YOLO segmentation validation split.

Compares ground truth labels with predictions to identify false negatives
(missed detections) and false positives (incorrect detections), providing
actionable guidance for improving training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_pipeline.ultralytics_extra import raise_ultralytics_missing


@dataclass
class ErrorConfig:
    """Configuration for error analysis on validation split."""

    weights: Path
    dataset_yaml: Path
    imgsz: int = 640
    conf: float = 0.25
    iou_threshold: float = 0.5
    device: str | None = None
    project: Path | None = None
    name: str = "error_analysis"
    report_json: Path | None = None
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.weights = Path(self.weights)
        self.dataset_yaml = Path(self.dataset_yaml)
        if self.project is not None:
            self.project = Path(self.project)
        if self.report_json is not None:
            self.report_json = Path(self.report_json)
        if self.extra is None:
            self.extra = {}


def _default_error_model_factory(weights: Path):
    """Load YOLO model for error analysis; raises if ultralytics unavailable."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise_ultralytics_missing(exc)
    return YOLO(str(weights))


def validate_error_inputs(config: ErrorConfig) -> None:
    """Validate that required inputs exist."""
    if not config.weights.exists():
        raise FileNotFoundError(f"Missing weights at {config.weights}")
    if not config.dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset.yaml at {config.dataset_yaml}")


def build_predict_kwargs(config: ErrorConfig) -> dict[str, Any]:
    """Build kwargs for YOLO predict() call during error analysis."""
    dataset_root = config.dataset_yaml.parent
    kwargs: dict[str, Any] = {
        "source": dataset_root / "images" / "val",
        "imgsz": config.imgsz,
        "conf": config.conf,
        "save": False,
        "stream": False,
    }
    if config.device is not None:
        kwargs["device"] = config.device
    if config.project is not None:
        kwargs["project"] = str(config.project)
    kwargs["name"] = config.name
    kwargs.update(config.extra)
    return kwargs


def compute_mask_iou(pred_coords: list[float], gt_coords: list[float]) -> float:
    """
    Compute approximate IoU between two polygon masks.

    Both inputs are flattened lists of normalized [x1, y1, x2, y2, ...].
    Returns a simple bounding-box IoU as a conservative proxy for polygon IoU.
    """
    if not pred_coords or not gt_coords:
        return 0.0

    def bbox_from_poly(coords: list[float]) -> tuple[float, float, float, float]:
        xs = coords[0::2]
        ys = coords[1::2]
        return (min(xs), min(ys), max(xs), max(ys))

    px1, py1, px2, py2 = bbox_from_poly(pred_coords)
    gx1, gy1, gx2, gy2 = bbox_from_poly(gt_coords)

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    if union_area == 0.0:
        return 0.0
    return inter_area / union_area


def analyze_image_errors(
    image_path: Path,
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    """
    Analyze errors for a single image.

    Args:
        image_path: Path to the image file (for reporting).
        predictions: List of dicts with keys 'class', 'confidence', 'polygon'.
        ground_truth: List of dicts with keys 'class', 'polygon'.
        iou_threshold: Minimum IoU to consider a prediction as matching GT.

    Returns:
        Dict with 'false_negatives' and 'false_positives' counts per class.
    """
    matched_gt = set()
    matched_pred = set()

    for pred_idx, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if pred["class"] != gt["class"]:
                continue
            iou = compute_mask_iou(pred["polygon"], gt["polygon"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold:
            matched_pred.add(pred_idx)
            matched_gt.add(best_gt_idx)

    fn_by_class: dict[int, int] = {}
    fp_by_class: dict[int, int] = {}

    for gt_idx, gt in enumerate(ground_truth):
        if gt_idx not in matched_gt:
            cls = gt["class"]
            fn_by_class[cls] = fn_by_class.get(cls, 0) + 1

    for pred_idx, pred in enumerate(predictions):
        if pred_idx not in matched_pred:
            cls = pred["class"]
            fp_by_class[cls] = fp_by_class.get(cls, 0) + 1

    return {
        "image": str(image_path.name),
        "false_negatives": fn_by_class,
        "false_positives": fp_by_class,
        "gt_count": len(ground_truth),
        "pred_count": len(predictions),
    }


def aggregate_errors(image_errors: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-image errors into summary statistics.

    Returns:
        Dict with total FN/FP counts per class and lists of worst images.
    """
    total_fn: dict[int, int] = {}
    total_fp: dict[int, int] = {}
    images_with_errors: list[dict[str, Any]] = []

    for img_err in image_errors:
        for cls, count in img_err["false_negatives"].items():
            total_fn[cls] = total_fn.get(cls, 0) + count
        for cls, count in img_err["false_positives"].items():
            total_fp[cls] = total_fp.get(cls, 0) + count

        total_errors = sum(img_err["false_negatives"].values()) + sum(
            img_err["false_positives"].values()
        )
        if total_errors > 0:
            images_with_errors.append(
                {
                    "image": img_err["image"],
                    "errors": total_errors,
                    "fn": sum(img_err["false_negatives"].values()),
                    "fp": sum(img_err["false_positives"].values()),
                }
            )

    images_with_errors.sort(key=lambda x: x["errors"], reverse=True)

    return {
        "total_false_negatives_by_class": total_fn,
        "total_false_positives_by_class": total_fp,
        "total_images_analyzed": len(image_errors),
        "images_with_errors": len(images_with_errors),
        "worst_images": images_with_errors[:20],
    }


def parse_yolo_label(label_path: Path, img_width: int = 640, img_height: int = 640) -> list[dict[str, Any]]:
    """
    Parse YOLO segmentation label file.

    Returns list of dicts with keys 'class' and 'polygon' (normalized coords).
    """
    if not label_path.exists():
        return []
    objects = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            objects.append({"class": cls, "polygon": coords})
    return objects


def run_error_analysis(
    config: ErrorConfig,
    *,
    dry_run: bool = False,
    model_factory=_default_error_model_factory,
) -> dict[str, Any]:
    """
    Run error analysis on the validation split.

    Args:
        config: ErrorConfig with paths and thresholds.
        dry_run: If True, skip actual prediction and return config only.
        model_factory: Callable to load YOLO model (for testing).

    Returns:
        Dict with config, summary statistics, and worst images.
    """
    from ml_pipeline.train import materialize_resolved_dataset_yaml

    validate_error_inputs(config)

    save_dir = (config.project or Path.cwd()) / config.name
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_yaml = save_dir / "dataset.resolved.yaml"
    materialize_resolved_dataset_yaml(config.dataset_yaml, save_dir)

    predict_kwargs = build_predict_kwargs(config)

    report: dict[str, Any] = {
        "config": {
            "weights": str(config.weights),
            "dataset_yaml": str(config.dataset_yaml),
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou_threshold": config.iou_threshold,
            "device": config.device,
            "project": str(config.project) if config.project else None,
            "name": config.name,
            "extra": config.extra,
        },
        "predict_kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in predict_kwargs.items()},
        "dry_run": dry_run,
        "save_dir": str(save_dir),
    }

    if dry_run:
        return report

    model = model_factory(config.weights)
    results = model.predict(**predict_kwargs)

    dataset_root = config.dataset_yaml.parent
    val_labels_dir = dataset_root / "labels" / "val"

    image_errors = []

    for result in results:
        img_path = Path(result.path)
        label_path = val_labels_dir / f"{img_path.stem}.txt"
        ground_truth = parse_yolo_label(label_path)

        predictions = []
        if result.masks is not None and result.boxes is not None:
            for mask_data, box_data in zip(result.masks.xy, result.boxes):
                cls = int(box_data.cls.item())
                conf = float(box_data.conf.item())
                polygon = mask_data.flatten().tolist()
                predictions.append({"class": cls, "confidence": conf, "polygon": polygon})

        img_err = analyze_image_errors(img_path, predictions, ground_truth, config.iou_threshold)
        image_errors.append(img_err)

    summary = aggregate_errors(image_errors)
    report["summary"] = summary

    if config.report_json:
        config.report_json.parent.mkdir(parents=True, exist_ok=True)
        with open(config.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
