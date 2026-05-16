"""Interpret validation metrics against documented decision thresholds."""

from __future__ import annotations

from typing import Any

MASK_MAP_LOW = 0.30
MASK_MAP_HIGH = 0.50

_BAND_SUMMARIES = {
    "below_threshold": (
        "Mask mAP@0.5:0.95 is below 0.30: the model is likely undertrained or "
        "labels are noisy. Collect more annotated frames, especially for the "
        "weakest class, before relying on mask-based blurring."
    ),
    "usable": (
        "Mask mAP@0.5:0.95 is between 0.30 and 0.50: segmentation may support "
        "downstream blurring, but per-class gaps should be reviewed and more "
        "data added where needed."
    ),
    "comfortable": (
        "Mask mAP@0.5:0.95 is above 0.50: comfortable margin for instance "
        "segmentation on this kind of data."
    ),
    "unknown": (
        "Mask mAP@0.5:0.95 was not available. Re-run validation with an "
        "Ultralytics build that reports segmentation metrics."
    ),
}


def classify_mask_map(mask_map: float | None) -> str:
    """Map overall mask mAP@0.5:0.95 to a decision band."""
    if mask_map is None:
        return "unknown"
    if mask_map < MASK_MAP_LOW:
        return "below_threshold"
    if mask_map < MASK_MAP_HIGH:
        return "usable"
    return "comfortable"


def assess_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Turn raw ``metrics`` into an assessment and actionable recommendations.

    Thresholds follow the rules of thumb documented in ``validate.py``:
    below 0.30, between 0.30 and 0.50, and above 0.50 for mask mAP@0.5:0.95.
    """
    mask = metrics.get("mask") or {}
    mask_map = mask.get("map")
    mask_map50 = mask.get("map50")
    band = classify_mask_map(
        float(mask_map) if mask_map is not None else None
    )

    recommendations: list[str] = []

    if band == "below_threshold":
        recommendations.append(
            "Overall mask mAP@0.5:0.95 is below 0.30: prioritize more "
            "labeled images and/or longer training before video blur."
        )
        recommendations.append(
            "Warm-start training with gdpr-yolo-train --weights "
            "<path-to-best.pt> and increase --epochs."
        )
    elif band == "usable":
        recommendations.append(
            "Overall mask mAP@0.5:0.95 is in the usable range: review "
            "per-class scores and add labels for weak classes before production blur."
        )
    elif band == "comfortable":
        recommendations.append(
            "Overall mask mAP@0.5:0.95 is strong: proceed to predict/video "
            "steps, but still inspect failure cases on a sample of frames."
        )
    else:
        recommendations.append(
            "Could not read mask mAP@0.5:0.95; run gdpr-yolo-validate without "
            "--dry-run to produce metrics."
        )

    per_class = metrics.get("per_class") or {}
    ranked = _rank_classes_by_mask_map(per_class)
    if ranked:
        worst_name, worst_map = ranked[0]
        if worst_map is not None and worst_map < MASK_MAP_LOW:
            recommendations.append(
                f"Weakest class '{worst_name}' (mask mAP@0.5:0.95={worst_map:.3f}): "
                "add more polygons for this class in the segmentator and augmentor."
            )
        if len(ranked) > 1:
            best_name, best_map = ranked[-1]
            if (
                worst_map is not None
                and best_map is not None
                and best_map - worst_map >= 0.10
            ):
                recommendations.append(
                    f"Largest gap: '{worst_name}' ({worst_map:.3f}) vs "
                    f"'{best_name}' ({best_map:.3f}); focus labeling on the weaker class."
                )

    if mask_map50 is not None and mask_map is not None:
        if float(mask_map50) >= MASK_MAP_LOW and float(mask_map) < MASK_MAP_LOW:
            recommendations.append(
                "Mask mAP@0.5 is noticeably higher than mAP@0.5:0.95: masks are "
                "loosely aligned but boundary quality may be poor for tight blur regions."
            )

    return {
        "assessment": {
            "overall_mask_map": mask_map,
            "overall_mask_map50": mask_map50,
            "band": band,
            "summary": _BAND_SUMMARIES[band],
            "ready_for_video_blur": band in ("usable", "comfortable"),
        },
        "recommendations": recommendations,
    }


def interpret_validation_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return interpretation for a full validation report or a metrics-only dict."""
    metrics = report.get("metrics")
    if metrics is None:
        metrics = report
    return assess_metrics(metrics)


def _rank_classes_by_mask_map(
    per_class: dict[str, Any],
) -> list[tuple[str, float | None]]:
    items: list[tuple[str, float | None]] = []
    for name, values in per_class.items():
        raw = values.get("mask_map") if isinstance(values, dict) else None
        try:
            parsed = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            parsed = None
        items.append((name, parsed))
    return sorted(
        items,
        key=lambda pair: pair[1] if pair[1] is not None else float("inf"),
    )
