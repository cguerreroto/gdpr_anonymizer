"""Warm-start iteration driver for ml_pipeline.

Wraps :func:`run_training` and :func:`run_validation` so a single call
produces a new training run that warm-starts from a previous ``best.pt``,
runs validation against the same frozen split, and (optionally) compares
the new mask + box mAP numbers with a prior validation report. Tests
inject stub runners so the orchestration is covered without GPU,
network, or model downloads.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ml_pipeline.checkpoints import best_pt_in_run_dir
from ml_pipeline.train import TrainConfig, build_train_kwargs, run_training
from ml_pipeline.validate import (
    VALID_SPLITS,
    ValidateConfig,
    build_val_kwargs,
    run_validation,
)


VERDICT_THRESHOLD = 0.005

_VERSION_RE = re.compile(r"^(?P<base>.+)_v(?P<num>\d+)$")
_AUTO_SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<num>\d+)$")


TrainRunner = Callable[[TrainConfig], dict[str, Any]]
ValRunner = Callable[[ValidateConfig], dict[str, Any]]


@dataclass
class IterateConfig:
    """Inputs for a warm-start iteration: train + optional val + compare."""

    previous_run: Path
    dataset_yaml: Path
    project: Path | None = None
    name: str | None = None
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str | None = None
    patience: int = 50
    save_period: int = -1
    workers: int = 8
    exist_ok: bool = False
    skip_validation: bool = False
    val_name: str | None = None
    val_split: str = "val"
    compare_to: Path | None = None
    extra_train: dict[str, Any] = field(default_factory=dict)
    extra_val: dict[str, Any] = field(default_factory=dict)


def derive_next_run_name(previous_name: str) -> str:
    """Return the next run name by bumping a ``_vN`` suffix.

    A trailing Ultralytics auto-suffix ``-N`` (for example ``-3`` in
    ``yolo26n_seg_v1-3``) is stripped first, since it is not part of the
    user-chosen run name. Then:

    - ``<base>_vN`` becomes ``<base>_v<N+1>``.
    - Any other name gets ``_v2`` appended.
    """
    cleaned = previous_name.strip()
    auto = _AUTO_SUFFIX_RE.match(cleaned)
    if auto is not None:
        cleaned = auto.group("base")
    versioned = _VERSION_RE.match(cleaned)
    if versioned is not None:
        next_num = int(versioned.group("num")) + 1
        return f"{versioned.group('base')}_v{next_num}"
    return f"{cleaned}_v2"


def validate_iterate_inputs(config: IterateConfig) -> None:
    """Validate that the previous run dir, dataset yaml, and split are usable."""
    if not config.previous_run.is_dir():
        msg = f"Previous run directory not found: {config.previous_run}"
        raise FileNotFoundError(msg)
    if not config.dataset_yaml.is_file():
        msg = f"Missing dataset.yaml at {config.dataset_yaml}"
        raise FileNotFoundError(msg)
    if config.val_split not in VALID_SPLITS:
        msg = (
            f"Unknown split '{config.val_split}'. Expected one of: "
            f"{', '.join(sorted(VALID_SPLITS))}."
        )
        raise ValueError(msg)
    if config.compare_to is not None and not config.compare_to.is_file():
        msg = f"--compare-to file not found: {config.compare_to}"
        raise FileNotFoundError(msg)


def resolve_previous_weights(previous_run: Path) -> Path:
    """Locate ``previous_run/weights/best.pt`` or raise a clear error."""
    weights = best_pt_in_run_dir(previous_run)
    if weights is None:
        msg = (
            f"No weights/best.pt under {previous_run}. "
            "Pass a Ultralytics run directory whose 'weights/best.pt' exists "
            "(typically <project>/<name>)."
        )
        raise FileNotFoundError(msg)
    return weights


def _resolve_project_and_name(config: IterateConfig) -> tuple[Path, str]:
    project = config.project if config.project is not None else config.previous_run.parent
    name = config.name if config.name else derive_next_run_name(config.previous_run.name)
    return project, name


def _build_train_config(config: IterateConfig, weights: Path) -> TrainConfig:
    project, name = _resolve_project_and_name(config)
    return TrainConfig(
        dataset_yaml=config.dataset_yaml,
        weights=str(weights),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        project=project,
        name=name,
        patience=config.patience,
        save_period=config.save_period,
        workers=config.workers,
        exist_ok=config.exist_ok,
        extra=dict(config.extra_train),
    )


def _build_val_config(config: IterateConfig, weights: Path) -> ValidateConfig:
    project, name = _resolve_project_and_name(config)
    val_name = config.val_name if config.val_name else f"{name}_val"
    return ValidateConfig(
        weights=weights,
        dataset_yaml=config.dataset_yaml,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        project=project,
        name=val_name,
        split=config.val_split,
        extra=dict(config.extra_val),
    )


def _classify_delta(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    if delta > VERDICT_THRESHOLD:
        return "improved"
    if delta < -VERDICT_THRESHOLD:
        return "regressed"
    return "unchanged"


def _delta_entry(previous: Any, current: Any) -> dict[str, Any]:
    prev_value = _safe_float(previous)
    curr_value = _safe_float(current)
    if prev_value is None or curr_value is None:
        return {
            "previous": prev_value,
            "current": curr_value,
            "delta": None,
            "verdict": "unknown",
        }
    delta = curr_value - prev_value
    return {
        "previous": prev_value,
        "current": curr_value,
        "delta": delta,
        "verdict": _classify_delta(delta),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics_block(report_or_metrics: dict[str, Any]) -> dict[str, Any]:
    """Accept either a full validate report or a bare metrics dict."""
    if "metrics" in report_or_metrics and isinstance(
        report_or_metrics["metrics"], dict
    ):
        return report_or_metrics["metrics"]
    return report_or_metrics


def compare_validation_reports(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Build a delta + verdict report between two validation runs.

    Either argument may be a full validate report (with a top-level
    ``metrics`` key) or a bare metrics dict.
    """
    prev_metrics = _metrics_block(previous)
    curr_metrics = _metrics_block(current)

    overall: dict[str, Any] = {}
    for block, keys in (("mask", ("map", "map50", "map75")), ("box", ("map", "map50"))):
        prev_block = prev_metrics.get(block) or {}
        curr_block = curr_metrics.get(block) or {}
        for key in keys:
            metric_key = f"{block}_{key}"
            overall[metric_key] = _delta_entry(
                prev_block.get(key), curr_block.get(key)
            )

    per_class: dict[str, dict[str, Any]] = {}
    prev_per_class = prev_metrics.get("per_class") or {}
    curr_per_class = curr_metrics.get("per_class") or {}
    class_names = sorted(set(prev_per_class.keys()) | set(curr_per_class.keys()))
    for class_name in class_names:
        prev_entry = prev_per_class.get(class_name) or {}
        curr_entry = curr_per_class.get(class_name) or {}
        per_class[class_name] = {
            "mask_map": _delta_entry(
                prev_entry.get("mask_map"), curr_entry.get("mask_map")
            ),
            "box_map": _delta_entry(
                prev_entry.get("box_map"), curr_entry.get("box_map")
            ),
        }

    primary = overall.get("mask_map", {})
    overall_verdict = primary.get("verdict") or "unknown"
    if overall_verdict == "unknown":
        fallback = overall.get("mask_map50", {})
        overall_verdict = fallback.get("verdict") or "unknown"

    return {
        "overall": overall,
        "per_class": per_class,
        "overall_verdict": overall_verdict,
        "summary": _summarize_comparison(primary, overall_verdict),
    }


def _summarize_comparison(primary: dict[str, Any], verdict: str) -> str:
    prev = primary.get("previous")
    curr = primary.get("current")
    delta = primary.get("delta")
    if delta is None or prev is None or curr is None:
        return "Mask mAP@0.5:0.95 not available on both reports."
    sign = "+" if delta >= 0 else ""
    return (
        f"Mask mAP@0.5:0.95 {verdict}: {sign}{delta:.3f} "
        f"({prev:.3f} → {curr:.3f})."
    )


def run_iteration(
    config: IterateConfig,
    *,
    train_runner: TrainRunner | None = None,
    val_runner: ValRunner | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a warm-start iteration and return a serializable report.

    Parameters
    ----------
    config:
        Iteration configuration.
    train_runner:
        Callable that takes a :class:`TrainConfig` and returns a JSON-friendly
        report dict (typically :func:`run_training`). Tests inject a stub.
    val_runner:
        Callable that takes a :class:`ValidateConfig` and returns a JSON-friendly
        report dict (typically :func:`run_validation`). Tests inject a stub.
    dry_run:
        When true, build the train (and val) kwargs and skip both runners.
    """
    validate_iterate_inputs(config)
    weights = resolve_previous_weights(config.previous_run)
    project, next_name = _resolve_project_and_name(config)
    train_config = _build_train_config(config, weights)

    report: dict[str, Any] = {
        "config": _config_to_jsonable(config),
        "previous_run": str(config.previous_run.resolve()),
        "previous_weights": str(weights.resolve()),
        "next_run_name": next_name,
        "next_project": str(project.resolve()),
        "dry_run": dry_run,
    }

    if dry_run:
        report["train_kwargs"] = build_train_kwargs(train_config)
        if not config.skip_validation:
            placeholder_weights = (
                project / next_name / "weights" / "best.pt"
            )
            preview_val = _build_val_config(config, placeholder_weights)
            report["val_kwargs"] = build_val_kwargs(preview_val)
            report["val_kwargs"]["weights_placeholder"] = str(placeholder_weights)
        if config.compare_to is not None:
            report["compare_to"] = str(config.compare_to.resolve())
        return report

    runner = train_runner if train_runner is not None else run_training
    train_report = runner(train_config)
    report["train_report"] = train_report

    if config.skip_validation:
        return report

    new_weights_str = train_report.get("validate_weights")
    if not new_weights_str:
        report["validation_skipped_reason"] = (
            "Train report did not include 'validate_weights'; cannot validate."
        )
        return report

    val_config = _build_val_config(config, Path(new_weights_str))
    val_runner_fn = val_runner if val_runner is not None else run_validation
    val_report = val_runner_fn(val_config)
    report["validate_report"] = val_report

    if config.compare_to is not None:
        prev_data = json.loads(config.compare_to.read_text(encoding="utf-8"))
        report["comparison"] = compare_validation_reports(prev_data, val_report)

    return report


def _config_to_jsonable(config: IterateConfig) -> dict[str, Any]:
    data = asdict(config)
    data["previous_run"] = str(config.previous_run)
    data["dataset_yaml"] = str(config.dataset_yaml)
    if config.project is not None:
        data["project"] = str(config.project)
    if config.compare_to is not None:
        data["compare_to"] = str(config.compare_to)
    return data
