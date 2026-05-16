"""YOLO26-seg validation driver for ml_pipeline.

Wraps Ultralytics ``YOLO.val(...)`` for the ``segment`` task and exposes a
small JSON-friendly summary of mask and box mAP, plus a per-class
breakdown when the installed Ultralytics version provides it. Ultralytics
is imported lazily by the default factory so unit tests can inject a stub.

Decision thresholds (rules of thumb, not enforced):

- mask mAP@0.5:0.95 < 0.30: likely undertrained or noisy labels; collect
  more annotated frames, especially for the worst-performing class.
- mask mAP@0.5:0.95 in [0.30, 0.50]: usable for downstream blurring; review
  per-class deltas to decide where to add data.
- mask mAP@0.5:0.95 > 0.50: comfortable margin for instance segmentation
  on this kind of data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ml_pipeline.train import materialize_resolved_dataset_yaml


class _ValModelLike(Protocol):
    def val(self, **kwargs: Any) -> Any: ...


ValModelFactory = Callable[[str], _ValModelLike]

VALID_SPLITS = frozenset({"val", "test", "train"})


@dataclass
class ValidateConfig:
    weights: Path
    dataset_yaml: Path
    imgsz: int = 640
    batch: int = 16
    device: str | None = None
    project: Path = Path("runs")
    name: str = "yolo26n_seg_val"
    split: str = "val"
    conf: float | None = None
    iou: float | None = None
    save_json: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _default_val_model_factory(weights_ref: str) -> _ValModelLike:
    """Import ultralytics only when needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        msg = (
            "ultralytics is not installed. Install the optional [train] "
            "extra (for example: uv sync --extra train) to run validation."
        )
        raise RuntimeError(msg) from exc
    return YOLO(weights_ref)


def validate_inputs(config: ValidateConfig) -> None:
    if not config.dataset_yaml.is_file():
        msg = f"Missing dataset.yaml at {config.dataset_yaml}"
        raise FileNotFoundError(msg)
    if not config.weights.is_file():
        msg = f"Missing weights at {config.weights}"
        raise FileNotFoundError(msg)
    if config.split not in VALID_SPLITS:
        msg = (
            f"Unknown split '{config.split}'. Expected one of: "
            f"{', '.join(sorted(VALID_SPLITS))}."
        )
        raise ValueError(msg)


def build_val_kwargs(config: ValidateConfig) -> dict[str, Any]:
    """Build keyword arguments passed to YOLO.val()."""
    kwargs: dict[str, Any] = {
        "data": str(config.dataset_yaml),
        "imgsz": config.imgsz,
        "batch": config.batch,
        "project": str(config.project),
        "name": config.name,
        "split": config.split,
        "save_json": config.save_json,
        "task": "segment",
    }
    if config.device is not None:
        kwargs["device"] = config.device
    if config.conf is not None:
        kwargs["conf"] = config.conf
    if config.iou is not None:
        kwargs["iou"] = config.iou
    kwargs.update(config.extra)
    return kwargs


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _name_for(names: Any, idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(idx, idx))
    if isinstance(names, (list, tuple)) and idx < len(names):
        return str(names[idx])
    return str(idx)


def extract_metrics(results: Any) -> dict[str, Any]:
    """Pull mask and box mAP, plus per-class numbers, from Ultralytics metrics."""
    out: dict[str, Any] = {"mask": {}, "box": {}, "per_class": {}}

    seg = getattr(results, "seg", None)
    if seg is not None:
        out["mask"] = {
            "map": _safe_float(getattr(seg, "map", None)),
            "map50": _safe_float(getattr(seg, "map50", None)),
            "map75": _safe_float(getattr(seg, "map75", None)),
        }

    box = getattr(results, "box", None)
    if box is not None:
        out["box"] = {
            "map": _safe_float(getattr(box, "map", None)),
            "map50": _safe_float(getattr(box, "map50", None)),
        }

    names = getattr(results, "names", None) or {}

    if seg is not None:
        per_class_seg = getattr(seg, "maps", None)
        if per_class_seg is not None:
            try:
                values = list(per_class_seg)
            except TypeError:
                values = []
            for idx, ap in enumerate(values):
                bucket = out["per_class"].setdefault(_name_for(names, idx), {})
                bucket["mask_map"] = _safe_float(ap)

    if box is not None:
        per_class_box = getattr(box, "maps", None)
        if per_class_box is not None:
            try:
                values = list(per_class_box)
            except TypeError:
                values = []
            for idx, ap in enumerate(values):
                bucket = out["per_class"].setdefault(_name_for(names, idx), {})
                bucket["box_map"] = _safe_float(ap)

    return out


def run_validation(
    config: ValidateConfig,
    *,
    model_factory: ValModelFactory | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute validation and return a serializable report.

    Parameters
    ----------
    config:
        Validation configuration.
    model_factory:
        Callable returning an object with ``.val(**kwargs)``. Defaults to
        the lazy Ultralytics loader. Tests pass a stub.
    dry_run:
        When true, only build the kwargs and skip the call to ``val``.
    """
    validate_inputs(config)
    kwargs = build_val_kwargs(config)
    weights_ref = str(config.weights)

    report: dict[str, Any] = {
        "config": _config_to_jsonable(config),
        "weights": weights_ref,
        "val_kwargs": kwargs,
        "dry_run": dry_run,
    }

    if dry_run:
        return report

    output_dir = config.project / config.name
    resolved_yaml = materialize_resolved_dataset_yaml(
        config.dataset_yaml, output_dir
    )
    kwargs["data"] = str(resolved_yaml)
    report["resolved_dataset_yaml"] = str(resolved_yaml)
    report["val_kwargs"] = kwargs

    factory = model_factory or _default_val_model_factory
    model = factory(weights_ref)
    results = model.val(**kwargs)

    save_dir = getattr(results, "save_dir", None)
    report["save_dir"] = (
        str(save_dir) if save_dir is not None else str(output_dir)
    )
    report["metrics"] = extract_metrics(results)
    return report


def _config_to_jsonable(config: ValidateConfig) -> dict[str, Any]:
    data = asdict(config)
    data["weights"] = str(config.weights)
    data["dataset_yaml"] = str(config.dataset_yaml)
    data["project"] = str(config.project)
    return data
