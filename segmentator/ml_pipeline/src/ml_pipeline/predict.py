"""YOLO26-seg batch predict driver for ml_pipeline.

Wraps Ultralytics ``YOLO.predict(...)`` for the ``segment`` task on a folder
or single image. Saves Ultralytics overlay images for human review and can
optionally export masks as YOLO polygon ``.txt`` files. Ultralytics is
imported lazily by the default factory so unit tests can inject a stub.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from ml_pipeline.checkpoints import missing_weights_message
from ml_pipeline.ultralytics_extra import raise_ultralytics_missing


class _PredictModelLike(Protocol):
    def predict(self, **kwargs: Any) -> Any: ...


PredictModelFactory = Callable[[str], _PredictModelLike]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class PredictConfig:
    """Inputs for a YOLO26 segmentation predict run on still images."""

    weights: Path
    source: Path
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str | None = None
    project: Path = Path("runs")
    name: str = "yolo26n_seg_predict"
    exist_ok: bool = False
    save_overlays: bool = True
    save_polygons: bool = False
    classes: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _default_predict_model_factory(weights_ref: str) -> _PredictModelLike:
    """Import ultralytics only when needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise_ultralytics_missing(exc)
    return YOLO(weights_ref)


def list_source_images(source: Path) -> list[Path]:
    """Return a sorted list of image files for ``source``.

    Accepts either a single image file or a directory. Recurses into
    subdirectories so layouts like ``images/val/`` work directly.
    """
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            msg = (
                f"Source file is not a supported image: {source} "
                f"(expected one of {sorted(IMAGE_EXTENSIONS)})."
            )
            raise ValueError(msg)
        return [source]
    if source.is_dir():
        files = [
            p
            for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        files.sort()
        return files
    msg = f"Source path does not exist: {source}"
    raise FileNotFoundError(msg)


def validate_predict_inputs(config: PredictConfig) -> None:
    """Validate that weights and source exist before invoking Ultralytics."""
    if not config.weights.is_file():
        raise FileNotFoundError(
            missing_weights_message(config.weights, config.project)
        )
    if not config.source.exists():
        msg = f"Predict source not found: {config.source}"
        raise FileNotFoundError(msg)
    if config.imgsz <= 0:
        raise ValueError("--imgsz must be a positive integer.")
    if not 0.0 <= config.conf <= 1.0:
        raise ValueError("--conf must be in [0.0, 1.0].")
    if not 0.0 <= config.iou <= 1.0:
        raise ValueError("--iou must be in [0.0, 1.0].")


def build_predict_kwargs(config: PredictConfig) -> dict[str, Any]:
    """Build keyword arguments passed to YOLO.predict()."""
    kwargs: dict[str, Any] = {
        "source": str(config.source),
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "project": str(config.project),
        "name": config.name,
        "exist_ok": config.exist_ok,
        "save": config.save_overlays,
        "save_txt": False,
        "stream": False,
        "task": "segment",
    }
    if config.device is not None:
        kwargs["device"] = config.device
    if config.classes:
        kwargs["classes"] = list(config.classes)
    kwargs.update(config.extra)
    return kwargs


def _coords_iter(xyn: Any) -> Iterable[list[float]]:
    """Yield each polygon as a flat list of normalized floats.

    Ultralytics exposes ``masks.xyn`` as a list of ``(N, 2)`` numpy arrays.
    Tests pass a stub object with the same shape.
    """
    for arr in xyn:
        flatten = getattr(arr, "flatten", None)
        if flatten is not None:
            arr = flatten()
        tolist = getattr(arr, "tolist", None)
        coords = list(tolist()) if tolist is not None else list(arr)
        yield [float(v) for v in coords]


def _classes_for_result(result: Any) -> list[int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    cls = getattr(boxes, "cls", None)
    if cls is None:
        return []
    tolist = getattr(cls, "tolist", None)
    raw = tolist() if tolist is not None else list(cls)
    return [int(v) for v in raw]


def write_polygons_for_result(result: Any, out_dir: Path) -> Path | None:
    """Write predicted masks as YOLO polygon ``.txt`` for one result.

    The filename mirrors the source image stem. Returns the output path on
    success, or ``None`` when the result has no masks.
    """
    masks = getattr(result, "masks", None)
    xyn = getattr(masks, "xyn", None) if masks is not None else None
    if xyn is None:
        return None
    classes = _classes_for_result(result)
    image_path = Path(getattr(result, "path", "")) if getattr(result, "path", "") else None
    stem = image_path.stem if image_path else "prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.txt"
    polygons = list(_coords_iter(xyn))
    lines: list[str] = []
    for idx, polygon in enumerate(polygons):
        if not polygon:
            continue
        class_id = classes[idx] if idx < len(classes) else 0
        coords = " ".join(f"{v:.6f}" for v in polygon)
        lines.append(f"{class_id} {coords}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


def summarize_results(results: Iterable[Any]) -> dict[str, Any]:
    """Build a JSON-friendly summary of predicted detections per image."""
    images: list[dict[str, Any]] = []
    total_detections = 0
    detections_by_class: dict[int, int] = {}

    for result in results:
        path = getattr(result, "path", None)
        classes = _classes_for_result(result)
        masks = getattr(result, "masks", None)
        has_masks = masks is not None and getattr(masks, "xyn", None) is not None
        per_image_counts: dict[int, int] = {}
        for cls in classes:
            per_image_counts[cls] = per_image_counts.get(cls, 0) + 1
            detections_by_class[cls] = detections_by_class.get(cls, 0) + 1
        total_detections += len(classes)
        images.append(
            {
                "image": str(path) if path is not None else "",
                "detections": len(classes),
                "has_masks": bool(has_masks),
                "by_class": {str(k): v for k, v in sorted(per_image_counts.items())},
            }
        )

    return {
        "image_count": len(images),
        "images_with_detections": sum(1 for entry in images if entry["detections"] > 0),
        "total_detections": total_detections,
        "detections_by_class": {str(k): v for k, v in sorted(detections_by_class.items())},
        "images": images,
    }


def run_predict(
    config: PredictConfig,
    *,
    model_factory: PredictModelFactory | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute prediction and return a serializable report.

    Parameters
    ----------
    config:
        Predict configuration.
    model_factory:
        Callable returning an object with ``.predict(**kwargs)``. Defaults
        to the lazy Ultralytics loader. Tests pass a stub.
    dry_run:
        When true, only build the kwargs and skip the call to ``predict``.
    """
    validate_predict_inputs(config)
    kwargs = build_predict_kwargs(config)
    weights_ref = str(config.weights)

    report: dict[str, Any] = {
        "config": _config_to_jsonable(config),
        "weights": weights_ref,
        "predict_kwargs": kwargs,
        "dry_run": dry_run,
    }

    if dry_run:
        return report

    factory = model_factory or _default_predict_model_factory
    model = factory(weights_ref)
    raw_results = model.predict(**kwargs)
    results = list(raw_results) if not isinstance(raw_results, list) else raw_results

    output_dir = config.project / config.name
    save_dir = None
    for result in results:
        save_dir = getattr(result, "save_dir", None)
        if save_dir is not None:
            break
    report["save_dir"] = str(save_dir) if save_dir is not None else str(output_dir)

    if config.save_polygons:
        polygons_dir = Path(report["save_dir"]) / "polygons"
        written: list[str] = []
        for result in results:
            out_path = write_polygons_for_result(result, polygons_dir)
            if out_path is not None:
                written.append(str(out_path))
        report["polygons_dir"] = str(polygons_dir)
        report["polygons_written"] = written

    report["summary"] = summarize_results(results)
    return report


def _config_to_jsonable(config: PredictConfig) -> dict[str, Any]:
    data = asdict(config)
    data["weights"] = str(config.weights)
    data["source"] = str(config.source)
    data["project"] = str(config.project)
    return data
