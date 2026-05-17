"""YOLO26-seg video blur pipeline for ml_pipeline.

Reads a video frame by frame, runs YOLO26 segmentation predictions, and
writes a copy of the video with the union of predicted masks blurred or
pixelated. The default backend uses OpenCV for I/O and NumPy for mask
arithmetic; both are pulled in by the ``[train]`` extra (via Ultralytics).
Tests inject a stub pipeline runner so the orchestration is covered
without OpenCV, NumPy, or a real video file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from ml_pipeline.checkpoints import missing_weights_message
from ml_pipeline.ultralytics_extra import raise_ultralytics_missing


VALID_BLUR_METHODS = frozenset({"gaussian", "pixelate"})

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}


def blurred_video_filename(source: Path) -> str:
    """Return ``{stem}_blurred{suffix}`` for the given source video path."""
    return f"{source.stem}_blurred{source.suffix}"


def resolve_blurred_output_path(source: Path, output: Path | None = None) -> Path:
    """Resolve the output video path from the source and an optional override.

    When *output* is omitted, the blurred file is written next to *source* with
    the same extension and ``_blurred`` appended to the stem. When *output* is
    an existing directory, or has no recognized video extension, it is treated
    as a directory and the auto-named file is placed inside it. Otherwise
    *output* is used as the explicit file path.
    """
    name = blurred_video_filename(source)
    if output is None:
        return source.parent / name
    if output.is_dir():
        return output / name
    if output.suffix.lower() not in VIDEO_EXTENSIONS:
        return output / name
    return output


@dataclass
class FrameStats:
    """Per-frame counters used to aggregate video-level KPIs."""

    detections: int
    detection_classes: list[int] = field(default_factory=list)
    mask_pixels: int = 0
    frame_pixels: int = 0


@dataclass
class VideoBlurConfig:
    """Inputs for a YOLO26 segmentation video blur run."""

    weights: Path
    source: Path
    output: Path
    classes: list[int] | None = None
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str | None = None
    blur_method: str = "gaussian"
    blur_kernel: int = 51
    blur_sigma: float = 0.0
    pixelate_block: int = 16
    mask_dilate: int = 0
    fps_override: float | None = None
    fourcc: str = "mp4v"
    extra: dict[str, Any] = field(default_factory=dict)


class _VideoModelLike(Protocol):
    def predict(self, source: Any, **kwargs: Any) -> Any: ...


VideoModelFactory = Callable[[str], _VideoModelLike]
VideoPipelineRunner = Callable[
    [_VideoModelLike, "VideoBlurConfig", dict[str, Any]],
    list[FrameStats],
]


def _default_video_model_factory(weights_ref: str) -> _VideoModelLike:
    """Import ultralytics only when needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise_ultralytics_missing(exc)
    return YOLO(weights_ref)


def validate_video_inputs(config: VideoBlurConfig, *, dry_run: bool = False) -> None:
    """Validate that weights, source, and config knobs are usable."""
    if not config.weights.is_file():
        raise FileNotFoundError(
            missing_weights_message(config.weights, config.weights.parent)
        )
    if not dry_run and not config.source.is_file():
        msg = f"Video source not found: {config.source}"
        raise FileNotFoundError(msg)
    if config.source.suffix.lower() not in VIDEO_EXTENSIONS:
        msg = (
            f"Source does not look like a video file: {config.source} "
            f"(expected one of {sorted(VIDEO_EXTENSIONS)})."
        )
        raise ValueError(msg)
    if (
        not dry_run
        and config.source.is_file()
        and config.source.resolve() == config.output.resolve()
    ):
        raise ValueError(
            "Source and output paths must differ; refusing to overwrite the input."
        )
    if config.blur_method.strip().lower() not in VALID_BLUR_METHODS:
        msg = (
            f"Unsupported blur method '{config.blur_method}'. Expected one of: "
            f"{', '.join(sorted(VALID_BLUR_METHODS))}."
        )
        raise ValueError(msg)
    if config.blur_kernel <= 0:
        raise ValueError("--blur-kernel must be a positive integer.")
    if config.blur_kernel % 2 == 0:
        raise ValueError("--blur-kernel must be odd (Gaussian blur requirement).")
    if config.pixelate_block <= 0:
        raise ValueError("--pixelate-block must be a positive integer.")
    if config.mask_dilate < 0:
        raise ValueError("--mask-dilate must be zero or positive.")
    if config.imgsz <= 0:
        raise ValueError("--imgsz must be a positive integer.")
    if not 0.0 <= config.conf <= 1.0:
        raise ValueError("--conf must be in [0.0, 1.0].")
    if not 0.0 <= config.iou <= 1.0:
        raise ValueError("--iou must be in [0.0, 1.0].")


def build_predict_kwargs(config: VideoBlurConfig) -> dict[str, Any]:
    """Build per-frame keyword arguments passed to YOLO.predict()."""
    kwargs: dict[str, Any] = {
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "task": "segment",
        "verbose": False,
        "save": False,
        "stream": False,
    }
    if config.device is not None:
        kwargs["device"] = config.device
    if config.classes:
        kwargs["classes"] = list(config.classes)
    kwargs.update(config.extra)
    return kwargs


def aggregate_video_metrics(frame_stats: Iterable[FrameStats]) -> dict[str, Any]:
    """Compute video-level KPIs from per-frame stats."""
    stats = list(frame_stats)
    total_frames = len(stats)
    if total_frames == 0:
        return {
            "total_frames": 0,
            "frames_with_detections": 0,
            "frames_with_detections_pct": 0.0,
            "total_detections": 0,
            "detections_by_class": {},
            "average_mask_area_fraction": 0.0,
        }

    frames_with_det = sum(1 for s in stats if s.detections > 0)
    total_det = sum(s.detections for s in stats)
    by_class: dict[int, int] = {}
    for s in stats:
        for cls in s.detection_classes:
            by_class[cls] = by_class.get(cls, 0) + 1

    area_samples = [
        s.mask_pixels / s.frame_pixels for s in stats if s.frame_pixels > 0
    ]
    avg_area = sum(area_samples) / total_frames if area_samples else 0.0

    return {
        "total_frames": total_frames,
        "frames_with_detections": frames_with_det,
        "frames_with_detections_pct": (
            frames_with_det / total_frames if total_frames else 0.0
        ),
        "total_detections": total_det,
        "detections_by_class": {str(k): v for k, v in sorted(by_class.items())},
        "average_mask_area_fraction": avg_area,
    }


def run_video_blur(
    config: VideoBlurConfig,
    *,
    model_factory: VideoModelFactory | None = None,
    pipeline_runner: VideoPipelineRunner | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the video blur pipeline and return a serializable report.

    Parameters
    ----------
    config:
        Video blur configuration.
    model_factory:
        Callable returning an object with ``.predict(source, **kwargs)``.
        Defaults to the lazy Ultralytics loader. Tests pass a stub.
    pipeline_runner:
        Callable that consumes a model, a config, and predict kwargs, and
        returns a list of :class:`FrameStats`. Defaults to an OpenCV-based
        runner. Tests pass a stub to avoid OpenCV.
    dry_run:
        When true, skip the predict loop and return the resolved kwargs.
    """
    validate_video_inputs(config, dry_run=dry_run)
    predict_kwargs = build_predict_kwargs(config)
    weights_ref = str(config.weights)

    report: dict[str, Any] = {
        "config": _config_to_jsonable(config),
        "weights": weights_ref,
        "predict_kwargs": predict_kwargs,
        "dry_run": dry_run,
    }

    if dry_run:
        return report

    factory = model_factory or _default_video_model_factory
    model = factory(weights_ref)

    runner = pipeline_runner or _default_video_pipeline_runner
    frame_stats = list(runner(model, config, predict_kwargs))

    report["output"] = str(config.output)
    report["output_exists"] = config.output.exists()
    report["summary"] = aggregate_video_metrics(frame_stats)
    return report


def _result_classes(result: Any) -> list[int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    cls = getattr(boxes, "cls", None)
    if cls is None:
        return []
    tolist = getattr(cls, "tolist", None)
    raw = tolist() if tolist is not None else list(cls)
    return [int(v) for v in raw]


def _apply_blur(frame: Any, config: VideoBlurConfig) -> Any:
    """Return a blurred copy of ``frame`` according to the config."""
    import cv2

    method = config.blur_method.strip().lower()
    if method == "gaussian":
        kernel = config.blur_kernel
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(frame, (kernel, kernel), config.blur_sigma)
    if method == "pixelate":
        height, width = frame.shape[:2]
        block = max(1, config.pixelate_block)
        small = cv2.resize(
            frame,
            (max(1, width // block), max(1, height // block)),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    msg = f"Unsupported blur method: {config.blur_method}"
    raise ValueError(msg)


def _union_mask(
    result: Any,
    keep_indices: list[int],
    height: int,
    width: int,
    dilate_px: int,
) -> Any:
    """Combine selected per-instance masks into a single binary mask."""
    import cv2
    import numpy as np

    out = np.zeros((height, width), dtype=np.uint8)
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None:
        return out

    for idx in keep_indices:
        if idx >= len(data):
            continue
        mask = data[idx]
        cpu = getattr(mask, "cpu", None)
        if callable(cpu):
            mask = cpu()
        numpy_fn = getattr(mask, "numpy", None)
        if callable(numpy_fn):
            mask = numpy_fn()
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
            )
        binary = (mask > 0.5).astype(np.uint8)
        out = np.maximum(out, binary)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        out = cv2.dilate(out, kernel)

    return out


def _default_video_pipeline_runner(
    model: _VideoModelLike,
    config: VideoBlurConfig,
    predict_kwargs: dict[str, Any],
) -> list[FrameStats]:
    """OpenCV-backed reader/writer loop with per-frame mask blurring."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(config.source))
    if not cap.isOpened():
        msg = f"Cannot open video for reading: {config.source}"
        raise RuntimeError(msg)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if config.fps_override is not None:
        fps = config.fps_override
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    config.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*config.fourcc)
    writer = cv2.VideoWriter(str(config.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        msg = f"Cannot open video for writing: {config.output}"
        raise RuntimeError(msg)

    stats: list[FrameStats] = []
    classes_filter = set(config.classes) if config.classes else None
    frame_pixels = max(1, width * height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, **predict_kwargs)
            result = None
            if results:
                try:
                    result = results[0]
                except (TypeError, IndexError):
                    iterator = iter(results)
                    result = next(iterator, None)

            classes = _result_classes(result) if result is not None else []
            keep_indices = [
                i
                for i, cls in enumerate(classes)
                if classes_filter is None or cls in classes_filter
            ]
            kept_classes = [classes[i] for i in keep_indices]

            if not keep_indices or result is None:
                writer.write(frame)
                stats.append(
                    FrameStats(
                        detections=0,
                        detection_classes=[],
                        mask_pixels=0,
                        frame_pixels=frame_pixels,
                    )
                )
                continue

            union = _union_mask(
                result, keep_indices, height, width, config.mask_dilate
            )
            blurred = _apply_blur(frame, config)
            mask3 = union[:, :, None].astype(bool)
            out_frame = np.where(mask3, blurred, frame)
            writer.write(out_frame)

            stats.append(
                FrameStats(
                    detections=len(keep_indices),
                    detection_classes=kept_classes,
                    mask_pixels=int(union.sum()),
                    frame_pixels=frame_pixels,
                )
            )
    finally:
        cap.release()
        writer.release()

    return stats


def _config_to_jsonable(config: VideoBlurConfig) -> dict[str, Any]:
    data = asdict(config)
    data["weights"] = str(config.weights)
    data["source"] = str(config.source)
    data["output"] = str(config.output)
    return data
