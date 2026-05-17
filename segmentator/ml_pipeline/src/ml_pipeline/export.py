"""YOLO26-seg export driver for ml_pipeline.

Wraps Ultralytics ``YOLO.export(...)`` for the ``segment`` task and generates a
JSON-friendly report with the resolved kwargs and the exported file path.
ONNX is the default target because it is portable and runs in many
runtimes without a CUDA toolchain. Other formats (torchscript, engine,
coreml, openvino, ...) are accepted as opt-in passthrough values.

Ultralytics is imported lazily by the default factory so unit tests can
inject a stub.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from ml_pipeline.checkpoints import missing_weights_message
from ml_pipeline.export_extra import ensure_onnx_export_requirements
from ml_pipeline.ultralytics_extra import raise_ultralytics_missing


class _ExportModelLike(Protocol):
    def export(self, **kwargs: Any) -> Any: ...


ExportModelFactory = Callable[[str], _ExportModelLike]

VALID_FORMATS = frozenset(
    {
        "torchscript",
        "onnx",
        "openvino",
        "engine",
        "coreml",
        "saved_model",
        "pb",
        "tflite",
        "edgetpu",
        "tfjs",
        "paddle",
        "ncnn",
    }
)


FORMAT_EXTENSIONS = {
    "torchscript": ".torchscript",
    "onnx": ".onnx",
    "engine": ".engine",
    "coreml": ".mlpackage",
    "saved_model": "_saved_model",
    "pb": ".pb",
    "tflite": ".tflite",
    "edgetpu": "_edgetpu.tflite",
    "tfjs": "_web_model",
    "paddle": "_paddle_model",
    "ncnn": "_ncnn_model",
    "openvino": "_openvino_model",
}


@dataclass
class ExportConfig:
    """Inputs for a YOLO26 segmentation export run."""
    weights: Path
    fmt: str = "onnx"
    imgsz: int = 640
    half: bool = False
    int8: bool = False
    dynamic: bool = False
    simplify: bool = True
    opset: int | None = None
    nms: bool = False
    batch: int = 1
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _default_export_model_factory(weights_ref: str) -> _ExportModelLike:
    """Import ultralytics only when needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise_ultralytics_missing(exc)
    return YOLO(weights_ref)


def validate_export_inputs(config: ExportConfig) -> None:
    """Validate that weights exist and chosen format is supported."""
    if not config.weights.is_file():
        raise FileNotFoundError(
            missing_weights_message(config.weights, config.weights.parent)
        )
    fmt = config.fmt.strip().lower()
    if fmt not in VALID_FORMATS:
        msg = (
            f"Unsupported export format '{config.fmt}'. Expected one of: "
            f"{', '.join(sorted(VALID_FORMATS))}."
        )
        raise ValueError(msg)
    if config.imgsz <= 0:
        raise ValueError("--imgsz must be a positive integer.")
    if config.batch <= 0:
        raise ValueError("--batch must be a positive integer.")
    if config.half and config.int8:
        raise ValueError("--half and --int8 cannot be combined.")
    if fmt == "onnx":
        ensure_onnx_export_requirements()
    if fmt == "engine" and config.device == "cpu":
        msg = "TensorRT engine export requires a CUDA device, not 'cpu'."
        raise ValueError(msg)


def build_export_kwargs(config: ExportConfig) -> dict[str, Any]:
    """Build keyword arguments passed to YOLO.export()."""
    kwargs: dict[str, Any] = {
        "format": config.fmt.strip().lower(),
        "imgsz": config.imgsz,
        "half": config.half,
        "int8": config.int8,
        "dynamic": config.dynamic,
        "simplify": config.simplify,
        "nms": config.nms,
        "batch": config.batch,
        "task": "segment",
    }
    if config.opset is not None:
        kwargs["opset"] = config.opset
    if config.device is not None:
        kwargs["device"] = config.device
    kwargs.update(config.extra)
    return kwargs


def _expected_export_path(config: ExportConfig) -> Path:
    """Return Ultralytics' default output path for the chosen format.

    Ultralytics writes the exported artifact next to the source weights and
    derives the filename from the weights stem plus a format-specific
    suffix. Used as a fallback when ``YOLO.export`` does not return a path.
    """
    fmt = config.fmt.strip().lower()
    suffix = FORMAT_EXTENSIONS.get(fmt, f".{fmt}")
    return config.weights.with_name(f"{config.weights.stem}{suffix}")


def run_export(
    config: ExportConfig,
    *,
    model_factory: ExportModelFactory | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute export and return a serializable report.

    Parameters
    ----------
    config:
        Export configuration.
    model_factory:
        Callable returning an object with ``.export(**kwargs)``. Defaults
        to the lazy Ultralytics loader. Tests pass a stub.
    dry_run:
        When true, only build the kwargs and skip the call to ``export``.
    """
    validate_export_inputs(config)
    kwargs = build_export_kwargs(config)
    weights_ref = str(config.weights)

    report: dict[str, Any] = {
        "config": _config_to_jsonable(config),
        "weights": weights_ref,
        "export_kwargs": kwargs,
        "dry_run": dry_run,
        "expected_output": str(_expected_export_path(config)),
    }

    if dry_run:
        return report

    factory = model_factory or _default_export_model_factory
    model = factory(weights_ref)
    raw = model.export(**kwargs)

    output_path = _coerce_output_path(raw, config)
    report["output_path"] = str(output_path) if output_path is not None else None
    report["output_exists"] = bool(
        output_path is not None and Path(output_path).exists()
    )
    return report


def _coerce_output_path(raw: Any, config: ExportConfig) -> Path | None:
    """Normalize the value returned by ``YOLO.export``.
    Ultralytics typically returns a string path (or pathlib.Path) for
    single-file formats, and may return a directory for bundle formats
    such as openvino or coreml. When the value cannot be parsed, fall
    back to the Ultralytics default location so the report is still
    actionable.
    """
    if isinstance(raw, (str, Path)):
        return Path(raw)
    if isinstance(raw, (list, tuple)) and raw:
        first = raw[0]
        if isinstance(first, (str, Path)):
            return Path(first)
    return _expected_export_path(config)


def _config_to_jsonable(config: ExportConfig) -> dict[str, Any]:
    data = asdict(config)
    data["weights"] = str(config.weights)
    return data
