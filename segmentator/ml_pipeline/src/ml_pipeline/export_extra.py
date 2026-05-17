"""Shared messaging for ONNX export optional dependencies."""

from __future__ import annotations

EXPORT_INSTALL_CMD = "uv sync --extra train --extra export"

EXPORT_ONNX_MISSING_MESSAGE = (
    "ONNX export requires the optional [export] dependency group (onnx, onnxslim, "
    "onnxruntime). Ultralytics cannot auto-install these in a uv-managed venv. "
    "From segmentator/ml_pipeline run: "
    f"{EXPORT_INSTALL_CMD}"
)


def raise_export_onnx_missing(exc: ImportError) -> None:
    """Re-raise ``exc`` as ``RuntimeError`` with install instructions."""
    raise RuntimeError(EXPORT_ONNX_MISSING_MESSAGE) from exc


def ensure_onnx_export_requirements() -> None:
    """Verify ONNX stack is importable before calling ``YOLO.export``."""
    missing: list[str] = []
    for module in ("onnx", "onnxslim", "onnxruntime"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            f"{EXPORT_ONNX_MISSING_MESSAGE} Missing modules: {', '.join(missing)}."
        )
