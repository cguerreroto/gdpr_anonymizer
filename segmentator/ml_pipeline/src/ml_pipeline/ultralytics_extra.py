"""Shared messaging for the optional Ultralytics [train] extra."""

from __future__ import annotations

ULTRALYTICS_INSTALL_CMD = "uv sync --extra train"

ULTRALYTICS_MISSING_MESSAGE = (
    "ultralytics is not installed. The train and validate CLIs require the "
    "optional [train] dependency group. From segmentator/ml_pipeline run: "
    f"{ULTRALYTICS_INSTALL_CMD}"
)


def raise_ultralytics_missing(exc: ImportError) -> None:
    """Re-raise ``exc`` as ``RuntimeError`` with install instructions."""
    raise RuntimeError(ULTRALYTICS_MISSING_MESSAGE) from exc
