"""Tests for ml_pipeline.ultralytics_extra."""

from __future__ import annotations

import pytest

from ml_pipeline.ultralytics_extra import (
    ULTRALYTICS_INSTALL_CMD,
    ULTRALYTICS_MISSING_MESSAGE,
    raise_ultralytics_missing,
)
from ml_pipeline.train import _default_model_factory


def test_missing_message_documents_uv_sync_extra() -> None:
    assert ULTRALYTICS_INSTALL_CMD in ULTRALYTICS_MISSING_MESSAGE
    assert "segmentator/ml_pipeline" in ULTRALYTICS_MISSING_MESSAGE


def test_raise_ultralytics_missing_wraps_import_error() -> None:
    original = ImportError("no module named ultralytics")
    with pytest.raises(RuntimeError, match="uv sync --extra train") as exc_info:
        raise_ultralytics_missing(original)
    assert exc_info.value.__cause__ is original


def test_default_model_factory_surfaces_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ultralytics":
            raise ImportError("no ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="uv sync --extra train"):
        _default_model_factory("yolo26n-seg.pt")
