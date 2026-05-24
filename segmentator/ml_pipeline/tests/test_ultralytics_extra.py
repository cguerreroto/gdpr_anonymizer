"""Tests for ml_pipeline.ultralytics_extra."""

from __future__ import annotations

import pytest

from fakes import block_ultralytics_import, install_fake_ultralytics
from ml_pipeline.ultralytics_extra import (
    ULTRALYTICS_INSTALL_CMD,
    ULTRALYTICS_MISSING_MESSAGE,
    raise_ultralytics_missing,
)
from ml_pipeline.train import _default_model_factory


class _StubModel:
    def __init__(self, ref: str) -> None:
        self.ref = ref


def test_missing_message_documents_uv_sync_extra() -> None:
    assert ULTRALYTICS_INSTALL_CMD in ULTRALYTICS_MISSING_MESSAGE
    assert "segmentator/ml_pipeline" in ULTRALYTICS_MISSING_MESSAGE


def test_raise_ultralytics_missing_wraps_import_error() -> None:
    original = ImportError("no module named ultralytics")
    with pytest.raises(RuntimeError, match="uv sync --extra train") as exc_info:
        raise_ultralytics_missing(original)
    assert exc_info.value.__cause__ is original


def test_default_model_factory_loads_fake_yolo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs: list[str] = []
    install_fake_ultralytics(
        monkeypatch,
        yolo_factory=lambda ref: refs.append(ref) or _StubModel(ref),
    )

    model = _default_model_factory("yolo26n-seg.pt")

    assert refs == ["yolo26n-seg.pt"]
    assert isinstance(model, _StubModel)
    assert model.ref == "yolo26n-seg.pt"


def test_default_model_factory_surfaces_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    block_ultralytics_import(monkeypatch)
    with pytest.raises(RuntimeError, match="uv sync --extra train"):
        _default_model_factory("yolo26n-seg.pt")
