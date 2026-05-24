"""Tests for ml_pipeline.export and ml_pipeline.cli_export.

A stub factory replaces Ultralytics so the export driver and CLI can be
unit tested without GPU, network access, or a real .pt file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml_pipeline.cli_export import main as cli_main
from ml_pipeline.export import (
    ExportConfig,
    build_export_kwargs,
    run_export,
    validate_export_inputs,
)


@pytest.fixture(autouse=True)
def _stub_onnx_export_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real onnx/onnxslim/onnxruntime import probe by default.

    The export driver enforces these optional dependencies in
    ``validate_export_inputs`` when ``format == "onnx"``. The unit tests in
    this module use stub factories and never call Ultralytics, so they should
    not require the [export] extra to be installed. Tests that intentionally
    exercise the failure path can override this fixture by re-monkeypatching
    the same symbol.
    """
    monkeypatch.setattr(
        "ml_pipeline.export.ensure_onnx_export_requirements", lambda: None
    )


class _StubModel:
    def __init__(self, ref: str) -> None:
        self.ref = ref
        self.export_kwargs: dict[str, Any] | None = None
        self._return_value: Any = None

    def export(self, **kwargs: Any) -> Any:
        self.export_kwargs = kwargs
        return self._return_value


def _make_config(tmp_path: Path) -> ExportConfig:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    return ExportConfig(weights=weights)


def test_build_export_kwargs_defaults(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    kwargs = build_export_kwargs(cfg)
    assert kwargs["task"] == "segment"
    assert kwargs["format"] == "onnx"
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 1
    assert kwargs["half"] is False
    assert kwargs["int8"] is False
    assert kwargs["dynamic"] is False
    assert kwargs["simplify"] is True
    assert kwargs["nms"] is False
    assert "opset" not in kwargs
    assert "device" not in kwargs


def test_build_export_kwargs_with_overrides(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.fmt = "torchscript"
    cfg.imgsz = 512
    cfg.batch = 4
    cfg.half = True
    cfg.dynamic = True
    cfg.opset = 17
    cfg.device = "cpu"
    cfg.extra = {"agnostic_nms": True}
    kwargs = build_export_kwargs(cfg)
    assert kwargs["format"] == "torchscript"
    assert kwargs["imgsz"] == 512
    assert kwargs["batch"] == 4
    assert kwargs["half"] is True
    assert kwargs["dynamic"] is True
    assert kwargs["opset"] == 17
    assert kwargs["device"] == "cpu"
    assert kwargs["agnostic_nms"] is True


def test_build_export_kwargs_normalizes_format_case(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.fmt = "ONNX"
    kwargs = build_export_kwargs(cfg)
    assert kwargs["format"] == "onnx"


def test_validate_export_inputs_requires_onnx_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path)

    def fake_ensure() -> None:
        raise RuntimeError("install export extra")

    monkeypatch.setattr(
        "ml_pipeline.export.ensure_onnx_export_requirements", fake_ensure
    )
    with pytest.raises(RuntimeError, match="install export extra"):
        validate_export_inputs(cfg)


def test_export_onnx_missing_message() -> None:
    from ml_pipeline.export_extra import EXPORT_ONNX_MISSING_MESSAGE

    assert "uv sync --extra train --extra export" in EXPORT_ONNX_MISSING_MESSAGE


def test_raise_export_onnx_missing_chains_original() -> None:
    from ml_pipeline.export_extra import raise_export_onnx_missing

    with pytest.raises(RuntimeError) as info:
        raise_export_onnx_missing(ImportError("no onnx"))
    assert isinstance(info.value.__cause__, ImportError)


def test_ensure_onnx_export_requirements_raises_when_modules_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml_pipeline import export_extra

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"onnx", "onnxslim", "onnxruntime"}:
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="Missing modules"):
        export_extra.ensure_onnx_export_requirements()


def test_ensure_onnx_export_requirements_passes_when_modules_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml_pipeline import export_extra
    import sys

    for name in ("onnx", "onnxslim", "onnxruntime"):
        sys.modules.setdefault(name, type(sys)("dummy_" + name))

    export_extra.ensure_onnx_export_requirements()


def test_validate_export_inputs_rejects_missing_weights(tmp_path: Path) -> None:
    cfg = ExportConfig(weights=tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="Missing weights"):
        validate_export_inputs(cfg)


def test_validate_export_inputs_rejects_unknown_format(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.fmt = "bogus"
    with pytest.raises(ValueError, match="Unsupported export format"):
        validate_export_inputs(cfg)


def test_validate_export_inputs_rejects_half_and_int8(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.half = True
    cfg.int8 = True
    with pytest.raises(ValueError, match="--half and --int8"):
        validate_export_inputs(cfg)


def test_validate_export_inputs_rejects_engine_on_cpu(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.fmt = "engine"
    cfg.device = "cpu"
    with pytest.raises(ValueError, match="TensorRT"):
        validate_export_inputs(cfg)


def test_validate_export_inputs_rejects_bad_imgsz(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.imgsz = 0
    with pytest.raises(ValueError, match="--imgsz"):
        validate_export_inputs(cfg)


def test_validate_export_inputs_rejects_bad_batch(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    cfg.batch = -1
    with pytest.raises(ValueError, match="--batch"):
        validate_export_inputs(cfg)


def test_run_export_dry_run_skips_factory(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    called: dict[str, bool] = {"factory": False}

    def factory(_ref: str) -> _StubModel:
        called["factory"] = True
        return _StubModel(_ref)

    report = run_export(cfg, model_factory=factory, dry_run=True)
    assert called["factory"] is False
    assert report["dry_run"] is True
    assert "output_path" not in report
    assert report["expected_output"].endswith(".onnx")


def test_run_export_invokes_factory_and_reports_path(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    captured: dict[str, _StubModel] = {}
    expected_output = cfg.weights.with_suffix(".onnx")
    expected_output.write_bytes(b"fake-onnx")

    def factory(ref: str) -> _StubModel:
        model = _StubModel(ref)
        model._return_value = str(expected_output)
        captured["model"] = model
        return model

    report = run_export(cfg, model_factory=factory)
    assert captured["model"].export_kwargs is not None
    assert report["dry_run"] is False
    assert report["output_path"] == str(expected_output)
    assert report["output_exists"] is True


def test_run_export_falls_back_to_expected_path(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)

    def factory(ref: str) -> _StubModel:
        model = _StubModel(ref)
        model._return_value = None
        return model

    report = run_export(cfg, model_factory=factory)
    assert report["output_path"] == report["expected_output"]
    assert report["output_exists"] is False


def test_cli_dry_run_prints_kwargs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    code = cli_main([str(weights), "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["export_kwargs"]["format"] == "onnx"
    assert payload["export_kwargs"]["task"] == "segment"


def test_cli_writes_report_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    expected_output = weights.with_suffix(".onnx")
    expected_output.write_bytes(b"fake-onnx")

    def factory(ref: str) -> _StubModel:
        model = _StubModel(ref)
        model._return_value = str(expected_output)
        return model

    monkeypatch.setattr("ml_pipeline.export._default_export_model_factory", factory)

    report_path = tmp_path / "export.json"
    code = cli_main(
        [
            str(weights),
            "--format",
            "onnx",
            "--imgsz",
            "640",
            "--report-json",
            str(report_path),
        ]
    )
    assert code == 0
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["output_path"] == str(expected_output)
    assert saved["output_exists"] is True


def test_cli_no_simplify_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    code = cli_main([str(weights), "--no-simplify", "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["export_kwargs"]["simplify"] is False


def test_run_export_uses_first_path_from_list(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    expected_first = tmp_path / "alt.onnx"
    expected_first.write_bytes(b"alt")

    def factory(ref: str) -> _StubModel:
        model = _StubModel(ref)
        model._return_value = [str(expected_first), str(tmp_path / "second.onnx")]
        return model

    report = run_export(cfg, model_factory=factory)
    assert report["output_path"] == str(expected_first)
    assert report["output_exists"] is True


def test_run_export_invalid_return_falls_back(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)

    def factory(ref: str) -> _StubModel:
        model = _StubModel(ref)
        model._return_value = 42
        return model

    report = run_export(cfg, model_factory=factory)
    assert report["output_path"] == report["expected_output"]
