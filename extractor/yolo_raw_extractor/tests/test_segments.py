"""Tests for yolo_raw_extractor.segments."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import yaml

import yolo_raw_extractor.segments as segments_mod
from yolo_raw_extractor.segments import (
    build_alpha_segment,
    extract_segments,
    parse_args,
    write_metadata,
)


def test_parse_args_dataset_only() -> None:
    args = parse_args(["/tmp/dataset"])
    assert args.dataset_dir == Path("/tmp/dataset")
    assert args.output_dir is None


def test_parse_args_with_output_dir() -> None:
    args = parse_args(["/tmp/dataset", "--output-dir", "/tmp/out"])
    assert args.output_dir == Path("/tmp/out")


def test_build_alpha_segment_rectangle() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    polygon = np.array(
        [[5.0, 5.0], [25.0, 5.0], [25.0, 15.0], [5.0, 15.0]], dtype=np.float32
    )
    segment, bbox = build_alpha_segment(image, polygon)
    x, y, w, h = bbox
    assert w > 0 and h > 0
    assert segment.shape == (h, w, 4)
    assert segment.dtype == np.uint8
    assert np.any(segment[:, :, 3] > 0)


def test_build_alpha_segment_degenerate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the w/h guard; OpenCV's boundingRect rarely returns zero for polygons."""
    monkeypatch.setattr(segments_mod.cv2, "boundingRect", lambda _pts: (0, 0, 0, 0))
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    polygon = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Degenerate bounding box"):
        build_alpha_segment(image, polygon)


def test_write_metadata_writes_json_file(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"")
    polygon = np.array([[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]], dtype=np.float32)
    bbox = (5, 10, 30, 35)

    write_metadata(image_path, class_id=1, class_label="car", polygon=polygon, bbox=bbox)

    payload = json.loads((tmp_path / "frame.json").read_text(encoding="utf-8"))
    assert payload["class_id"] == 1
    assert payload["class_name"] == "car"
    assert payload["crop"] == {"x": 5, "y": 10, "width": 30, "height": 35}
    assert payload["source_image"] == "frame.png"
    assert payload["points"][0] == [5.0, 10.0]


def _build_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "shot.png").write_bytes(b"")
    (lbl_dir / "shot.txt").write_text(
        "0 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0\n", encoding="utf-8"
    )
    cfg = {
        "path": ".",
        "train": "images/train",
        "names": {0: "person"},
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return root


def test_extract_segments_writes_pngs_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    image_arr = np.zeros((100, 200, 3), dtype=np.uint8)
    monkeypatch.setattr(segments_mod.cv2, "imread", lambda _path, _flags: image_arr)

    written: list[str] = []

    def fake_imwrite(path: str, _segment: np.ndarray) -> bool:
        written.append(path)
        Path(path).write_bytes(b"")
        return True

    monkeypatch.setattr(segments_mod.cv2, "imwrite", fake_imwrite)

    extracted = extract_segments(dataset_dir)

    assert extracted == 1
    assert len(written) == 1
    output_path = Path(written[0])
    assert output_path.parent.name == "person"
    assert output_path.parent.parent.name == "train"
    assert output_path.with_suffix(".json").exists()


def test_extract_segments_skips_unreadable_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    monkeypatch.setattr(segments_mod.cv2, "imread", lambda _path, _flags: None)
    monkeypatch.setattr(segments_mod.cv2, "imwrite", lambda _path, _seg: True)

    assert extract_segments(dataset_dir) == 0


def test_extract_segments_no_labels_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "ds"
    img_dir = root / "images" / "train"
    img_dir.mkdir(parents=True)
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}}
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with caplog.at_level("INFO"):
        assert extract_segments(root) == 0

    assert any("No labels found" in rec.message for rec in caplog.records)


def test_extract_segments_missing_image_dir_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "ds"
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}}
    (root).mkdir(parents=True)
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert extract_segments(root) == 0

    assert any("Image directory missing" in rec.message for rec in caplog.records)


def test_extract_segments_skips_bad_label_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "ds"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "shot.png").write_bytes(b"")
    (lbl_dir / "shot.txt").write_text("0 0.1 0.2\n", encoding="utf-8")
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}}
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(
        segments_mod.cv2, "imread", lambda _path, _flags: np.zeros((5, 5, 3), np.uint8)
    )

    with caplog.at_level("WARNING"):
        assert extract_segments(root) == 0

    assert any("Failed to process" in rec.message for rec in caplog.records)


def test_extract_segments_imwrite_failure_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    monkeypatch.setattr(
        segments_mod.cv2, "imread", lambda _path, _flags: np.zeros((50, 80, 3), np.uint8)
    )
    monkeypatch.setattr(segments_mod.cv2, "imwrite", lambda _path, _seg: False)

    with caplog.at_level("WARNING"):
        extracted = extract_segments(dataset_dir)

    assert extracted == 0
    assert any("Failed to write segment" in rec.message for rec in caplog.records)


def test_extract_segments_uses_custom_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    output_root = tmp_path / "alt-output"
    monkeypatch.setattr(
        segments_mod.cv2, "imread", lambda _path, _flags: np.zeros((5, 5, 3), np.uint8)
    )

    written: list[str] = []

    def fake_imwrite(path: str, _segment: np.ndarray) -> bool:
        written.append(path)
        Path(path).write_bytes(b"")
        return True

    monkeypatch.setattr(segments_mod.cv2, "imwrite", fake_imwrite)

    extract_segments(dataset_dir, output_dir=output_root)

    assert written
    assert Path(written[0]).is_relative_to(output_root.resolve())


def test_run_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(segments_mod, "extract_segments", lambda *_a, **_kw: 4)
    assert segments_mod._run([str(tmp_path)]) == 0


def test_run_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("oops")

    monkeypatch.setattr(segments_mod, "extract_segments", boom)
    assert segments_mod._run([str(tmp_path)]) == 1


def test_main_raises_systemexit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(segments_mod, "_run", lambda _argv: 3)
    with pytest.raises(SystemExit) as info:
        segments_mod.main(["/tmp/ds"])
    assert info.value.code == 3
