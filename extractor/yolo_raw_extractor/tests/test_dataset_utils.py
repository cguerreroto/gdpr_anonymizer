"""Tests for yolo_raw_extractor.dataset_utils."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import yaml

from yolo_raw_extractor.dataset_utils import (
    derive_label_dir,
    find_image_path,
    load_dataset_config,
    parse_label_line,
    sanitize_class_name,
)


def test_sanitize_class_name() -> None:
    assert sanitize_class_name("Foo Bar") == "foo-bar"
    assert sanitize_class_name("A/B") == "ab"
    assert sanitize_class_name("Class_1") == "class_1"


def test_parse_label_line_valid() -> None:
    line = "2 0.0 0.0 1.0 0.0 0.5 1.0"
    class_id, poly = parse_label_line(line, width=100, height=200)
    assert class_id == 2
    assert poly.shape == (3, 2)
    np.testing.assert_allclose(poly[0], [0.0, 0.0])
    np.testing.assert_allclose(poly[1], [100.0, 0.0])
    np.testing.assert_allclose(poly[2], [50.0, 200.0])


def test_parse_label_line_bbox() -> None:
    line = "0 0.5 0.5 0.2 0.4"
    class_id, poly = parse_label_line(line, width=100, height=100)
    assert class_id == 0
    assert poly.shape == (4, 2)
    np.testing.assert_allclose(poly[0], [40.0, 30.0])
    np.testing.assert_allclose(poly[2], [60.0, 70.0])


def test_parse_label_line_errors() -> None:
    with pytest.raises(ValueError, match="bbox or polygon"):
        parse_label_line("0 0.1 0.2", width=10, height=10)
    with pytest.raises(ValueError, match="bbox \\(cx,cy,w,h\\) or polygon"):
        parse_label_line("0 0.1 0.2 0.3 0.4 0.5 0.6 0.7", width=10, height=10)


def test_derive_label_dir_when_labels_exists(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    assert derive_label_dir(img_dir) == lbl_dir


def test_derive_label_dir_fallback(tmp_path: Path) -> None:
    p = tmp_path / "somewhere" / "images" / "train"
    p.mkdir(parents=True)
    assert derive_label_dir(p) == p


def test_find_image_path(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    stem = Path("shot_001")
    (img_dir / f"{stem.name}.png").write_bytes(b"")
    found = find_image_path(tmp_path / "labels" / f"{stem.name}.txt", img_dir)
    assert found == img_dir / f"{stem.name}.png"


def test_load_dataset_config(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    train_img = root / "images" / "train"
    train_img.mkdir(parents=True)
    cfg = {
        "path": ".",
        "train": "images/train",
        "names": {0: "person", "1": "car"},
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    splits, names = load_dataset_config(root)
    assert "train" in splits
    assert splits["train"] == train_img.resolve()
    assert names[0] == "person"
    assert names[1] == "car"


def test_load_dataset_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        load_dataset_config(tmp_path)


def test_load_dataset_config_skips_unparseable_class_keys(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    train_img = root / "images" / "train"
    train_img.mkdir(parents=True)
    cfg = {
        "train": "images/train",
        "names": {"abc": "ignored", 7: "person"},
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    _splits, names = load_dataset_config(root)

    assert names == {7: "person"}


def test_find_image_path_returns_none_when_no_match(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    found = find_image_path(tmp_path / "labels" / "missing.txt", img_dir)
    assert found is None


def test_load_dataset_config_absolute_split_path(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    train_img = tmp_path / "abs-train"
    train_img.mkdir(parents=True)
    root.mkdir(parents=True)
    cfg = {
        "train": str(train_img),
        "names": {0: "person"},
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    splits, _names = load_dataset_config(root)

    assert splits["train"] == train_img
