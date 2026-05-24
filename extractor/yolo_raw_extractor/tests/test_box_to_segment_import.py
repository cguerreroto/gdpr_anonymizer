"""Tests for YOLO bbox-to-segmentation dataset import."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_raw_extractor.box_to_segment_import import (
    bbox_to_polygon_tokens,
    build_person_class_map,
    convert_label_line,
    discover_bbox_splits,
    import_bbox_to_segment,
)


def _write_bbox_src_export(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "face"},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    for split, folder in (("train", "train"), ("valid", "valid"), ("test", "test")):
        image_dir = root / folder / "images"
        label_dir = root / folder / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        stem = f"{split}_sample"
        (image_dir / f"{stem}.jpg").write_bytes(b"fake")
        (label_dir / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.4\n",
            encoding="utf-8",
        )


def test_bbox_to_polygon_tokens() -> None:
    tokens = bbox_to_polygon_tokens(0.5, 0.5, 0.2, 0.4)
    assert tokens == [
        "0.400000",
        "0.300000",
        "0.600000",
        "0.300000",
        "0.600000",
        "0.700000",
        "0.400000",
        "0.700000",
    ]


def test_convert_label_line_bbox_to_person() -> None:
    class_map = {0: 0}
    line = convert_label_line("0 0.5 0.5 0.2 0.4", class_map, person_class_id=0)
    assert line is not None
    assert line.startswith("0 ")
    assert len(line.split()) == 9


def test_convert_label_line_skips_unknown_class() -> None:
    class_map = {0: 0}
    assert convert_label_line("1 0.5 0.5 0.2 0.4", class_map, person_class_id=0) is None


def test_build_person_class_map_face() -> None:
    names = {0: "face", 1: "car"}
    mapping = build_person_class_map(names, map_all_classes=False, person_class_id=0)
    assert mapping == {0: 0}


def test_discover_bbox_splits(tmp_path: Path) -> None:
    bbox_src = tmp_path / "bbox_src"
    _write_bbox_src_export(bbox_src)
    splits = discover_bbox_splits(bbox_src)
    assert set(splits) == {"train", "val", "test"}


def test_import_bbox_to_segment(tmp_path: Path) -> None:
    bbox_src = tmp_path / "bbox_src"
    dataset = tmp_path / "dataset_raw"
    _write_bbox_src_export(bbox_src)
    (dataset / "images" / "train").mkdir(parents=True)

    stats = import_bbox_to_segment(bbox_src, dataset, prefix="imp_")
    assert stats.images_copied == 3
    assert stats.objects_imported == 3

    train_label = dataset / "images" / "train" / "imp_train_sample.txt"
    assert train_label.exists()
    first_line = train_label.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("0 ")
    assert len(first_line.split()) == 9


def test_import_requires_matching_class(tmp_path: Path) -> None:
    bbox_src = tmp_path / "bbox_src"
    _write_bbox_src_export(bbox_src)
    cfg = yaml.safe_load((bbox_src / "data.yaml").read_text(encoding="utf-8"))
    cfg["names"] = {0: "vehicle"}
    (bbox_src / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="No source classes matched"):
        import_bbox_to_segment(bbox_src, tmp_path / "dataset")
