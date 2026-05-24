"""Tests for YOLO bbox-to-segmentation dataset import."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import yolo_raw_extractor.box_to_segment_import as bbox_mod
from yolo_raw_extractor.box_to_segment_import import (
    ImportStats,
    _safe_stem,
    bbox_to_polygon_tokens,
    build_person_class_map,
    convert_label_file,
    convert_label_line,
    discover_bbox_splits,
    import_bbox_to_segment,
    load_bbox_class_names,
    parse_args,
    write_dataset_yaml,
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


def test_parse_args_full_options() -> None:
    args = parse_args(
        [
            "/tmp/src",
            "/tmp/dst",
            "--prefix",
            "x_",
            "--no-prefix",
            "--person-class-id",
            "2",
            "--map-all-classes",
            "--dry-run",
            "--overwrite",
        ]
    )
    assert args.bbox_src_dir == Path("/tmp/src")
    assert args.dataset_dir == Path("/tmp/dst")
    assert args.prefix == "x_"
    assert args.no_prefix is True
    assert args.person_class_id == 2
    assert args.map_all_classes is True
    assert args.dry_run is True
    assert args.overwrite is True


def test_import_stats_post_init_default() -> None:
    stats = ImportStats()
    assert stats.splits == {}


def test_import_stats_preserves_provided_splits() -> None:
    stats = ImportStats(splits={"train": 3})
    assert stats.splits == {"train": 3}


def test_load_bbox_src_config_returns_empty_when_missing(tmp_path: Path) -> None:
    assert bbox_mod._load_bbox_src_config(tmp_path) == {}


def test_resolve_bbox_src_path_absolute(tmp_path: Path) -> None:
    target = tmp_path / "absolute" / "images"
    target.mkdir(parents=True)
    resolved = bbox_mod._resolve_bbox_src_path(tmp_path / "anywhere", str(target))
    assert resolved == target.resolve()


def test_resolve_bbox_src_path_with_parent_alias(tmp_path: Path) -> None:
    bbox_src = tmp_path / "src"
    target = tmp_path / "src" / "train" / "images"
    target.mkdir(parents=True)
    resolved = bbox_mod._resolve_bbox_src_path(bbox_src, "../train/images")
    assert resolved == target.resolve()


def test_split_image_label_dirs_returns_none_for_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "train"
    empty.mkdir()
    (empty / "readme.txt").write_text("no images here", encoding="utf-8")
    assert bbox_mod._split_image_label_dirs(empty) is None


def test_split_image_label_dirs_returns_none_for_known_subfolder() -> None:
    assert bbox_mod._split_image_label_dirs(Path("/tmp/images")) is None


def test_split_image_label_dirs_flat_layout(tmp_path: Path) -> None:
    flat = tmp_path / "train"
    flat.mkdir()
    (flat / "img.jpg").write_bytes(b"")
    pair = bbox_mod._split_image_label_dirs(flat)
    assert pair == (flat, flat)


def test_discover_bbox_splits_raises_when_configured_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absolutely" / "missing" / "images"
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"train": str(missing), "names": {0: "face"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Configured split path does not exist"):
        discover_bbox_splits(tmp_path)


def test_discover_bbox_splits_val_fallback_to_valid_folder(tmp_path: Path) -> None:
    missing = tmp_path / "absolutely" / "missing" / "val" / "images"
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"val": str(missing), "names": {0: "face"}}),
        encoding="utf-8",
    )
    img_dir = tmp_path / "valid" / "images"
    lbl_dir = tmp_path / "valid" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    splits = discover_bbox_splits(tmp_path)

    assert splits["val"] == (img_dir.resolve(), lbl_dir.resolve())


def test_discover_bbox_splits_skips_files_and_unknown_dirs(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    misc = tmp_path / "misc"
    misc.mkdir()
    train_img = tmp_path / "train" / "images"
    train_lbl = tmp_path / "train" / "labels"
    train_img.mkdir(parents=True)
    train_lbl.mkdir(parents=True)

    splits = discover_bbox_splits(tmp_path)

    assert set(splits) == {"train"}


def test_guess_label_dir_uses_parallel_labels_tree(tmp_path: Path) -> None:
    root = tmp_path / "export"
    images = root / "images" / "val"
    labels = root / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    assert bbox_mod._guess_label_dir(images) == labels


def test_discover_bbox_splits_uses_dir_walk_when_yaml_missing(tmp_path: Path) -> None:
    for split in ("train", "valid"):
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

    splits = discover_bbox_splits(tmp_path)

    assert set(splits) == {"train", "val"}


def test_discover_bbox_splits_raises_when_nothing_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No YOLO detection splits"):
        discover_bbox_splits(tmp_path)


def test_discover_bbox_splits_yaml_path_fallback(tmp_path: Path) -> None:
    cfg = {
        "train": "missing/folder",
        "names": {0: "face"},
    }
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    img_dir = tmp_path / "train" / "images"
    lbl_dir = tmp_path / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    splits = discover_bbox_splits(tmp_path)

    assert splits["train"] == (img_dir.resolve(), lbl_dir.resolve())


def test_guess_label_dir_uses_sibling_labels(tmp_path: Path) -> None:
    images = tmp_path / "train" / "data"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    assert bbox_mod._guess_label_dir(images) == labels


def test_guess_label_dir_returns_image_dir_as_fallback(tmp_path: Path) -> None:
    images = tmp_path / "train" / "data"
    images.mkdir(parents=True)
    assert bbox_mod._guess_label_dir(images) == images


def test_build_person_class_map_map_all() -> None:
    names = {0: "face", 1: "car", 2: "tree"}
    mapping = build_person_class_map(names, map_all_classes=True, person_class_id=5)
    assert mapping == {0: 5, 1: 5, 2: 5}


def test_load_bbox_class_names_list(tmp_path: Path) -> None:
    cfg = {"names": ["face", "car"]}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert load_bbox_class_names(tmp_path) == {0: "face", 1: "car"}


def test_load_bbox_class_names_dict_with_bad_keys(tmp_path: Path) -> None:
    cfg = {"names": {"abc": "skip", 0: "face"}}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert load_bbox_class_names(tmp_path) == {0: "face"}


def test_convert_label_line_polygon_path() -> None:
    class_map = {0: 0}
    line = convert_label_line(
        "0 0.1 0.2 0.3 0.4 0.5 0.6", class_map, person_class_id=0
    )
    assert line is not None
    assert line.startswith("0 ")
    assert len(line.split()) == 7


def test_convert_label_line_blank_returns_none() -> None:
    assert convert_label_line("   ", {0: 0}, person_class_id=0) is None


def test_convert_label_line_too_few_tokens() -> None:
    assert convert_label_line("0 0.1 0.2", {0: 0}, person_class_id=0) is None


def test_convert_label_line_invalid_coord_count() -> None:
    line = convert_label_line(
        "0 0.1 0.2 0.3 0.4 0.5", {0: 0}, person_class_id=0
    )
    assert line is None


def test_convert_label_file_counts_skipped(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text(
        "0 0.5 0.5 0.2 0.4\n"
        "1 0.5 0.5 0.2 0.4\n"
        "\n"
        "0 0.5 0.5 0.2 0.4\n",
        encoding="utf-8",
    )
    converted, imported, skipped = convert_label_file(
        label, {0: 0}, person_class_id=0
    )
    assert imported == 2
    assert skipped == 1
    assert len(converted) == 2


def test_safe_stem_replaces_invalid_chars_and_adds_prefix() -> None:
    assert _safe_stem("My Image! 01", "imp_") == "imp_My_Image_01"


def test_safe_stem_handles_blank() -> None:
    assert _safe_stem("...", "imp_") == "imp_image"


def test_safe_stem_no_prefix_when_already_present() -> None:
    assert _safe_stem("imp_face_01", "imp_") == "imp_face_01"


def test_write_dataset_yaml(tmp_path: Path) -> None:
    path = write_dataset_yaml(tmp_path, person_class_id=3)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["nc"] == 1
    assert payload["names"][3] == "Person"
    assert payload["train"] == "images/train"


def test_import_dry_run_does_not_copy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bbox_src = tmp_path / "bbox_src"
    dataset = tmp_path / "dataset_raw"
    _write_bbox_src_export(bbox_src)

    with caplog.at_level("INFO"):
        stats = import_bbox_to_segment(bbox_src, dataset, dry_run=True)

    assert stats.images_copied == 0
    assert stats.objects_imported == 3
    assert not (dataset / "images" / "train").exists()
    assert any("Would import" in rec.message for rec in caplog.records)


def test_import_skips_existing_destination_without_overwrite(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bbox_src = tmp_path / "bbox_src"
    dataset = tmp_path / "dataset_raw"
    _write_bbox_src_export(bbox_src)
    train_dir = dataset / "images" / "train"
    train_dir.mkdir(parents=True)
    existing = train_dir / "imp_train_sample.txt"
    existing.write_text("placeholder", encoding="utf-8")

    with caplog.at_level("WARNING"):
        stats = import_bbox_to_segment(bbox_src, dataset, prefix="imp_")

    assert any("destination exists" in rec.message for rec in caplog.records)
    assert existing.read_text(encoding="utf-8") == "placeholder"
    assert "train" in stats.splits


def test_import_overwrite_replaces_existing_destination(tmp_path: Path) -> None:
    bbox_src = tmp_path / "bbox_src"
    dataset = tmp_path / "dataset_raw"
    _write_bbox_src_export(bbox_src)
    train_dir = dataset / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "imp_train_sample.txt").write_text("old", encoding="utf-8")

    stats = import_bbox_to_segment(bbox_src, dataset, prefix="imp_", overwrite=True)

    assert stats.images_copied == 3
    new_label = (train_dir / "imp_train_sample.txt").read_text(encoding="utf-8")
    assert new_label.startswith("0 ")


def test_import_skips_label_without_matching_image(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bbox_src = tmp_path / "bbox_src"
    bbox_src.mkdir()
    img_dir = bbox_src / "train" / "images"
    lbl_dir = bbox_src / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (lbl_dir / "ghost.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (bbox_src / "data.yaml").write_text(
        yaml.safe_dump({"train": "train/images", "names": {0: "face"}}),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        stats = import_bbox_to_segment(bbox_src, tmp_path / "dataset")

    assert stats.images_copied == 0
    assert any("no matching image" in rec.message for rec in caplog.records)


def test_import_skips_empty_converted_labels(tmp_path: Path) -> None:
    bbox_src = tmp_path / "bbox_src"
    bbox_src.mkdir()
    img_dir = bbox_src / "train" / "images"
    lbl_dir = bbox_src / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "stem.jpg").write_bytes(b"")
    (lbl_dir / "stem.txt").write_text("0 0.1 0.2\n", encoding="utf-8")
    (bbox_src / "data.yaml").write_text(
        yaml.safe_dump({"train": "train/images", "names": {0: "face"}}),
        encoding="utf-8",
    )

    stats = import_bbox_to_segment(bbox_src, tmp_path / "dataset")

    assert stats.images_copied == 0


def test_run_returns_zero_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_import(*_a: object, **_kw: object) -> ImportStats:
        return ImportStats()

    monkeypatch.setattr(bbox_mod, "import_bbox_to_segment", fake_import)
    assert bbox_mod._run([str(tmp_path), str(tmp_path / "dst")]) == 0


def test_run_returns_one_on_known_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_kw: object) -> ImportStats:
        raise ValueError("nope")

    monkeypatch.setattr(bbox_mod, "import_bbox_to_segment", boom)
    assert bbox_mod._run([str(tmp_path), str(tmp_path / "dst")]) == 1


def test_main_raises_systemexit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bbox_mod, "_run", lambda _argv: 7)
    with pytest.raises(SystemExit) as info:
        bbox_mod.main(["/tmp/src", "/tmp/dst"])
    assert info.value.code == 7
