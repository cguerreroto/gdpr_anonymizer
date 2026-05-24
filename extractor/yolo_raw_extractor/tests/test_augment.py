"""Tests for orchestration helpers in yolo_raw_extractor.augment."""

from __future__ import annotations

import json
import random
from pathlib import Path
import pytest
import yaml
import yolo_raw_extractor.augment as aug
import numpy as np


def _build_minimal_dataset(root: Path, with_segments: bool = False) -> None:
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "frame.jpg").write_bytes(b"")
    (lbl_dir / "frame.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
    )
    cfg = {
        "path": ".",
        "train": "images/train",
        "names": {0: "person"},
        "name": "demo",
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    if with_segments:
        seg_dir = root / "segments" / "train" / "person"
        seg_dir.mkdir(parents=True)
        asset = np.zeros((20, 20, 4), dtype=np.uint8)
        asset[:, :, 3] = 255
        (seg_dir / "asset.png").write_bytes(b"")
        (seg_dir / "asset.json").write_text(
            json.dumps(
                {
                    "class_id": 0,
                    "class_name": "person",
                    "points": [[0, 0], [20, 0], [20, 20], [0, 20]],
                    "crop": {"x": 0, "y": 0, "width": 20, "height": 20},
                    "source_image": "frame.jpg",
                }
            ),
            encoding="utf-8",
        )


def _make_segment_asset(
    size: int = 12,
    *,
    channels: int = 4,
    alpha: int = 255,
    points: np.ndarray | None = None,
    class_id: int = 0,
) -> aug.SegmentAsset:
    if channels >= 4:
        image = np.zeros((size, size, 4), dtype=np.uint8)
        image[:, :, :3] = 128
        image[:, :, 3] = alpha
    else:
        image = np.zeros((size, size, channels), dtype=np.uint8)
    if points is None:
        points = np.array(
            [[1.0, 1.0], [float(size - 1), 1.0], [float(size - 1), float(size - 1)]],
            dtype=np.float32,
        )
    return aug.SegmentAsset(class_id, "person", image, points)


def test_place_random_segment_places_valid_asset() -> None:
    canvas = np.full((80, 80, 3), 50, dtype=np.uint8)
    asset = _make_segment_asset(size=12)
    rng = random.Random(1)

    success, entry = aug.place_random_segment(canvas, [asset], rng)

    assert success is True
    assert entry is not None
    assert entry.class_id == 0
    assert entry.polygon.shape[0] >= 3
    assert not np.all(canvas == 50)


def test_place_random_segment_rejects_asset_without_alpha() -> None:
    canvas = np.zeros((40, 40, 3), dtype=np.uint8)
    asset = _make_segment_asset(size=10, channels=3)

    success, entry = aug.place_random_segment(canvas, [asset], random.Random(0))

    assert success is False
    assert entry is None


def test_place_random_segment_rejects_patch_larger_than_canvas() -> None:
    canvas = np.zeros((8, 8, 3), dtype=np.uint8)
    asset = _make_segment_asset(size=40)

    success, entry = aug.place_random_segment(canvas, [asset], random.Random(0))

    assert success is False
    assert entry is None


def test_place_random_segment_rejects_zero_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas = np.zeros((40, 40, 3), dtype=np.uint8)
    asset = _make_segment_asset(size=12, alpha=0)
    transparent = np.zeros((14, 14, 4), dtype=np.uint8)

    def fake_rotate(_patch: np.ndarray, _angle: float) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return transparent, matrix

    monkeypatch.setattr(aug, "rotate_with_bounds", fake_rotate)

    success, entry = aug.place_random_segment(canvas, [asset], random.Random(0))

    assert success is False
    assert entry is None


def test_place_random_segment_rejects_empty_points() -> None:
    canvas = np.zeros((40, 40, 3), dtype=np.uint8)
    asset = aug.SegmentAsset(
        0,
        "person",
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((0, 2), dtype=np.float32),
    )

    success, entry = aug.place_random_segment(canvas, [asset], random.Random(0))

    assert success is False
    assert entry is None


def test_insert_segments_adds_label_entries() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assets = [_make_segment_asset(size=12)]
    base_entries = [
        aug.LabelEntry(
            class_id=0,
            polygon=np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0]], dtype=np.float32),
        )
    ]

    canvas, inserted, entries = aug.insert_segments(
        image, base_entries, assets, random.Random(0)
    )

    assert inserted is True
    assert len(entries) > len(base_entries)
    assert canvas.shape == image.shape


def test_add_offwhite_rectangle_returns_none_when_patch_exceeds_canvas() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    polygon = np.array(
        [[0.0, 0.0], [11.0, 0.0], [11.0, 11.0], [0.0, 11.0]], dtype=np.float32
    )
    entries = [aug.LabelEntry(class_id=0, polygon=polygon)]

    result = aug.add_offwhite_rectangle(image, entries, random.Random(0))

    assert result is None


def test_augment_image_logs_when_insertion_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    img_path = tmp_path / "frame.jpg"
    lbl_path = tmp_path / "frame.txt"
    lbl_path.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
    monkeypatch.setattr(
        aug.cv2, "imread", lambda _path, _flags: np.zeros((40, 40, 3), np.uint8)
    )
    monkeypatch.setattr(aug.cv2, "imwrite", lambda _path, _img: True)

    with caplog.at_level("INFO"):
        aug.augment_image(
            img_path,
            lbl_path,
            tmp_path,
            [],
            random.Random(0),
            np.random.default_rng(0),
        )

    assert any("Insertion skipped" in record.message for record in caplog.records)


def test_augment_dataset_missing_labels_directory_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}}
    (src / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with caplog.at_level("WARNING"):
        aug.augment_dataset(src, tmp_path / "dst", seed=0)

    assert any("Missing labels directory" in record.message for record in caplog.records)


def test_augment_dataset_skips_empty_label_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = tmp_path / "src"
    img_dir = src / "images" / "train"
    lbl_dir = src / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "frame.jpg").write_bytes(b"")
    (lbl_dir / "frame.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
    )
    (lbl_dir / "empty.txt").write_text("", encoding="utf-8")
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}}
    (src / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with caplog.at_level("DEBUG"):
        aug.augment_dataset(src, tmp_path / "dst", seed=0)

    assert any("Skipping empty label file" in record.message for record in caplog.records)


def test_augment_dataset_skips_label_without_matching_image(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = tmp_path / "src"
    img_dir = src / "images" / "train"
    lbl_dir = src / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (lbl_dir / "orphan.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
    )
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}}
    (src / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with caplog.at_level("WARNING"):
        aug.augment_dataset(src, tmp_path / "dst", seed=0)

    assert any("missing image match" in record.message for record in caplog.records)


def test_parse_args_minimal() -> None:
    args = aug.parse_args(["/tmp/in", "/tmp/out"])
    assert args.dataset_dir == Path("/tmp/in")
    assert args.output_dir == Path("/tmp/out")
    assert args.seed == 0


def test_parse_args_with_seed() -> None:
    args = aug.parse_args(["/tmp/in", "/tmp/out", "--seed", "42"])
    assert args.seed == 42


def test_uniform_with_gap_lower_branch() -> None:
    rng = random.Random()
    rng.random = lambda: 0.4  # type: ignore[assignment]
    rng.uniform = lambda a, _b: a  # type: ignore[assignment]
    assert aug._uniform_with_gap(rng) == 0.5


def test_uniform_with_gap_upper_branch() -> None:
    rng = random.Random()
    rng.random = lambda: 0.7  # type: ignore[assignment]
    rng.uniform = lambda a, _b: a  # type: ignore[assignment]
    assert aug._uniform_with_gap(rng) == 1.2


def test_clone_entries_independent_copies() -> None:
    entry = aug.LabelEntry(class_id=2, polygon=np.array([[1.0, 2.0]], dtype=np.float32))
    clones = aug.clone_entries([entry])
    clones[0].polygon[0, 0] = 9.0
    assert entry.polygon[0, 0] == 1.0
    assert clones[0].class_id == 2


def test_ensure_output_layout_creates_split_dirs(tmp_path: Path) -> None:
    images_root = aug.ensure_output_layout(tmp_path)
    for split in ("train", "val", "test"):
        assert (images_root / split).is_dir()


def test_adjust_saturation_exposure_returns_image_shape() -> None:
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    rng = random.Random(0)
    result = aug.adjust_saturation_exposure(image, rng)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_occlude_segments_empty_entries_returns_copy() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    result = aug.occlude_segments(image, [], rng, np_rng)
    assert np.array_equal(result, image)
    assert result is not image


def test_occlude_segments_replaces_pixels_with_noise() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    polygon = np.array(
        [[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]], dtype=np.float32
    )
    entries = [aug.LabelEntry(class_id=0, polygon=polygon)]
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    result = aug.occlude_segments(image, entries, rng, np_rng)
    assert result.shape == image.shape
    assert np.any(result != image)


def test_rotate_with_bounds_changes_canvas_size() -> None:
    patch = np.zeros((10, 20, 4), dtype=np.uint8)
    rotated, matrix = aug.rotate_with_bounds(patch, 45.0)
    assert matrix.shape == (2, 3)
    assert rotated.shape[0] >= patch.shape[0]
    assert rotated.shape[1] >= patch.shape[1]


def test_apply_affine_transform_identity_matrix() -> None:
    points = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    result = aug.apply_affine_transform(points, matrix)
    np.testing.assert_allclose(result, points)


def test_insert_segments_no_assets_returns_copy_and_false() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    entries: list[aug.LabelEntry] = []
    canvas, inserted, result_entries = aug.insert_segments(
        image, entries, [], random.Random(0)
    )
    assert inserted is False
    assert result_entries == entries
    assert np.array_equal(canvas, image)


def test_add_offwhite_rectangle_no_entries_returns_none() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert aug.add_offwhite_rectangle(image, [], random.Random(0)) is None


def test_add_offwhite_rectangle_entry_smaller_than_canvas() -> None:
    image = np.full((100, 100, 3), 50, dtype=np.uint8)
    polygon = np.array(
        [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]], dtype=np.float32
    )
    entries = [aug.LabelEntry(class_id=0, polygon=polygon)]
    rng = random.Random(0)
    result = aug.add_offwhite_rectangle(image, entries, rng)
    assert result is not None
    assert result.shape == image.shape


def test_load_segment_assets_missing_directory_returns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        result = aug.load_segment_assets(tmp_path, {0: "person"})
    assert result == []
    assert any("Segments directory not found" in r.message for r in caplog.records)


def test_load_segment_assets_loads_valid_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seg_dir = tmp_path / "segments" / "train" / "person"
    seg_dir.mkdir(parents=True)
    (seg_dir / "asset.png").write_bytes(b"")
    (seg_dir / "asset.json").write_text(
        json.dumps({"points": [[0, 0], [10, 0], [5, 10]]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        aug.cv2, "imread", lambda _path, _flags: np.zeros((10, 10, 4), np.uint8)
    )

    assets = aug.load_segment_assets(tmp_path, {0: "person"})

    assert len(assets) == 1
    assert assets[0].class_id == 0
    assert assets[0].class_name == "person"


def test_load_segment_assets_skips_unreadable_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seg_dir = tmp_path / "segments" / "train" / "person"
    seg_dir.mkdir(parents=True)
    (seg_dir / "asset.png").write_bytes(b"")
    (seg_dir / "asset.json").write_text(
        json.dumps({"points": [[0, 0], [10, 0], [5, 10]]}), encoding="utf-8"
    )
    monkeypatch.setattr(aug.cv2, "imread", lambda _path, _flags: None)

    assert aug.load_segment_assets(tmp_path, {0: "person"}) == []


def test_save_variant_writes_image_and_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []

    def fake_imwrite(path: str, _img: np.ndarray) -> bool:
        written.append(path)
        Path(path).write_bytes(b"")
        return True

    monkeypatch.setattr(aug.cv2, "imwrite", fake_imwrite)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    entries = [
        aug.LabelEntry(class_id=0, polygon=np.array([[0.0, 0.0]], dtype=np.float32))
    ]
    aug.save_variant(image, entries, 10, 10, "stem", "orig", ".jpg", tmp_path)

    assert (tmp_path / "stem_orig.jpg").exists()
    assert (tmp_path / "stem_orig.txt").exists()
    assert written


def test_save_variant_imwrite_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aug.cv2, "imwrite", lambda _p, _i: False)
    with pytest.raises(RuntimeError, match="Failed to write augmented image"):
        aug.save_variant(
            np.zeros((4, 4, 3), np.uint8), [], 4, 4, "s", "x", ".jpg", tmp_path
        )


def test_copy_dataset_metadata(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}, "name": "demo"}
    (src / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    aug.copy_dataset_metadata(src, dst)

    out = yaml.safe_load((dst / "dataset.yaml").read_text(encoding="utf-8"))
    assert out["path"] == "."
    assert out["train"] == "images/train"
    assert out["val"] == "images/val"
    assert out["test"] == "images/test"
    assert out["name"] == "demo_augmented"


def test_copy_dataset_metadata_missing_yaml_raises(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        aug.copy_dataset_metadata(src, dst)


def test_augment_image_missing_image_logs_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(aug.cv2, "imread", lambda _path, _flags: None)
    with caplog.at_level("WARNING"):
        aug.augment_image(
            tmp_path / "missing.jpg",
            tmp_path / "missing.txt",
            tmp_path,
            [],
            random.Random(0),
            np.random.default_rng(0),
        )
    assert any("Unable to read image" in r.message for r in caplog.records)


def test_augment_image_missing_label_logs_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        aug.cv2, "imread", lambda _p, _f: np.zeros((10, 10, 3), np.uint8)
    )
    with caplog.at_level("WARNING"):
        aug.augment_image(
            tmp_path / "image.jpg",
            tmp_path / "missing.txt",
            tmp_path,
            [],
            random.Random(0),
            np.random.default_rng(0),
        )
    assert any("Label file missing" in r.message for r in caplog.records)


def test_augment_dataset_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _build_minimal_dataset(src)
    monkeypatch.setattr(
        aug.cv2, "imread", lambda _p, _f: np.zeros((40, 40, 3), np.uint8)
    )

    written: list[str] = []

    def fake_imwrite(path: str, _img: np.ndarray) -> bool:
        written.append(path)
        Path(path).write_bytes(b"")
        return True

    monkeypatch.setattr(aug.cv2, "imwrite", fake_imwrite)

    aug.augment_dataset(src, dst, seed=0)

    assert (dst / "seed.txt").read_text(encoding="utf-8").strip() == "0"
    assert (dst / "dataset.yaml").exists()
    suffixes = {Path(p).stem.split("_", 1)[-1] for p in written}
    assert "orig" in suffixes
    assert any(s.startswith("col") for s in suffixes)


def test_augment_dataset_missing_labels_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    img_dir = src / "images" / "train"
    img_dir.mkdir(parents=True)
    cfg = {"path": ".", "train": "images/train", "names": {0: "person"}}
    (src / "dataset.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with caplog.at_level("INFO"):
        aug.augment_dataset(src, tmp_path / "dst", seed=0)

    assert any("No labels found" in r.message for r in caplog.records)


def test_run_returns_zero_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aug, "augment_dataset", lambda *_a, **_kw: None)
    assert aug._run([str(tmp_path), str(tmp_path / "out")]) == 0


def test_run_returns_one_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(aug, "augment_dataset", boom)
    assert aug._run([str(tmp_path), str(tmp_path / "out")]) == 1


def test_main_raises_systemexit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aug, "_run", lambda _argv: 5)
    with pytest.raises(SystemExit) as info:
        aug.main(["/tmp/in", "/tmp/out"])
    assert info.value.code == 5
