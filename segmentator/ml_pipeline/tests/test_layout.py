"""Tests for ml_pipeline.layout."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from ml_pipeline.layout import (
    audit_split,
    load_split_dirs,
    normalize_split,
    run_audit_and_normalize,
    write_report,
)


def test_load_split_dirs_valid(sample_dataset: Path) -> None:
    splits = load_split_dirs(sample_dataset)
    assert set(splits) == {"train", "val"}
    assert splits["train"] == (sample_dataset / "images" / "train").resolve()
    assert splits["val"] == (sample_dataset / "images" / "val").resolve()


def test_load_split_dirs_missing_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        load_split_dirs(tmp_path)


def test_load_split_dirs_no_splits(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no train, val, or test"):
        load_split_dirs(tmp_path)


def test_audit_split_images_missing_label(sample_dataset: Path) -> None:
    audit = audit_split(
        sample_dataset,
        "train",
        sample_dataset / "images" / "train",
    )
    assert audit.images_missing_label == ["b"]
    assert audit.labels_missing_image == []


def test_symlink_normalize_creates_labels(sample_dataset: Path) -> None:
    report = run_audit_and_normalize(
        sample_dataset,
        mode="symlink",
        dry_run=False,
        force=False,
    )

    train_labels = sample_dataset / "labels" / "train" / "a.txt"
    assert train_labels.is_symlink()
    assert train_labels.resolve() == (sample_dataset / "images" / "train" / "a.txt").resolve()

    val_labels = sample_dataset / "labels" / "val" / "c.txt"
    assert val_labels.is_symlink()

    data = report["splits"]["train"]
    assert "b" in data["images_missing_label"]
    assert data["labels_missing_image"] == []


def test_copy_mode_duplicates_label(sample_dataset: Path) -> None:
    run_audit_and_normalize(
        sample_dataset,
        mode="copy",
        dry_run=False,
        force=False,
    )
    src = sample_dataset / "images" / "train" / "a.txt"
    dest = sample_dataset / "labels" / "train" / "a.txt"
    assert src.is_file()
    assert dest.is_file()
    assert not dest.is_symlink()
    assert dest.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_force_replaces_existing_symlink(sample_dataset: Path) -> None:
    run_audit_and_normalize(
        sample_dataset,
        mode="symlink",
        dry_run=False,
        force=False,
    )
    dest = sample_dataset / "labels" / "train" / "a.txt"
    dest.unlink()
    dest.write_text("stale\n", encoding="utf-8")

    audit = audit_split(
        sample_dataset,
        "train",
        sample_dataset / "images" / "train",
    )
    normalize_split(audit, mode="symlink", dry_run=False, force=True)
    assert dest.is_symlink()
    assert dest.resolve() == (sample_dataset / "images" / "train" / "a.txt").resolve()


def test_dry_run_does_not_create_labels(sample_dataset: Path) -> None:
    run_audit_and_normalize(
        sample_dataset,
        mode="symlink",
        dry_run=True,
        force=False,
    )
    assert not (sample_dataset / "labels").exists()


def test_empty_split_skips_labels_dir(tmp_path: Path) -> None:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "test").mkdir(parents=True)
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "test": "images/test"},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "images" / "train" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "images" / "train" / "a.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.15 0.2\n",
        encoding="utf-8",
    )

    run_audit_and_normalize(tmp_path, mode="symlink", dry_run=False, force=False)
    assert (tmp_path / "labels" / "train" / "a.txt").is_symlink()
    assert not (tmp_path / "labels" / "test").exists()


def test_write_report(tmp_path: Path, sample_dataset: Path) -> None:
    report = run_audit_and_normalize(
        sample_dataset,
        mode="symlink",
        dry_run=True,
        force=False,
    )
    out = tmp_path / "audit.json"
    write_report(out, report)
    assert out.is_file()
    assert '"dataset_root"' in out.read_text(encoding="utf-8")


def test_load_split_dirs_with_explicit_path_field(tmp_path: Path) -> None:
    nested = tmp_path / "data"
    (nested / "images" / "train").mkdir(parents=True)
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": "data", "train": "images/train"},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    splits = load_split_dirs(tmp_path)
    assert splits["train"] == (nested / "images" / "train").resolve()


def test_load_split_dirs_with_absolute_path_field(tmp_path: Path) -> None:
    nested = tmp_path / "abs"
    (nested / "images" / "train").mkdir(parents=True)
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": str(nested), "train": "images/train"},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    splits = load_split_dirs(tmp_path)
    assert splits["train"] == (nested / "images" / "train").resolve()


def test_load_split_dirs_with_absolute_split_path(tmp_path: Path) -> None:
    abs_train = tmp_path / "abs_train"
    abs_train.mkdir()
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump({"train": str(abs_train)}, sort_keys=False),
        encoding="utf-8",
    )
    splits = load_split_dirs(tmp_path)
    assert splits["abs_train"] == abs_train.resolve()


def test_audit_split_finds_labels_in_separate_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    (label_dir / "shot.txt").write_text(
        "0 0.1 0.1 0.2 0.2 0.3 0.3\n", encoding="utf-8"
    )

    audit = audit_split(tmp_path, "train", image_dir)

    assert audit.txt_in_labels_dir
    assert audit.images_missing_label == []


def test_normalize_split_skips_when_no_image_dir(tmp_path: Path) -> None:
    from ml_pipeline.layout import SplitAudit

    audit = SplitAudit(
        split="train",
        image_dir=tmp_path / "missing",
        labels_out_dir=tmp_path / "labels" / "train",
    )
    result = normalize_split(audit, mode="symlink", dry_run=False, force=False)
    assert result is audit
    assert result.labels_created == []


def test_normalize_split_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "shot.txt").write_text("0", encoding="utf-8")

    audit = audit_split(tmp_path, "train", image_dir)

    def boom(*_args: object, **_kw: object) -> None:
        raise OSError("forced failure")

    monkeypatch.setattr("ml_pipeline.layout._ensure_copy", boom)
    normalize_split(audit, mode="copy", dry_run=False, force=False)

    assert audit.labels_failed
    assert audit.labels_failed[0][0] == "shot"


def test_normalize_split_unknown_mode_raises(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "shot.txt").write_text("0", encoding="utf-8")
    audit = audit_split(tmp_path, "train", image_dir)
    with pytest.raises(ValueError, match="Unknown mode"):
        normalize_split(audit, mode="bogus", dry_run=False, force=False)  # type: ignore[arg-type]


def test_normalize_split_move_mode_relocates_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    (image_dir / "shot.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3", encoding="utf-8")
    audit = audit_split(tmp_path, "train", image_dir)

    normalize_split(audit, mode="move", dry_run=False, force=False)

    assert (tmp_path / "labels" / "train" / "shot.txt").is_file()
    assert not (image_dir / "shot.txt").exists()


def test_normalize_split_skips_existing_when_not_force(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "shot.txt").write_text("0", encoding="utf-8")
    (label_dir / "shot.txt").write_text("kept", encoding="utf-8")
    audit = audit_split(tmp_path, "train", image_dir)

    normalize_split(audit, mode="copy", dry_run=False, force=False)

    assert "shot" in audit.labels_skipped_existing
    assert (label_dir / "shot.txt").read_text(encoding="utf-8") == "kept"
