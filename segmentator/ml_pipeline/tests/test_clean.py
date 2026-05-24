"""Tests for ml_pipeline.clean and ml_pipeline.cli_clean."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from ml_pipeline.clean import (
    CleanConfig,
    list_run_dirs_to_remove,
    quarantine_stems,
    run_clean,
    stems_from_errors_report,
    validate_clean_config,
)
from ml_pipeline.cli_clean import main


def _make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"keep_{split}.jpg").write_bytes(b"img")
        (root / "images" / split / f"drop_{split}.jpg").write_bytes(b"img")
        (root / "labels" / split / f"keep_{split}.txt").write_text("0 0.1 0.1 0.2 0.2\n")
        (root / "labels" / split / f"drop_{split}.txt").write_text("1 0.5 0.5 0.6 0.6\n")
    (root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names: ['Person', 'Car']\n"
    )
    return root


def test_stems_from_errors_report(tmp_path: Path):
    report_path = tmp_path / "errors.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "worst_images": [
                        {"image": "a.jpg"},
                        {"image": "b.png"},
                    ]
                }
            }
        )
    )
    assert stems_from_errors_report(report_path) == ["a", "b"]
    assert stems_from_errors_report(report_path, worst_n=1) == ["a"]


def test_quarantine_stems_moves_files(tmp_path: Path):
    root = _make_dataset(tmp_path)
    quarantine = tmp_path / "quarantine"

    result = quarantine_stems(
        root,
        ["val"],
        ["drop_val"],
        quarantine_dir=quarantine,
        action="move",
        dry_run=False,
    )

    assert len(result["moved"]) == 2
    assert not (root / "images" / "val" / "drop_val.jpg").exists()
    assert not (root / "labels" / "val" / "drop_val.txt").exists()
    assert (quarantine / "val" / "drop_val.jpg").is_file()
    assert (quarantine / "val" / "drop_val.txt").is_file()
    assert (root / "images" / "val" / "keep_val.jpg").is_file()


def test_quarantine_dry_run_leaves_files(tmp_path: Path):
    root = _make_dataset(tmp_path)
    quarantine_stems(
        root,
        ["train"],
        ["drop_train"],
        quarantine_dir=tmp_path / "q",
        action="move",
        dry_run=True,
    )
    assert (root / "images" / "train" / "drop_train.jpg").is_file()


def test_run_prefix_matches_versioned_dirs(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ("yolo26n_seg_v1", "yolo26n_seg_v1-2", "yolo26n_seg_v1-3", "other"):
        (runs / name).mkdir()
    to_remove, kept = list_run_dirs_to_remove(
        runs,
        run_prefixes=["yolo26n_seg_v1"],
        run_names=[],
        runs_all=False,
        keep_run_names=[],
        keep_latest_per_prefix=False,
    )
    assert {p.name for p in to_remove} == {
        "yolo26n_seg_v1",
        "yolo26n_seg_v1-2",
        "yolo26n_seg_v1-3",
    }
    assert kept == []


def test_keep_latest_per_prefix(tmp_path: Path):
    import os
    import time

    runs = tmp_path / "runs"
    runs.mkdir()
    old = runs / "yolo26n_seg_v1-2"
    new = runs / "yolo26n_seg_v1-3"
    old.mkdir()
    new.mkdir()
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))

    to_remove, kept = list_run_dirs_to_remove(
        runs,
        run_prefixes=["yolo26n_seg_v1"],
        run_names=[],
        runs_all=False,
        keep_run_names=[],
        keep_latest_per_prefix=True,
    )
    assert {p.name for p in kept} == {"yolo26n_seg_v1-3"}
    assert {p.name for p in to_remove} == {"yolo26n_seg_v1-2"}


def test_validate_requires_yes_for_delete(tmp_path: Path):
    root = _make_dataset(tmp_path)
    config = CleanConfig(
        dataset_root=root,
        splits=["val"],
        stems=["drop_val"],
        action="delete",
        yes=False,
    )
    with pytest.raises(ValueError, match="requires --yes"):
        validate_clean_config(config)


def test_validate_requires_selection_mode(tmp_path: Path):
    root = _make_dataset(tmp_path)
    config = CleanConfig(dataset_root=root, splits=["val"], yes=True)
    with pytest.raises(ValueError, match="selection mode"):
        validate_clean_config(config)


def test_run_clean_from_errors(tmp_path: Path):
    root = _make_dataset(tmp_path)
    errors = tmp_path / "errors.json"
    errors.write_text(
        json.dumps(
            {
                "summary": {
                    "worst_images": [{"image": "drop_val.jpg"}],
                }
            }
        )
    )
    report = run_clean(
        CleanConfig(
            dataset_root=root,
            splits=["val"],
            from_errors=errors,
            quarantine_dir=tmp_path / "q",
            yes=True,
        )
    )
    assert "samples" in report["dataset"]
    assert not (root / "images" / "val" / "drop_val.jpg").exists()


def test_run_clean_runs_only(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "yolo26n_seg_v1_val").mkdir()
    (runs / "yolo26n_seg_v1_val-2").mkdir()
    report = run_clean(
        CleanConfig(
            runs_dir=runs,
            run_prefixes=["yolo26n_seg_v1_val"],
            yes=True,
        )
    )
    assert len(report["runs"]["removed"]) == 2
    assert not any(runs.iterdir())


def test_cli_dry_run_dataset(tmp_path: Path, capsys):
    root = _make_dataset(tmp_path)
    code = main(
        [
            str(root),
            "--val",
            "--stems",
            "drop_val",
            "--dry-run",
        ]
    )
    assert code == 0
    assert (root / "images" / "val" / "drop_val.jpg").is_file()
    captured = capsys.readouterr()
    assert "drop_val" in captured.out


def test_cli_runs_requires_yes(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "old_run").mkdir()
    code = main(
        [
            "--runs-dir",
            str(runs),
            "--run-names",
            "old_run",
        ]
    )
    assert code == 1


def test_stems_from_errors_report_skips_invalid_entries(tmp_path: Path):
    report_path = tmp_path / "errors.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "worst_images": [
                        "not_a_dict",
                        {"score": 0.5},
                        {"image": "ok.jpg"},
                    ]
                }
            }
        )
    )
    assert stems_from_errors_report(report_path) == ["ok"]


def test_quarantine_stems_unknown_split_raises(tmp_path: Path):
    root = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        quarantine_stems(
            root,
            ["bogus"],
            ["drop_val"],
            quarantine_dir=tmp_path / "q",
            action="move",
            dry_run=False,
        )


def test_quarantine_stems_records_missing(tmp_path: Path):
    root = _make_dataset(tmp_path)
    result = quarantine_stems(
        root,
        ["val"],
        ["does_not_exist"],
        quarantine_dir=tmp_path / "q",
        action="move",
        dry_run=False,
    )
    assert result["missing"] == [{"split": "val", "stem": "does_not_exist"}]
    assert result["moved"] == []


def test_quarantine_stems_delete_action_unlinks(tmp_path: Path):
    root = _make_dataset(tmp_path)
    result = quarantine_stems(
        root,
        ["val"],
        ["drop_val"],
        quarantine_dir=None,
        action="delete",
        dry_run=False,
    )
    assert result["quarantine_dir"] is None
    assert not (root / "images" / "val" / "drop_val.jpg").exists()
    assert not (root / "labels" / "val" / "drop_val.txt").exists()
    assert all(entry["destination"] == "" for entry in result["moved"])


def test_quarantine_stems_overwrites_existing_destination(tmp_path: Path):
    root = _make_dataset(tmp_path)
    quarantine = tmp_path / "quarantine"
    (quarantine / "val").mkdir(parents=True)
    (quarantine / "val" / "drop_val.jpg").write_bytes(b"old")
    quarantine_stems(
        root,
        ["val"],
        ["drop_val"],
        quarantine_dir=quarantine,
        action="move",
        dry_run=False,
    )
    assert (quarantine / "val" / "drop_val.jpg").read_bytes() != b"old"


def test_stems_in_splits_returns_sorted_unique(tmp_path: Path):
    from ml_pipeline.clean import stems_in_splits

    root = _make_dataset(tmp_path)
    stems = stems_in_splits(root, ["train", "val"])
    assert stems == sorted(stems)
    assert "keep_train" in stems
    assert "drop_val" in stems


def test_stems_in_splits_skips_missing_split_dirs(tmp_path: Path):
    from ml_pipeline.clean import stems_in_splits

    root = _make_dataset(tmp_path)
    (root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names: ['Person', 'Car']\n"
    )
    stems = stems_in_splits(root, ["test"])
    assert stems == []


def test_remove_label_caches_returns_paths(tmp_path: Path):
    from ml_pipeline.clean import remove_label_caches

    root = tmp_path / "ds"
    labels = root / "labels" / "train"
    labels.mkdir(parents=True)
    cache = labels / "train.cache"
    cache.write_bytes(b"")

    removed_dry = remove_label_caches(root, dry_run=True)
    assert str(cache) in removed_dry
    assert cache.exists()

    removed = remove_label_caches(root, dry_run=False)
    assert str(cache) in removed
    assert not cache.exists()


def test_remove_label_caches_no_labels_dir_returns_empty(tmp_path: Path):
    from ml_pipeline.clean import remove_label_caches

    assert remove_label_caches(tmp_path, dry_run=False) == []


def test_list_run_dirs_to_remove_missing_runs_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Runs directory"):
        list_run_dirs_to_remove(
            tmp_path / "missing",
            run_prefixes=[],
            run_names=[],
            runs_all=False,
            keep_run_names=[],
            keep_latest_per_prefix=False,
        )


def test_list_run_dirs_to_remove_runs_all_with_keep(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "a").mkdir()
    (runs / "b").mkdir()
    to_remove, kept = list_run_dirs_to_remove(
        runs,
        run_prefixes=[],
        run_names=[],
        runs_all=True,
        keep_run_names=["a"],
        keep_latest_per_prefix=False,
    )
    assert {p.name for p in to_remove} == {"b"}
    assert {p.name for p in kept} == {"a"}


def test_list_run_dirs_to_remove_run_names(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "keep_run").mkdir()
    (runs / "drop_run").mkdir()
    to_remove, _kept = list_run_dirs_to_remove(
        runs,
        run_prefixes=[],
        run_names=["drop_run"],
        runs_all=False,
        keep_run_names=[],
        keep_latest_per_prefix=False,
    )
    assert {p.name for p in to_remove} == {"drop_run"}


def test_clean_run_dirs_dry_run_keeps_files(tmp_path: Path):
    from ml_pipeline.clean import clean_run_dirs

    runs = tmp_path / "runs"
    target = runs / "old"
    target.mkdir(parents=True)

    removed = clean_run_dirs(runs, [target], dry_run=True)
    assert removed == [str(target)]
    assert target.exists()


def test_validate_nothing_to_clean_raises():
    with pytest.raises(ValueError, match="Nothing to clean"):
        validate_clean_config(CleanConfig())


def test_validate_dataset_root_required(tmp_path: Path):
    config = CleanConfig(splits=["val"], stems=["x"], yes=True)
    with pytest.raises(ValueError, match="dataset_root is required"):
        validate_clean_config(config)


def test_validate_dataset_root_not_dir(tmp_path: Path):
    bogus = tmp_path / "not-a-dir"
    bogus.write_text("file", encoding="utf-8")
    config = CleanConfig(
        dataset_root=bogus,
        splits=["val"],
        stems=["x"],
        yes=True,
    )
    with pytest.raises(FileNotFoundError, match="not a directory"):
        validate_clean_config(config)


def test_validate_from_errors_file_required(tmp_path: Path):
    root = _make_dataset(tmp_path)
    config = CleanConfig(
        dataset_root=root,
        splits=["val"],
        from_errors=tmp_path / "missing.json",
        yes=True,
    )
    with pytest.raises(FileNotFoundError, match="Error report not found"):
        validate_clean_config(config)


def test_validate_only_one_selection_mode(tmp_path: Path):
    root = _make_dataset(tmp_path)
    errors = tmp_path / "errors.json"
    errors.write_text(json.dumps({"summary": {"worst_images": []}}))
    config = CleanConfig(
        dataset_root=root,
        splits=["val"],
        stems=["x"],
        from_errors=errors,
        yes=True,
    )
    with pytest.raises(ValueError, match="only one of"):
        validate_clean_config(config)


def test_validate_all_in_split_requires_yes(tmp_path: Path):
    root = _make_dataset(tmp_path)
    config = CleanConfig(dataset_root=root, splits=["val"], all_in_split=True)
    with pytest.raises(ValueError, match="--all-in-split requires"):
        validate_clean_config(config)


def test_validate_runs_all_excludes_explicit(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    config = CleanConfig(
        runs_dir=runs,
        runs_all=True,
        run_names=["x"],
        yes=True,
    )
    with pytest.raises(ValueError, match="not both"):
        validate_clean_config(config)


def test_run_clean_remove_label_cache_only(tmp_path: Path):
    root = _make_dataset(tmp_path)
    (root / "labels" / "train" / "train.cache").write_bytes(b"")
    report = run_clean(
        CleanConfig(
            dataset_root=root,
            remove_label_cache=True,
            yes=True,
        )
    )
    assert report["dataset"]["label_caches_removed"]


def test_run_clean_all_in_split(tmp_path: Path):
    root = _make_dataset(tmp_path)
    report = run_clean(
        CleanConfig(
            dataset_root=root,
            splits=["val"],
            all_in_split=True,
            quarantine_dir=tmp_path / "q",
            yes=True,
        )
    )
    moved = report["dataset"]["samples"]["moved"]
    assert moved
    assert all(entry["split"] == "val" for entry in moved)


def test_cli_writes_report_json(tmp_path: Path):
    root = _make_dataset(tmp_path)
    report_path = tmp_path / "report.json"
    code = main(
        [
            str(root),
            "--train",
            "--stems",
            "drop_train",
            "--dry-run",
            "--report-json",
            str(report_path),
        ]
    )
    assert code == 0
    assert report_path.exists()


def test_cli_train_split_flag(tmp_path: Path):
    root = _make_dataset(tmp_path)
    code = main(
        [
            str(root),
            "--train",
            "--stems",
            "drop_train",
            "--dry-run",
        ]
    )
    assert code == 0


def test_cli_handles_invalid_config(tmp_path: Path, capsys):
    code = main(["/nonexistent/root", "--train", "--stems", "x", "--yes"])
    assert code == 1
    assert "Error" in capsys.readouterr().err
