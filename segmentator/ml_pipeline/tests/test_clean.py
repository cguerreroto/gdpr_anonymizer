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
