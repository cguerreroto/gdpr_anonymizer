"""Tests for ml_pipeline.checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

from ml_pipeline.checkpoints import (
    best_pt_in_run_dir,
    find_best_pt_under_project,
    missing_weights_message,
)


def test_best_pt_in_run_dir_returns_path_when_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "yolo26n_seg_v1-3"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    best.write_bytes(b"")
    assert best_pt_in_run_dir(run_dir) == best


def test_best_pt_in_run_dir_returns_none_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    assert best_pt_in_run_dir(run_dir) is None


def test_find_best_pt_under_project_sorts_newest_first(tmp_path: Path) -> None:
    project = tmp_path / "runs"
    older = project / "run_a" / "weights"
    newer = project / "run_b" / "weights"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    old_best = older / "best.pt"
    new_best = newer / "best.pt"
    old_best.write_bytes(b"")
    new_best.write_bytes(b"")
    os.utime(old_best, (1_000_000, 1_000_000))
    os.utime(new_best, (2_000_000, 2_000_000))
    found = find_best_pt_under_project(project)
    assert found[0] == new_best
    assert found[1] == old_best


def test_missing_weights_message_lists_candidates(tmp_path: Path) -> None:
    project = tmp_path / "runs"
    best = project / "yolo26n_seg_v1-3" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"")
    requested = project / "yolo26n_seg_v1" / "weights" / "best.pt"
    msg = missing_weights_message(requested, project)
    assert "Missing weights" in msg
    assert "yolo26n_seg_v1-3" in msg
    assert str(best.resolve()) in msg
    assert "validate_weights" in msg


def test_missing_weights_message_when_no_checkpoints_exist(tmp_path: Path) -> None:
    project = tmp_path / "runs"
    project.mkdir()
    msg = missing_weights_message(project / "nope.pt", project)
    assert "No weights/best.pt files found" in msg
