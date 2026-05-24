"""Shared pytest fixtures for ml_pipeline tests."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Minimal YOLO tree: train has one labeled image and one image-only; val has one labeled pair."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "Person", 1: "Car"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "images" / "train" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "images" / "train" / "a.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.15 0.2\n",
        encoding="utf-8",
    )
    (tmp_path / "images" / "train" / "b.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "images" / "val" / "c.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "images" / "val" / "c.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.15 0.2\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def fake_cv2_numpy(monkeypatch: pytest.MonkeyPatch):
    """Install fake cv2 and numpy modules (see tests/fakes.py)."""
    from fakes import install_fake_cv2_numpy

    return install_fake_cv2_numpy(monkeypatch)
