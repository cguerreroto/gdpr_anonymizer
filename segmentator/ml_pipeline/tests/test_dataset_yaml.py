"""Tests for ml_pipeline.dataset_yaml."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from ml_pipeline.cli_dataset_yaml import main as cli_main
from ml_pipeline.dataset_yaml import (
    audit_dataset_yaml,
    build_fixed_config,
    carve_val_from_train,
    collect_class_ids,
    load_dataset_yaml,
    run_dataset_yaml_hygiene,
    write_dataset_yaml,
)


def test_collect_class_ids(sample_dataset: Path) -> None:
    assert collect_class_ids(sample_dataset) == {0}


def test_audit_missing_nc_and_names_gap(tmp_path: Path) -> None:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "images" / "train" / "a.txt").write_text(
        "1 0.1 0.1 0.2 0.1 0.15 0.2\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "names": {0: "Person"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg = load_dataset_yaml(tmp_path)
    audit = audit_dataset_yaml(cfg, {1})
    assert any("nc is not set" in issue for issue in audit["issues"])
    assert any("names missing" in issue for issue in audit["issues"])


def test_build_fixed_config_adds_nc_and_names() -> None:
    cfg = {"path": ".", "train": "images/train", "names": {0: "Person"}}
    fixed, changes = build_fixed_config(cfg, {0, 1})
    assert fixed["nc"] == 2
    assert fixed["names"][0] == "Person"
    assert fixed["names"][1] == "Car"
    assert any("nc" in c for c in changes)


def test_run_hygiene_apply_writes_yaml(sample_dataset: Path) -> None:
    (sample_dataset / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "Person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = run_dataset_yaml_hygiene(
        sample_dataset,
        apply_fixes=True,
        dry_run=False,
    )
    assert report["proposed_changes"]
    cfg = load_dataset_yaml(sample_dataset)
    assert cfg["nc"] == 1
    assert cfg["names"][0] == "Person"


def test_carve_val_fraction_dry_run(sample_dataset: Path) -> None:
    before = list((sample_dataset / "images" / "val").glob("*.jpg"))
    report = carve_val_from_train(
        sample_dataset,
        fraction=0.5,
        seed=0,
        dry_run=True,
        move=False,
    )
    assert report["selected_count"] >= 1
    assert list((sample_dataset / "images" / "val").glob("*.jpg")) == before


def test_carve_val_fraction_copy(sample_dataset: Path) -> None:
    carve_val_from_train(
        sample_dataset,
        fraction=0.5,
        seed=0,
        dry_run=False,
        move=False,
    )
    assert (sample_dataset / "images" / "val").is_dir()
    val_images = list((sample_dataset / "images" / "val").glob("*.jpg"))
    assert val_images


def test_cli_strict_without_apply(sample_dataset: Path) -> None:
    (sample_dataset / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "val": "images/val"},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    code = cli_main([str(sample_dataset), "--strict"])
    assert code == 1


def test_cli_apply_clears_strict_issues(sample_dataset: Path) -> None:
    (sample_dataset / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "val": "images/val"},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    code = cli_main([str(sample_dataset), "--apply"])
    assert code == 0
    cfg = load_dataset_yaml(sample_dataset)
    assert "nc" in cfg
    assert "names" in cfg


def test_carve_fraction_invalid(sample_dataset: Path) -> None:
    with pytest.raises(ValueError, match="fraction"):
        carve_val_from_train(
            sample_dataset,
            fraction=1.5,
            seed=0,
            dry_run=True,
            move=False,
        )
