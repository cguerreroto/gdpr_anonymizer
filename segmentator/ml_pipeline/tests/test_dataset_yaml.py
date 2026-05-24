"""Tests for ml_pipeline.dataset_yaml."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from ml_pipeline.cli_dataset_yaml import main as cli_main
from ml_pipeline.dataset_yaml import (
    _label_paths_for_split,
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


def test_cli_carve_writes_report_json(
    sample_dataset: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"
    code = cli_main(
        [
            str(sample_dataset),
            "--apply",
            "--carve-val-fraction",
            "0.5",
            "--report-json",
            str(report_path),
        ]
    )
    assert code == 0
    assert report_path.exists()


def test_cli_carve_dry_run_reports(sample_dataset: Path) -> None:
    code = cli_main(
        [
            str(sample_dataset),
            "--carve-val-fraction",
            "0.5",
            "--dry-run",
        ]
    )
    assert code == 0


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


def test_cli_strict_apply_reports_remaining_issues(
    sample_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_hygiene(
        _root: Path,
        *,
        apply_fixes: bool,
        dry_run: bool,
        default_names: dict[int, str] | None = None,
    ) -> dict:
        _ = default_names
        calls.append((apply_fixes, dry_run))
        if len(calls) == 1:
            return {
                "audit": {"issues": ["nc is not set."]},
                "proposed_changes": ["Set nc"],
            }
        return {"audit": {"issues": ["still broken"]}}

    monkeypatch.setattr(
        "ml_pipeline.cli_dataset_yaml.run_dataset_yaml_hygiene",
        fake_hygiene,
    )

    code = cli_main([str(sample_dataset), "--apply", "--strict"])

    assert code == 1
    assert "issues remain after apply" in capsys.readouterr().err
    assert calls == [(True, False), (False, True)]


def test_carve_fraction_invalid(sample_dataset: Path) -> None:
    with pytest.raises(ValueError, match="fraction"):
        carve_val_from_train(
            sample_dataset,
            fraction=1.5,
            seed=0,
            dry_run=True,
            move=False,
        )


def test_load_dataset_yaml_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing dataset.yaml"):
        load_dataset_yaml(tmp_path)


def test_parse_names_field_list_form() -> None:
    from ml_pipeline.dataset_yaml import parse_names_field

    assert parse_names_field(["a", "b"]) == {0: "a", 1: "b"}


def test_parse_names_field_dict_with_bad_key() -> None:
    from ml_pipeline.dataset_yaml import parse_names_field

    assert parse_names_field({"abc": "skip", 0: "Person"}) == {0: "Person"}


def test_parse_names_field_unknown_type_returns_empty() -> None:
    from ml_pipeline.dataset_yaml import parse_names_field

    assert parse_names_field(42) == {}
    assert parse_names_field(None) == {}


def test_audit_warns_on_unused_names_and_path_gap(tmp_path: Path) -> None:
    cfg = {
        "path": ".",
        "names": {0: "Person", 1: "Car", 7: "Tree"},
        "nc": 1,
    }
    audit = audit_dataset_yaml(cfg, {0})
    assert any("unused" in w for w in audit["warnings"])
    assert any("nc=1 does not match" in i for i in audit["issues"])


def test_audit_nc_not_integer() -> None:
    cfg = {"names": {0: "Person"}, "nc": "abc"}
    audit = audit_dataset_yaml(cfg, {0})
    assert any("not an integer" in i for i in audit["issues"])


def test_audit_warns_when_no_classes(tmp_path: Path) -> None:
    cfg = {"names": {0: "Person"}, "nc": 1}
    audit = audit_dataset_yaml(cfg, set())
    assert any("No class indices" in w for w in audit["warnings"])


def test_build_fixed_config_sets_path(tmp_path: Path) -> None:
    cfg = {"names": {0: "Person"}, "nc": 1, "path": ""}
    fixed, changes = build_fixed_config(cfg, {0})
    assert fixed["path"] == "."
    assert any("Set path" in c for c in changes)


def test_build_fixed_config_empty_class_ids(tmp_path: Path) -> None:
    cfg = {"names": {0: "Person"}, "nc": 1}
    fixed, _changes = build_fixed_config(cfg, set())
    assert fixed["nc"] == 1


def test_build_fixed_config_no_class_ids_falls_back_to_default_count(
    tmp_path: Path,
) -> None:
    cfg = {"path": ".", "train": "images/train"}
    fixed, changes = build_fixed_config(cfg, set())
    assert fixed["nc"] == 2
    assert any("Set nc" in c for c in changes)


def test_carve_val_missing_train_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train image"):
        carve_val_from_train(
            tmp_path,
            fraction=0.5,
            seed=0,
            dry_run=True,
            move=False,
        )


def test_carve_val_move_mode(sample_dataset: Path) -> None:
    before = list((sample_dataset / "images" / "train").glob("*.jpg"))
    carve_val_from_train(
        sample_dataset,
        fraction=0.5,
        seed=0,
        dry_run=False,
        move=True,
    )
    after = list((sample_dataset / "images" / "train").glob("*.jpg"))
    assert len(after) < len(before)


def test_carve_val_uses_separate_label_dir(tmp_path: Path) -> None:
    train_imgs = tmp_path / "images" / "train"
    train_lbls = tmp_path / "labels" / "train"
    train_imgs.mkdir(parents=True)
    train_lbls.mkdir(parents=True)
    for stem in ("a", "b"):
        (train_imgs / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        (train_lbls / f"{stem}.txt").write_text(
            "0 0.1 0.1 0.2 0.2 0.3 0.3\n", encoding="utf-8"
        )
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "Person"},
                "nc": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    train_imgs_with_txt = tmp_path / "images" / "train"
    for stem in ("a", "b"):
        (train_imgs_with_txt / f"{stem}.txt").write_text("placeholder", encoding="utf-8")

    report = carve_val_from_train(
        tmp_path,
        fraction=0.5,
        seed=0,
        dry_run=False,
        move=False,
    )
    assert report["selected_count"] >= 1


def test_write_report_creates_parent_dirs(tmp_path: Path) -> None:
    from ml_pipeline.dataset_yaml import write_report

    target = tmp_path / "nested" / "report.json"
    write_report(target, {"ok": True})
    payload = target.read_text(encoding="utf-8")
    assert "ok" in payload


def test_label_paths_prefers_labels_directory(tmp_path: Path) -> None:
    train_imgs = tmp_path / "images" / "train"
    train_lbls = tmp_path / "labels" / "train"
    train_imgs.mkdir(parents=True)
    train_lbls.mkdir(parents=True)
    (train_lbls / "a.txt").write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")

    paths = _label_paths_for_split(tmp_path, "train", train_imgs)

    assert paths == [train_lbls / "a.txt"]


def test_label_paths_falls_back_to_image_dir_txt(tmp_path: Path) -> None:
    train_imgs = tmp_path / "images" / "train"
    train_imgs.mkdir(parents=True)
    (train_imgs / "a.txt").write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")

    paths = _label_paths_for_split(tmp_path, "train", train_imgs)

    assert paths == [train_imgs / "a.txt"]


def test_collect_class_ids_skips_blank_label_lines(tmp_path: Path) -> None:
    train_imgs = tmp_path / "images" / "train"
    train_imgs.mkdir(parents=True)
    (train_imgs / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (train_imgs / "a.txt").write_text("0 0.1 0.1 0.2 0.2\n\n  \n", encoding="utf-8")
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images/train"}, sort_keys=False),
        encoding="utf-8",
    )

    assert collect_class_ids(tmp_path) == {0}


def test_audit_warns_when_nc_below_max_class_id() -> None:
    cfg = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Person", 1: "Car"},
        "nc": 1,
    }
    audit = audit_dataset_yaml(cfg, {0, 1})
    assert any("below max(class id)+1" in warning for warning in audit["warnings"])


def test_build_fixed_config_adds_default_split_paths() -> None:
    cfg = {"path": ".", "names": {0: "Person"}, "nc": 1}
    fixed, changes = build_fixed_config(cfg, {0})
    assert fixed["train"] == "images/train"
    assert fixed["val"] == "images/val"
    assert fixed["test"] == "images/test"
    assert any("Added train" in change for change in changes)


def test_carve_val_uses_labels_train_when_label_not_in_image_dir(tmp_path: Path) -> None:
    train_imgs = tmp_path / "images" / "train"
    train_lbls = tmp_path / "labels" / "train"
    train_imgs.mkdir(parents=True)
    train_lbls.mkdir(parents=True)
    (train_imgs / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (train_lbls / "a.txt").write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    (train_imgs / "a.txt").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "Person"},
                "nc": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = carve_val_from_train(
        tmp_path,
        fraction=0.5,
        seed=0,
        dry_run=False,
        move=False,
    )

    assert report["selected_count"] == 1
    assert (tmp_path / "labels" / "val" / "a.txt").exists()
