from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any
import yaml
from ml_pipeline.layout import IMAGE_EXTENSIONS, load_split_dirs

DEFAULT_CLASS_NAMES: dict[int, str] = {0: "Person", 1: "Car"}


def load_dataset_yaml(dataset_root: Path) -> dict[str, Any]:
    cfg_path = dataset_root / "dataset.yaml"
    if not cfg_path.is_file():
        msg = f"Missing dataset.yaml at {cfg_path}"
        raise FileNotFoundError(msg)
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_names_field(raw: Any) -> dict[int, str]:
    if not raw:
        return {}
    if isinstance(raw, list):
        return {idx: str(name) for idx, name in enumerate(raw)}
    if isinstance(raw, dict):
        out: dict[int, str] = {}
        for key, value in raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            out[idx] = str(value)
        return out
    return {}


def _label_paths_for_split(dataset_root: Path, split: str, image_dir: Path) -> list[Path]:
    labels_dir = dataset_root / "labels" / split
    if labels_dir.is_dir():
        paths = sorted(labels_dir.glob("*.txt"))
        if paths:
            return paths
    if image_dir.is_dir():
        return sorted(image_dir.glob("*.txt"))
    return []


def collect_class_ids(dataset_root: Path) -> set[int]:
    """Union of class indices referenced in segmentation label files."""
    class_ids: set[int] = set()
    split_dirs = load_split_dirs(dataset_root)
    for split_name, image_dir in split_dirs.items():
        for label_path in _label_paths_for_split(dataset_root, split_name, image_dir):
            with label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    token = stripped.split()[0]
                    class_ids.add(int(float(token)))
    return class_ids


def audit_dataset_yaml(cfg: dict[str, Any], class_ids: set[int]) -> dict[str, Any]:
    names = parse_names_field(cfg.get("names"))
    nc = cfg.get("nc")
    issues: list[str] = []
    warnings: list[str] = []

    if not class_ids:
        warnings.append("No class indices found in label files.")

    missing_name_keys = sorted(cid for cid in class_ids if cid not in names)
    if missing_name_keys:
        issues.append(f"names missing entries for class ids: {missing_name_keys}")

    unused_names = sorted(k for k in names if class_ids and k not in class_ids)
    if unused_names:
        warnings.append(f"names defines unused class ids: {unused_names}")

    if nc is None:
        issues.append("nc is not set.")
    else:
        try:
            nc_int = int(nc)
        except (TypeError, ValueError):
            issues.append(f"nc is not an integer: {nc!r}")
        else:
            if names and nc_int != len(names):
                issues.append(f"nc={nc_int} does not match names count ({len(names)}).")
            if class_ids and nc_int <= max(class_ids):
                if nc_int != max(class_ids) + 1 and nc_int != len(class_ids):
                    warnings.append(
                        f"nc={nc_int} is below max(class id)+1 ({max(class_ids) + 1})."
                    )

    for split in ("train", "val", "test"):
        if split not in cfg:
            warnings.append(f"split path {split!r} is not defined in dataset.yaml.")

    return {
        "class_ids_in_labels": sorted(class_ids),
        "names": names,
        "nc": nc,
        "issues": issues,
        "warnings": warnings,
    }


def build_fixed_config(
    cfg: dict[str, Any],
    class_ids: set[int],
    *,
    default_names: dict[int, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return a copy of cfg with nc and names aligned to label files."""
    defaults = default_names or DEFAULT_CLASS_NAMES
    fixed = dict(cfg)
    changes: list[str] = []

    names = parse_names_field(cfg.get("names"))
    for cid in sorted(class_ids):
        if cid not in names:
            fallback = defaults.get(cid, f"class_{cid}")
            names[cid] = fallback
            changes.append(f"Added names[{cid}] = {fallback!r}")

    if class_ids:
        expected_nc = max(class_ids) + 1
        if len(names) > expected_nc:
            expected_nc = len(names)
    else:
        expected_nc = len(names) if names else len(defaults)

    if fixed.get("nc") != expected_nc:
        changes.append(f"Set nc from {fixed.get('nc')!r} to {expected_nc}")
        fixed["nc"] = expected_nc

    fixed["names"] = {k: names[k] for k in sorted(names)}

    if fixed.get("path") in (None, ""):
        fixed["path"] = "."
        changes.append("Set path to '.'")

    for split, default_rel in (
        ("train", "images/train"),
        ("val", "images/val"),
        ("test", "images/test"),
    ):
        if split not in fixed:
            fixed[split] = default_rel
            changes.append(f"Added {split}: {default_rel!r}")

    return fixed, changes


def write_dataset_yaml(dataset_root: Path, cfg: dict[str, Any]) -> Path:
    cfg_path = dataset_root / "dataset.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg_path


def run_dataset_yaml_hygiene(
    dataset_root: Path,
    *,
    apply_fixes: bool,
    dry_run: bool,
    default_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    cfg = load_dataset_yaml(dataset_root)
    class_ids = collect_class_ids(dataset_root)
    audit = audit_dataset_yaml(cfg, class_ids)
    fixed_cfg, changes = build_fixed_config(cfg, class_ids, default_names=default_names)

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "dry_run": dry_run,
        "apply_fixes": apply_fixes,
        "audit": audit,
        "proposed_changes": changes,
        "fixed_config": fixed_cfg if apply_fixes or dry_run else None,
    }

    if apply_fixes and not dry_run:
        write_dataset_yaml(dataset_root, fixed_cfg)
        report["written"] = str(dataset_root / "dataset.yaml")

    return report


def _find_image_for_stem(stem: str, image_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def carve_val_from_train(
    dataset_root: Path,
    *,
    fraction: float,
    seed: int,
    dry_run: bool,
    move: bool,
) -> dict[str, Any]:
    """Move or copy a deterministic fraction of labeled train samples into val."""
    if not 0.0 < fraction < 1.0:
        msg = "fraction must be between 0 and 1"
        raise ValueError(msg)

    dataset_root = dataset_root.expanduser().resolve()
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"

    if not train_dir.is_dir():
        msg = f"Missing train image directory: {train_dir}"
        raise FileNotFoundError(msg)

    stems: list[str] = []
    for txt in sorted(train_dir.glob("*.txt")):
        if _find_image_for_stem(txt.stem, train_dir) is not None:
            stems.append(txt.stem)

    rng = random.Random(seed)
    rng.shuffle(stems)
    count = max(1, int(len(stems) * fraction)) if stems else 0
    selected = sorted(stems[:count])

    actions: list[dict[str, str]] = []
    for stem in selected:
        image = _find_image_for_stem(stem, train_dir)
        assert image is not None
        label_src = train_dir / f"{stem}.txt"
        if not label_src.is_file() and (train_labels / f"{stem}.txt").is_file():
            label_src = train_labels / f"{stem}.txt"

        image_dest = val_dir / image.name
        label_dest = val_labels / f"{stem}.txt"

        if not dry_run:
            val_dir.mkdir(parents=True, exist_ok=True)
            val_labels.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(image), str(image_dest))
                if label_src.is_file():
                    shutil.move(str(label_src), str(label_dest))
            else:
                shutil.copy2(image, image_dest)
                if label_src.is_file():
                    shutil.copy2(label_src, label_dest)

        actions.append(
            {
                "stem": stem,
                "image": str(image_dest),
                "label": str(label_dest),
                "mode": "move" if move else "copy",
            }
        )

    return {
        "dataset_root": str(dataset_root),
        "fraction": fraction,
        "seed": seed,
        "dry_run": dry_run,
        "move": move,
        "train_labeled_count": len(stems),
        "selected_count": len(selected),
        "selected_stems": selected,
        "actions": actions,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
