from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

SplitName = str
NormalizeMode = Literal["symlink", "copy", "move"]


@dataclass
class SplitAudit:
    split: SplitName
    image_dir: Path
    labels_out_dir: Path
    image_paths: list[Path] = field(default_factory=list)
    txt_in_image_dir: list[Path] = field(default_factory=list)
    txt_in_labels_dir: list[Path] = field(default_factory=list)
    images_missing_label: list[str] = field(default_factory=list)
    labels_missing_image: list[str] = field(default_factory=list)
    labels_created: list[str] = field(default_factory=list)
    labels_skipped_existing: list[str] = field(default_factory=list)
    labels_failed: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "image_dir": str(self.image_dir),
            "labels_out_dir": str(self.labels_out_dir),
            "image_count": len(self.image_paths),
            "txt_in_image_dir_count": len(self.txt_in_image_dir),
            "txt_in_labels_dir_count": len(self.txt_in_labels_dir),
            "images_missing_label": self.images_missing_label,
            "labels_missing_image": self.labels_missing_image,
            "labels_created": self.labels_created,
            "labels_skipped_existing": self.labels_skipped_existing,
            "labels_failed": [{"stem": s, "error": e} for s, e in self.labels_failed],
        }


def _resolve_split_image_dir(dataset_root: Path, train_val_test_path: str) -> Path:
    p = Path(train_val_test_path)
    if p.is_absolute():
        return p.resolve()
    return (dataset_root / p).resolve()


def load_split_dirs(dataset_root: Path) -> dict[SplitName, Path]:
    """Return split name -> absolute image directory from dataset.yaml."""
    cfg_path = dataset_root / "dataset.yaml"
    if not cfg_path.is_file():
        msg = f"Missing dataset.yaml at {cfg_path}"
        raise FileNotFoundError(msg)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = dataset_root
    path_field = cfg.get("path")
    if path_field and str(path_field).strip() not in (".", ""):
        p = Path(str(path_field))
        base = (dataset_root / p).resolve() if not p.is_absolute() else p.resolve()
    splits: dict[SplitName, Path] = {}
    for key in ("train", "val", "test"):
        rel = cfg.get(key)
        if not rel:
            continue
        img_dir = _resolve_split_image_dir(base, str(rel))
        split_name = img_dir.name
        splits[split_name] = img_dir
    if not splits:
        msg = "dataset.yaml has no train, val, or test image directory entries"
        raise ValueError(msg)
    return splits


def _find_image_for_stem(stem: str, image_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return out


def audit_split(dataset_root: Path, split: SplitName, image_dir: Path) -> SplitAudit:
    labels_out_dir = (dataset_root / "labels" / split).resolve()
    txt_in_image = sorted(image_dir.glob("*.txt")) if image_dir.is_dir() else []
    txt_in_labels = sorted(labels_out_dir.glob("*.txt")) if labels_out_dir.is_dir() else []

    stems_from_txts: set[str] = set()
    for p in txt_in_image:
        stems_from_txts.add(p.stem)
    for p in txt_in_labels:
        stems_from_txts.add(p.stem)

    images_missing_label: list[str] = []
    for img_path in _list_images(image_dir):
        if img_path.stem not in stems_from_txts:
            images_missing_label.append(img_path.stem)
    images_missing_label.sort()

    labels_missing_image: list[str] = []
    for stem in sorted(stems_from_txts):
        if _find_image_for_stem(stem, image_dir) is None:
            labels_missing_image.append(stem)

    return SplitAudit(
        split=split,
        image_dir=image_dir,
        labels_out_dir=labels_out_dir,
        image_paths=_list_images(image_dir),
        txt_in_image_dir=txt_in_image,
        txt_in_labels_dir=txt_in_labels,
        images_missing_label=images_missing_label,
        labels_missing_image=labels_missing_image,
    )


def _ensure_symlink(src: Path, dest: Path, dry_run: bool) -> None:
    rel = Path(os.path.relpath(src, dest.parent))
    if dry_run:
        return
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(rel, target_is_directory=False)


def _ensure_copy(src: Path, dest: Path, dry_run: bool) -> None:
    if dry_run:
        return
    shutil.copy2(src, dest)


def _ensure_move(src: Path, dest: Path, dry_run: bool) -> None:
    if dry_run:
        return
    shutil.move(str(src), str(dest))


def normalize_split(
    audit: SplitAudit,
    *,
    mode: NormalizeMode,
    dry_run: bool,
    force: bool,
) -> SplitAudit:
    """Populate labels/<split>/ from co-located image-side .txt files."""
    if not audit.image_dir.is_dir():
        return audit
    if dry_run:
        return audit
    if not audit.txt_in_image_dir:
        return audit
    audit.labels_out_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in audit.txt_in_image_dir:
        stem = txt_path.stem
        dest = audit.labels_out_dir / txt_path.name
        if dest.exists() or dest.is_symlink():
            if not force:
                audit.labels_skipped_existing.append(stem)
                continue
            if not dry_run:
                dest.unlink()

        try:
            if mode == "symlink":
                _ensure_symlink(txt_path.resolve(), dest, dry_run)
            elif mode == "copy":
                _ensure_copy(txt_path, dest, dry_run)
            elif mode == "move":
                _ensure_move(txt_path, dest, dry_run)
            else:
                msg = f"Unknown mode {mode!r}"
                raise ValueError(msg)
            audit.labels_created.append(stem)
        except OSError as e:
            audit.labels_failed.append((stem, str(e)))

    return audit


def run_audit_and_normalize(
    dataset_root: Path,
    *,
    mode: NormalizeMode,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    split_dirs = load_split_dirs(dataset_root)
    audits: list[SplitAudit] = []
    for split_name, image_dir in sorted(split_dirs.items()):
        audit = audit_split(dataset_root, split_name, image_dir)
        audit = normalize_split(audit, mode=mode, dry_run=dry_run, force=force)
        audits.append(audit)

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "mode": mode,
        "dry_run": dry_run,
        "splits": {a.split: a.to_dict() for a in audits},
    }
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
