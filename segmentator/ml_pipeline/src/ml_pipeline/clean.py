"""Move or delete dataset samples and Ultralytics run directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ml_pipeline.layout import IMAGE_EXTENSIONS, load_split_dirs

Action = Literal["move", "delete"]
SPLIT_ALIASES = {"validation": "val", "validate": "val"}


@dataclass
class CleanConfig:
    """Configuration for gdpr-yolo-clean."""

    dataset_root: Path | None = None
    splits: list[str] = field(default_factory=list)
    stems: list[str] = field(default_factory=list)
    from_errors: Path | None = None
    worst_n: int | None = None
    all_in_split: bool = False
    quarantine_dir: Path | None = None
    action: Action = "move"
    remove_label_cache: bool = False
    runs_dir: Path | None = None
    run_prefixes: list[str] = field(default_factory=list)
    run_names: list[str] = field(default_factory=list)
    runs_all: bool = False
    keep_run_names: list[str] = field(default_factory=list)
    keep_latest_per_prefix: bool = False
    dry_run: bool = False
    yes: bool = False


def normalize_split_name(name: str) -> str:
    key = name.strip().lower()
    return SPLIT_ALIASES.get(key, key)


def stems_from_errors_report(path: Path, *, worst_n: int | None = None) -> list[str]:
    """Return image stems listed in an error-analysis JSON report."""
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    worst = report.get("summary", {}).get("worst_images") or []
    if worst_n is not None:
        worst = worst[:worst_n]
    stems: list[str] = []
    for entry in worst:
        if not isinstance(entry, dict):
            continue
        image_name = entry.get("image")
        if not image_name:
            continue
        stems.append(Path(str(image_name)).stem)
    return stems


def _find_image_for_stem(stem: str, image_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _paths_for_stem(dataset_root: Path, split: str, stem: str, image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    image_path = _find_image_for_stem(stem, image_dir)
    if image_path is not None:
        paths.append(image_path)
    co_located = image_dir / f"{stem}.txt"
    if co_located.is_file():
        paths.append(co_located)
    label_path = dataset_root / "labels" / split / f"{stem}.txt"
    if label_path.is_file():
        paths.append(label_path)
    return paths


def _default_quarantine_dir(dataset_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return dataset_root / "_quarantine" / stamp


def _relocate(path: Path, dest: Path, *, action: Action, dry_run: bool) -> None:
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if action == "delete":
        if path.is_symlink() or path.is_file():
            path.unlink()
        return
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.move(str(path), str(dest))


def quarantine_stems(
    dataset_root: Path,
    splits: list[str],
    stems: list[str],
    *,
    quarantine_dir: Path | None,
    action: Action,
    dry_run: bool,
) -> dict[str, Any]:
    """Move or delete image and label files for the given stems in each split."""
    split_dirs = load_split_dirs(dataset_root)
    unknown = [s for s in splits if s not in split_dirs]
    if unknown:
        msg = f"Unknown split(s) {unknown!r}. Available: {sorted(split_dirs)}"
        raise ValueError(msg)

    target_root = quarantine_dir or _default_quarantine_dir(dataset_root)
    moved: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for split in splits:
        image_dir = split_dirs[split]
        for stem in stems:
            paths = _paths_for_stem(dataset_root, split, stem, image_dir)
            if not paths:
                missing.append({"split": split, "stem": stem})
                continue
            for src in paths:
                if action == "move":
                    dest = target_root / split / src.name
                else:
                    dest = src  # unused for delete
                _relocate(src, dest, action=action, dry_run=dry_run)
                moved.append(
                    {
                        "split": split,
                        "stem": stem,
                        "source": str(src),
                        "destination": str(dest) if action == "move" else "",
                        "action": action,
                    }
                )

    return {
        "dataset_root": str(dataset_root),
        "quarantine_dir": str(target_root) if action == "move" else None,
        "action": action,
        "moved": moved,
        "missing": missing,
    }


def stems_in_splits(dataset_root: Path, splits: list[str]) -> list[str]:
    """Collect all image stems present in the selected splits."""
    split_dirs = load_split_dirs(dataset_root)
    stems: set[str] = set()
    for split in splits:
        image_dir = split_dirs[split]
        if not image_dir.is_dir():
            continue
        for path in image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                stems.add(path.stem)
    return sorted(stems)


def remove_label_caches(dataset_root: Path, *, dry_run: bool) -> list[str]:
    """Remove Ultralytics label cache files under labels/."""
    labels_root = dataset_root / "labels"
    removed: list[str] = []
    if not labels_root.is_dir():
        return removed
    for cache_file in labels_root.rglob("*.cache"):
        removed.append(str(cache_file))
        if not dry_run:
            cache_file.unlink()
    return removed


def _run_dir_matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}-")


def list_run_dirs_to_remove(
    runs_dir: Path,
    *,
    run_prefixes: list[str],
    run_names: list[str],
    runs_all: bool,
    keep_run_names: list[str],
    keep_latest_per_prefix: bool,
) -> tuple[list[Path], list[Path]]:
    """Return (to_remove, kept) immediate child directories under runs_dir."""
    if not runs_dir.is_dir():
        msg = f"Runs directory does not exist: {runs_dir}"
        raise FileNotFoundError(msg)

    children = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep_set = {name for name in keep_run_names}

    if runs_all:
        to_remove = [p for p in children if p.name not in keep_set]
        kept = [p for p in children if p.name in keep_set]
        return to_remove, kept

    selected: set[Path] = set()
    for name in run_names:
        candidate = runs_dir / name
        if candidate.is_dir():
            selected.add(candidate)

    for prefix in run_prefixes:
        matches = [p for p in children if _run_dir_matches_prefix(p.name, prefix)]
        if not matches:
            continue
        if keep_latest_per_prefix:
            latest = max(matches, key=lambda p: p.stat().st_mtime)
            keep_set.add(latest.name)
        for match in matches:
            selected.add(match)

    to_remove = sorted(
        [p for p in selected if p.name not in keep_set],
        key=lambda p: p.name,
    )
    kept = sorted([p for p in selected if p.name in keep_set], key=lambda p: p.name)
    return to_remove, kept


def clean_run_dirs(
    runs_dir: Path,
    to_remove: list[Path],
    *,
    dry_run: bool,
) -> list[str]:
    removed: list[str] = []
    for path in to_remove:
        removed.append(str(path))
        if dry_run:
            continue
        shutil.rmtree(path)
    return removed


def validate_clean_config(config: CleanConfig) -> None:
    """Validate flags and raise when the invocation is ambiguous or unsafe."""
    dataset_requested = bool(
        config.splits
        or config.stems
        or config.from_errors
        or config.all_in_split
        or config.remove_label_cache
    )
    runs_requested = bool(
        config.runs_dir
        and (
            config.run_prefixes
            or config.run_names
            or config.runs_all
        )
    )

    if not dataset_requested and not runs_requested:
        msg = (
            "Nothing to clean. Select dataset splits with --train/--val/--test and "
            "a target (--stems, --from-errors, or --all-in-split), and/or select "
            "runs with --runs-dir and --run-prefix/--run-names/--runs-all."
        )
        raise ValueError(msg)

    if dataset_requested and config.dataset_root is None:
        msg = "dataset_root is required for dataset cleaning."
        raise ValueError(msg)

    if config.dataset_root is not None and not config.dataset_root.is_dir():
        msg = f"Dataset root is not a directory: {config.dataset_root}"
        raise FileNotFoundError(msg)

    if config.from_errors is not None and not config.from_errors.is_file():
        msg = f"Error report not found: {config.from_errors}"
        raise FileNotFoundError(msg)

    selection_count = sum(
        [
            bool(config.stems),
            config.from_errors is not None,
            config.all_in_split,
        ]
    )
    if config.splits and selection_count == 0 and not config.remove_label_cache:
        msg = (
            "Split flags were given without a selection mode. Use --stems, "
            "--from-errors, or --all-in-split."
        )
        raise ValueError(msg)
    if selection_count > 1:
        msg = "Use only one of --stems, --from-errors, or --all-in-split."
        raise ValueError(msg)

    if config.all_in_split and not config.yes:
        msg = "--all-in-split requires --yes."
        raise ValueError(msg)

    if config.action == "delete" and not config.yes:
        msg = "Permanent delete (--delete) requires --yes."
        raise ValueError(msg)

    if runs_requested:
        if config.runs_all and (config.run_prefixes or config.run_names):
            msg = "Use either --runs-all or explicit --run-prefix/--run-names, not both."
            raise ValueError(msg)
        destructive_runs = config.runs_all or bool(config.run_prefixes or config.run_names)
        if destructive_runs and not config.yes:
            msg = "Removing run directories requires --yes."
            raise ValueError(msg)


def run_clean(config: CleanConfig) -> dict[str, Any]:
    """Execute dataset quarantine and/or runs cleanup."""
    validate_clean_config(config)
    report: dict[str, Any] = {"dry_run": config.dry_run}

    if config.dataset_root is not None:
        splits = [normalize_split_name(s) for s in config.splits]
        stems = list(config.stems)
        if config.from_errors is not None:
            stems = stems_from_errors_report(
                config.from_errors, worst_n=config.worst_n
            )
        if config.all_in_split:
            stems = stems_in_splits(config.dataset_root, splits)

        dataset_report: dict[str, Any] = {}
        if splits and stems:
            dataset_report["samples"] = quarantine_stems(
                config.dataset_root,
                splits,
                stems,
                quarantine_dir=config.quarantine_dir,
                action=config.action,
                dry_run=config.dry_run,
            )
        if config.remove_label_cache:
            dataset_report["label_caches_removed"] = remove_label_caches(
                config.dataset_root, dry_run=config.dry_run
            )
        if dataset_report:
            report["dataset"] = dataset_report

    if config.runs_dir is not None and (
        config.run_prefixes or config.run_names or config.runs_all
    ):
        to_remove, kept = list_run_dirs_to_remove(
            config.runs_dir,
            run_prefixes=config.run_prefixes,
            run_names=config.run_names,
            runs_all=config.runs_all,
            keep_run_names=config.keep_run_names,
            keep_latest_per_prefix=config.keep_latest_per_prefix,
        )
        removed = clean_run_dirs(
            config.runs_dir, to_remove, dry_run=config.dry_run
        )
        report["runs"] = {
            "runs_dir": str(config.runs_dir),
            "removed": removed,
            "kept": [str(p) for p in kept],
        }

    return report
