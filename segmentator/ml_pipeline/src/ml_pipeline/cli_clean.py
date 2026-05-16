"""CLI entrypoint for dataset quarantine and Ultralytics runs cleanup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml_pipeline.clean import CleanConfig, run_clean


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-clean",
        description=(
            "Quarantine or delete dataset samples from train/val/test splits, "
            "and remove old Ultralytics run directories under a project folder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Quarantine worst val images from error analysis:\n"
            "    gdpr-yolo-clean <dataset-root> --val --from-errors errors.json --yes\n"
            "  Remove duplicate Ultralytics run folders, keep newest per prefix:\n"
            "    gdpr-yolo-clean --runs-dir <runs-root> --run-prefix yolo26n_seg_v1 "
            "--keep-latest --yes\n"
        ),
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=None,
        help="Dataset root with dataset.yaml (optional when only cleaning runs).",
    )

    split_group = parser.add_argument_group("dataset splits")
    split_group.add_argument(
        "--train",
        action="store_true",
        help="Apply to the train split.",
    )
    split_group.add_argument(
        "--val",
        "--validation",
        action="store_true",
        dest="val",
        help="Apply to the val split.",
    )
    split_group.add_argument(
        "--test",
        action="store_true",
        help="Apply to the test split.",
    )

    select_group = parser.add_argument_group("dataset selection")
    select_group.add_argument(
        "--stems",
        metavar="STEM",
        default=None,
        help="Comma-separated image stems to quarantine or delete.",
    )
    select_group.add_argument(
        "--from-errors",
        type=Path,
        metavar="REPORT.json",
        default=None,
        help="Read stems from summary.worst_images in an error-analysis JSON report.",
    )
    select_group.add_argument(
        "--worst",
        type=int,
        default=None,
        metavar="N",
        help="With --from-errors, limit to the first N worst images.",
    )
    select_group.add_argument(
        "--all-in-split",
        action="store_true",
        help="Target every image in the selected splits (requires --yes).",
    )

    dataset_action = parser.add_argument_group("dataset action")
    dataset_action.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help=(
            "Directory for quarantined files (default: "
            "<dataset-root>/_quarantine/<UTC-timestamp>/)."
        ),
    )
    dataset_action.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete files instead of moving them (requires --yes).",
    )
    dataset_action.add_argument(
        "--cache",
        action="store_true",
        help="Remove Ultralytics label cache files under labels/**/*.cache.",
    )

    runs_group = parser.add_argument_group("Ultralytics runs")
    runs_group.add_argument(
        "--runs-dir",
        type=Path,
        metavar="DIR",
        default=None,
        help="Project directory containing train/val run subfolders.",
    )
    runs_group.add_argument(
        "--run-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Remove run folders named PREFIX or PREFIX-N (repeatable). "
            "Example: yolo26n_seg_v1 matches yolo26n_seg_v1-3."
        ),
    )
    runs_group.add_argument(
        "--run-names",
        default=None,
        metavar="NAME,...",
        help="Remove explicit run folder names (comma-separated).",
    )
    runs_group.add_argument(
        "--runs-all",
        action="store_true",
        help="Remove every subdirectory under --runs-dir except names passed to --keep.",
    )
    runs_group.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="NAME",
        help="Do not remove this run folder name (repeatable).",
    )
    runs_group.add_argument(
        "--keep-latest",
        action="store_true",
        help="With --run-prefix, keep the newest matching folder and remove the rest.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without changing files.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive operations.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the JSON action report to this path.",
    )

    args = parser.parse_args(argv)

    splits: list[str] = []
    if args.train:
        splits.append("train")
    if args.val:
        splits.append("val")
    if args.test:
        splits.append("test")

    config = CleanConfig(
        dataset_root=args.dataset_root,
        splits=splits,
        stems=_parse_csv_list(args.stems),
        from_errors=args.from_errors,
        worst_n=args.worst,
        all_in_split=args.all_in_split,
        quarantine_dir=args.quarantine_dir,
        action="delete" if args.delete else "move",
        remove_label_cache=args.cache,
        runs_dir=args.runs_dir,
        run_prefixes=list(args.run_prefix or []),
        run_names=_parse_csv_list(args.run_names),
        runs_all=args.runs_all,
        keep_run_names=list(args.keep or []),
        keep_latest_per_prefix=args.keep_latest,
        dry_run=args.dry_run,
        yes=args.yes,
    )

    try:
        report = run_clean(config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
