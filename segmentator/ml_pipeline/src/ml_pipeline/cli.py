from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml_pipeline.layout import run_audit_and_normalize, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-normalize-labels",
        description=(
            "Audit a YOLO dataset tree and mirror co-located label .txt files "
            "under labels/<split>/ for trainers that expect parallel images/ and labels/."
        ),
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory containing dataset.yaml, images/, and (often) co-located .txt labels.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy", "move"),
        default="symlink",
        help=(
            "How to place each label under labels/<split>/: symlink (default, single file), "
            "copy (duplicate; leaves originals in images/), or move (removes from images/)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without creating or changing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files under labels/<split>/ when they already exist.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the audit JSON report to this path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any image lacks a label or any label lacks a matching image.",
    )
    args = parser.parse_args(argv)

    report = run_audit_and_normalize(
        args.dataset_root,
        mode=args.mode,
        dry_run=args.dry_run,
        force=args.force,
    )

    if args.report_json:
        write_report(args.report_json, report)

    print(json.dumps(report, indent=2))

    if args.strict:
        for split, data in report["splits"].items():
            if data["images_missing_label"] or data["labels_missing_image"]:
                print(
                    f"Strict mode: split {split!r} has missing pairs.",
                    file=sys.stderr,
                )
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
