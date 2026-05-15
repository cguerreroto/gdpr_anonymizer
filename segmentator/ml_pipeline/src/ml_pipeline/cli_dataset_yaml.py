from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml_pipeline.dataset_yaml import (
    carve_val_from_train,
    run_dataset_yaml_hygiene,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-fix-dataset-yaml",
        description=(
            "Audit dataset.yaml against label class ids, optionally set nc and names, "
            "and optionally carve a fraction of labeled train samples into val."
        ),
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory containing dataset.yaml and images/ (and usually labels/).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write an updated dataset.yaml when audit reports fixable issues.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report proposed changes without writing dataset.yaml.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the JSON report to this path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when audit issues remain after a dry audit.",
    )
    parser.add_argument(
        "--carve-val-fraction",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Copy or move this fraction of labeled train items into val (deterministic).",
    )
    parser.add_argument(
        "--carve-seed",
        type=int,
        default=0,
        help="Random seed for --carve-val-fraction stem selection.",
    )
    parser.add_argument(
        "--carve-move",
        action="store_true",
        help="Move files into val instead of copying (only with --carve-val-fraction).",
    )
    args = parser.parse_args(argv)

    report = run_dataset_yaml_hygiene(
        args.dataset_root,
        apply_fixes=args.apply,
        dry_run=args.dry_run or not args.apply,
    )

    if args.carve_val_fraction is not None:
        carve_report = carve_val_from_train(
            args.dataset_root,
            fraction=args.carve_val_fraction,
            seed=args.carve_seed,
            dry_run=args.dry_run or not args.apply,
            move=args.carve_move,
        )
        report["carve_val"] = carve_report

    if args.report_json:
        write_report(args.report_json, report)

    print(json.dumps(report, indent=2))

    if args.strict:
        issues = report.get("audit", {}).get("issues", [])
        if issues and not args.apply:
            print("Strict mode: unresolved dataset.yaml issues.", file=sys.stderr)
            return 1
        if args.apply and not args.dry_run:
            post = run_dataset_yaml_hygiene(
                args.dataset_root,
                apply_fixes=False,
                dry_run=True,
            )
            if post.get("audit", {}).get("issues"):
                print("Strict mode: issues remain after apply.", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
