"""CLI entrypoint for YOLO segmentation error analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml_pipeline.errors import ErrorConfig, run_error_analysis


def main() -> int:
    """
    Run false negative and false positive analysis on validation split.

    Requires ultralytics (install with: uv sync --extra train).
    """
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-analyze-errors",
        description=(
            "Analyze false negatives (FN) and false positives (FP) on the validation split. "
            "Compares ground truth labels with predictions to identify images needing more labels. "
            "Requires ultralytics (install with: uv sync --extra train)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to dataset root containing dataset.yaml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained weights (best.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for prediction (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (e.g. cpu, 0, 0,1)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project directory for saving analysis results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="error_analysis",
        help="Run name (default: error_analysis)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Path to save JSON error report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running analysis",
    )

    args = parser.parse_args()

    dataset_yaml = args.dataset_root / "dataset.yaml"
    if not dataset_yaml.exists():
        dataset_yaml = args.dataset_root
        if not dataset_yaml.exists():
            print(
                f"Error: dataset.yaml not found at {args.dataset_root} or {args.dataset_root / 'dataset.yaml'}",
                file=sys.stderr,
            )
            return 1

    config = ErrorConfig(
        weights=args.weights,
        dataset_yaml=dataset_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_threshold=args.iou_threshold,
        device=args.device,
        project=args.project,
        name=args.name,
        report_json=args.report_json,
    )

    try:
        report = run_error_analysis(config, dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(f"Error analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
